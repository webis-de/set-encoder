import re
from argparse import ArgumentParser
from itertools import islice
from pathlib import Path

import torch
import ir_datasets
import numpy as np
import pandas as pd
import sklearn.cluster as sc
import transformers
from lightning_ir.data.dataset import DASHED_DATASET_MAP
from tqdm import tqdm


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def load_run(run_path: Path, depth: int, add_qrels: bool) -> pd.DataFrame:
    run = pd.read_csv(
        run_path,
        sep=r"\s+",
        names=["query_id", "Q0", "doc_id", "rank", "score", "run_id"],
        dtype={"query_id": str, "doc_id": str},
    ).drop(columns=["Q0", "run_id"])
    run = run.loc[run["rank"] <= depth]

    dataset = ir_datasets.load(
        DASHED_DATASET_MAP[
            re.sub(r"__.+__", "", run_path.name[: -(len("".join(run_path.suffixes)))])
        ]
    )
    qrels = pd.DataFrame(dataset.qrels_iter())
    docs_store = dataset.docs_store()
    queries = pd.DataFrame(dataset.queries_iter()).rename(columns={"text": "query"})
    run = run.merge(
        qrels, on=["query_id", "doc_id"], how="outer" if add_qrels else "left"
    )
    run = run.merge(queries, on="query_id", how="left")
    run["text"] = run["doc_id"].map(lambda x: docs_store.get(x).default_text())
    return run


def cluster_docs(
    df: pd.DataFrame,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
) -> pd.DataFrame:
    all_embeddings = []
    pg = tqdm(total=len(df))
    for docs in batched(df["text"].tolist(), n=256):
        inputs = tokenizer(docs, return_tensors="pt", padding=True, truncation=True)
        with torch.inference_mode():
            embeddings = model(**inputs.to(model.device))["pooler_output"]
        all_embeddings.append(embeddings.cpu())
        pg.update(len(docs))

    embeddings = torch.cat(all_embeddings).numpy()
    clusters = np.full((embeddings.shape[0],), fill_value=-1, dtype=np.int32)
    for _, query_df in tqdm(df.groupby("query_id")):
        all_idcs = query_df.index.values
        run_idcs = query_df.loc[~query_df["rank"].isna()].index.values
        clustering = sc.KMeans(10, random_state=0)
        clustering.fit(embeddings[run_idcs])
        c = clustering.predict(embeddings[all_idcs])
        clusters[all_idcs] = c
    assert not np.any(clusters == -1)
    df["iteration"] = clusters
    return df


def main(args=None):
    parser = ArgumentParser()

    parser.add_argument("--run_paths", type=Path, required=True, nargs="+")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--num_subtopics", type=int, default=10)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--save_qrels", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--lexical_threshold", type=float, default=0.5)

    args = parser.parse_args(args)

    model = transformers.AutoModel.from_pretrained(args.model_name_or_path)
    if torch.cuda.is_available():
        model = model.to("cuda")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    for run_path in args.run_paths:
        run = load_run(run_path, args.depth, args.save_qrels)
        run = cluster_docs(run, model, tokenizer)
        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = args.output_dir / run_path.name
        else:
            output_path = run_path
        if args.save_qrels:
            output_path = output_path.with_suffix(".qrels")
            run[["query_id", "iteration", "doc_id", "relevance"]].dropna(
                subset="relevance"
            ).astype({"relevance": int}).to_csv(
                output_path, sep=" ", header=False, index=False
            )
        else:
            output_path = output_path.with_stem(
                f"__subtopic__{run_path.stem}"
            ).with_suffix(".jsonl.gz")
            run.to_json(
                output_path,
                orient="records",
                lines=True,
                compression="gzip",
            )


if __name__ == "__main__":
    main()
