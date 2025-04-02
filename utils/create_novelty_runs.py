import re
from argparse import ArgumentParser
from itertools import islice
from pathlib import Path
from typing import Literal

import ir_datasets
import nltk
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

tqdm.pandas()

DASHED_DATASET_MAP = {dataset.replace("/", "-"): dataset for dataset in ir_datasets.registry._registered}


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def load_run(run_path: Path, depth: int) -> pd.DataFrame:
    run = pd.read_csv(
        run_path,
        sep=r"\s+",
        names=["query_id", "Q0", "doc_id", "rank", "score", "run_id"],
        dtype={"query_id": str, "doc_id": str},
    ).drop(columns=["Q0"])
    run = run.loc[run["rank"] <= depth]

    dataset = ir_datasets.load(
        DASHED_DATASET_MAP[re.sub(r"__.+__", "", run_path.name[: -(len("".join(run_path.suffixes)))])]
    )
    docs_store = dataset.docs_store()
    queries = pd.DataFrame(dataset.queries_iter()).rename(columns={"text": "query"})
    print("adding docs")
    run = run.merge(queries, on="query_id", how="left")
    doc_ids = run["doc_id"].drop_duplicates()
    docs = doc_ids.progress_map(lambda x: docs_store.get(x).default_text()).rename("text")
    docs = docs.to_frame().set_index(doc_ids)
    run = run.merge(docs, left_on="doc_id", right_index=True, how="left")
    return run


def load_qrels(qrel_id: str) -> pd.DataFrame:
    dataset = ir_datasets.load(qrel_id)
    qrels = pd.DataFrame(dataset.qrels_iter())
    qrels["text"] = qrels["doc_id"].map(lambda x: dataset.docs_store().get(x).default_text())
    return qrels


def cluster_docs(df: pd.DataFrame, threshold: float, distance: Literal["jaccard", "edit"]) -> pd.DataFrame:
    clusters = np.full((df.shape[0],), fill_value=-1, dtype=np.int32)
    for _, query_df in tqdm(df.groupby("query_id")):
        if query_df.shape[0] > 1_000 and "relevance" in query_df:
            query_df = query_df.loc[query_df.loc[:, "relevance"] > 0]
        if query_df.shape[0] > 5_000:
            continue
        idcs = query_df.index.values
        words = query_df["text"].map(nltk.word_tokenize).str[:512].values
        idx_a, idx_b = np.triu(np.ones((len(words), len(words))), k=1).nonzero()
        pairwise_distances = np.full((len(words), len(words)), 10e6)
        if distance == "jaccard":
            distances = np.array([[nltk.jaccard_distance(set(a), set(b)) for a, b in zip(words[idx_a], words[idx_b])]])
        elif distance == "edit":
            distances = np.array([nltk.edit_distance(a, b) for a, b in zip(words[idx_a], words[idx_b])])
        else:
            raise ValueError(f"Unknown distance: {distance}")
        pairwise_distances[idx_a, idx_b] = distances
        pairwise_distances[idx_b, idx_a] = distances
        ac = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="complete",
            distance_threshold=threshold,
        )
        g = ac.fit_predict(pairwise_distances)
        clusters[idcs] = g
    df["iteration"] = clusters
    return df


def main(args=None):
    parser = ArgumentParser()

    parser.add_argument("--run_paths", type=Path, nargs="*")
    parser.add_argument("--qrel_ids", type=str, nargs="*")
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--distance", choices=["jaccard", "edit"], default="jaccard")

    args = parser.parse_args(args)

    run_paths = args.run_paths or []
    for run_path in run_paths:
        run = load_run(run_path, args.depth)
        run = cluster_docs(run, args.threshold, args.distance)
        run["relevance"] = 1 + run["rank"].max() - run["rank"]
        output_path = run_path.with_stem(f"__novelty__{run_path.stem}")
        run.loc[:, ["query_id", "iteration", "doc_id", "rank", "score", "run_id"]].to_csv(
            output_path, sep=" ", header=False, index=False
        )
    qrel_ids = args.qrel_ids or []
    for qrel_id in qrel_ids:
        qrels = load_qrels(qrel_id)
        qrels = cluster_docs(qrels, args.threshold, args.distance)
        output_path = (args.output_dir / qrel_id.replace("/", "-")).with_suffix(".qrels")
        qrels[["query_id", "iteration", "doc_id", "relevance"]].dropna(subset="relevance").astype(
            {"relevance": int}
        ).to_csv(output_path, sep=" ", header=False, index=False)


if __name__ == "__main__":
    main()
