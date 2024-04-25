import json
import re
from argparse import ArgumentParser
from pathlib import Path

import ir_datasets
import torch
import pandas as pd
from lightning_ir.data.dataset import DASHED_DATASET_MAP
from tqdm import tqdm


def run_file_to_rank_llm(
    run_file: Path, depth: int, output_dir: Path | None, overwrite: bool
):
    if output_dir is None:
        output_dir = run_file.parent
    output_file = output_dir / run_file.with_suffix(".json").name
    if output_file.exists() and not overwrite:
        return
    dataset_name = run_file.name[: -(len("".join(run_file.suffixes)))]
    dataset_name = re.sub(r"__.*__", "", dataset_name)
    dataset = ir_datasets.load(DASHED_DATASET_MAP[dataset_name])
    queries = pd.DataFrame(dataset.queries_iter()).set_index("query_id")["text"]
    docs = dataset.docs_store()
    run_df = pd.read_csv(
        run_file,
        sep=r"\s+",
        header=None,
        names=["query_id", "Q0", "doc_id", "rank", "score", "system"],
        dtype={"query_id": str, "doc_id": str},
    ).drop(columns=["Q0", "system"])
    run_df = run_df.sort_values(by=["query_id", "score"], ascending=[True, False])
    data = []
    for query_id, group in tqdm(run_df.groupby("query_id")):
        query = queries.loc[query_id]
        group = group.sort_values(by="score", ascending=False).iloc[:depth]
        hits = []
        for idx, doc_id in enumerate(group["doc_id"]):
            doc = docs.get(doc_id).default_text()
            hits.append(
                {
                    "docid": doc_id,
                    "content": doc,
                    "qid": query_id,
                    "score": depth - idx,
                    "rank": idx + 1,
                }
            )
        data.append({"query": query, "hits": hits})
    with output_file.open("w") as f:
        json.dump(data, f)


def main(args=None):
    parser = ArgumentParser()

    parser.add_argument("--run_files", type=Path, default=None, nargs="+")
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(args)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_files is not None:
        for run_file in args.run_files:
            run_file_to_rank_llm(run_file, args.depth, args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()
