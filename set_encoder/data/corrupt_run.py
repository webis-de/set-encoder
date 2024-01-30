from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from ir_dataset_utils import DASHED_DATASET_MAP
import ir_datasets
import numpy as np

RUN_HEADER = ["query_id", "q0", "doc_id", "rank", "score", "system"]


def main(args=None):
    parser = ArgumentParser()

    parser.add_argument("--run_path", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--corruption", type=str, choices=["random", "best", "worst"])
    parser.add_argument("--depth", type=int, default=100)

    args = parser.parse_args(args)

    run = pd.read_csv(
        args.run_path,
        sep=r"\s+",
        names=RUN_HEADER,
        dtype={"query_id": str, "doc_id": str},
    )
    run = run.loc[run["rank"] <= args.depth]
    qrels = pd.DataFrame(
        ir_datasets.load(DASHED_DATASET_MAP[args.run_path.stem]).qrels_iter()
    ).drop(columns=["iteration"], errors="ignore")
    run = run.merge(
        qrels,
        left_on=["query_id", "doc_id"],
        right_on=["query_id", "doc_id"],
        how="left",
    )
    run["relevance"] = run["relevance"].fillna(0.5)

    if args.corruption == "random":
        run["score"] = np.random.random(size=len(run))
    elif args.corruption == "best":
        run["score"] = run["relevance"]
    elif args.corruption == "worst":
        run["score"] = -1 * run["relevance"]
    else:
        raise ValueError(f"Unknown corruption type: {args.corruption}")

    run["rank"] = run.groupby("query_id")["score"].rank(method="first", ascending=False)
    run = run.sort_values(by=["query_id", "rank"])
    run = run[RUN_HEADER]
    run.to_csv(args.output_file, sep="\t", header=False, index=False)


if __name__ == "__main__":
    main()
