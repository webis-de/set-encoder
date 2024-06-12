from argparse import ArgumentParser
from pathlib import Path

import torch
import ir_datasets
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


def main(args=None):
    parser = ArgumentParser()

    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    patterns = [
        "clueweb09/en/trec-web-2009/diversity",
        "clueweb09/en/trec-web-2010/diversity",
        "clueweb09/en/trec-web-2011/diversity",
        "clueweb09/en/trec-web-2012/diversity",
        "clueweb12/trec-web-2013/diversity",
        "clueweb12/trec-web-2014/diversity",
    ]

    for pattern in tqdm(patterns):
        dataset = ir_datasets.load(pattern)
        docs_store = dataset.docs_store()
        output_path = (
            args.output_dir / f"{dataset._dataset_id.replace('/', '-')}.jsonl.gz"
        )
        if output_path.exists() and not args.overwrite:
            print(f"Skipping {pattern}")
            continue

        queries = pd.DataFrame(dataset.queries_iter())
        queries = queries.loc[queries["subtopics"] != tuple()]
        queries = queries.loc[:, ["query_id", "query"]]

        qrels = pd.DataFrame(dataset.qrels_iter())
        qrels = qrels.loc[qrels.loc[:, "query_id"].isin(queries.loc[:, "query_id"])]

        docs = qrels[["doc_id"]].drop_duplicates()
        docs["text"] = docs["doc_id"].progress_map(
            lambda x: docs_store.get(x).default_text()
        )

        subtopic_relevance_sum = qrels.groupby(["query_id", "doc_id"])[
            "relevance"
        ].sum()
        rank = (
            subtopic_relevance_sum.groupby("query_id", sort=False)
            .rank(method="first", ascending=False)
            .rename("rank")
            .astype(int)
        ).reset_index()

        data = qrels
        data = pd.merge(data, rank, on=["query_id", "doc_id"])
        data = pd.merge(data, queries, on="query_id")
        data = pd.merge(data, docs, on="doc_id")
        data["score"] = -data["rank"]
        data = data.rename(
            columns={"query_id": "qid", "doc_id": "docno", "subtopic_id": "iteration"}
        )
        data.to_json(output_path, orient="records", lines=True, compression="gzip")

    clueweb09_years = [2009, 2010, 2011, 2012]
    clueweb12_paths = [
        args.output_dir / f"clueweb12-trec-web-{year}-diversity.jsonl.gz"
        for year in (2013, 2014)
    ]
    for year in tqdm(clueweb09_years):
        output_path = (
            args.output_dir / f"clueweb09-en-trec-web-without-{year}-diversity.jsonl.gz"
        )
        if output_path.exists() and not args.overwrite:
            print(f"Skipping {year}")
            continue
        other_years = [y for y in clueweb09_years if y != year]
        paths = [
            args.output_dir / f"clueweb09-en-trec-web-{y}-diversity.jsonl.gz"
            for y in other_years
        ] + clueweb12_paths
        data = pd.concat(
            [pd.read_json(p, lines=True, compression="gzip") for p in paths]
        )
        data.to_json(output_path, orient="records", lines=True, compression="gzip")


if __name__ == "__main__":
    main()
