import gzip
import json
from argparse import ArgumentParser
from pathlib import Path

import ir_datasets
from tqdm import tqdm


def main(args=None):
    parser = ArgumentParser()

    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--years", type=int, nargs="+", default=[2009, 2010, 2011, 2012, 2013, 2014]
    )
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(args)

    years = args.years

    args.output_dir.mkdir(parents=True, exist_ok=True)

    year_to_pattern = {
        2009: "clueweb09/en/trec-web-2009/diversity",
        2010: "clueweb09/en/trec-web-2010/diversity",
        2011: "clueweb09/en/trec-web-2011/diversity",
        2012: "clueweb09/en/trec-web-2012/diversity",
        2013: "clueweb12/trec-web-2013/diversity",
        2014: "clueweb12/trec-web-2014/diversity",
    }

    docs = ir_datasets.load("clueweb09/en").docs_store()
    for year in tqdm(years):
        pattern = year_to_pattern[year]
        dataset = ir_datasets.load(pattern)
        output_path = (
            args.output_dir / f"{dataset._dataset_id.replace('/', '-')}.jsonl.gz"
        )
        if output_path.exists() and not args.overwrite:
            continue
        queries = {q.query_id: q.query for q in dataset.queries_iter()}
        subtopics = {q.query_id: q.subtopics for q in dataset.queries_iter()}
        with gzip.open(output_path, "wt") as f:
            for qrel in tqdm(
                dataset.qrels_iter(),
                total=dataset.qrels_count(),
                position=1,
            ):
                if not subtopics[qrel.query_id]:
                    continue
                try:
                    doc = docs.get(qrel.doc_id)
                except KeyError:
                    continue
                data = {
                    "qid": qrel.query_id,
                    "query": queries[qrel.query_id],
                    "docno": qrel.doc_id,
                    "text": doc.default_text(),
                    "rank": 1,
                    "score": 0,
                }
                f.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()
