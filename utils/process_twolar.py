import json
from argparse import ArgumentParser
from pathlib import Path

import datasets
from tqdm import tqdm


def main(args=None):
    parser = ArgumentParser()

    parser.add_argument("--out_file", type=Path, required=True)

    args = parser.parse_args(args)

    twolar = datasets.load_dataset("Dundalia/TWOLAR_ds")

    with args.out_file.open("w") as f:
        for sample in tqdm(twolar["train"]):
            data = {"query_id": sample["query_id"], "query": sample["query"]}
            for doc in sample["retrieved_passages"]:
                data["doc_id"] = doc["docid"]
                data["text"] = doc["text"]
                data["rank"] = doc["rank"]
                f.write(json.dumps(data) + "\n")
                # TODO here


if __name__ == "__main__":
    main()
