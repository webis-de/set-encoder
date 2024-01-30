from zipfile import ZipFile
import json
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
import re

# Download links for data here https://github.com/sunnweiwei/RankGPT


def main(args=None):
    parser = ArgumentParser()

    parser.add_argument("--marco_train_Xk", type=Path, required=True)
    parser.add_argument("--marco_train_Xk_gpt", type=Path, required=True)
    parser.add_argument("--out_run_file", type=Path, required=True)

    args = parser.parse_args(args)

    with args.marco_train_Xk_gpt.open("r") as rank_file:
        rank_data = json.load(rank_file)

    invalid_lines = 0
    idx = 0
    num_lines = int(args.marco_train_Xk_gpt.name.split("-")[2][:-1]) * 1000

    with ZipFile(args.marco_train_Xk, "r") as zip_file:
        with zip_file.open(args.marco_train_Xk.with_suffix("").name) as data_file:
            with args.out_run_file.open("w") as out_file:
                for idx, (data_line, raw_ranks) in tqdm(
                    enumerate(zip(data_file, rank_data)), total=num_lines
                ):
                    if ">" not in raw_ranks:
                        print(f"Error in line number {idx + 1}: {raw_ranks}")
                        invalid_lines += 1
                        continue
                    try:
                        ranks = list(
                            map(
                                int, re.findall(r"\d+", re.sub(r"\[|\]", "", raw_ranks))
                            )
                        )
                    except Exception:
                        # print(f"Error in line number {idx + 1}: {raw_ranks}")
                        invalid_lines += 1
                        continue
                    # remove duplicates
                    ranks = list(dict.fromkeys(ranks))
                    if len(ranks) != 20 or len(set(ranks)) != len(ranks):
                        # print(f"Error in line number {idx + 1}: {raw_ranks}")
                        invalid_lines += 1
                        continue
                    data = json.loads(data_line)
                    passages = data["retrieved_passages"]
                    positive_passages = data["positive_passages"]
                    doc_ids = [passage["docid"] for passage in passages]
                    ranked_doc_ids = [doc_ids[rank - 1] for rank in ranks]
                    for positive_passage in positive_passages:
                        if positive_passage["docid"] not in ranked_doc_ids:
                            ranked_doc_ids.insert(0, positive_passage["docid"])
                    for rank, doc_id in enumerate(ranked_doc_ids):
                        rank += 1
                        out_file.write(
                            f"{data['query_id']} Q0 {doc_id} {rank} {1.0/rank} gpt\n"
                        )
                        if rank > 20:
                            break
    assert (
        idx + 1 == num_lines
    ), "Number of lines in file doesn't match file description"
    print(f"Number of invalid lines: {invalid_lines}")


if __name__ == "__main__":
    main()
