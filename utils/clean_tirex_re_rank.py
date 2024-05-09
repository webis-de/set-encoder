import gzip
import json
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm


def main(args=None):
    parser = ArgumentParser()

    parser.add_argument("--tirex_re_rank_dir", type=Path, required=True)

    args = parser.parse_args(args)

    for in_file in tqdm(sorted(args.tirex_re_rank_dir.glob("*/rerank.jsonl.gz"))):
        out_file = args.tirex_re_rank_dir / (
            "-".join(in_file.parent.name.split("-")[:-2]) + ".jsonl.gz"
        )
        if "argsme" in str(out_file):
            out_file = Path(str(out_file).replace("argsme", "argsme-2020-04-01"))
        with gzip.open(in_file, "rt") as in_f:
            with gzip.open(out_file, "wt") as out_f:
                for line in tqdm(in_f, position=1, leave=False):
                    data = json.loads(line)
                    del data["original_query"]
                    del data["original_document"]
                    out_f.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()
