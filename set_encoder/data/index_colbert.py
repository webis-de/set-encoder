import argparse
import pathlib

from colbert import Indexer
from colbert.infra import ColBERTConfig, Run, RunConfig


def main(args=None):

    parser = argparse.ArgumentParser()

    parser.add_argument("--index_name", type=str, required=True)
    parser.add_argument("--collection_path", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint_path", type=pathlib.Path, required=True)
    parser.add_argument("--experiment_root_dir", type=pathlib.Path, required=True)

    parser.add_argument("--doc_max_len", type=int, default=300)
    parser.add_argument("--nbits", type=int, default=2)
    parser.add_argument("--k_means_iters", type=int, default=4)
    parser.add_argument("--num_procs", type=int, default=1)

    args = parser.parse_args(args)

    with Run().context(
        RunConfig(
            nranks=args.num_procs,
            root=str(args.experiment_root_dir),
        )
    ):

        config = ColBERTConfig(
            doc_maxlen=args.doc_max_len,
            nbits=args.nbits,
            root=str(args.experiment_root_dir),
            kmeans_niters=args.k_means_iters,
        )
        indexer = Indexer(checkpoint=str(args.checkpoint_path), config=config)
        indexer.index(
            name=f"{args.index_name}.nbits{args.nbits}",
            collection=str(args.collection_path),
            overwrite=True,
        )


if __name__ == "__main__":
    main()
