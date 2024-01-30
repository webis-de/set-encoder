import argparse
from pathlib import Path
from typing import List, Optional, Iterator, Tuple, Set
import shutil

import ir_datasets
import pandas as pd
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexer
from colbert import Searcher as _ColBERTSearcher
from colbert import Indexer as ColBERTIndexer
from colbert.data import Queries
from colbert.infra import ColBERTConfig, Run, RunConfig

from ir_dataset_utils import get_base


def load_collection(
    dataset: ir_datasets.Dataset, doc_fields: Optional[List[str]]
) -> Iterator[Tuple[str, str]]:
    for doc in tqdm(
        dataset.docs_iter(), total=dataset.docs_count(), desc="loading collection"
    ):
        if doc_fields:
            contents = " ".join(
                [getattr(doc, field) for field in doc_fields if hasattr(doc, field)]
            )
        else:
            contents = doc.default_text()
        yield doc.doc_id, contents


class Searcher:
    def search(
        self,
        dataset: ir_datasets.Dataset,
        k: int,
        num_procs: int,
        doc_ids: Optional[List[int]],
        query_ids: Optional[Set[str]],
        add_query: bool,
    ) -> Iterator[pd.DataFrame]:
        raise NotImplementedError()


class PyseriniSearcher(Searcher):
    def __init__(self, index_dir: Path) -> None:
        super().__init__()
        self.index_dir = index_dir
        self._searcher = None

    @property
    def searcher(self) -> LuceneSearcher:
        if self._searcher is None:
            self._searcher = LuceneSearcher(str(self.index_dir))
        return self._searcher

    def search(
        self,
        dataset: ir_datasets.Dataset,
        k: int,
        num_procs: int,
        doc_ids: Optional[List[int]],
        query_ids: Optional[Set[str]],
        add_query: bool,
    ) -> Iterator[pd.DataFrame]:
        # queries = []
        # query_ids = []
        # for query_id, query in dataset.queries_iter():
        #     query_ids.append(str(query_id))
        #     queries.append(query)
        # hit_dict = self.searcher.batch_search(queries, query_ids, k=k, threads=num_procs)
        for query in tqdm(dataset.queries_iter(), total=dataset.queries_count()):
            data = []
            if query_ids and query.query_id not in query_ids:
                continue
            # for query_id, hits in hit_dict.items():
            hits = self.searcher.search(query.text, k=k)
            for idx, hit in enumerate(hits):
                rank = idx + 1
                score = hit.score
                doc_id = hit.docid
                data.append([query.query_id, doc_id, rank, score])
            df = pd.DataFrame(data, columns=["qid", "doc_id", "rank", "score"])
            if add_query:
                df["query"] = query.text
            yield df

    def index(
        self,
        dataset: ir_datasets.Dataset,
        doc_fields: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> None:
        if self.index_dir.exists():
            if overwrite:
                shutil.rmtree(self.index_dir)
            else:
                return
        self.index_dir.mkdir(exist_ok=True)
        print("indexing collection...")
        index = LuceneIndexer(str(self.index_dir))
        for doc_id, contents in load_collection(dataset, doc_fields):
            index.add_doc_dict({"id": doc_id, "contents": contents})
        index.close()


class ColBERTSearcher(Searcher):
    def __init__(self, index_dir: Path, checkpoint_path: Path) -> None:
        super().__init__()
        self.index_dir = index_dir
        self.config = ColBERTConfig(
            root=str(index_dir.parent),
            index_name=index_dir.name,
            index_path=str(index_dir),
            index_root=str(index_dir.parent),
            doc_maxlen=300,
        )
        self.checkpoint_path = checkpoint_path
        self._searcher = None

    @property
    def searcher(self) -> _ColBERTSearcher:
        if self._searcher is None:
            self._searcher = _ColBERTSearcher(
                index=self.index_dir.name,
                config=self.config,
                checkpoint=str(self.checkpoint_path),
            )
        return self._searcher

    def search(
        self,
        dataset: ir_datasets.Dataset,
        k: int,
        num_procs: int,
        doc_ids: Optional[List[int]],
        query_ids: Optional[Set[str]],
        add_query: bool,
    ) -> Iterator[pd.DataFrame]:
        query_dict = {
            query.query_id: query.text
            for query in dataset.queries_iter()
            if query_ids is None or query.query_id in query_ids
        }
        queries = Queries(data=query_dict)
        ranking = self.searcher.search_all(queries, k=k)

        for query_id, hits in ranking.data.items():
            data = []
            for hit in hits:
                data.append([query_id, *hit])
            df = pd.DataFrame(data, columns=["qid", "doc_id", "rank", "score"])
            if doc_ids is not None:
                df["doc_id"] = df["doc_id"].map(lambda x: doc_ids[x])
            if add_query:
                df["query"] = query_dict[query_id]
            yield df

    def index(
        self,
        dataset: ir_datasets.Dataset,
        doc_fields: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> None:
        if self.index_dir.exists() and not overwrite:
            return
        indexer = ColBERTIndexer(
            checkpoint=str(self.checkpoint_path), config=self.config
        )
        collection = []
        print("indexing collection...")
        for _, contents in load_collection(dataset, doc_fields):
            collection.append(contents)
        indexer.index(
            name=self.index_dir.name,
            collection=collection,
            collection_name=self.index_dir.name,
            overwrite=overwrite,
        )


def create_run_file(
    searcher: Searcher,
    dataset: ir_datasets.Dataset,
    run_file: Path,
    k: int,
    num_procs: int,
    doc_ids: Optional[List[int]],
    query_ids: Optional[Set[str]],
    add_query: bool,
) -> None:
    system = None
    if isinstance(searcher, PyseriniSearcher):
        system = "pyserini-BM25"
    elif isinstance(searcher, ColBERTSearcher):
        system = "ColBERT"
    with run_file.open("w") as f:
        for hits in searcher.search(
            dataset,
            k=k,
            num_procs=num_procs,
            doc_ids=doc_ids,
            query_ids=query_ids,
            add_query=add_query,
        ):
            hits["system"] = system
            hits["q0"] = 0
            hits = hits.loc[:, ["qid", "q0", "doc_id", "rank", "score", "system"]]
            hits.to_csv(
                f,
                sep="\t",
                index=False,
                header=False,
            )


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--searcher", type=str, choices=["bm25", "colbert"], required=True
    )
    parser.add_argument("--ir_datasets", type=str, nargs="+", required=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--index_dir", type=Path, required=True)
    parser.add_argument("--checkpoint_path", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--k", type=int, default=500)
    parser.add_argument("--num_procs", type=int, default=1)
    parser.add_argument("--doc_fields", type=str, nargs="*")
    parser.add_argument("--query_ids", type=str, nargs="*")
    parser.add_argument("--add_query", action="store_true")

    args = parser.parse_args(args)

    if args.searcher == "colbert" and args.checkpoint_path is None:
        raise ValueError("must specify checkpoint_path for colbert")

    for dataset_name in args.ir_datasets:
        run_file = (
            args.run_dir
            / args.searcher
            / ("-".join([dataset_name.replace("/", "-")]) + ".run")
        )
        run_file.parent.mkdir(exist_ok=True)

        if run_file.exists() and not args.overwrite:
            print(f"run file {run_file} already exists, skipping")
            continue
        dataset = ir_datasets.load(dataset_name)
        dataset_base = get_base(dataset_name)

        context = None
        index_dir = args.index_dir / args.searcher
        index_dir.mkdir(exist_ok=True)
        index_dir = index_dir / dataset_base
        if args.searcher == "bm25":
            searcher = PyseriniSearcher(index_dir)
        elif args.searcher == "colbert":
            root_dir = str(index_dir.parent)
            experiment_name = dataset_base
            context = Run().context(
                RunConfig(
                    nranks=args.num_procs,
                    experiment=experiment_name,
                    root=root_dir,
                    index_root=root_dir,
                )
            )
            context.__enter__()
            searcher = ColBERTSearcher(index_dir, args.checkpoint_path)
        else:
            raise ValueError(f"unknown searcher type {args.searcher}")

        searcher.index(dataset, args.doc_fields, args.reindex)

        doc_ids = None
        if args.searcher == "colbert":
            doc_ids = []
            for doc_id, _ in load_collection(dataset, args.doc_fields):
                doc_ids.append(doc_id)

        query_ids = None
        if args.query_ids:
            if len(args.query_ids) == 1 and Path(args.query_ids[0]).exists():
                query_ids = set(Path(args.query_ids[0]).read_text().split("\n"))
            else:
                query_ids = set(args.query_ids)

        create_run_file(
            searcher,
            dataset,
            run_file,
            args.k,
            args.num_procs,
            doc_ids,
            query_ids,
            args.add_query,
        )

        if context is not None:
            context.__exit__(None, None, None)


if __name__ == "__main__":
    main()
