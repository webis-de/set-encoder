from collections.abc import MutableMapping, Mapping
import collections

collections.MutableMapping = MutableMapping
collections.Mapping = Mapping

import codecs
from pathlib import Path
from typing import NamedTuple, Union
import re

import ir_datasets
import pandas as pd
from ir_datasets.datasets.base import Dataset
from ir_datasets.formats import (
    BaseDocPairs,
    BaseDocs,
    BaseQueries,
    BaseScoredDocs,
    GenericDoc,
    GenericQuery,
    TrecQrels,
)
from ir_datasets.indices import Docstore
from ir_datasets.util import Cache, DownloadConfig
from ir_datasets.datasets.beir import (
    BeirSciDoc,
    BeirToucheDoc,
    BeirUrlQuery,
    BeirSciQuery,
    BeirToucheQuery,
)


def doc_default_text(self):
    return f"{self.title} {self.text}"


def query_default_text(self):
    return self.text


BeirSciDoc.default_text = doc_default_text
BeirToucheDoc.default_text = doc_default_text
BeirUrlQuery.default_text = query_default_text
BeirSciQuery.default_text = query_default_text
BeirToucheQuery.default_text = query_default_text


class ScoredDocPair(NamedTuple):
    query_id: str
    doc_id_a: str
    doc_id_b: str
    score_a: float
    score_b: float


class KDDocPairs(BaseDocPairs):
    def __init__(self, docpairs_dlc, negate_score=False):
        self._docpairs_dlc = docpairs_dlc
        self._negate_score = negate_score

    def docpairs_path(self):
        return self._docpairs_dlc.path()

    def docpairs_iter(self):
        with self._docpairs_dlc.stream() as f:
            f = codecs.getreader("utf8")(f)
            for line in f:
                cols = line.rstrip().split()
                pos_score, neg_score, qid, pid1, pid2 = cols
                pos_score = float(pos_score)
                neg_score = float(neg_score)
                if self._negate_score:
                    pos_score = -pos_score
                    neg_score = -neg_score
                yield ScoredDocPair(qid, pid1, pid2, pos_score, neg_score)

    def docpairs_cls(self):
        return ScoredDocPair


def register_kd_docpairs():
    if "msmarco-passage/train/kd-docpairs" in ir_datasets.registry._registered:
        return
    base_path = ir_datasets.util.home_path() / "msmarco-passage"
    dlc = DownloadConfig.context("msmarco-passage", base_path)
    dlc._contents["train/kd-docpairs"] = {
        "url": "https://zenodo.org/record/4068216/files/bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv?download=1",
        "expected_md5": "4d99696386f96a7f1631076bcc53ac3c",
        "cache_path": "train/kd-docpairs",
    }
    ir_dataset = ir_datasets.load("msmarco-passage/train")
    collection = ir_dataset.docs_handler()
    queries = ir_dataset.queries_handler()
    qrels = ir_dataset.qrels_handler()
    docpairs = KDDocPairs(
        Cache(dlc["train/kd-docpairs"], base_path / "train" / "kd.run")
    )
    dataset = Dataset(collection, queries, qrels, docpairs)
    ir_datasets.registry.register("msmarco-passage/train/kd-docpairs", Dataset(dataset))


def register_tripjudge():
    if (
        "tripclick/test/head/tripjudge-2" in ir_datasets.registry._registered
        and "tripclick/test/head/tripjudge-4" in ir_datasets.registry._registered
    ):
        return
    base_path = ir_datasets.util.home_path() / "tripclick"
    dlc = DownloadConfig.context("tripclick", base_path)
    dlc._contents["test/head/tripjudge-2"] = {
        "url": "https://raw.githubusercontent.com/sophiaalthammer/tripjudge/main/data/qrels_2class.txt",
        "expected_md5": "a98e57c4f0daa3699ef4208cebe18f09",
        "cache_path": "test/head/tripjudge-2",
    }
    dlc._contents["test/head/tripjudge-4"] = {
        "url": "https://raw.githubusercontent.com/sophiaalthammer/tripjudge/main/data/qrels_4class.txt",
        "expected_md5": "0a051265e092f7cc8cfbde126d9e7665",
        "cache_path": "test/head/tripjudge-4",
    }
    ir_dataset = ir_datasets.load("tripclick/test/head")
    collection = ir_dataset.docs_handler()
    queries = ir_dataset.queries_handler()
    # TODO filter queries
    qrels_2_defs = {
        0: "labeled as irrelevant",
        1: "labeled as relevant",
    }
    qrels_4_defs = {
        0: "Wrong If the document has nothing to do with the query, and does not help in any way to answer it",
        1: "Topic If the document talks about the general area or topic of a query, might provide some background info, but ultimately does not answer it",
        2: "Partial The document contains a partial answer, but you think that there should be more to it",
        3: "Perfect The document contains a full answer: easy to understand and it directly answers the question in full",
    }
    qrels_2 = TrecQrels(
        Cache(
            dlc["test/head/tripjudge-2"], base_path / "test" / "head" / "tripjudge-2"
        ),
        qrels_2_defs,
    )
    qrels_4 = TrecQrels(
        Cache(
            dlc["test/head/tripjudge-4"], base_path / "test" / "head" / "tripjudge-4"
        ),
        qrels_4_defs,
    )
    dataset = Dataset(collection, queries, qrels_2)
    ir_datasets.registry.register("tripclick/test/head/tripjudge-2", Dataset(dataset))
    dataset = Dataset(collection, queries, qrels_4)
    ir_datasets.registry.register("tripclick/test/head/tripjudge-4", Dataset(dataset))


register_kd_docpairs()
register_tripjudge()

DASHED_DATASET_MAP = {
    dataset.replace("/", "-"): dataset for dataset in ir_datasets.registry._registered
}


def register_rerank_data_to_ir_datasets(re_rank_file_path: Path, ir_dataset_id: str):
    if "json" in re_rank_file_path.name:
        lines = "jsonl" in re_rank_file_path.name
        re_rank_df = pd.read_json(
            re_rank_file_path, lines=lines, dtype={"docno": str, "qid": str}
        )
        re_rank_df = re_rank_df.drop(
            ["original_document", "original_query"], axis=1, errors="ignore"
        )
        re_rank_df = re_rank_df.astype({"docno": str, "qid": str})
    elif re_rank_file_path.suffix == ".parquet":
        re_rank_df = pd.read_parquet(
            re_rank_file_path, columns=["qid", "docno", "rank", "score"]
        )
        re_rank_df = re_rank_df.astype({"docno": str, "qid": str})
    else:
        re_rank_df = pd.read_csv(
            re_rank_file_path,
            sep=r"\s+",
            names=["qid", "q0", "docno", "rank", "score", "system"],
            dtype={"docno": str, "qid": str},
        ).loc[:, ["qid", "docno", "rank", "score"]]

    re_rank_df = re_rank_df.rename(
        {"qid": "query_id", "docno": "doc_id"}, axis=1, copy=False
    )

    try:
        original_ir_dataset_id = re.sub(r"__.+__", "", ir_dataset_id.split("/")[-1])
        original_ir_dataset = ir_datasets.load(
            DASHED_DATASET_MAP[original_ir_dataset_id]
        )
    except KeyError:
        original_ir_dataset = None

    handlers = {}
    if original_ir_dataset is not None:
        if original_ir_dataset.has_docs():
            handlers["docs"] = original_ir_dataset.docs_handler()
        if original_ir_dataset.has_queries():
            handlers["queries"] = original_ir_dataset.queries_handler()
        if original_ir_dataset.has_qrels():
            handlers["qrels"] = original_ir_dataset.qrels_handler()
        if original_ir_dataset.has_scoreddocs():
            handlers["scoreddocs"] = original_ir_dataset.scoreddocs_handler()
        if original_ir_dataset.has_docpairs():
            handlers["docpairs"] = original_ir_dataset.docpairs_handler()
        if original_ir_dataset.has_qlogs():
            handlers["qlogs"] = original_ir_dataset.qlogs_handler()
    if "text" in re_rank_df.columns:
        handlers["docs"] = __docs(re_rank_df)
    if "docs" not in handlers:
        raise ValueError("No text column in re-rank file and no original dataset")
    if "query" in re_rank_df.columns:
        handlers["queries"] = __queries(re_rank_df)
    if "queries" not in handlers:
        raise ValueError("No query column in re-rank file and no original dataset")
    scoreddocs = __scored_docs(re_rank_df)
    handlers["scoreddocs"] = scoreddocs
    dataset = Dataset(*handlers.values(), scoreddocs)
    ir_datasets.registry.register(ir_dataset_id, dataset)

    __check_registration_was_successful(ir_dataset_id)


def __docs(df: pd.DataFrame):
    class DFDocstore(Docstore):
        def __init__(self, df: pd.DataFrame):
            self.df = df.loc[:, "text"]

        def get(self, doc_id: str):
            return GenericDoc(doc_id=doc_id, text=self.df.loc[doc_id])

    class DynamicDocs(BaseDocs):
        def __init__(self, df: pd.DataFrame):
            self.df = df.drop_duplicates("doc_id").set_index("doc_id")

        def docs_iter(self):
            for row in df.itertuples():
                yield GenericDoc(doc_id=row.doc_id, text=row.text)

        def docs_count(self):
            return len(self.df)

        def docs_store(self):
            return DFDocstore(self.df)

    return DynamicDocs(df)


def __queries(df: pd.DataFrame):
    class DynamicQueries(BaseQueries):
        def __init__(self, df: pd.DataFrame):
            self.df = df.drop_duplicates("query_id").set_index("query_id")

        def queries_iter(self):
            for row in df.itertuples():
                yield GenericQuery(query_id=row.query_id, text=row.query)

    return DynamicQueries(df)


def __scored_docs(df: pd.DataFrame):
    class GenericScoredDocWithRank(NamedTuple):
        query_id: str
        doc_id: str
        score: float
        rank: int

    class DynamicScoredDocs(BaseScoredDocs):
        def __init__(self, df: pd.DataFrame):
            # self.df = df.drop_duplicates(("doc_id", "query_id"))
            self.df = df

        def scoreddocs_iter(self):
            for row in df.itertuples():
                yield GenericScoredDocWithRank(
                    row.query_id, row.doc_id, row.score, row.rank
                )

    return DynamicScoredDocs(df)


def __check_registration_was_successful(ir_dataset_id):
    import ir_datasets

    dataset = ir_datasets.load(ir_dataset_id)

    assert dataset.has_docs(), f"dataset {ir_dataset_id} has no documents"
    assert dataset.has_queries(), f"dataset {ir_dataset_id} has no queries"
    assert dataset.has_scoreddocs(), f"dataset {ir_dataset_id} has no scored_docs"


def load(dataset: Union[str, Path]) -> ir_datasets.Dataset:
    try:
        return ir_datasets.load(dataset)
    except (KeyError, TypeError):
        path = Path(dataset)
        dataset_id = str(path)[: -len("".join(path.suffixes))]  # remove all suffixes
        try:
            return ir_datasets.load(dataset_id)
        except KeyError:
            register_rerank_data_to_ir_datasets(path, dataset_id)
    return ir_datasets.load(dataset_id)


def get_base(dataset_id: str) -> str:
    try:
        ir_dataset = ir_datasets.load(dataset_id)
    except KeyError:
        ir_dataset = ir_datasets.load(DASHED_DATASET_MAP[dataset_id])
    if not ir_dataset.has_docs():
        raise ValueError(f"Dataset {dataset_id} does not have docs")
    while True:
        parent = "/".join(dataset_id.split("/")[:-1])
        try:
            ir_dataset = ir_datasets.load(parent)
        except KeyError:
            return dataset_id.replace("/", "-")
        if not ir_dataset.has_docs():
            return dataset_id.replace("/", "-")
        dataset_id = parent
        if dataset_id == "":
            raise ValueError("Dataset has no base")
