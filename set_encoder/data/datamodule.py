import random
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Union

import ir_datasets
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import transformers

from .ir_dataset_utils import get_base
from .ir_dataset_utils import load as load_ir_dataset


def _read_ir_dataset(
    dataset: ir_datasets.Dataset,
    data_type: Union[Literal["queries"], Literal["qrels"], Literal["scoreddocs"]],
) -> pd.DataFrame:
    if data_type == "queries":
        if not dataset.has_queries():
            raise ValueError(f"dataset {dataset.dataset_id()} does not have queries")
        data = pd.DataFrame(dataset.queries_iter())
        data = data.rename(columns={"query": "text"}, errors="ignore")
        data = data.loc[:, ["query_id", "text"]]
        data["text"] = data["text"].fillna("").str.strip()
        if data.dtypes["query_id"] != object or data.dtypes["text"] != object:
            data = data.astype({"query_id": str, "text": str})
    elif data_type == "qrels":
        if not dataset.has_qrels():
            raise ValueError(f"dataset {dataset.dataset_id()} does not have qrels")
        data = pd.DataFrame(dataset.qrels_iter())
        if data.dtypes["query_id"] != object or data.dtypes["doc_id"] != object:
            data = data.astype({"query_id": str, "doc_id": str})
    elif data_type == "scoreddocs":
        if not dataset.has_scoreddocs():
            raise ValueError(f"dataset {dataset.dataset_id()} does not have scoreddocs")
        if hasattr(dataset.scoreddocs_handler(), "df"):
            data = dataset.scoreddocs_handler().df
        else:
            data = pd.DataFrame(dataset.scoreddocs_iter())
        if data.dtypes["query_id"] != object or data.dtypes["doc_id"] != object:
            data = data.astype({"query_id": str, "doc_id": str})
        if "rank" not in data:
            data["rank"] = (
                data.groupby(["query_id", "score"])
                .rank(method="first", ascending=False)
                .astype(int)
            )
        if data["rank"].iloc[0] != 1:
            # heuristic for sorting, if by chance the first rank is 1, bad luck
            data = data.sort_values(["query_id", "rank"])
    else:
        raise ValueError(f"invalid data_type: {data_type}")

    return data


class CacheLoader:
    __instance = None
    __data = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def read_ir_dataset(
        self,
        dataset: ir_datasets.Dataset,
        data_type: Union[Literal["queries"], Literal["qrels"], Literal["scoreddocs"]],
    ) -> pd.DataFrame:
        dataset_id = dataset.dataset_id() + "/" + data_type
        if data_type in ("queries", "qrels"):
            dataset_id = re.sub(r"__.+__", "", dataset_id)
        if CacheLoader.__data is None:
            CacheLoader.__data = {}

        if dataset_id in CacheLoader.__data:
            return CacheLoader.__data[dataset_id]

        # TODO use pd.read_csv to load queries etc faster, see evaluate-set_encoder.ipynb
        data = _read_ir_dataset(dataset, data_type)

        CacheLoader.__data[dataset_id] = data
        return data

    def clear(self):
        CACHE_LOADER.__data = None


CACHE_LOADER = CacheLoader()
read_ir_dataset = CACHE_LOADER.read_ir_dataset


class DocStore:
    def __init__(
        self,
        docstore: ir_datasets.indices.Docstore,
        doc_fields: Optional[Iterable[str]],
    ) -> None:
        self.docstore = docstore
        self.doc_fields = doc_fields

    def __getitem__(self, doc_id: str) -> str:
        doc = self.docstore.get(doc_id)
        if self.doc_fields:
            contents = " ".join(
                [
                    getattr(doc, field)
                    for field in self.doc_fields
                    if hasattr(doc, field)
                ]
            )
        else:
            contents = doc.default_text()
        return contents

    @property
    def path(self) -> str:
        if hasattr(self.docstore, "path"):
            return self.docstore.path
        elif hasattr(self.docstore, "_path"):
            return self.docstore._path
        else:
            raise AttributeError("docstore has no path attribute")

    def get(self, doc_id, field=None):
        return self.docstore.get(doc_id, field)

    def get_many(self, doc_ids, field=None):
        return self.docstore.get_many(doc_ids, field)


class IRDataset:
    @staticmethod
    def get_docs(
        ir_datasets: Iterable[ir_datasets.Dataset],
        doc_fields: Optional[Iterable[str]] = None,
    ) -> Dict[str, DocStore]:
        return {
            get_base(ir_dataset.dataset_id()): DocStore(
                ir_dataset.docs_store(), doc_fields
            )
            for ir_dataset in ir_datasets
        }

    @staticmethod
    def load_queries(
        ir_datasets: Iterable[ir_datasets.Dataset],
        query_ids: Optional[Dict[str, Iterable[str]]] = None,
    ) -> Dict[str, Dict[str, str]]:
        queries_dict = defaultdict(dict)
        for ir_dataset in ir_datasets:
            queries = read_ir_dataset(ir_dataset, "queries")
            base = get_base(ir_dataset.dataset_id())
            if query_ids is not None:
                queries = queries[queries["query_id"].isin(query_ids[base])]
            queries_dict[base].update(queries.set_index("query_id")["text"].to_dict())
        return dict(queries_dict)

    @staticmethod
    def load_runs(
        ir_datasets: Iterable[ir_datasets.Dataset],
    ) -> Dict[str, pd.DataFrame]:
        runs_dict = defaultdict(list)
        for ir_dataset in ir_datasets:
            run = read_ir_dataset(ir_dataset, "scoreddocs")
            runs_dict[get_base(ir_dataset.dataset_id())].append(run)
        out_dict = {}
        for dataset_id, runs in runs_dict.items():
            out_dict[dataset_id] = pd.concat(runs, copy=False)
        return out_dict

    @staticmethod
    def load_qrels(
        ir_datasets: Iterable[ir_datasets.Dataset],
    ) -> Dict[str, pd.DataFrame]:
        qrels_dict = defaultdict(list)
        for ir_dataset in ir_datasets:
            qrels = read_ir_dataset(ir_dataset, "qrels")
            qrels_dict[get_base(ir_dataset.dataset_id())].append(qrels)
        out_dict = {}
        for dataset_id, qrels in qrels_dict.items():
            out_dict[dataset_id] = pd.concat(qrels, copy=False)
        return out_dict


class ListwiseDataset(torch.utils.data.Dataset, IRDataset):
    """depth is the number of documents to consider per query, sample size is the the
    number of documents sampled from the depth of documents. if num_relevant_samples
    is set to -1, the number of relevant documents at that sample size is used.

    example: depth=10, sample_size=4, num_relevant_samples=-1

    rels = [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    num_relevant_samples = 2 # number of relevant documents at sample_size=4
    """

    def __init__(
        self,
        ir_datasets: Iterable[ir_datasets.Dataset],
        sample_size: int,
        num_relevant_samples: int,
        min_num_relevant_samples: int,
        non_relevant_sampling_strategy: str,
        relevant_sampling_strategy: str,
        depth: int,
        remove_unjudged_docs: bool = False,
        keep_non_retrieved: bool | Literal["relevant"] = True,
        shuffle_docs: bool = False,
        doc_fields: Optional[Iterable[str]] = None,
        load_qrels: bool = True,
        use_ranks_as_qrels: bool = False,
        use_qrels_as_run: bool = False,
    ):
        if depth > -1 and sample_size > depth:
            raise ValueError("sample_size must be smaller or equal to depth")
        if use_qrels_as_run:
            self.runs = self.load_qrels(ir_datasets)
            for dataset_id in self.runs:
                self.runs[dataset_id] = self.runs[dataset_id].drop(
                    ["relevance", "subtopic_id"], axis=1, errors="ignore"
                )
                self.runs[dataset_id]["score"] = 0
                self.runs[dataset_id]["rank"] = 1
        else:
            self.runs = self.load_runs(ir_datasets)
        self.qrels = None
        if load_qrels:
            if use_ranks_as_qrels:
                qrels = {
                    dataset_id: self.runs[dataset_id]
                    .copy()
                    .rename(columns={"rank": "relevance"})
                    .assign(
                        relevance=lambda x: x["relevance"].max()
                        - x["relevance"].astype(int)
                        + 1
                    )
                    .drop("score", axis=1, errors="ignore")
                    for dataset_id in self.runs
                }
            else:
                qrels = self.load_qrels(ir_datasets)
            self.qrels = {}
            num_mismatched_query_ids = 0
            for dataset_id in qrels:
                mismatched_query_ids = set(
                    qrels[dataset_id].query_id
                ).symmetric_difference(set(self.runs[dataset_id].query_id))
                num_mismatched_query_ids += len(mismatched_query_ids)
                if min_num_relevant_samples:
                    relevant = qrels[dataset_id].loc[qrels[dataset_id].relevance >= 1]
                    num_relevant = relevant.groupby("query_id", sort=False).size()
                    invalid_queries = num_relevant.loc[
                        num_relevant < min_num_relevant_samples
                    ]
                    mismatched_query_ids.update(invalid_queries.index)
                if "subtopic_id" in qrels[dataset_id]:
                    no_subtopic_queries = (
                        (qrels[dataset_id].subtopic_id == "0")
                        .groupby(qrels[dataset_id].query_id)
                        .all()
                    )
                    mismatched_query_ids.update(
                        no_subtopic_queries.loc[no_subtopic_queries].index.values
                    )
                if mismatched_query_ids:
                    self.runs[dataset_id] = self.runs[dataset_id].loc[
                        ~self.runs[dataset_id].query_id.isin(mismatched_query_ids)
                    ]
                    qrels[dataset_id] = qrels[dataset_id].loc[
                        ~qrels[dataset_id].query_id.isin(mismatched_query_ids)
                    ]
                if num_mismatched_query_ids:
                    warnings.warn(
                        f"mismatch between run and qrel data, found for {dataset_id}: "
                        f"{num_mismatched_query_ids} mismatched query ids",
                        RuntimeWarning,
                    )
                self.qrels[dataset_id] = qrels[dataset_id].groupby(
                    "query_id", sort=False
                )

        self.groups = {
            base: self.runs[base].groupby("query_id", sort=False) for base in self.runs
        }
        self.run_query_ids = {
            base: list(group.groups.keys()) for base, group in self.groups.items()
        }
        self.docs = self.get_docs(ir_datasets, doc_fields)
        self.queries = self.load_queries(ir_datasets, self.run_query_ids)

        self.sample_size = sample_size
        self.depth = depth
        self.num_relevant_samples = num_relevant_samples
        self.shuffle_docs = shuffle_docs
        self.non_relevant_sampling_strategy = non_relevant_sampling_strategy
        self.relevant_sampling_strategy = relevant_sampling_strategy
        self.remove_unjudged_docs = remove_unjudged_docs
        self.keep_non_retrieved = keep_non_retrieved

    def __len__(self) -> int:
        return sum(len(query_ids) for query_ids in self.run_query_ids.values())

    def get_base_and_index(self, index: int) -> Tuple[str, int]:
        for base, query_ids in self.run_query_ids.items():
            if index < len(query_ids):
                return base, index
            index -= len(query_ids)
            if index < 0:
                break
        raise IndexError("index out of range")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        base, base_index = self.get_base_and_index(index)
        query_id = self.run_query_ids[base][base_index]
        group = self.groups[base].get_group(query_id).copy()
        if self.qrels is not None:
            how = "outer" if self.keep_non_retrieved else "left"
            qrels = self.qrels[base].get_group(query_id)
            group = pd.merge(group, qrels, how=how, on=["query_id", "doc_id"])
            group = group.sort_values("rank")
            if self.remove_unjudged_docs:
                group = group.loc[group["relevance"].notnull()]
        else:
            group["relevance"] = -1
        optimal_df = group.loc[group["relevance"] > 0].sort_values(
            "relevance", ascending=False
        )
        if self.depth != -1:
            valid = group["rank"] <= self.depth
            if self.keep_non_retrieved == "relevant":
                valid = valid | (group["relevance"] >= 1)
            group = group.loc[valid]
        group = group.drop_duplicates(subset=["doc_id"])
        group.loc[:, ["relevance", "rank"]] = group.loc[
            :, ["relevance", "rank"]
        ].fillna(-1)
        query = self.queries[base][query_id]
        doc_sample, subtopic_ids = self.get_sample(df=group)
        doc_ids = list(doc_sample)
        if self.shuffle_docs:
            # don't need to shuffle labels, we grab them later by docid
            random.shuffle(doc_ids)
        else:
            ranks = group.set_index("doc_id")["rank"]
            max_rank = ranks.max()
            if max_rank == -1:
                max_rank = 1
            ranks[ranks == -1] = np.random.randint(1, max_rank + 1, (ranks == -1).sum())
            doc_ids = sorted(doc_ids, key=lambda x: ranks[x])
        docs = []
        labels = []
        subtopics = []
        for doc_id in doc_ids:
            docs.append(self.docs[base][doc_id])
            labels.append(doc_sample[doc_id])
            subtopics.append(subtopic_ids[doc_id] if subtopic_ids else None)
        sample = {
            "query": query,
            "docs": docs,
            "labels": labels,
            "subtopics": subtopics if subtopic_ids else None,
            "doc_ids": doc_ids,
            "query_id": query_id,
        }
        sample["optimal_labels"] = optimal_df.set_index("doc_id")[
            "relevance"
        ].values.tolist()
        if subtopic_ids:
            sample["optimal_subtopics"] = (
                optimal_df.set_index("doc_id")["subtopic_id"]
                .values.astype(int)
                .tolist()
            )
        else:
            sample["optimal_subtopics"] = None
        return sample

    @staticmethod
    def sample_random(df: pd.DataFrame, non_relevant_size: int) -> pd.DataFrame:
        non_relevant_df = df.copy()
        non_relevant_df.loc[non_relevant_df.relevance == -1, "relevance"] = 0
        non_relevant_df = non_relevant_df.loc[non_relevant_df.relevance == 0]
        non_relevant_size = min(non_relevant_size, non_relevant_df.shape[0])
        non_relevant_samples = non_relevant_df.sample(non_relevant_size).set_index(
            "doc_id"
        )
        return non_relevant_samples

    @staticmethod
    def sample_first(
        df: pd.DataFrame,
        non_relevant_size: int,
    ) -> pd.DataFrame:
        non_relevant_df = df.copy()
        non_relevant_df.loc[non_relevant_df.relevance == -1, "relevance"] = 0
        non_relevant_df = non_relevant_df.loc[non_relevant_df.relevance == 0]
        non_relevant_size = min(non_relevant_size, non_relevant_df.shape[0])
        non_relevant_samples = non_relevant_df.iloc[:non_relevant_size].set_index(
            "doc_id"
        )
        return non_relevant_samples

    @staticmethod
    def sample_first_neg(
        df: pd.DataFrame,
        non_relevant_size: int,
        first: bool,
        fill_up: bool,
    ) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)
        df = df.loc[df["rank"] != -1]
        pos_df = df.loc[df.relevance >= 1]
        if not pos_df.shape[0]:
            # only additional docs relevant (not found by retrieval model)
            first_non_relevant = df.iloc[0].name
        else:
            # first non-relevant document ranked lower than lowest relevant document
            last_relevant_doc = pos_df.iloc[-1].name
            first_non_relevant = None
            if last_relevant_doc != df.iloc[-1].name:
                first_non_relevant = last_relevant_doc + 1
        non_relevant_samples = {}
        df.loc[df.relevance == -1, "relevance"] = 0
        if first_non_relevant is not None:
            non_relevant_df = df.loc[first_non_relevant:].copy()
            non_relevant_df = non_relevant_df.loc[non_relevant_df.relevance == 0]
            _non_relevant_size = min(non_relevant_size, non_relevant_df.shape[0])
            non_relevant_samples = {}
            if _non_relevant_size:
                if first:
                    non_relevant_samples = non_relevant_df.iloc[
                        :_non_relevant_size
                    ].set_index("doc_id")
                else:
                    non_relevant_samples = non_relevant_df.sample(
                        _non_relevant_size
                    ).set_index("doc_id")
        if len(non_relevant_samples) < non_relevant_size and fill_up:
            non_relevant_df = df.loc[df.relevance == 0]
            _non_relevant_size = non_relevant_size - len(non_relevant_samples)
            _non_relevant_size = min(_non_relevant_size, non_relevant_df.shape[0])
            remaining_non_relevant_samples = {}
            if _non_relevant_size:
                remaining_non_relevant_samples = non_relevant_df.sample(
                    _non_relevant_size
                ).set_index("doc_id")
            non_relevant_samples = {
                **non_relevant_samples,
                **remaining_non_relevant_samples,
            }
        return non_relevant_samples

    @staticmethod
    def sample_ignore_neg(
        df: pd.DataFrame,
        non_relevant_size: int,
        first: bool,
    ) -> pd.DataFrame:
        non_relevant_df = df.loc[df.relevance == -1].copy()
        non_relevant_df["relevance"] = 0
        non_relevant_size = min(non_relevant_size, non_relevant_df.shape[0])
        if first:
            non_relevant_samples = non_relevant_df.iloc[:non_relevant_size].set_index(
                "doc_id"
            )
        else:
            non_relevant_samples = non_relevant_df.sample(non_relevant_size).set_index(
                "doc_id"
            )
        return non_relevant_samples

    def get_relevant_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        rel_df = df.loc[df.relevance >= 1]
        if rel_df.shape[0] == 0:
            return pd.DataFrame()
        num_relevant_samples = self.num_relevant_samples
        if num_relevant_samples == -1:
            valid_bool = df["relevance"] >= 1
            if self.relevant_sampling_strategy == "first":
                # only sample relevant docs from top k=sample_size
                # can happen if sample_size < depth
                valid_bool = valid_bool & (df["rank"] <= self.sample_size)
            num_relevant_samples = valid_bool.sum()
        num_relevant_samples = min(num_relevant_samples, rel_df.shape[0])
        num_relevant_samples = min(num_relevant_samples, self.sample_size)
        if self.relevant_sampling_strategy == "first":
            rel_df = rel_df.head(num_relevant_samples)
        elif self.relevant_sampling_strategy == "random":
            rel_df = rel_df.sample(num_relevant_samples)
        else:
            raise ValueError(
                f"Invalid relevant sampling strategy: {self.relevant_sampling_strategy}"
            )
        rel_samples = rel_df.set_index("doc_id")
        return rel_samples

    def get_non_relevant_sample(
        self, df: pd.DataFrame, num_relevant_samples: int
    ) -> pd.DataFrame:
        non_relevant_size = self.sample_size - num_relevant_samples
        if self.non_relevant_sampling_strategy == "random_first_neg":
            non_relevant_samples = self.sample_first_neg(
                df,
                non_relevant_size,
                first=False,
                fill_up=False,
            )
        elif self.non_relevant_sampling_strategy == "first_first_neg":
            non_relevant_samples = self.sample_first_neg(
                df,
                non_relevant_size,
                first=True,
                fill_up=False,
            )
        elif self.non_relevant_sampling_strategy == "random_ignore_neg":
            non_relevant_samples = self.sample_ignore_neg(
                df,
                non_relevant_size,
                first=False,
            )
        elif self.non_relevant_sampling_strategy == "first_ignore_neg":
            non_relevant_samples = self.sample_ignore_neg(
                df, non_relevant_size, first=True
            )
        elif self.non_relevant_sampling_strategy == "random_first_neg_ignore_neg":
            df = df.copy()
            df.loc[df["relevance"] == 0, "rank"] = -1
            non_relevant_samples = self.sample_first_neg(
                df, non_relevant_size, first=False, fill_up=False
            )
        elif self.non_relevant_sampling_strategy == "first_first_neg_ignore_neg":
            df = df.copy()
            df.loc[df["relevance"] == 0, "rank"] = -1
            non_relevant_samples = self.sample_first_neg(
                df, non_relevant_size, first=True, fill_up=False
            )
        elif self.non_relevant_sampling_strategy == "first":
            non_relevant_samples = self.sample_first(df, non_relevant_size)
        elif self.non_relevant_sampling_strategy == "random":
            non_relevant_samples = self.sample_random(df, non_relevant_size)
        else:
            raise ValueError(
                f"invalid value for non-relevant sampling_strategy "
                f"{self.non_relevant_sampling_strategy}"
            )
        return non_relevant_samples

    def get_sample(
        self, df: pd.DataFrame
    ) -> Tuple[Dict[str, int], Dict[str, int] | None]:
        if self.relevant_sampling_strategy == "full_first":
            sample_size = min(self.sample_size, df.shape[0])
            sample_df = df.head(sample_size).set_index("doc_id")
        elif self.relevant_sampling_strategy == "full_random":
            sample_size = min(self.sample_size, df.shape[0])
            sample_df = df.sample(sample_size).set_index("doc_id")
        else:
            relevant_samples = self.get_relevant_sample(df)
            non_relevant_samples = self.get_non_relevant_sample(
                df, len(relevant_samples)
            )
            sample_df = pd.concat([relevant_samples, non_relevant_samples])
        subtopic_ids = None
        if "subtopic_id" in sample_df:
            subtopic_ids = (
                sample_df["subtopic_id"].astype(float).fillna(0).astype(int).to_dict()
            )
        return sample_df["relevance"].to_dict(), subtopic_ids


class DocPairDataset(torch.utils.data.IterableDataset, IRDataset):
    def __init__(
        self,
        ir_datasets: Iterable[ir_datasets.Dataset],
        doc_fields: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__()
        self.ir_datasets = ir_datasets
        self.docs = self.get_docs(ir_datasets, doc_fields)
        self.queries = self.load_queries(ir_datasets)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for ir_dataset in self.ir_datasets:
            ir_dataset_base = get_base(ir_dataset.dataset_id())
            for query_id, pos_doc_id, neg_doc_id, *scores in ir_dataset.docpairs_iter():
                query = self.queries[ir_dataset_base][query_id]
                pos_doc = self.docs[ir_dataset_base][pos_doc_id]
                neg_doc = self.docs[ir_dataset_base][neg_doc_id]
                labels = [1, 0]
                ranks = torch.tensor([1, 2])
                if scores:
                    labels = [float(score) for score in scores]
                yield {
                    "query": query,
                    "docs": [pos_doc, neg_doc],
                    "labels": labels,
                    "ranks": ranks,
                }


class SetEncoderDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        train_ir_dataset_paths: Optional[Iterable[Union[str, Path]]] = None,
        val_ir_dataset_paths: Optional[Iterable[Union[str, Path]]] = None,
        predict_ir_dataset_paths: Optional[Iterable[Union[str, Path]]] = None,
        doc_fields: Optional[Iterable[str]] = None,
        use_triples: bool = False,
        truncate: bool = True,
        max_length: int = 512,
        min_doc_length: int = 0,
        batch_size: int = 1,
        val_batch_size: int = 1,
        predict_batch_size: int = 1,
        train_sample_size: int = 10,
        train_sample_depth: int = -1,
        depth: int = 100,
        num_relevant_samples: int = 1,
        min_num_relevant_samples: int = 0,
        shuffle_queries: bool = True,
        shuffle_docs: bool = True,
        relevant_sampling_strategy: Literal[
            "full_random", "full_first", "random", "first"
        ] = "random",
        non_relevant_sampling_strategy: Literal[
            "random_first_neg",
            "first_first_neg",
            "random_ignore_neg",
            "first_ignore_neg",
            "random_first_neg_ignore_neg",
            "first_first_neg_ignore_neg",
            "first",
            "random",
        ] = "random",
        remove_unjudged_docs: bool = False,
        use_ranks_as_qrels: bool = False,
        use_qrels_as_run: bool = False,
        keep_non_retrieved: bool | Literal["relevant"] = True,
        num_workers: int = 0,
    ):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

        self.train_ir_dataset_paths = train_ir_dataset_paths
        self.val_ir_dataset_paths = val_ir_dataset_paths
        self.predict_ir_dataset_paths = predict_ir_dataset_paths
        self.doc_fields = doc_fields
        self.use_triples = use_triples

        self.truncate = truncate
        self.max_length = max_length
        self.min_doc_length = min_doc_length
        if self.min_doc_length > self.max_length:
            raise ValueError("min_doc_length must be smaller or equal to max_length")
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.predict_batch_size = predict_batch_size

        self.train_sample_size = train_sample_size
        self.train_sample_depth = train_sample_depth
        self.depth = depth
        self.num_relevant_samples = num_relevant_samples
        self.min_num_relevant_samples = min_num_relevant_samples

        self.shuffle_queries = shuffle_queries
        self.shuffle_docs = shuffle_docs
        self.relevant_sampling_strategy = relevant_sampling_strategy
        self.non_relevant_sampling_strategy = non_relevant_sampling_strategy
        self.remove_unjudged_docs = remove_unjudged_docs
        self.use_ranks_as_qrels = use_ranks_as_qrels
        self.use_qrels_as_run = use_qrels_as_run
        self.keep_non_retrieved = keep_non_retrieved
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_datasets = []
        self.predict_datasets = []
        super().__init__()

    def setup_train(self, ir_datasets: Iterable[ir_datasets.Dataset]) -> None:
        if self.use_triples:
            self.train_dataset = DocPairDataset(ir_datasets, self.doc_fields)
        else:
            self.train_dataset = ListwiseDataset(
                ir_datasets=ir_datasets,
                sample_size=self.train_sample_size,
                num_relevant_samples=self.num_relevant_samples,
                min_num_relevant_samples=self.min_num_relevant_samples,
                non_relevant_sampling_strategy=self.non_relevant_sampling_strategy,
                relevant_sampling_strategy=self.relevant_sampling_strategy,
                depth=self.train_sample_depth,
                shuffle_docs=self.shuffle_docs,
                doc_fields=self.doc_fields,
                load_qrels=True,
                use_ranks_as_qrels=self.use_ranks_as_qrels,
                remove_unjudged_docs=self.remove_unjudged_docs,
                keep_non_retrieved=self.keep_non_retrieved,
                use_qrels_as_run=self.use_qrels_as_run,
            )

    def setup_validate(self, ir_datasets: Iterable[ir_datasets.Dataset]) -> None:
        self.val_datasets = []
        for ir_dataset in ir_datasets:
            self.val_datasets.append(
                ListwiseDataset(
                    ir_datasets=[ir_dataset],
                    sample_size=self.depth,
                    num_relevant_samples=-1,
                    min_num_relevant_samples=0,
                    non_relevant_sampling_strategy="first",
                    relevant_sampling_strategy="full_first",
                    depth=self.depth,
                    shuffle_docs=self.shuffle_docs,
                    doc_fields=self.doc_fields,
                    load_qrels=True,
                    keep_non_retrieved=False,
                )
            )

    def setup_predict(self, ir_datasets: Iterable[ir_datasets.Dataset]) -> None:
        self.predict_datasets = []
        for ir_dataset in ir_datasets:
            self.predict_datasets.append(
                ListwiseDataset(
                    ir_datasets=[ir_dataset],
                    sample_size=self.depth,
                    num_relevant_samples=-1,
                    min_num_relevant_samples=0,
                    non_relevant_sampling_strategy="first",
                    relevant_sampling_strategy="full_first",
                    depth=self.depth,
                    shuffle_docs=self.shuffle_docs,
                    doc_fields=self.doc_fields,
                    load_qrels=False,
                    keep_non_retrieved=False,
                )
            )

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", "validate", None):
            if stage in ("fit", None):
                assert self.train_ir_dataset_paths is not None
                datasets = [
                    load_ir_dataset(ir_dataset_path)
                    for ir_dataset_path in self.train_ir_dataset_paths
                ]
                self.setup_train(datasets)
            datasets = (
                [
                    load_ir_dataset(ir_dataset_path)
                    for ir_dataset_path in self.val_ir_dataset_paths
                ]
                if self.val_ir_dataset_paths is not None
                else []
            )
            self.setup_validate(datasets)

        if stage == "predict":
            assert self.predict_ir_dataset_paths is not None
            datasets = [
                load_ir_dataset(ir_dataset_path)
                for ir_dataset_path in self.predict_ir_dataset_paths
            ]
            self.setup_predict(datasets)

        CACHE_LOADER.clear()

    def train_dataloader(self):
        assert self.train_dataset is not None
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_and_tokenize,
            shuffle=(
                self.shuffle_queries
                if isinstance(self.train_dataset, ListwiseDataset)
                else False
            ),
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return [
            torch.utils.data.DataLoader(
                val_data,
                batch_size=self.val_batch_size,
                collate_fn=self.collate_and_tokenize,
                num_workers=self.num_workers,
            )
            for val_data in self.val_datasets
        ]

    def predict_dataloader(self):
        return [
            torch.utils.data.DataLoader(
                predict_data,
                batch_size=self.predict_batch_size,
                collate_fn=self.collate_and_tokenize,
                num_workers=self.num_workers,
            )
            for predict_data in self.predict_datasets
        ]

    def collate_and_tokenize(self, samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        num_docs = []
        queries = []
        docs = []
        labels = []
        subtopics = []
        optimal_labels = []
        optimal_subtopics = []
        query_ids = []
        doc_ids = []
        for sample in samples:
            num_docs.append(len(sample["docs"]))
            query = sample["query"]
            queries.extend([query] * num_docs[-1])
            docs.extend(sample["docs"])
            labels.extend(sample["labels"])
            if sample["subtopics"]:
                subtopics.extend(sample["subtopics"])
            optimal_labels.extend(
                sample["optimal_labels"][: num_docs[-1]]
                + [0] * (num_docs[-1] - len(sample["optimal_labels"]))
            )
            if sample["optimal_subtopics"]:
                optimal_subtopics.extend(
                    sample["optimal_subtopics"][: num_docs[-1]]
                    + [0] * (num_docs[-1] - len(sample["optimal_subtopics"]))
                )
            query_ids.append(sample["query_id"])
            doc_ids.append(sample["doc_ids"])
        encoded = self.tokenizer(
            queries,
            docs,
            truncation=self.truncate,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        batch = {**encoded}
        batch["labels"] = torch.tensor(labels)
        batch["subtopics"] = torch.tensor(subtopics) if subtopics else None
        batch["optimal_labels"] = torch.tensor(optimal_labels)
        batch["optimal_subtopics"] = (
            torch.tensor(optimal_subtopics) if subtopics else None
        )
        batch["num_docs"] = num_docs
        batch["query_id"] = query_ids
        batch["doc_ids"] = doc_ids
        return batch
