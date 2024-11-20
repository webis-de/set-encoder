import random
from pathlib import Path
from typing import Iterator, Literal, Tuple

import ir_datasets
import pandas as pd
import torch
from ir_datasets.datasets.base import Dataset
from ir_datasets.formats import TrecSubQrels
from ir_datasets.util import Cache
from lightning_ir.data.dataset import (
    GenericDocPair,
    RankSample,
    RunDataset,
    ScoredDocTuple,
    TupleDataset,
)


def register_trec_dl_subtopics():
    data_dir = Path("/mnt/ceph/storage/data-tmp/current/fschlatt/set_encoder/data/baseline-runs/" "colbert-subtopics")
    for year in ("2019", "2020"):
        base_dataset_id = f"msmarco-passage/trec-dl-{year}/judged"
        dataset_id = base_dataset_id + "/subtopics"
        if dataset_id in ir_datasets.registry:
            continue
        qrels_path = (data_dir / base_dataset_id.replace("/", "-")).with_suffix(".qrels")
        ir_dataset = ir_datasets.load(base_dataset_id)
        collection = ir_dataset.docs_handler()
        queries = ir_dataset.queries_handler()
        qrels = TrecSubQrels(Cache(None, qrels_path), "")
        dataset = Dataset(collection, queries, qrels)
        ir_datasets.registry.register(dataset_id, dataset)


def register_trec_dl_novelty():
    data_dir = Path("/mnt/ceph/storage/data-tmp/current/fschlatt/set_encoder/data/baseline-runs/" "trec-dl-novelty")
    for suffix, year in zip(("", "", "-v2", "-v2"), ("2019", "2020", "2021", "2022")):
        base_dataset_id = f"msmarco-passage{suffix}/trec-dl-{year}/judged"
        dataset_id = base_dataset_id + "/novelty"
        if dataset_id in ir_datasets.registry:
            continue
        qrels_path = (data_dir / base_dataset_id.replace("/", "-")).with_suffix(".qrels")
        ir_dataset = ir_datasets.load(base_dataset_id)
        collection = ir_dataset.docs_handler()
        queries = ir_dataset.queries_handler()
        qrels = TrecSubQrels(Cache(None, qrels_path), "")
        dataset = Dataset(collection, queries, qrels)
        ir_datasets.registry.register(dataset_id, dataset)


class SubtopicRunDataset(RunDataset):

    def __init__(
        self,
        run_path: Path,
        depth: int,
        sample_size: int,
    ) -> None:
        super().__init__(run_path, depth, sample_size, "top")
        self.targets = "sub_topics"

    def load_qrels(self, *args, **kwargs) -> pd.DataFrame | None:
        return None

    def __getitem__(self, idx: int) -> RankSample:
        query_id = str(self.query_ids[idx])
        group = self.run_groups.get_group(query_id).copy()
        query = self.queries[query_id]
        group = group.head(self.sample_size)

        doc_ids = tuple(group["doc_id"])
        docs = tuple(self.docs.get(doc_id).default_text() for doc_id in doc_ids)

        targets = torch.tensor(group["iteration"].values)
        return RankSample(query_id, query, doc_ids, docs, targets)


class RepeatRunDataset(RunDataset):

    def __getitem__(self, idx: int) -> RankSample:
        sample = super().__getitem__(idx)
        doc_ids = sample.doc_ids
        docs = sample.docs
        repeat_idx = random.randint(1, len(doc_ids) - 1)
        doc_ids = doc_ids + (doc_ids[repeat_idx],)
        docs = docs + (docs[repeat_idx],)
        targets = None
        if sample.targets is not None:
            repeat_targets = torch.zeros((len(doc_ids), 1))
            repeat_targets[[repeat_idx, -1]] = 1
            original_targets = sample.targets
            original_targets = torch.cat([original_targets, torch.zeros((1, 1))])
            targets = torch.cat([original_targets, repeat_targets], axis=1)
        sample = RankSample(sample.query_id, sample.query, doc_ids, docs, targets)
        return sample


class DummyDataset(TupleDataset):

    def __init__(
        self,
        tuples_dataset: str,
        num_docs: int = 2,
        targets: Literal["duplicate"] = "duplicate",
    ) -> None:
        super().__init__(tuples_dataset, targets, num_docs)

    def parse_sample(self, sample: ScoredDocTuple | GenericDocPair) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        if isinstance(sample, GenericDocPair):
            doc_ids = (sample.doc_id_a, sample.doc_id_b)
        elif isinstance(sample, ScoredDocTuple):
            doc_ids = sample.doc_ids
        else:
            raise ValueError("Invalid sample type.")
        docs = tuple(self.docs.get(doc_id).default_text() for doc_id in doc_ids)
        return doc_ids, docs

    def __iter__(self) -> Iterator[RankSample]:
        for sample in self.ir_dataset.docpairs_iter():
            query_id = sample.query_id
            query = self.queries.loc[query_id]
            doc_ids, docs = self.parse_sample(sample)
            idx = torch.randint(len(doc_ids), (1,))
            doc_ids = doc_ids + (doc_ids[idx],) * (self.num_docs - 1)
            docs = docs + (docs[idx],) * (self.num_docs - 1)
            targets = torch.zeros(len(docs))
            targets[idx] = 1
            targets[-(self.num_docs - 1) :] = 1
            yield RankSample(query_id, query, doc_ids, docs, targets)
