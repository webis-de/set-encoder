from pathlib import Path

import pandas as pd
import torch
from lightning_ir.data.dataset import RunDataset, RunSample


class SubtopicRunDataset(RunDataset):

    def __init__(
        self,
        run_path: Path,
        depth: int,
        sample_size: int,
    ) -> None:
        super().__init__(run_path, depth, sample_size, "top")
        self.targets = "sub_topics"

    def load_qrels(self) -> pd.DataFrame | None:
        return None

    def __getitem__(self, idx: int) -> RunSample:
        query_id = str(self.query_ids[idx])
        group = self.run_groups.get_group(query_id).copy()
        query = self.queries[query_id]
        group = group.head(self.sample_size)

        doc_ids = tuple(group["doc_id"])
        docs = tuple(self.docs.get(doc_id).default_text() for doc_id in doc_ids)

        targets = torch.tensor(group["iteration"].values)
        return RunSample(query_id, query, doc_ids, docs, targets)
