from pathlib import Path

import pytest
import torch
from lightning_ir.data import RankBatch
from lightning_ir.loss.loss import RankNet

from set_encoder.set_encoder import SetEncoderConfig, SetEncoderModule

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def module() -> SetEncoderModule:
    return SetEncoderModule(
        "bert-base-uncased",
        config=SetEncoderConfig(add_extra_token=True, sample_missing_docs=False),
        loss_functions=[RankNet()],
    ).eval()


def test_doc_order_invariance(module: SetEncoderModule) -> None:
    _docs = (
        "The meaning of life is to be happy.",
        "I don't know what the meaning of life is.",
        "Death is meaningless.",
        "Death is meaningless.",
        "Foo bla bar." * 10,
    )
    docs = (_docs, _docs[::-1])
    doc_ids = (("1", "2", "3", "4", "5"), ("5", "4", "3", "2", "1"))
    queries = [
        "What is the meaning of life?",
    ] * 2
    rank_batch = RankBatch(("1", "2"), tuple(queries), doc_ids, docs)
    with torch.no_grad():
        output = module(rank_batch)
    scores_1 = output.scores[: len(_docs)]
    scores_2 = output.scores[len(_docs) :].flip(0)
    assert torch.allclose(scores_1, scores_2, atol=1e-4, rtol=1e-4)
