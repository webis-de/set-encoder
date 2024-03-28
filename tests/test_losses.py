import torch
import pytest

from set_encoder.model.loss_utils import (
    RankNet,
    ApproxNDCG,
    ApproxMRR,
    ListNet,
    ListMLE,
    NeuralNDCG,
    NeuralMRR,
    LocalizedContrastive,
    PAD_VALUE,
)


SEQ_LENS = [4, 8]


@pytest.fixture
def scores() -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(
        [torch.randn(seq_len) for seq_len in SEQ_LENS],
        padding_value=PAD_VALUE,
        batch_first=True,
    ).requires_grad_(True)


@pytest.fixture
def labels() -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(
        [torch.randint(0, 4, (seq_len,)) for seq_len in SEQ_LENS],
        padding_value=PAD_VALUE,
        batch_first=True,
    )


def test_ranknet(scores: torch.Tensor, labels: torch.Tensor) -> None:
    loss_func = RankNet()
    loss = loss_func.compute(scores, labels)
    assert loss
    loss.backward()


def test_approx_ndcg(scores: torch.Tensor, labels: torch.Tensor) -> None:
    loss_func = ApproxNDCG()
    loss = loss_func.compute(scores, labels)
    assert loss
    loss.backward()


def test_approx_mrr(scores: torch.Tensor, labels: torch.Tensor) -> None:
    loss_func = ApproxMRR()
    loss = loss_func.compute(scores, labels)
    assert loss
    loss.backward()


def test_listnet(scores: torch.Tensor, labels: torch.Tensor) -> None:
    loss_func = ListNet()
    loss = loss_func.compute(scores, labels)
    assert loss
    loss.backward()


def test_listmle(scores: torch.Tensor, labels: torch.Tensor) -> None:
    loss_func = ListMLE()
    loss = loss_func.compute(scores, labels)
    assert loss
    loss.backward()


def test_neuralndcg(scores: torch.Tensor, labels: torch.Tensor) -> None:
    loss_func = NeuralNDCG()
    loss = loss_func.compute(scores, labels)
    assert loss
    loss.backward()


def test_neuralmrr(scores: torch.Tensor, labels: torch.Tensor) -> None:
    loss_func = NeuralMRR()
    loss = loss_func.compute(scores, labels)
    assert loss
    loss.backward()


def test_localizedcontrastive(scores: torch.Tensor, labels: torch.Tensor) -> None:
    loss_func = LocalizedContrastive()
    loss = loss_func.compute(scores, labels)
    assert loss
    loss.backward()
