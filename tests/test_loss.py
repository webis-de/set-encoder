from typing import Type

import pytest
import torch

from set_encoder.loss import ApproxAlphaNDCG, ApproxERRIA, ApproxLossFunction


@pytest.fixture(scope="module")
def logits():
    return torch.randn(5, 10, requires_grad=True)


@pytest.fixture(scope="module")
def labels():
    return torch.randint(0, 4, (5, 10, 8))


@pytest.mark.parametrize("LossFunc", [ApproxAlphaNDCG, ApproxERRIA])
def test_loss(
    logits: torch.Tensor, labels: torch.Tensor, LossFunc: Type[ApproxLossFunction]
):
    loss_func = LossFunc()
    loss = loss_func.compute_loss(logits, labels)
    assert loss >= 0
    assert loss.requires_grad
