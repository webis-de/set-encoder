from typing import Literal, Type
import pathlib

import pytest
import torch

from set_encoder.data.datamodule import SetEncoderDataModule
from set_encoder.model.set_encoder_module import SetEncoderModule
from set_encoder.model import loss_utils


class TestListwiseSetEncoderModule:
    @pytest.fixture()
    def model(self, model_name: str) -> SetEncoderModule:
        set_encoder_module = SetEncoderModule(model_name)
        return set_encoder_module

    def test_init(
        self,
        model: SetEncoderModule,
    ):
        assert model

    @pytest.mark.parametrize(
        "loss_function_cls",
        [
            loss_utils.RankNet,
            loss_utils.ApproxNDCG,
            loss_utils.ApproxMRR,
            loss_utils.ListNet,
            loss_utils.ListMLE,
            loss_utils.NeuralNDCG,
            loss_utils.NeuralMRR,
            loss_utils.LocalizedContrastive,
        ],
    )
    def test_training_step(
        self,
        model: SetEncoderModule,
        flash_datamodule: SetEncoderDataModule,
        loss_function_cls: Type[loss_utils.LossFunc],
    ):
        dataloader = flash_datamodule.train_dataloader()
        batch = next(iter(dataloader))
        model.loss_function = loss_function_cls()
        loss = model.training_step(batch, 0)
        assert loss.requires_grad
        assert loss > 0
        assert torch.isnan(loss) == False

    @pytest.mark.parametrize("validation_metric", ["mrr", "ndcg"])
    def test_validation_step(
        self,
        model: SetEncoderModule,
        flash_datamodule: SetEncoderDataModule,
        validation_metric: Literal["mrr", "ndcg"],
    ):
        dataloader = flash_datamodule.val_dataloader()[0]
        model.validation_metric = validation_metric
        batch = next(iter(dataloader))
        with torch.inference_mode():
            model.validation_step(batch, 0)
            assert len(model.validation_step_outputs)
            for _, value in model.validation_step_outputs:
                value = sum(value) / len(value)
                assert value
                assert value <= 1
