from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Union

import pytest
import torch
from _pytest.fixtures import SubRequest
from lightning_ir.cross_encoder.model import CrossEncoderConfig
from lightning_ir.data.datamodule import LightningIRDataModule
from lightning_ir.data.dataset import RunDataset, TupleDataset
from lightning_ir.loss.loss import RankNet, SupervisedMarginMSE

from set_encoder.set_encoder import (
    SetEncoderBertModel,
    SetEncoderBertModule,
    SetEncoderElectraModel,
    SetEncoderElectraModule,
    SetEncoderRobertaModel,
    SetEncoderRobertaModule,
)
from set_encoder.tokenizer import SetEncoderTokenizer

DATA_DIR = Path(__file__).parent / "data"

MODULE_MAP = {
    SetEncoderBertModel: SetEncoderBertModule,
    SetEncoderElectraModel: SetEncoderElectraModule,
    SetEncoderRobertaModel: SetEncoderRobertaModule,
}
MODELS = Union[SetEncoderBertModel, SetEncoderElectraModel, SetEncoderRobertaModel]
MODULES = Union[SetEncoderBertModule, SetEncoderElectraModule, SetEncoderRobertaModule]
MODEL_NAME_OR_PATH_MAP = {
    SetEncoderBertModule: "sentence-transformers/all-MiniLM-L6-v2",
    SetEncoderElectraModule: "google/electra-small-discriminator",
    SetEncoderRobertaModule: "FacebookAI/roberta-base",
    SetEncoderBertModel: "sentence-transformers/all-MiniLM-L6-v2",
    SetEncoderElectraModel: "google/electra-small-discriminator",
    SetEncoderRobertaModel: "FacebookAI/roberta-base",
}


@pytest.fixture(scope="module", params=product(list(MODULE_MAP.keys()), [True, False]))
def model(request: SubRequest) -> MODELS:
    Model, add_extra_token = request.param
    model_name_or_path = MODEL_NAME_OR_PATH_MAP[Model]
    config = Model.config_class.from_pretrained(
        model_name_or_path, num_hidden_layers=1, add_extra_token=add_extra_token
    )
    _model = Model.from_pretrained(model_name_or_path, config=config)
    if config.add_extra_token:
        _model.encoder.resize_token_embeddings(_model.config.vocab_size + 1, 8)
    return _model


@pytest.fixture(
    scope="module",
    params=[RankNet(), SupervisedMarginMSE()],
)
def module(model: MODELS, request: SubRequest) -> MODULES:
    loss_function = request.param
    _module = MODULE_MAP[type(model)](
        model, loss_functions=[loss_function], evaluation_metrics=["nDCG@10", "loss"]
    )
    return _module


def test_doc_order_invariance(model: MODELS) -> None:
    tokenizer = SetEncoderTokenizer.from_pretrained(
        model.config.name_or_path, **model.config.to_tokenizer_dict()
    )

    queries = [
        "What is the meaning of life?",
        "What is the meaning of life?",
        "What is the meaning of death?",
    ] * 2
    docs = [
        "The meaning of life is to be happy.",
        "I don't know what the meaning of life is.",
        "Death is meaningless.",
        "I don't know what the meaning of life is.",
        "The meaning of life is to be happy.",
        "Death is meaningless.",
    ]
    num_docs = [2, 1] * 2
    encoded = tokenizer(queries, docs, return_tensors="pt", padding=True)
    with torch.no_grad():
        output = model(**encoded, num_docs=num_docs)
    logits_1 = output[: len(queries) // 2]
    logits_2 = output[len(queries) // 2 :]
    logits_2[:2] = logits_2[:2].flip(0)
    assert torch.allclose(logits_1, logits_2, atol=1e-4, rtol=1e-4)


@lru_cache
def tuples_datamodule(model: MODELS) -> LightningIRDataModule:
    inference_datasets = [
        RunDataset(
            run_path,
            depth=10,
            sample_size=10,
            sampling_strategy="top",
            targets="relevance",
        )
        for run_path in [
            DATA_DIR / "clueweb09-en-trec-web-2009-diversity.jsonl",
            DATA_DIR / "msmarco-passage-trec-dl-2019-judged.run",
        ]
    ]
    datamodule = LightningIRDataModule(
        model_name_or_path=model.config.name_or_path,
        config=model.config,
        num_workers=0,
        train_batch_size=3,
        inference_batch_size=3,
        train_dataset=TupleDataset(
            "msmarco-passage/train/kd-docpairs", targets="score", num_docs=2
        ),
        inference_datasets=inference_datasets,
    )
    datamodule.setup(stage="fit")
    return datamodule


def test_training_step(module: MODULES):
    datamodule = tuples_datamodule(module.model)
    dataloader = datamodule.train_dataloader()
    batch = next(iter(dataloader))
    loss = module.training_step(batch, 0)
    assert loss


def test_validation(module: MODULES):
    datamodule = tuples_datamodule(module.model)
    dataloader = datamodule.val_dataloader()[0]
    for batch, batch_idx in zip(dataloader, range(2)):
        module.validation_step(batch, batch_idx, 0)

    metrics = module.on_validation_epoch_end()
    assert metrics is not None
    for key, value in metrics.items():
        metric = key.split("/")[1]
        assert metric in {"nDCG@10"} or "validation" in metric
        assert value


def test_seralize_deserialize(model: MODELS, tmpdir_factory: pytest.TempdirFactory):
    save_dir = tmpdir_factory.mktemp(model.config_class.model_type)
    model.save_pretrained(save_dir)
    kwargs = {}
    new_model = type(model).from_pretrained(save_dir, **kwargs)
    for key, value in model.config.__dict__.items():
        if key in (
            "torch_dtype",
            "_name_or_path",
            "_commit_hash",
            "transformers_version",
            "model_type",
        ):
            continue
        assert getattr(new_model.config, key) == value
    for key, value in model.state_dict().items():
        assert new_model.state_dict()[key].equal(value)
