import pytest
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from set_encoder.model.set_encoder import SetEncoderClassFactory


@pytest.mark.parametrize(
    "model_name",
    [
        "bert-base-uncased",
        "roberta-base",
        "google/electra-base-discriminator",
        "microsoft/deberta-v3-base",
    ],
)
def test_same_as_model(model_name: str) -> None:
    config = AutoConfig.from_pretrained(model_name)
    model_class = AutoModelForSequenceClassification._model_mapping[type(config)]
    SetEncoder = SetEncoderClassFactory(model_class)
    set_encoder = SetEncoder.from_pretrained(model_name, other_doc_attention=False)
    set_encoder = set_encoder.eval()
    base_model = model_class.from_pretrained(model_name, num_labels=1)
    base_model = base_model.eval()
    for base_param, set_encoder_param in zip(
        base_model.parameters(), set_encoder.parameters()
    ):
        set_encoder_param.data.copy_(base_param.data)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    queries = [
        "What is the meaning of life?",
        "What is the meaning of life?",
        "What is the meaning of death?",
    ]
    docs = [
        "The meaning of life is to be happy.",
        "The meaning of life is to be happy.",
        "Death is meaningless.",
    ]
    num_docs = [2, 1]
    encoded = tokenizer(queries, docs, return_tensors="pt", padding=True)

    with torch.no_grad():
        base_output = base_model(**encoded)
        set_encoder_output = set_encoder(**encoded, num_docs=num_docs)

    assert torch.allclose(
        base_output.logits, set_encoder_output.logits, atol=1e-4, rtol=1e-4
    )


def test_rank_position_embeddings() -> None:
    model_name = "bert-base-uncased"
    config = AutoConfig.from_pretrained(model_name)
    model_class = AutoModelForSequenceClassification._model_mapping[type(config)]
    SetEncoder = SetEncoderClassFactory(model_class)
    set_encoder = SetEncoder.from_pretrained(
        model_name, rank_position_embeddings=True, depth=100
    )
    set_encoder = set_encoder.eval()
    base_model = model_class.from_pretrained(model_name)
    base_model = base_model.eval()
    set_encoder.classifier = base_model.classifier
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    queries = [
        "What is the meaning of life?",
        "What is the meaning of life?",
        "What is the meaning of death?",
    ]
    docs = [
        "The meaning of life is to be happy.",
        "The meaning of life is to be happy.",
        "Death is meaningless.",
    ]
    num_docs = [2, 1]
    encoded = tokenizer(queries, docs, return_tensors="pt", padding=True)

    with torch.no_grad():
        base_output = base_model(**encoded)
        set_encoder_output = set_encoder(**encoded, num_docs=num_docs)

    assert not torch.allclose(
        base_output.logits, set_encoder_output.logits, atol=1e-4, rtol=1e-4
    )


def test_other_doc() -> None:
    model_name = "bert-base-uncased"
    config = AutoConfig.from_pretrained(model_name)
    model_class = AutoModelForSequenceClassification._model_mapping[type(config)]
    SetEncoder = SetEncoderClassFactory(model_class)
    set_encoder = SetEncoder.from_pretrained(model_name, other_doc_attention=True)
    set_encoder = set_encoder.eval()
    base_model = model_class.from_pretrained(model_name)
    base_model = base_model.eval()
    set_encoder.classifier = base_model.classifier
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    queries = [
        "What is the meaning of life?",
        "What is the meaning of life?",
        "What is the meaning of death?",
    ]
    docs = [
        "The meaning of life is to be happy.",
        "The meaning of life is to be happy.",
        "Death is meaningless.",
    ]
    num_docs = [2, 1]
    encoded = tokenizer(queries, docs, return_tensors="pt", padding=True)

    with torch.no_grad():
        base_output = base_model(**encoded)
        set_encoder_output = set_encoder(**encoded, num_docs=num_docs)

    assert not torch.allclose(
        base_output.logits, set_encoder_output.logits, atol=1e-4, rtol=1e-4
    )


def test_extra_other_doc_token() -> None:
    model_name = "bert-base-uncased"
    config = AutoConfig.from_pretrained(model_name)
    model_class = AutoModelForSequenceClassification._model_mapping[type(config)]
    SetEncoder = SetEncoderClassFactory(model_class)
    set_encoder_1 = SetEncoder.from_pretrained(model_name, other_doc_attention=True)
    set_encoder_2 = SetEncoder.from_pretrained(
        model_name, extra_other_doc_token=True, other_doc_attention=True
    )
    set_encoder_1 = set_encoder_1.eval()
    set_encoder_2 = set_encoder_2.eval()
    set_encoder_2.classifier = set_encoder_1.classifier
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    queries = [
        "What is the meaning of life?",
        "What is the meaning of life?",
        "What is the meaning of death?",
    ]
    docs = [
        "The meaning of life is to be happy.",
        "The meaning of life is to be happy.",
        "Death is meaningless.",
    ]
    num_docs = [2, 1]
    encoded = tokenizer(queries, docs, return_tensors="pt", padding=True)

    with torch.no_grad():
        set_encoder_1_output = set_encoder_1(**encoded, num_docs=num_docs)
        set_encoder_2_output = set_encoder_2(**encoded, num_docs=num_docs)

    assert not torch.allclose(
        set_encoder_1_output.logits, set_encoder_2_output.logits, atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize("extra_other_doc_token", [True, False])
def test_doc_order_invariance(extra_other_doc_token: bool) -> None:
    model_name = "bert-base-uncased"
    config = AutoConfig.from_pretrained(model_name)
    model_class = AutoModelForSequenceClassification._model_mapping[type(config)]
    SetEncoder = SetEncoderClassFactory(model_class)
    set_encoder = SetEncoder.from_pretrained(
        model_name,
        other_doc_attention=True,
        extra_other_doc_token=extra_other_doc_token,
    )
    set_encoder = set_encoder.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        set_encoder_output = set_encoder(**encoded, num_docs=num_docs)
    logits_1 = set_encoder_output.logits[: len(queries) // 2]
    logits_2 = set_encoder_output.logits[len(queries) // 2 :]
    logits_2[:2] = logits_2[:2].flip(0)
    assert torch.allclose(logits_1, logits_2, atol=1e-4, rtol=1e-4)
