from pathlib import Path

import torch
from set_encoder.model.set_encoder import SetEncoderModelForSequenceClassification

import re


def extend_position_embeddings(run_path: Path, max_position_embeddings: int) -> None:
    # NOTE DON'T PASS ALREADY CONVERTED MODEL!!!
    model = SetEncoderModelForSequenceClassification.from_pretrained(
        run_path / "files" / "huggingface_checkpoint",
        max_position_embeddings=max_position_embeddings,
    )
    ckpt_paths = list((run_path / "files" / "checkpoints").glob("epoch=*.ckpt"))
    assert len(ckpt_paths) == 1
    ckpt_path = ckpt_paths[0]
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    pl_config_path = run_path / "files" / "pl_config.yaml"
    pl_config_text = pl_config_path.read_text()
    match = re.findall(r"(?<!base_)max_position_embeddings: \d+", pl_config_text)
    assert len(match) == 1
    pl_config_text = pl_config_text.replace(
        match[0],
        f"max_position_embeddings: {max_position_embeddings}",
    )
    pl_config_path.write_text(pl_config_text)

    extended_state_dict = model.state_dict()
    new_state_dict = checkpoint["state_dict"].copy()
    for key in extended_state_dict.keys():
        if extended_state_dict[key].shape != new_state_dict["set_encoder." + key].shape:
            new_state_dict["set_encoder." + key] = extended_state_dict[key]
        else:
            assert torch.allclose(
                extended_state_dict[key], new_state_dict["set_encoder." + key]
            )
    checkpoint["state_dict"] = new_state_dict
    torch.save(checkpoint, ckpt_path)
    model.save_pretrained(run_path / "files" / "huggingface_checkpoint")


def reduce_position_embeddings(run_path: Path, max_position_embeddings: int) -> None:
    model = SetEncoderModelForSequenceClassification.from_pretrained(
        run_path / "files" / "huggingface_checkpoint",
        max_position_embeddings=max_position_embeddings,
    )
    model.config.base_max_position_embeddings = max_position_embeddings
    ckpt_paths = list((run_path / "files" / "checkpoints").glob("epoch=*.ckpt"))
    assert len(ckpt_paths) == 1
    ckpt_path = ckpt_paths[0]
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    pl_config_path = run_path / "files" / "pl_config.yaml"
    pl_config_text = pl_config_path.read_text()
    match = re.findall(r"(?<!base_)max_position_embeddings: \d+", pl_config_text)
    assert len(match) == 1
    pl_config_text = pl_config_text.replace(
        match[0],
        f"max_position_embeddings: {max_position_embeddings}",
    )
    pl_config_path.write_text(pl_config_text)

    state_dict = checkpoint["state_dict"]
    state_dict["set_encoder.bert.embeddings.position_embeddings.weight"] = state_dict[
        "set_encoder.bert.embeddings.position_embeddings.weight"
    ][:max_position_embeddings]
    state_dict["set_encoder.bert.embeddings.position_ids"] = state_dict[
        "set_encoder.bert.embeddings.position_ids"
    ][:, :max_position_embeddings]
    torch.save(checkpoint, ckpt_path)

    model.bert.embeddings.position_embeddings.num_embeddings = max_position_embeddings
    model.bert.embeddings.position_embeddings.weight = torch.nn.Parameter(
        model.bert.embeddings.position_embeddings.weight[:max_position_embeddings]
    )
    model.bert.embeddings.position_ids = model.bert.embeddings.position_ids[
        :, :max_position_embeddings
    ]
    model.save_pretrained(run_path / "files" / "huggingface_checkpoint")
