from typing import Sequence, Type, Union

from lightning_ir.data.data import CrossEncoderTrainBatch
import torch
from lightning_ir.cross_encoder.model import CrossEncoderConfig, CrossEncoderModel
from lightning_ir.cross_encoder.module import CrossEncoderModule
from lightning_ir.cross_encoder.mono import (
    MonoBertModel,
    MonoElectraModel,
    MonoRobertaModel,
)
from lightning_ir.loss.loss import LossFunction
from transformers import (
    AutoConfig,
    AutoModel,
    BertModel,
    BertPreTrainedModel,
    ElectraModel,
    ElectraPreTrainedModel,
    PretrainedConfig,
    PreTrainedModel,
    RobertaModel,
    RobertaPreTrainedModel,
)

from set_encoder.set_encoder_bert import BertSetEncoderMixin
from set_encoder.set_encoder_electra import ElectraSetEncoderMixin
from set_encoder.set_encoder_mixin import SetEncoderMixin
from set_encoder.set_encoder_roberta import RoBERTaSetEncoderMixin


class SetEncoderConfig:
    depth: int = 100
    fill_random_docs: bool = True


def SetEncoderClassFactory(
    TransformerModel: Type[PreTrainedModel],
) -> Type[PreTrainedModel]:
    Mixin = get_mixin(TransformerModel)
    assert issubclass(TransformerModel.config_class, PretrainedConfig)
    model_name = TransformerModel.config_class.__name__.replace("Config", "").replace(
        "Mono", ""
    )

    ModelSetEncoderConfig = type(
        f"SetEncoder{model_name}Config",
        (TransformerModel.config_class, SetEncoderConfig),
        {"model_type": f"set-encoder-{model_name.lower()}"},
    )

    def __init__(
        self,
        config: PretrainedConfig,
        use_flash: bool = True,
    ) -> None:
        config.num_labels = 1
        TransformerModel.__init__(self, config)
        Mixin.__init__(
            self,
            TransformerModel.forward,
            use_flash,
            config.depth if config.fill_random_docs else None,
        )

    set_encoder_class = type(
        f"SetEncoder{model_name}Model",
        (Mixin, TransformerModel),
        {"__init__": __init__, "config_class": ModelSetEncoderConfig},
    )
    return set_encoder_class


def get_mixin(TransformerModel: Type[PreTrainedModel]) -> Type[SetEncoderMixin]:
    if issubclass(TransformerModel, BertPreTrainedModel):
        return BertSetEncoderMixin
    elif issubclass(TransformerModel, RobertaPreTrainedModel):
        return RoBERTaSetEncoderMixin
    elif issubclass(TransformerModel, ElectraPreTrainedModel):
        return ElectraSetEncoderMixin
    else:
        raise ValueError(
            f"Model type {TransformerModel.__name__} not supported by SetEncoder"
        )


class Pooler(torch.nn.Module):
    def __init__(self, encoder: BertModel | ElectraModel | RobertaModel) -> None:
        super().__init__()

    def forward(self, batch: CrossEncoderTrainBatch) -> torch.Tensor:
        num_docs = [len(doc_ids) for doc_ids in batch.doc_ids]
        logits = self.model.forward(
            batch.encoding.input_ids,
            batch.encoding.get("attention_mask", None),
            batch.encoding.get("token_type_ids", None),
            num_docs=num_docs,
        )
        logits = logits.view(len(batch.query_ids), -1)
        return logits
        self.encoder = encoder

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.last_hidden_state[:, 0, :]


SetEncoderBertModel = SetEncoderClassFactory(MonoBertModel)
SetEncoderElectraModel = SetEncoderClassFactory(MonoElectraModel)
SetEncoderRobertaModel = SetEncoderClassFactory(MonoRobertaModel)
SetEncoderBertConfig = SetEncoderBertModel.config_class
SetEncoderElectraConfig = SetEncoderElectraModel.config_class
SetEncoderRobertaConfig = SetEncoderRobertaModel.config_class


class SetEncoderBertModule(CrossEncoderModule):
    config_class = SetEncoderBertModel.config_class

    def __init__(
        self,
        model: CrossEncoderModel | None = None,
        model_name_or_path: str | None = None,
        config: CrossEncoderConfig | None = None,
        loss_functions: Sequence[LossFunction] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
        use_flash: bool = True,
    ) -> None:
        if model is None:
            if model_name_or_path is None:
                if config is None:
                    raise ValueError(
                        "Either model, model_name_or_path, or config must be provided."
                    )
                if not isinstance(config, SetEncoderBertConfig):
                    raise ValueError(
                        "To initialize a new model pass a SetEncoderBertConfig."
                    )
                model = SetEncoderBertModel(config, use_flash=use_flash)
            else:
                model = SetEncoderBertModel.from_pretrained(
                    model_name_or_path, config=config, use_flash=use_flash
                )
        else:
            if not isinstance(model, SetEncoderBertModel):
                raise ValueError("Incorrect model type. Expected SetEncoderBertModel.")
        super().__init__(model, loss_functions, evaluation_metrics)

    def forward(self, batch: CrossEncoderTrainBatch) -> torch.Tensor:
        num_docs = [len(doc_ids) for doc_ids in batch.doc_ids]
        logits = self.model.forward(
            batch.encoding.input_ids,
            batch.encoding.get("attention_mask", None),
            batch.encoding.get("token_type_ids", None),
            num_docs=num_docs,
        )
        logits = logits.view(len(batch.query_ids), -1)
        return logits


class SetEncoderElectraModule(CrossEncoderModule):
    config_class = SetEncoderElectraConfig

    def __init__(
        self,
        model: CrossEncoderModel | None = None,
        model_name_or_path: str | None = None,
        config: CrossEncoderConfig | None = None,
        loss_functions: Sequence[LossFunction] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
        use_flash: bool = True,
    ) -> None:
        if model is None:
            if model_name_or_path is None:
                if config is None:
                    raise ValueError(
                        "Either model, model_name_or_path, or config must be provided."
                    )
                if not isinstance(config, SetEncoderElectraConfig):
                    raise ValueError(
                        "To initialize a new model pass a SetEncoderElectraConfig."
                    )
                model = SetEncoderElectraModel(config, use_flash=use_flash)
            else:
                model = SetEncoderElectraModel.from_pretrained(
                    model_name_or_path, config=config, use_flash=use_flash
                )
        else:
            if not isinstance(model, SetEncoderElectraModel):
                raise ValueError(
                    "Incorrect model type. Expected SetEncoderElectraModel."
                )
        super().__init__(model, loss_functions, evaluation_metrics)

    def forward(self, batch: CrossEncoderTrainBatch) -> torch.Tensor:
        num_docs = [len(doc_ids) for doc_ids in batch.doc_ids]
        logits = self.model.forward(
            batch.encoding.input_ids,
            batch.encoding.get("attention_mask", None),
            batch.encoding.get("token_type_ids", None),
            num_docs=num_docs,
        )
        logits = logits.view(len(batch.query_ids), -1)
        return logits


class SetEncoderRobertaModule(CrossEncoderModule):
    config_class = SetEncoderRobertaConfig

    def __init__(
        self,
        model: CrossEncoderModel | None = None,
        model_name_or_path: str | None = None,
        config: CrossEncoderConfig | None = None,
        loss_functions: Sequence[LossFunction] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
        use_flash: bool = True,
    ) -> None:
        if model is None:
            if model_name_or_path is None:
                if config is None:
                    raise ValueError(
                        "Either model, model_name_or_path, or config must be provided."
                    )
                if not isinstance(config, SetEncoderRobertaConfig):
                    raise ValueError(
                        "To initialize a new model pass a SetEncoderRobertaConfig."
                    )
                model = SetEncoderRobertaModel(config, use_flash=use_flash)
            else:
                model = SetEncoderRobertaModel.from_pretrained(
                    model_name_or_path, config=config, use_flash=use_flash
                )
        else:
            if not isinstance(model, SetEncoderRobertaModel):
                raise ValueError(
                    "Incorrect model type. Expected SetEncoderRobertaModel."
                )
        super().__init__(model, loss_functions, evaluation_metrics)

    def forward(self, batch: CrossEncoderTrainBatch) -> torch.Tensor:
        num_docs = [len(doc_ids) for doc_ids in batch.doc_ids]
        logits = self.model.forward(
            batch.encoding.input_ids,
            batch.encoding.get("attention_mask", None),
            batch.encoding.get("token_type_ids", None),
            num_docs=num_docs,
        )
        logits = logits.view(len(batch.query_ids), -1)
        return logits


AutoConfig.register(SetEncoderBertConfig.model_type, SetEncoderBertConfig)
AutoModel.register(SetEncoderBertConfig, SetEncoderBertModel)
AutoConfig.register(SetEncoderElectraConfig.model_type, SetEncoderElectraConfig)
AutoModel.register(SetEncoderElectraConfig, SetEncoderElectraModel)
AutoConfig.register(SetEncoderRobertaConfig.model_type, SetEncoderRobertaConfig)
AutoModel.register(SetEncoderRobertaConfig, SetEncoderRobertaModel)
