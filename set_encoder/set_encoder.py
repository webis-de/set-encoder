from typing import Sequence, Type

import torch
from lightning_ir.cross_encoder.model import CrossEncoderConfig, CrossEncoderModel
from lightning_ir.cross_encoder.module import CrossEncoderModule, LightningIRModule
from lightning_ir.cross_encoder.mono import (
    MonoBertModel,
    MonoElectraModel,
    MonoRobertaModel,
)
from lightning_ir.data.data import CrossEncoderRunBatch
from lightning_ir.loss.loss import LossFunction
from transformers import (
    AutoConfig,
    AutoModel,
    BertPreTrainedModel,
    ElectraPreTrainedModel,
    PretrainedConfig,
    PreTrainedModel,
    RobertaPreTrainedModel,
)

from set_encoder.set_encoder_bert import BertSetEncoderMixin
from set_encoder.set_encoder_electra import ElectraSetEncoderMixin
from set_encoder.set_encoder_mixin import SetEncoderMixin
from set_encoder.set_encoder_roberta import RoBERTaSetEncoderMixin
from set_encoder.tokenizer import SetEncoderTokenizer


def SetEncoderClassFactory(
    TransformerModel: Type[PreTrainedModel],
) -> Type[PreTrainedModel]:
    Mixin = get_mixin(TransformerModel)
    config_class = TransformerModel.config_class
    assert issubclass(config_class, PretrainedConfig)
    model_name = config_class.__name__.replace("Config", "").replace("Mono", "")

    def config_init(
        self,
        *args,
        depth: int = 100,
        add_extra_token: bool = False,
        other_sequence_embedding: bool = False,
        **kwargs,
    ):
        config_class.__init__(self, *args, **kwargs)
        self.depth = depth
        self.add_extra_token = add_extra_token
        self.other_sequence_embedding = other_sequence_embedding

    ModelSetEncoderConfig = type(
        f"SetEncoder{model_name}Config",
        (config_class,),
        {
            "__init__": config_init,
            "model_type": f"set-encoder-{model_name.lower()}",
            "ADDED_ARGS": (
                CrossEncoderConfig.ADDED_ARGS
                + ["depth", "add_extra_token", "other_sequence_embedding"]
            ),
            "TOKENIZER_ARGS": (CrossEncoderConfig.TOKENIZER_ARGS + ["add_extra_token"]),
            "Tokenizer": SetEncoderTokenizer,
        },
    )

    def model_init(
        self,
        config: PretrainedConfig,
        use_flash: bool = True,
    ) -> None:
        TransformerModel.__init__(self, config)
        Mixin.__init__(self, config, TransformerModel.forward, use_flash)

    set_encoder_class = type(
        f"SetEncoder{model_name}Model",
        (Mixin, TransformerModel),
        {"__init__": model_init, "config_class": ModelSetEncoderConfig},
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


SetEncoderBertModel = SetEncoderClassFactory(MonoBertModel)
SetEncoderElectraModel = SetEncoderClassFactory(MonoElectraModel)
SetEncoderRobertaModel = SetEncoderClassFactory(MonoRobertaModel)
SetEncoderBertConfig = SetEncoderBertModel.config_class
SetEncoderElectraConfig = SetEncoderElectraModel.config_class
SetEncoderRobertaConfig = SetEncoderRobertaModel.config_class


class SetEncoderModule(CrossEncoderModule):

    def __init__(
        self,
        model: CrossEncoderModel,
        loss_functions: Sequence[LossFunction] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
    ):
        super().__init__(model, loss_functions, evaluation_metrics)
        if (
            self.config.add_extra_token
            and len(self.tokenizer) != self.config.vocab_size
        ):
            self.model.encoder.resize_token_embeddings(len(self.tokenizer), 8)

    def forward(self, batch: CrossEncoderRunBatch) -> torch.Tensor:
        num_docs = [len(doc_ids) for doc_ids in batch.doc_ids]
        logits = self.model.forward(
            batch.encoding.input_ids,
            batch.encoding.get("attention_mask", None),
            batch.encoding.get("token_type_ids", None),
            num_docs=num_docs,
        )
        return logits


class SetEncoderBertModule(SetEncoderModule):
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
                kwargs = {}
                if config is not None:
                    kwargs = config.to_added_args_dict()
                config = SetEncoderBertConfig.from_pretrained(
                    model_name_or_path, **kwargs
                )
                model = SetEncoderBertModel.from_pretrained(
                    model_name_or_path, config=config, use_flash=use_flash
                )
        else:
            if not isinstance(model, SetEncoderBertModel):
                raise ValueError("Incorrect model type. Expected SetEncoderBertModel.")
        super().__init__(model, loss_functions, evaluation_metrics)


class SetEncoderElectraModule(SetEncoderModule):
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
                kwargs = {}
                if config is not None:
                    kwargs = config.to_added_args_dict()
                config = SetEncoderElectraConfig.from_pretrained(
                    model_name_or_path, **kwargs
                )
                model = SetEncoderElectraModel.from_pretrained(
                    model_name_or_path, config=config, use_flash=use_flash
                )
        else:
            if not isinstance(model, SetEncoderElectraModel):
                raise ValueError(
                    "Incorrect model type. Expected SetEncoderElectraModel."
                )
        super().__init__(model, loss_functions, evaluation_metrics)


class SetEncoderRobertaModule(SetEncoderModule):
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
                kwargs = {}
                if config is not None:
                    kwargs = config.to_added_args_dict()
                config = SetEncoderRobertaConfig.from_pretrained(
                    model_name_or_path, **kwargs
                )
                model = SetEncoderRobertaModel.from_pretrained(
                    model_name_or_path, config=config, use_flash=use_flash
                )
        else:
            if not isinstance(model, SetEncoderRobertaModel):
                raise ValueError(
                    "Incorrect model type. Expected SetEncoderRobertaModel."
                )
        super().__init__(model, loss_functions, evaluation_metrics)


AutoConfig.register(SetEncoderBertConfig.model_type, SetEncoderBertConfig)
AutoModel.register(SetEncoderBertConfig, SetEncoderBertModel)
AutoConfig.register(SetEncoderElectraConfig.model_type, SetEncoderElectraConfig)
AutoModel.register(SetEncoderElectraConfig, SetEncoderElectraModel)
AutoConfig.register(SetEncoderRobertaConfig.model_type, SetEncoderRobertaConfig)
AutoModel.register(SetEncoderRobertaConfig, SetEncoderRobertaModel)
