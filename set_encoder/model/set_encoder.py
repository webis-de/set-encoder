from typing import Type

from transformers import PretrainedConfig, PreTrainedModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.models.electra.modeling_electra import ElectraPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers.models.t5.modeling_t5 import T5PreTrainedModel

from set_encoder.model.set_encoder_bert import BertSetEncoderMixin
from set_encoder.model.set_encoder_electra import ElectraSetEncoderMixin
from set_encoder.model.set_encoder_mixin import SetEncoderMixin
from set_encoder.model.set_encoder_roberta import RoBERTaSetEncoderMixin


def SetEncoderClassFactory(
    TransformerModel: Type[PreTrainedModel],
) -> Type[PreTrainedModel]:
    Mixin = get_mixin(TransformerModel)

    assert issubclass(TransformerModel.config_class, PretrainedConfig)
    SetEncoderConfig = type(
        "SetEncoderConfig",
        (TransformerModel.config_class,),
        {
            "depth": 100,
            "other_doc_attention": False,
            "average_doc_embeddings": False,
            "num_labels": 1,
        },
    )

    def __init__(
        self,
        config: PretrainedConfig,
        use_flash: bool = True,
        fill_random_docs: bool = True,
    ) -> None:
        config.num_labels = 1
        TransformerModel.__init__(self, config)
        Mixin.__init__(
            self,
            TransformerModel.forward,
            use_flash,
            config.depth if fill_random_docs else None,
        )

    set_encoder_class = type(
        "SetEncoderModel",
        (Mixin, TransformerModel),
        {"__init__": __init__, "config_class": SetEncoderConfig},
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
