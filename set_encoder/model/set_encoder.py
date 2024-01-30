from typing import Type

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel
from transformers.models.electra.modeling_electra import ElectraPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

from set_encoder.model.set_encoder_bert import BertSetEncoderMixin
from set_encoder.model.set_encoder_deberta import DebertaV2SetEncoderMixin
from set_encoder.model.set_encoder_electra import ElectraSetEncoderMixin
from set_encoder.model.set_encoder_mixin import SetEncoderMixin
from set_encoder.model.set_encoder_roberta import RoBERTaSetEncoderMixin

# from set_encoder.model.set_encoder_deberta import
# from set_encoder.model.set_encoder_t5 import


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

    def __init__(self, config: PretrainedConfig, use_flash: bool = True) -> None:
        config.num_labels = 1
        TransformerModel.__init__(self, config)
        Mixin.__init__(self, use_flash)
        self.forward = self.model_forward_wrapper(self.forward)
        encoder = getattr(self, self.encoder_name)
        encoder.get_extended_attention_mask = self.extended_attention_mask_wrapper(
            encoder.get_extended_attention_mask
        )
        for name, module in self.named_modules():
            if self.config.rank_position_embeddings and name.endswith(
                f"{self.encoder_name}.embeddings"
            ):
                module.add_module(
                    "rank_position_embeddings",
                    torch.nn.Embedding(self.config.depth, self.config.hidden_size),
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
    # elif issubclass(TransformerModel, DebertaPreTrainedModel):
    #     return DeBERTaSetEncoderMixin
    elif issubclass(TransformerModel, DebertaV2PreTrainedModel):
        return DebertaV2SetEncoderMixin
    # elif issubclass(TransformerModel, T5PreTrainedModel):
    #     return T5SetEncoderMixin
    elif issubclass(TransformerModel, ElectraPreTrainedModel):
        return ElectraSetEncoderMixin
    else:
        raise ValueError(
            f"Model type {TransformerModel.__name__} not supported by SetEncoder"
        )
