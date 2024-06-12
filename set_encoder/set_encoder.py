from dataclasses import dataclass
from typing import Dict, Sequence, Type

import torch
from lightning_ir.cross_encoder.model import (
    CrossEncoderConfig,
    CrossEncoderModel,
    CrossEncoderOuput,
)
from lightning_ir.cross_encoder.module import CrossEncoderModule
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
from set_encoder.loss import RepeatLossFunction
from set_encoder.data import register_trec_dl_novelty, register_trec_dl_subtopics

register_trec_dl_subtopics()
register_trec_dl_novelty()


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
        sample_missing_docs: bool = True,
        **kwargs,
    ):
        config_class.__init__(self, *args, **kwargs)
        self.depth = depth
        self.add_extra_token = add_extra_token
        self.other_sequence_embedding = other_sequence_embedding
        self.sample_missing_docs = sample_missing_docs

    ModelSetEncoderConfig = type(
        f"SetEncoder{model_name}Config",
        (config_class,),
        {
            "__init__": config_init,
            "model_type": f"set-encoder-{model_name.lower()}",
            "ADDED_ARGS": (
                CrossEncoderConfig.ADDED_ARGS
                + [
                    "depth",
                    "add_extra_token",
                    "other_sequence_embedding",
                    "sample_missing_docs",
                ]
            ),
            "TOKENIZER_ARGS": (CrossEncoderConfig.TOKENIZER_ARGS + ["add_extra_token"]),
            "Tokenizer": SetEncoderTokenizer,
        },
    )

    def model_init(
        self,
        config: PretrainedConfig,
    ) -> None:
        TransformerModel.__init__(self, config)
        Mixin.__init__(self, config, TransformerModel.forward)

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


@dataclass
class SetEncoderOutput(CrossEncoderOuput):
    repeat_logits: torch.Tensor | None = None


class SetEncoderModule(CrossEncoderModule):

    def __init__(
        self,
        model: CrossEncoderModel,
        loss_functions: Sequence[LossFunction] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
        repeat_linear_layer: bool = False,
    ):
        super().__init__(model, loss_functions, evaluation_metrics)
        self.model.encoder.embeddings.position_embeddings.requires_grad_(False)
        if (
            self.config.add_extra_token
            and len(self.tokenizer) != self.config.vocab_size
        ):
            self.model.encoder.resize_token_embeddings(len(self.tokenizer), 8)
            # word_embeddings = self.model.encoder.get_input_embeddings()
            # word_embeddings.weight.data[self.tokenizer.interaction_token_id] = (
            #     word_embeddings.weight.data[self.tokenizer.cls_token_id]
            # )
            # self.model.encoder.set_input_embeddings(word_embeddings)
            # position_embeddings = self.model.encoder.embeddings.position_embeddings
            # final_position = position_embeddings.weight.data[0].clone()
            # position_embeddings.weight.data[2:] = position_embeddings.weight.data[
            #     1:-1
            # ].clone()
            # position_embeddings.weight.data[1] = final_position
            # self.model.encoder.embeddings.position_embeddings = position_embeddings

        self.repeat_linear = None
        if repeat_linear_layer:
            self.repeat_linear = torch.nn.Linear(
                self.model.encoder.config.hidden_size, 1
            )

    def forward(self, batch: CrossEncoderRunBatch) -> SetEncoderOutput:
        num_docs = [len(doc_ids) for doc_ids in batch.doc_ids]
        output = self.model.forward(
            batch.encoding.input_ids,
            batch.encoding.get("attention_mask", None),
            batch.encoding.get("token_type_ids", None),
            num_docs=num_docs,
        )
        repeat_logits = None
        if self.repeat_linear is not None:
            repeat_logits = self.repeat_linear(output.last_hidden_state[:, 0])
        return SetEncoderOutput(
            scores=output.scores,
            repeat_logits=repeat_logits,
            last_hidden_state=output.last_hidden_state,
        )

    def compute_losses(
        self,
        batch: CrossEncoderRunBatch,
        loss_functions: Sequence[LossFunction] | None,
    ) -> Dict[str, torch.Tensor]:
        if loss_functions is None:
            if self.loss_functions is None:
                raise ValueError("Loss function is not set")
            loss_functions = self.loss_functions
        output = self.forward(batch)
        scores = output.scores
        if scores is None or batch.targets is None:
            raise ValueError("scores and targets must be set in the output and batch")

        repeat_logits = output.repeat_logits
        repeat_targets = None

        scores = scores.view(len(batch.query_ids), -1)
        targets = batch.targets.view(*scores.shape, -1)
        if targets.shape[-1] == 2:
            if repeat_logits is not None:
                repeat_logits = repeat_logits.view(len(batch.query_ids), -1)[:, 1:-1]
                repeat_targets = targets[:, 1:-1, 1]
            scores = scores[:, :-1]
            targets = targets[:, :-1, [0]]

        losses = {}
        for loss_function in loss_functions:
            if isinstance(loss_function, RepeatLossFunction):
                if repeat_logits is not None and repeat_targets is not None:
                    losses[loss_function.__class__.__name__] = (
                        loss_function.compute_loss(repeat_logits, repeat_targets)
                    )
            else:
                losses[loss_function.__class__.__name__] = loss_function.compute_loss(
                    scores, targets
                )
        return losses


class SetEncoderBertModule(SetEncoderModule):
    config_class = SetEncoderBertModel.config_class

    def __init__(
        self,
        model: CrossEncoderModel | None = None,
        model_name_or_path: str | None = None,
        config: CrossEncoderConfig | None = None,
        loss_functions: Sequence[LossFunction] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
        repeat_linear_layer: bool = False,
    ) -> None:
        if config is not None:
            config._attn_implementation = "eager"
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
                model = SetEncoderBertModel(config)
            else:
                kwargs = {}
                if config is not None:
                    kwargs = config.to_added_args_dict()
                config = SetEncoderBertConfig.from_pretrained(
                    model_name_or_path, **kwargs, _attn_implementation="eager"
                )
                model = SetEncoderBertModel.from_pretrained(
                    model_name_or_path, config=config
                )
        else:
            if not isinstance(model, SetEncoderBertModel):
                raise ValueError("Incorrect model type. Expected SetEncoderBertModel.")
        super().__init__(model, loss_functions, evaluation_metrics, repeat_linear_layer)


class SetEncoderElectraModule(SetEncoderModule):
    config_class = SetEncoderElectraConfig

    def __init__(
        self,
        model: CrossEncoderModel | None = None,
        model_name_or_path: str | None = None,
        config: CrossEncoderConfig | None = None,
        loss_functions: Sequence[LossFunction] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
        repeat_linear_layer: bool = False,
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
                model = SetEncoderElectraModel(config)
            else:
                kwargs = {}
                if config is not None:
                    kwargs = config.to_added_args_dict()
                config = SetEncoderElectraConfig.from_pretrained(
                    model_name_or_path, **kwargs
                )
                model = SetEncoderElectraModel.from_pretrained(
                    model_name_or_path, config=config
                )
        else:
            if not isinstance(model, SetEncoderElectraModel):
                raise ValueError(
                    "Incorrect model type. Expected SetEncoderElectraModel."
                )
        super().__init__(model, loss_functions, evaluation_metrics, repeat_linear_layer)


class SetEncoderRobertaModule(SetEncoderModule):
    config_class = SetEncoderRobertaConfig

    def __init__(
        self,
        model: CrossEncoderModel | None = None,
        model_name_or_path: str | None = None,
        config: CrossEncoderConfig | None = None,
        loss_functions: Sequence[LossFunction] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
        repeat_linear_layer: bool = False,
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
                model = SetEncoderRobertaModel(config)
            else:
                kwargs = {}
                if config is not None:
                    kwargs = config.to_added_args_dict()
                config = SetEncoderRobertaConfig.from_pretrained(
                    model_name_or_path, **kwargs
                )
                model = SetEncoderRobertaModel.from_pretrained(
                    model_name_or_path, config=config
                )
        else:
            if not isinstance(model, SetEncoderRobertaModel):
                raise ValueError(
                    "Incorrect model type. Expected SetEncoderRobertaModel."
                )
        super().__init__(model, loss_functions, evaluation_metrics, repeat_linear_layer)


AutoConfig.register(SetEncoderBertConfig.model_type, SetEncoderBertConfig)
AutoModel.register(SetEncoderBertConfig, SetEncoderBertModel)
AutoConfig.register(SetEncoderElectraConfig.model_type, SetEncoderElectraConfig)
AutoModel.register(SetEncoderElectraConfig, SetEncoderElectraModel)
AutoConfig.register(SetEncoderRobertaConfig.model_type, SetEncoderRobertaConfig)
AutoModel.register(SetEncoderRobertaConfig, SetEncoderRobertaModel)
