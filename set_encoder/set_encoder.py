from functools import partial
from typing import Dict, Sequence, Tuple

import torch
from lightning_ir import (
    CrossEncoderConfig,
    CrossEncoderModel,
    CrossEncoderModule,
    CrossEncoderOutput,
    TrainBatch,
)
from lightning_ir.data import RankBatch
from lightning_ir.loss.loss import LossFunction
from transformers import AutoConfig, AutoModel, AutoTokenizer, BatchEncoding

from .loss import RepeatLossFunction
from .tokenizer import SetEncoderTokenizer


class SetEncoderConfig(CrossEncoderConfig):
    model_type = "set-encoder"
    tokenizer_class = SetEncoderTokenizer

    ADDED_ARGS = CrossEncoderConfig.ADDED_ARGS.union({"depth", "add_extra_token", "sample_missing_docs"})
    TOKENIZER_ARGS = CrossEncoderConfig.TOKENIZER_ARGS.union({"add_extra_token"})

    def __init__(
        self,
        *args,
        depth: int = 100,
        add_extra_token: bool = False,
        sample_missing_docs: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.depth = depth
        self.add_extra_token = add_extra_token
        self.sample_missing_docs = sample_missing_docs


class SetEncoderModel(CrossEncoderModel):
    config_class = SetEncoderConfig
    self_attention_pattern = "self"

    def __init__(self, config: SetEncoderConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config: SetEncoderConfig
        self.attn_implementation = "eager"

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, ...],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        num_docs: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if num_docs is not None:
            eye = (1 - torch.eye(self.config.depth, device=device)).long()
            if not self.config.sample_missing_docs:
                eye = eye[:, : max(num_docs)]
            other_doc_attention_mask = torch.cat([eye[:n] for n in num_docs])
            attention_mask = torch.cat(
                [attention_mask, other_doc_attention_mask.to(attention_mask)],
                dim=-1,
            )
            input_shape = tuple(attention_mask.shape)
        return super().get_extended_attention_mask(attention_mask, input_shape, device, dtype)

    def forward(self, encoding: BatchEncoding, num_docs: Sequence[int]) -> CrossEncoderOutput:
        self.get_extended_attention_mask = partial(self.get_extended_attention_mask, num_docs=num_docs)
        for name, module in self.named_modules():
            if name.endswith(self.self_attention_pattern):
                module.forward = partial(self.attention_forward, self, module, num_docs=num_docs)
        return super().forward(encoding)

    @staticmethod
    def attention_forward(
        _self,
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None,
        *args,
        num_docs: Sequence[int],
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        key_value_hidden_states = hidden_states
        if num_docs is not None:
            key_value_hidden_states = _self.cat_other_doc_hidden_states(hidden_states, num_docs)
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(key_value_hidden_states))
        value = self.transpose_for_scores(self.value(key_value_hidden_states))

        context = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attention_mask.to(query.dtype) if attention_mask is not None else None,
            self.dropout.p if self.training else 0,
        )

        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(new_context_shape)
        return (context,)

    def cat_other_doc_hidden_states(
        self,
        hidden_states: torch.Tensor,
        num_docs: Sequence[int],
    ) -> torch.Tensor:
        idx = 1 if self.config.add_extra_token else 0
        split_other_doc_hidden_states = torch.split(hidden_states[:, idx], list(num_docs))
        repeated_other_doc_hidden_states = []
        for idx, h_states in enumerate(split_other_doc_hidden_states):
            missing_docs = 0 if self.config.depth is None else self.config.depth - num_docs[idx]
            if missing_docs and self.config.sample_missing_docs:
                mean = h_states.mean(0, keepdim=True).expand(missing_docs, -1)
                if num_docs[idx] == 1:
                    std = torch.zeros_like(mean)
                else:
                    std = h_states.std(0, keepdim=True).expand(missing_docs, -1)
                sampled_h_states = torch.normal(mean, std).to(h_states)
                h_states = torch.cat([h_states, sampled_h_states])
            repeated_other_doc_hidden_states.append(h_states.unsqueeze(0).expand(num_docs[idx], -1, -1))
        other_doc_hidden_states = torch.cat(repeated_other_doc_hidden_states)
        key_value_hidden_states = torch.cat([hidden_states, other_doc_hidden_states], dim=1)
        return key_value_hidden_states


class SetEncoderModule(CrossEncoderModule):

    def __init__(
        self,
        model_name_or_path: str | None = None,
        config: CrossEncoderConfig | None = None,
        model: CrossEncoderModel | None = None,
        loss_functions: Sequence[LossFunction | Tuple[LossFunction, float]] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
        repeat_linear_layer: bool = False,
    ):
        super().__init__(model_name_or_path, config, model, loss_functions, evaluation_metrics)
        self.model: SetEncoderModel
        if self.config.add_extra_token and len(self.tokenizer) != self.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer), 8)

        self.repeat_linear = None
        if repeat_linear_layer:
            self.repeat_linear = torch.nn.Linear(self.model.encoder.config.hidden_size, 1)

    def forward(self, batch: RankBatch) -> CrossEncoderOutput:
        queries = list(batch.queries)
        docs = [d for docs in batch.docs for d in docs]
        num_docs = [len(docs) for docs in batch.docs]
        encoding = self.prepare_input(queries, docs, num_docs)
        output = self.model.forward(encoding["encoding"], num_docs)
        return output

    def compute_losses(
        self, batch: TrainBatch, loss_functions: Sequence[LossFunction] | None
    ) -> Dict[str, torch.Tensor]:
        if loss_functions is None:
            if self.loss_functions is None:
                raise ValueError("Loss function is not set")
            loss_functions = self.loss_functions
        output = self.forward(batch)
        scores = output.scores
        if scores is None or batch.targets is None:
            raise ValueError("scores and targets must be set in the output and batch")

        repeat_logits = None
        repeat_targets = None
        if self.repeat_linear is not None:
            repeat_logits = self.repeat_linear(output.embeddings)

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
                    losses[loss_function.__class__.__name__] = loss_function.compute_loss(repeat_logits, repeat_targets)
            else:
                losses[loss_function.__class__.__name__] = loss_function.compute_loss(scores, targets)
        return losses


AutoConfig.register(SetEncoderConfig.model_type, SetEncoderConfig)
AutoModel.register(SetEncoderConfig, SetEncoderModel)
AutoTokenizer.register(SetEncoderConfig, SetEncoderTokenizer)
