import math
from functools import partial, wraps
from typing import Callable, List, Tuple

import torch
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertSelfAttention

from .set_encoder_mixin import SetEncoderMixin


class BertSetEncoderMixin(SetEncoderMixin):
    encoder_name = "bert"
    self_attention_pattern = "self"

    def __init__(
        self,
        original_forward: Callable[
            ...,
            Tuple[torch.Tensor] | BaseModelOutputWithPoolingAndCrossAttentions,
        ],
        use_flash: bool,
        depth: int | None,
    ) -> None:
        super().__init__(original_forward, use_flash, depth)
        encoder = getattr(self, self.encoder_name)
        encoder.get_extended_attention_mask = self.extended_attention_mask_wrapper(
            encoder.get_extended_attention_mask
        )

    def forward(self, *args, num_docs: List[int] | None = None, **kwargs):
        encoder = getattr(self, self.encoder_name)
        encoder.get_extended_attention_mask = partial(
            encoder.get_extended_attention_mask,
            num_docs=num_docs if self.config.other_doc_attention else None,
        )
        return super().forward(*args, num_docs=num_docs, **kwargs)

    def extended_attention_mask_wrapper(
        self,
        get_extended_attention_mask: Callable[..., torch.Tensor],
    ):
        @wraps(get_extended_attention_mask)
        def wrapper(
            self,
            attention_mask: torch.Tensor,
            input_shape: Tuple[int],
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
            num_docs: List[int] | None = None,
        ):
            if self.config.other_doc_attention and num_docs is not None:
                if self.depth is None:
                    max_num_docs = max(num_docs)
                else:
                    max_num_docs = self.depth
                other_doc_mask = torch.zeros(
                    input_shape[0], max_num_docs, device=device, dtype=torch.bool
                )
                cum_idx = 0
                for n in num_docs:
                    other_doc_mask[cum_idx : cum_idx + n, n:] = True
                    cum_idx += n
                eye = torch.eye(max_num_docs, device=device).bool()
                repeated_same_doc_mask = [eye[:n] for n in num_docs]
                same_doc_mask = torch.cat(repeated_same_doc_mask)
                other_doc_attention_mask = ~(other_doc_mask | same_doc_mask)
                attention_mask = torch.cat(
                    [attention_mask, other_doc_attention_mask.to(attention_mask)],
                    dim=-1,
                )
                input_shape = tuple(attention_mask.shape)
            return get_extended_attention_mask(
                attention_mask, input_shape, device, dtype
            )

        return partial(wrapper, self)

    def flash_attention_forward(
        _self,
        self: BertSelfAttention,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_value: Tuple[Tuple[torch.FloatTensor]] | None = None,
        output_attentions: bool | None = False,
        num_docs: List[int] | None = None,
    ) -> Tuple[torch.Tensor]:
        key_value_hidden_states = hidden_states
        if num_docs is not None:
            key_value_hidden_states = _self.cat_other_doc_hidden_states(
                hidden_states, num_docs
            )
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

    def attention_forward(
        _self,
        self: BertSelfAttention,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_value: Tuple[Tuple[torch.FloatTensor]] | None = None,
        output_attentions: bool | None = False,
        num_docs: List[int] | None = None,
        extra_other_doc_token: bool = False,
    ) -> Tuple[torch.Tensor]:
        key_value_hidden_states = hidden_states
        if num_docs is not None:
            key_value_hidden_states = self._cat_other_doc_hidden_states(
                hidden_states,
                (self.other_doc_layer if hasattr(self, "other_doc_layer") else None),
                num_docs,
                extra_other_doc_token,
            )
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(key_value_hidden_states))
        value = self.transpose_for_scores(self.value(key_value_hidden_states))

        attention_scores = torch.matmul(query, key.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs
