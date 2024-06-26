from functools import partial, wraps
from typing import Callable, List, Tuple

import torch
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertSelfAttention
from lightning_ir.cross_encoder.model import CrossEncoderConfig

from .set_encoder_mixin import SetEncoderMixin

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


class BertSetEncoderMixin(SetEncoderMixin):
    encoder_name = "bert"
    self_attention_pattern = "self"

    def __init__(
        self,
        config: CrossEncoderConfig,
        original_forward: Callable[
            ...,
            Tuple[torch.Tensor] | BaseModelOutputWithPoolingAndCrossAttentions,
        ],
    ) -> None:
        super().__init__(config, original_forward)
        self.base_encoder.get_extended_attention_mask = (
            self.extended_attention_mask_wrapper(
                self.base_encoder.get_extended_attention_mask
            )
        )

    @property
    def base_encoder(self):
        if hasattr(self, self.encoder_name):
            encoder = getattr(self, self.encoder_name)
        elif hasattr(self, "encoder"):
            encoder = self
        else:
            raise ValueError(
                f"Neither {self.encoder_name} nor encoder attribute found in {self}"
            )
        return encoder

    def forward(self, *args, num_docs: List[int] | None = None, **kwargs):
        self.base_encoder.get_extended_attention_mask = partial(
            self.base_encoder.get_extended_attention_mask, num_docs=num_docs
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
            return get_extended_attention_mask(
                attention_mask, input_shape, device, dtype
            )

        return partial(wrapper, self)

    def attention_forward(
        _self,
        self: BertSelfAttention,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None,
        *args,
        num_docs: List[int] | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        key_value_hidden_states = hidden_states
        if num_docs is not None:
            key_value_hidden_states = _self.cat_other_doc_hidden_states(
                hidden_states, num_docs
            )
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(key_value_hidden_states))
        value = self.transpose_for_scores(self.value(key_value_hidden_states))

        if attention_mask is not None and not attention_mask.any():
            attention_mask = None

        if (
            flash_attn_func is not None
            and hidden_states.is_cuda
            and attention_mask is None
        ):
            context = (
                flash_attn_func(
                    query.bfloat16().transpose(1, 2),
                    key.bfloat16().transpose(1, 2),
                    value.bfloat16().transpose(1, 2),
                    self.dropout.p if self.training else 0,
                )
                .transpose(1, 2)
                .to(query.dtype)
            )
        else:
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
