from typing import Callable, List, Tuple, Union
from abc import ABC, abstractmethod

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import torch
from functools import partial, wraps


class SetEncoderMixin(ABC):
    encoder_name: str
    self_attention_pattern: str

    ADDITIONAL_KWARGS = [
        "depth",
        "other_doc_attention",
        "other_doc_layer",
        "rank_position_embeddings",
        "average_doc_embeddings",
        "extra_other_doc_token",
    ]

    def __init__(self, use_flash: bool) -> None:
        self.use_flash = use_flash

    @abstractmethod
    def flash_attention_forward(*args, **kwargs):
        ...

    @abstractmethod
    def attention_forward(*args, **kwargs):
        ...

    @abstractmethod
    def embedding_forward(*args, **kwargs):
        ...

    def model_forward_wrapper(
        self,
        forward: Callable[
            ...,
            Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions],
        ],
    ) -> Callable[
        ..., Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]
    ]:
        attention_forward = (
            self.flash_attention_forward if self.use_flash else self.attention_forward
        )

        @wraps(forward)
        def wrapper(self, *args, num_docs: List[int] | None = None, **kwargs):
            encoder = getattr(self, self.encoder_name)
            encoder.get_extended_attention_mask = partial(
                encoder.get_extended_attention_mask, num_docs=num_docs
            )
            for name, module in self.named_modules():
                if name.endswith(self.self_attention_pattern):
                    module.forward = partial(
                        attention_forward,
                        module,
                        num_docs=num_docs if self.config.other_doc_attention else None,
                        extra_other_doc_token=self.config.extra_other_doc_token,
                    )
                elif name.endswith(f"{self.encoder_name}.embeddings"):
                    module.forward = partial(
                        self.embedding_forward,
                        module,
                        num_docs=num_docs,
                        depth=self.config.depth,
                        average_doc_embeddings=self.config.average_doc_embeddings,
                        rank_position_embeddings=self.config.rank_position_embeddings,
                    )
            return forward(*args, **kwargs)

        return partial(wrapper, self)

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
                max_num_docs = max(num_docs)
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
                other_doc_attention_mask = other_doc_mask | same_doc_mask
                other_doc_attention_mask = ~other_doc_attention_mask
                attention_mask = torch.cat(
                    [attention_mask, other_doc_attention_mask.to(attention_mask)],
                    dim=-1,
                )
                input_shape = tuple(attention_mask.shape)
            # if self.config.extra_other_doc_token:
            #     attention_mask[:, 1] = 0
            return get_extended_attention_mask(
                attention_mask, input_shape, device, dtype
            )

        return partial(wrapper, self)

    @staticmethod
    def cat_other_doc_hidden_states(
        hidden_states: torch.Tensor,
        other_doc_layer: torch.nn.Linear | None,
        num_docs: List[int],
        extra_other_doc_token: bool = False,
    ) -> torch.Tensor:
        idx = 1 if extra_other_doc_token else 0
        split_other_doc_hidden_states = torch.split(hidden_states[:, idx], num_docs)
        repeated_other_doc_hidden_states = [
            h_states
            for idx, h_states in enumerate(split_other_doc_hidden_states)
            for _ in range(num_docs[idx])
        ]
        other_doc_hidden_states = torch.nn.utils.rnn.pad_sequence(
            repeated_other_doc_hidden_states,
            batch_first=True,
            padding_value=0,
        )
        if other_doc_layer is not None:
            other_doc_hidden_states = other_doc_layer(other_doc_hidden_states)
        key_value_hidden_states = torch.cat(
            [hidden_states, other_doc_hidden_states], dim=1
        )
        return key_value_hidden_states
