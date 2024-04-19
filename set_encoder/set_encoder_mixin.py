from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, List, Tuple

import torch
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class SetEncoderMixin(torch.nn.Module, ABC):
    self_attention_pattern: str

    def __init__(
        self,
        original_forward: Callable[
            ...,
            Tuple[torch.Tensor] | BaseModelOutputWithPoolingAndCrossAttentions,
        ],
        use_flash: bool,
        depth: int | None,
    ) -> None:
        self.original_forward = original_forward
        self.use_flash = use_flash
        self.depth = depth

    def forward(self, *args, num_docs: List[int] | None = None, **kwargs):
        attention_forward = (
            self.flash_attention_forward if self.use_flash else self.attention_forward
        )
        for name, module in self.named_modules():
            if name.endswith(self.self_attention_pattern):
                module.forward = partial(attention_forward, module, num_docs=num_docs)
        return self.original_forward(self, *args, **kwargs)

    @abstractmethod
    def flash_attention_forward(*args, **kwargs): ...

    @abstractmethod
    def attention_forward(*args, **kwargs): ...

    def cat_other_doc_hidden_states(
        self,
        hidden_states: torch.Tensor,
        num_docs: List[int],
    ) -> torch.Tensor:
        split_other_doc_hidden_states = torch.split(hidden_states[:, 0], num_docs)
        repeated_other_doc_hidden_states = []
        for idx, h_states in enumerate(split_other_doc_hidden_states):
            missing_docs = 0 if self.depth is None else self.depth - num_docs[idx]
            if missing_docs:
                mean = h_states.mean(0, keepdim=True).expand(missing_docs, -1)
                if num_docs[idx] == 1:
                    std = torch.zeros_like(mean)
                else:
                    std = h_states.std(0, keepdim=True).expand(missing_docs, -1)
                sampled_h_states = torch.normal(mean, std).to(h_states)
                h_states = torch.cat([h_states, sampled_h_states])
            for _ in range(num_docs[idx]):
                repeated_other_doc_hidden_states.append(h_states)
        other_doc_hidden_states = torch.nn.utils.rnn.pad_sequence(
            repeated_other_doc_hidden_states,
            batch_first=True,
            padding_value=0,
        )
        key_value_hidden_states = torch.cat(
            [hidden_states, other_doc_hidden_states], dim=1
        )
        return key_value_hidden_states