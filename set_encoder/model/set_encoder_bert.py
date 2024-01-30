import torch
import math
from typing import Tuple, List, Literal

from .set_encoder_mixin import SetEncoderMixin
from transformers.models.bert.modeling_bert import BertEmbeddings, BertSelfAttention


class BertSetEncoderMixin(SetEncoderMixin):
    encoder_name = "bert"
    self_attention_pattern = "self"

    def flash_attention_forward(
        self,
        self_attention_layer: BertSelfAttention,
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
            key_value_hidden_states = self.cat_other_doc_hidden_states(
                hidden_states, num_docs
            )
        query = self_attention_layer.transpose_for_scores(
            self_attention_layer.query(hidden_states)
        )
        key = self_attention_layer.transpose_for_scores(
            self_attention_layer.key(key_value_hidden_states)
        )
        value = self_attention_layer.transpose_for_scores(
            self_attention_layer.value(key_value_hidden_states)
        )

        context = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attention_mask.to(query.dtype) if attention_mask is not None else None,
            self_attention_layer.dropout.p if self_attention_layer.training else 0,
        )

        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self_attention_layer.all_head_size,)
        context = context.view(new_context_shape)
        return (context,)

    def embedding_forward(
        self,
        embedding_layer: BertEmbeddings,
        input_ids: torch.LongTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values_length: int = 0,
        num_docs: List[int] | None = None,
        rank_position_embeddings: bool | Literal["random", "sorted"] = False,
        depth: int = 100,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = embedding_layer.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(embedding_layer, "token_type_ids"):
                buffered_token_type_ids = embedding_layer.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape,
                    dtype=torch.long,
                    device=embedding_layer.position_ids.device,
                )

        if inputs_embeds is None:
            inputs_embeds = embedding_layer.word_embeddings(input_ids)
        token_type_embeddings = embedding_layer.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if embedding_layer.position_embedding_type == "absolute":
            position_embeddings = embedding_layer.position_embeddings(position_ids)
            embeddings += position_embeddings

        if num_docs is not None and hasattr(
            embedding_layer, "rank_position_embeddings"
        ):
            rank_tensors = []
            for _num_docs in num_docs:
                _ranks = torch.randperm(depth)[:_num_docs]
                if rank_position_embeddings and rank_position_embeddings != "random":
                    _ranks = _ranks.sort()[0]
                rank_tensors.append(_ranks)
            ranks = torch.cat(rank_tensors).to(embeddings.device)
            rank_embeddings = embedding_layer.rank_position_embeddings(ranks)
            # embeddings = embeddings + rank_embeddings[:, None]
            embeddings[:, 0] = embeddings[:, 0] + rank_embeddings

        embeddings = embedding_layer.LayerNorm(embeddings)
        embeddings = embedding_layer.dropout(embeddings)
        return embeddings

    def attention_forward(
        self,
        self_attention_layer: BertSelfAttention,
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
            key_value_hidden_states = self.cat_other_doc_hidden_states(
                hidden_states,
                self_attention_layer.other_doc_layer
                if hasattr(self_attention_layer, "other_doc_layer")
                else None,
                num_docs,
                extra_other_doc_token,
            )
        query = self_attention_layer.transpose_for_scores(
            self_attention_layer.query(hidden_states)
        )
        key = self_attention_layer.transpose_for_scores(
            self_attention_layer.key(key_value_hidden_states)
        )
        value = self_attention_layer.transpose_for_scores(
            self_attention_layer.value(key_value_hidden_states)
        )

        attention_scores = torch.matmul(query, key.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(
            self_attention_layer.attention_head_size
        )
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self_attention_layer.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self_attention_layer.all_head_size,
        )
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs
