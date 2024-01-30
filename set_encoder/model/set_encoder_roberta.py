import torch
from typing import List, Literal

from transformers.models.roberta.modeling_roberta import (
    create_position_ids_from_input_ids,
)
from .set_encoder_bert import BertSetEncoderMixin


class RoBERTaSetEncoderMixin(BertSetEncoderMixin):
    encoder_name = "roberta"

    def embedding_forward(
        self,
        embedding_layer,
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
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids, embedding_layer.padding_idx, past_key_values_length
                )
            else:
                position_ids = embedding_layer.create_position_ids_from_inputs_embeds(
                    inputs_embeds
                )

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
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

        if num_docs is not None and hasattr(self, "rank_position_embeddings"):
            rank_tensors = []
            for _num_docs in num_docs:
                _ranks = torch.randperm(depth)[:_num_docs]
                if rank_position_embeddings == "sorted":
                    _ranks = _ranks.sort()[0]
                rank_tensors.append(_ranks)
            ranks = torch.cat(rank_tensors).to(embeddings.device)
            rank_embeddings = embedding_layer.rank_position_embeddings(ranks)
            embeddings = embeddings + rank_embeddings[:, None]
            # embeddings[:, 0] = embeddings[:, 0] + rank_embeddings

        embeddings = embedding_layer.LayerNorm(embeddings)
        embeddings = embedding_layer.dropout(embeddings)
        return embeddings
