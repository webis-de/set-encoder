from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from lightning_ir import CrossEncoderModule, CrossEncoderOutput, TrainBatch
from lightning_ir.loss.loss import LossFunction
from lightning_ir.models.set_encoder import SetEncoderConfig, SetEncoderModel

from .data import RepeatRunDataset


@dataclass
class RepeatOutput(CrossEncoderOutput):
    repeat_logits: torch.Tensor | None = None


@dataclass
class RepeatTrainBatch(TrainBatch):
    repeat_targets: torch.Tensor | None = None


class SetEncoderModule(CrossEncoderModule):

    def __init__(
        self,
        model_name_or_path: str | None = None,
        config: SetEncoderConfig | None = None,
        model: SetEncoderModel | None = None,
        loss_functions: Sequence[LossFunction | Tuple[LossFunction, float]] | None = None,
        evaluation_metrics: Sequence[str] | None = None,
    ):
        super().__init__(model_name_or_path, config, model, loss_functions, evaluation_metrics)
        if self.config.add_extra_token and len(self.tokenizer) != self.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer), 8)

        self.repeat_linear = torch.nn.Linear(self.model.encoder.config.hidden_size, 1)

    def _compute_losses(self, batch: TrainBatch, output: CrossEncoderOutput) -> List[torch.Tensor]:
        output.repeat_logits = None
        repeat = isinstance(self.trainer.datamodule.train_dataset, RepeatRunDataset)
        assert output.scores is not None and batch.targets is not None and output.embeddings is not None
        num_queries = len(batch.queries)
        if repeat:
            scores = output.scores.view(num_queries, -1)[:, :-1].reshape(-1)
            targets = batch.targets.view(num_queries, -1, 2)[:, :-1, 0].reshape(-1)
            repeat_targets = batch.targets.view(num_queries, -1, 2)[:, :-1, 1].reshape(-1)
            repeat_logits = self.repeat_linear(
                output.embeddings.view(num_queries, -1, self.model.encoder.config.hidden_size)[:, :-1, :].reshape(
                    -1, self.model.encoder.config.hidden_size
                )
            )
            output = RepeatOutput(scores, output.embeddings, repeat_logits)
            batch = RepeatTrainBatch(
                batch.queries, batch.docs, batch.query_ids, batch.doc_ids, batch.qrels, targets, repeat_targets
            )

        return super()._compute_losses(batch, output)
