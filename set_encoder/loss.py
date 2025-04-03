import torch
from lightning_ir import LightningIROutput, TrainBatch
from lightning_ir.loss.loss import PairwiseLossFunction, ScoringLossFunction

from .module import RepeatOutput, RepeatTrainBatch


class DuplicateAwareRankNet(PairwiseLossFunction):

    def compute_loss(self, output: LightningIROutput, batch: TrainBatch) -> torch.Tensor:
        assert batch.targets is not None
        scores = self.process_scores(output)
        # assume ranking is sorted
        targets = torch.arange(scores.shape[1], 0, -1, device=scores.device)[None].expand_as(scores)
        subtopic_ids = batch.targets[..., 0]

        max_subtopic = subtopic_ids.max()
        in_subtopic = subtopic_ids[..., None] == torch.arange(max_subtopic + 1, device=scores.device).view(1, 1, -1)
        subtopic_max_scores = torch.scatter_reduce(
            torch.zeros(scores.shape[0], max_subtopic + 1).to(scores),
            1,
            subtopic_ids,
            scores,
            "amax",
            include_self=False,
        )
        subtopic_max_target = torch.scatter_reduce(
            torch.zeros(scores.shape[0], max_subtopic + 1).to(targets),
            1,
            subtopic_ids,
            targets,
            "amax",
            include_self=False,
        )
        is_subtopic_max_score = (in_subtopic & (scores[..., None] == subtopic_max_scores[:, None])).any(-1)
        is_subtopic_non_max_score = (in_subtopic & (scores[..., None] < subtopic_max_scores[:, None])).any(-1)
        da_targets = targets.clone()
        # set the target for all non-max scores per subtopic to 0
        da_targets[is_subtopic_non_max_score] = 0
        # set the target for the max score per subtopic to the max target per subtopic
        da_targets[is_subtopic_max_score] = subtopic_max_target.gather(1, subtopic_ids)[is_subtopic_max_score]

        # ranknet
        query_idcs, pos_idcs, neg_idcs = self.get_pairwise_idcs(da_targets)
        pos = scores[query_idcs, pos_idcs]
        neg = scores[query_idcs, neg_idcs]
        margin = pos - neg
        loss = torch.nn.functional.binary_cross_entropy_with_logits(margin, torch.ones_like(margin))
        return loss


class RepeatCrossEntropyLoss(ScoringLossFunction):

    def __init__(self) -> None:
        super().__init__()

    def compute_loss(self, output: RepeatOutput, batch: RepeatTrainBatch) -> torch.Tensor:
        scores = getattr(output, "repeat_logits", None)
        targets = getattr(batch, "repeat_targets", None)
        if scores is None or targets is None:
            raise ValueError(
                "No repeat logits or repeat tagets found in output or batch. "
                "Use the RepeatRunDataset to finetune the model for duplicate detection."
            )
        pos_weight = torch.tensor((scores.shape[1] - 2) / 2).to(scores.device)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            scores.view(-1), targets.view(-1), pos_weight=pos_weight
        )
