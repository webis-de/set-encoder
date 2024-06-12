import torch
from lightning_ir.loss.loss import (
    ApproxLossFunction,
    ApproxRankMSE,
    LossFunction,
    PairwiseLossFunction,
)
from set_encoder.data import register_trec_dl_novelty, register_trec_dl_subtopics

register_trec_dl_novelty()
register_trec_dl_subtopics()


class SortedApproxRankMSE(ApproxRankMSE):

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        approx_ranks = self.get_approx_ranks(logits, self.temperature)
        ranks = torch.arange(logits.shape[1], device=logits.device) + 1
        ranks = ranks.expand_as(approx_ranks)
        loss = torch.nn.functional.mse_loss(
            approx_ranks, ranks.to(approx_ranks), reduction="none"
        )
        if self.discount == "log2":
            weight = 1 / torch.log2(ranks + 1)
        elif self.discount == "reciprocal":
            weight = 1 / ranks
        else:
            weight = 1
        loss = loss * weight
        loss = loss.mean()
        return loss


class SubtopicMeanMinRank(ApproxLossFunction):

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = self.process_targets(logits, targets)
        approx_ranks = self.get_approx_ranks(logits, self.temperature)
        expanded_approx_ranks = (
            torch.nn.functional.one_hot(targets) * approx_ranks[..., None]
        )
        expanded_approx_ranks = expanded_approx_ranks.masked_fill(
            expanded_approx_ranks == 0, 10_000
        )
        loss = expanded_approx_ranks.min(1).values.mean()
        return loss


class SupervisedSubtopicMeanMinRank(ApproxLossFunction):

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = self.process_targets(logits, targets)
        approx_ranks = self.get_approx_ranks(logits, self.temperature)
        min_subtopic_ranks = (
            (targets[..., None] == torch.arange(targets.max(), device=targets.device))
            .logical_not()
            .long()
            .argmin(1)
        )
        min_ranks = torch.gather(approx_ranks, 1, min_subtopic_ranks)
        loss = min_ranks.mean()
        return loss


class DuplicateAwareRankNet(PairwiseLossFunction):

    def compute_loss(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        subtopic_ids = torch.nonzero(targets)[:, 2].view_as(scores)
        targets = targets.max(-1).values
        max_subtopic = int(subtopic_ids.max().item())
        in_subtopic = subtopic_ids[..., None] == torch.arange(
            max_subtopic + 1, device=scores.device
        ).view(1, 1, -1)
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
        is_subtopic_max_score = in_subtopic & (
            scores.unsqueeze(-1) == subtopic_max_scores
        )
        is_subtopic_non_max_score = in_subtopic & (
            scores.unsqueeze(-1) < subtopic_max_scores
        )
        da_targets = targets.clone()
        # replace all non-max scores per "subtopic" with 0
        da_targets[is_subtopic_non_max_score.any(-1)] = 0
        # replace the max score per "subtopic" with the highest target per subtopic
        da_targets[is_subtopic_max_score.any(-1)] = subtopic_max_target[
            is_subtopic_max_score.any(1)
        ][subtopic_ids[is_subtopic_max_score.any(-1)]]

        # ranknet
        query_idcs, pos_idcs, neg_idcs = self.get_pairwise_idcs(da_targets)
        pos = scores[query_idcs, pos_idcs]
        neg = scores[query_idcs, neg_idcs]
        margin = pos - neg
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            margin, torch.ones_like(margin)
        )
        return loss


class RepeatLossFunction(LossFunction):
    pass


class RepeatCrossEntropyLoss(RepeatLossFunction):

    def __init__(self) -> None:
        super().__init__()

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = torch.tensor((logits.shape[1] - 2) / 2).to(logits.device)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight
        )


class ApproxAlphaNDCG(ApproxLossFunction):
    def __init__(self, temperature: float = 1, alpha: float = 0.5) -> None:
        super().__init__(temperature)
        self.alpha = alpha

    def process_targets(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return targets

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = self.process_targets(logits, targets)
        approx_ranks = self.get_approx_ranks(logits, self.temperature)
        alpha_ndcg = self.get_alpha_ndcg(approx_ranks, targets)
        loss = 1 - alpha_ndcg
        return loss.mean()

    def get_alpha_ndcg(
        self, ranks: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        targets = targets.clamp(min=0, max=1)
        optimal_ranks = self.greedy_sort(targets)
        alpha_dcg = self.get_alpha_dcg(ranks, targets)
        alpha_idcg = self.get_alpha_dcg(optimal_ranks, targets)
        return alpha_dcg / alpha_idcg.clamp(min=1e-12)

    def get_alpha_dcg(self, ranks: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        sorted_idcs = torch.argsort(ranks)
        ranks = torch.gather(ranks, 1, sorted_idcs)
        targets = torch.gather(targets, 1, sorted_idcs[..., None].expand_as(targets))
        coverage = targets.cumsum(dim=-2).roll(1, 1)
        coverage[:, 0] = 0
        gains = (targets * (1 - self.alpha) ** coverage).sum(-1) / torch.log2(
            1 + ranks.float()
        )
        return gains.sum(dim=-1)

    def greedy_sort(self, targets: torch.Tensor) -> torch.Tensor:
        batch_size, depth, num_subtopics = targets.shape
        optimal_ranks = torch.zeros(
            (batch_size, depth), dtype=torch.long, device=targets.device
        )
        coverage = torch.zeros((batch_size, 1, num_subtopics), device=targets.device)
        targets = targets.clone()
        for r in range(1, depth + 1):
            gain = (targets * (1 - self.alpha) ** coverage).sum(-1)
            idcs = gain.argmax(-1)
            optimal_ranks[torch.arange(batch_size), idcs] = r
            coverage += targets[torch.arange(batch_size), idcs].unsqueeze(1)
            targets[torch.arange(batch_size), idcs] = -1
        return optimal_ranks


class ApproxERRIA(ApproxLossFunction):

    def __init__(
        self,
        temperature: float = 1,
        normalize: bool = False,
        max_relevance: int | None = None,
    ) -> None:
        super().__init__(temperature)
        self.normalize = normalize
        self.max_relevance = max_relevance

    def process_targets(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return targets

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = self.process_targets(logits, targets)
        approx_ranks = self.get_approx_ranks(logits, self.temperature)
        err_ia = self.get_err_ia(approx_ranks, targets)
        loss = 1 - err_ia
        return loss.mean()

    def get_err_ia(self, ranks: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.clamp(min=0)
        idcs = torch.argsort(ranks)
        ranks = torch.gather(ranks, 1, idcs)
        targets = torch.gather(targets, 1, idcs[..., None].expand_as(targets))
        max_relevance = (
            torch.tensor(self.max_relevance, device=ranks.device)
            if self.max_relevance
            else targets.max()
        )
        relevance_prob = (torch.pow(2, targets) - 1) / torch.pow(2, max_relevance)
        unsatisfied_prob = (1 - relevance_prob).cumprod(dim=1).roll(1, 1)
        unsatisfied_prob[:, 0] = 1
        err_ia = (relevance_prob * unsatisfied_prob / ranks[..., None]).sum(1).mean(1)
        return err_ia
