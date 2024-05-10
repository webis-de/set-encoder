import torch
from lightning_ir.loss.loss import ApproxLossFunction


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