from typing import Literal, Optional
import math

import torch


PAD_VALUE = -10000
EPS = 1e-6


class LossFunc:
    def __init__(self, reduction: Optional[Literal["mean", "sum"]] = "mean"):
        self.reduction = reduction

    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        subtopics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def aggregate(
        self,
        loss: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.reduction is None:
            return loss
        if mask is not None:
            loss = loss[~mask]
        if not loss.numel():
            return torch.tensor(0.0, requires_grad=True, device=loss.device)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        raise ValueError(f"Unknown reduction {self.reduction}")


def get_approx_ranks(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    mask = (logits[..., None] == PAD_VALUE) | (logits[:, None] == PAD_VALUE)
    score_diff = logits[:, None] - logits[..., None]
    normalized_score_diff = torch.sigmoid(score_diff / temperature)
    normalized_score_diff = normalized_score_diff.masked_fill(mask, 0)
    # set diagonal to 0
    normalized_score_diff = normalized_score_diff * (
        1 - torch.eye(logits.shape[1], device=logits.device)
    )
    approx_ranks = normalized_score_diff.sum(-1) + 1
    approx_ranks[mask[:, 0]] = 0
    return approx_ranks


def neural_sort(
    logits: torch.Tensor, mask: Optional[torch.Tensor], temperature: float
) -> torch.Tensor:
    # https://github.com/ermongroup/neuralsort/blob/master/pytorch/neuralsort.py
    logits = logits.unsqueeze(-1)
    dim = logits.shape[1]
    one = torch.ones((dim, 1), device=logits.device)

    A_logits = torch.abs(logits - logits.permute(0, 2, 1))
    B = torch.matmul(A_logits, torch.matmul(one, torch.transpose(one, 0, 1)))
    scaling = dim + 1 - 2 * (torch.arange(dim, device=logits.device) + 1)
    C = torch.matmul(logits, scaling.to(logits).unsqueeze(0))

    P_max = (C - B).permute(0, 2, 1)
    if mask is not None:
        P_max = P_max.masked_fill(
            mask[:, None, :] | mask[:, :, None], torch.finfo(P_max.dtype).min
        )
        P_max = P_max.masked_fill(mask[:, None, :] & mask[:, :, None], 1.0)
    P_hat = torch.nn.functional.softmax(P_max / temperature, dim=-1)

    P_hat = sinkhorn_scaling(P_hat, mask)

    return P_hat


def sinkhorn_scaling(
    mat: torch.Tensor,
    mask: Optional[torch.Tensor],
    tol: float = 1e-5,
    max_iter: int = 50,
):
    # https://github.com/allegro/allRank/blob/master/allrank/models/losses/loss_utils.py#L8
    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)
        mat = mat.masked_fill(mask[:, None, :] & mask[:, :, None], 1.0)
    idx = 0
    while True:
        if (
            torch.max(torch.abs(mat.sum(dim=2) - 1.0)) < tol
            and torch.max(torch.abs(mat.sum(dim=1) - 1.0)) < tol
        ) or idx > max_iter:
            break
        mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=EPS)
        mat = mat / mat.sum(dim=2, keepdim=True).clamp(min=EPS)
        idx += 1

    return mat


def get_mrr(
    ranks: torch.Tensor, labels: torch.Tensor, k: Optional[int] = None
) -> torch.Tensor:
    labels = labels.clamp(None, 1)
    mask = (ranks == PAD_VALUE) | (ranks == 0) | (labels == PAD_VALUE)
    ranks = ranks.masked_fill(mask, 1)
    reciprocal_ranks = 1 / ranks
    mrr = reciprocal_ranks * labels
    mrr = mrr.masked_fill(mask, 0)
    if k is not None:
        mrr = mrr.masked_fill(ranks > k, 0)
    mrr = mrr.max(dim=-1)[0]
    return mrr


def get_dcg(
    ranks: torch.Tensor,
    labels: torch.Tensor,
    k: int | None = None,
    scale_gains: bool = True,
) -> torch.Tensor:
    ranks = ranks.clone()
    mask = (ranks == PAD_VALUE) | (ranks == 0) | (labels == PAD_VALUE)
    ranks = ranks.masked_fill(mask, 1)
    log_ranks = torch.log2(1 + ranks)
    discounts = 1 / log_ranks
    discounts = discounts.masked_fill(mask, 0)
    if scale_gains:
        gains = 2**labels - 1
    else:
        gains = labels
    dcgs = gains * discounts
    if k is not None:
        dcgs = dcgs.masked_fill(ranks > k, 0)
    return dcgs.sum(dim=-1)


def get_ndcg(
    ranks: torch.Tensor,
    labels: torch.Tensor,
    k: int | None = None,
    scale_gains: bool = True,
    optimal_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    if optimal_labels is None:
        optimal_labels = labels
    optimal_ranks = torch.argsort(torch.argsort(optimal_labels, descending=True))
    optimal_ranks = optimal_ranks + 1
    dcg = get_dcg(ranks, labels, k, scale_gains)
    idcg = get_dcg(optimal_ranks, optimal_labels, k, scale_gains)
    ndcg = dcg / (idcg + EPS)
    return ndcg


def get_alpha_dcg(
    ranks: torch.Tensor,
    labels: torch.Tensor,
    subtopics: torch.Tensor,
    k: int | None = None,
    scale_gains: bool = False,
    alpha: float = 0.5,
) -> torch.Tensor:
    ranks = ranks.clone()
    idcs = torch.argsort(ranks)
    ranks = torch.gather(ranks, -1, idcs)
    labels = torch.gather(labels, -1, idcs)
    subtopics = torch.gather(subtopics, -1, idcs)
    if k is not None:
        ranks = ranks[:, :k]
        labels = labels[:, :k]
        subtopics = subtopics[:, :k]
    mask = (ranks == PAD_VALUE) | (ranks == 0) | (labels == PAD_VALUE)
    ranks = ranks.masked_fill(mask, 1)
    log_ranks = torch.log2(1 + ranks)
    discounts = 1 / log_ranks
    discounts = discounts.masked_fill(mask, 0)
    if scale_gains:
        gains = 2**labels - 1
    else:
        gains = labels
    subtopic_matrix = torch.nn.functional.one_hot(subtopics)
    cumulative_subtopics = (
        subtopic_matrix.cumsum(dim=1).gather(-1, subtopics[..., None]).squeeze(-1)
    ) - 1
    gains = gains * torch.pow(1 - alpha, cumulative_subtopics)
    dcgs = gains * discounts
    return dcgs.sum(dim=-1)


def get_alpha_ndcg(
    ranks: torch.Tensor,
    labels: torch.Tensor,
    subtopics: torch.Tensor,
    k: int | None = None,
    scale_gains: bool = False,
    optimal_labels: torch.Tensor | None = None,
    optimal_subtopics: torch.Tensor | None = None,
    alpha: float = 0.5,
):
    if optimal_labels is None:
        optimal_labels = labels
    if optimal_subtopics is None:
        optimal_subtopics = subtopics
    optimal_ranks = torch.argsort(torch.argsort(optimal_labels, descending=True))
    optimal_ranks = optimal_ranks + 1
    dcg = get_alpha_dcg(ranks, labels, subtopics, k, scale_gains, alpha)
    idcg = get_alpha_dcg(
        optimal_ranks, optimal_labels, optimal_subtopics, k, scale_gains, alpha
    )
    ndcg = dcg / (idcg + EPS)
    return ndcg


class MarginMSE(LossFunc):
    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        subtopics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert logits.shape[-1] == 2
        mask = ((logits == PAD_VALUE) | (labels == PAD_VALUE)).any(-1)
        logits_diff = logits[:, 0] - logits[:, 1]
        label_diff = labels[:, 0] - labels[:, 1]
        loss = torch.nn.functional.mse_loss(logits_diff, label_diff, reduction="none")
        return self.aggregate(loss, mask)


class RankNet(LossFunc):
    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        subtopics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        greater = labels[..., None] > labels[:, None]
        logits_mask = logits == PAD_VALUE
        label_mask = labels == PAD_VALUE
        mask = (
            logits_mask[..., None]
            | logits_mask[:, None]
            | label_mask[..., None]
            | label_mask[:, None]
            | ~greater
        )
        diff = logits[..., None] - logits[:, None]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            diff, greater.to(diff), reduction="none"
        )
        loss = loss.masked_fill(mask, 0)
        return self.aggregate(loss, mask)


class IARankNet(RankNet):
    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        subtopics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if subtopics is None:
            raise ValueError("subtopics must be provided for IARankNet")
        rank_net = super().compute(logits, labels, subtopics)
        depth = subtopics.shape[-1]
        sort_idcs = torch.argsort(logits, dim=-1, descending=True)
        logits = torch.gather(logits, -1, sort_idcs)
        labels = torch.gather(labels, -1, sort_idcs)
        subtopics = torch.gather(subtopics, -1, sort_idcs)
        num_subtopics = []
        _optimal_subtopic_count = []
        for idx, _subtopics in enumerate(subtopics):
            _, unique_idcs = _subtopics.unique(return_inverse=True)
            subtopics[idx] = unique_idcs
            _num_subtopics = unique_idcs.unique().shape[0]
            num_subtopics.append(_num_subtopics)
            num_even_topic_dists = math.ceil(depth / _num_subtopics)
            _optimal_subtopic_count.append(
                torch.arange(
                    num_even_topic_dists, device=logits.device
                ).repeat_interleave(_num_subtopics)[:depth]
                + 1
            )
        optimal_subtopic_count = torch.stack(_optimal_subtopic_count)
        max_subtopic = max(num_subtopics)
        cum_subtopic_count = (
            (
                subtopics[..., None]
                .expand(-1, -1, max_subtopic)
                .eq(torch.arange(max_subtopic, device=logits.device))
            )
            .cumsum(dim=1)
            .gather(-1, subtopics[..., None])
            .squeeze(-1)
        )
        diff_subtopic = (
            subtopics[:, None]
            .not_equal(subtopics[..., None])
            .logical_and(
                subtopics[..., None]
                .greater(0)
                .logical_and(subtopics[:, None].greater(0))
            )
        )
        subtopic_count_larger = cum_subtopic_count > optimal_subtopic_count
        subtopic_count_smaller = cum_subtopic_count < optimal_subtopic_count
        incorrect_order = (
            subtopic_count_larger[..., None] & subtopic_count_smaller[:, None]
        )
        diff = logits[..., None] - logits[:, None]
        diff = diff[incorrect_order & diff_subtopic]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            diff, torch.zeros_like(diff), reduction="none"
        )
        return self.aggregate(loss) + rank_net


class ApproxNDCG(LossFunc):
    def __init__(
        self,
        temperature: float = 1,
        reduction: Optional[Literal["mean", "sum"]] = "mean",
        scale_gains: bool = True,
    ):
        super().__init__(reduction)
        self.temperature = temperature
        self.scale_gains = scale_gains

    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        subtopics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        approx_ranks = get_approx_ranks(logits, self.temperature)
        ndcg = get_ndcg(approx_ranks, labels, k=None, scale_gains=self.scale_gains)
        loss = 1 - ndcg
        return self.aggregate(loss)


class ApproxMRR(LossFunc):
    def __init__(
        self,
        temperature: float = 1,
        reduction: Optional[Literal["mean", "sum"]] = "mean",
    ):
        super().__init__(reduction)
        self.temperature = temperature

    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        subtopics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        approx_ranks = get_approx_ranks(logits, self.temperature)
        mrr = get_mrr(approx_ranks, labels, k=None)
        loss = 1 - mrr
        return self.aggregate(loss)


class ApproxRankMSE(LossFunc):
    def __init__(
        self,
        temperature: float = 1,
        reduction: Optional[Literal["mean", "sum"]] = "mean",
        discounted: bool = False,
    ):
        super().__init__(reduction)
        self.temperature = temperature
        self.discounted = discounted

    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        subtopics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        approx_ranks = get_approx_ranks(logits, self.temperature)
        ranks = torch.argsort(labels, descending=True) + 1
        mask = (labels == PAD_VALUE) | (logits == PAD_VALUE)
        loss = torch.nn.functional.mse_loss(
            approx_ranks, ranks.to(approx_ranks), reduction="none"
        )
        if self.discounted:
            weight = 1 / torch.log2(ranks + 1)
            weight = weight.masked_fill(mask, 0)
            loss = loss * weight
        return self.aggregate(loss, mask)


class ListNet(LossFunc):
    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        subtopics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = (labels == PAD_VALUE) | (logits == PAD_VALUE)
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        labels = labels.to(logits)
        labels = labels.masked_fill(mask, torch.finfo(labels.dtype).min)
        logits = torch.nn.functional.softmax(logits, dim=-1)
        labels = torch.nn.functional.softmax(labels, dim=-1)
        loss = -(labels * torch.log(logits + EPS)).sum(dim=-1)
        return self.aggregate(loss)


class ListMLE(LossFunc):
    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        subtopics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        random_indices = torch.randperm(logits.shape[-1])
        logits_shuffled = logits[:, random_indices]
        label_shuffled = labels[:, random_indices]

        label_sorted, indices = torch.sort(label_shuffled, descending=True)

        mask = label_sorted == PAD_VALUE

        logits_sorted = torch.gather(logits_shuffled, -1, indices)
        logits_sorted = logits_sorted.masked_fill(mask, torch.finfo(logits.dtype).min)
        max_logits = logits_sorted.max(dim=-1, keepdim=True)[0]
        diff = logits_sorted - max_logits
        cumsum = torch.cumsum(diff.exp().flip(-1), dim=-1).flip(-1)
        loss = torch.log(cumsum + EPS) - diff
        loss = loss.masked_fill(mask, 0)
        return self.aggregate(loss)


class NeuralLoss(LossFunc):
    def __init__(
        self,
        temperature: float = 1.0,
        reduction: Optional[Literal["mean", "sum"]] = "mean",
    ):
        super().__init__(reduction)
        self.temperature = temperature

    def get_sorted_labels(
        self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        permutation_matrix = neural_sort(logits, mask, self.temperature)
        pred_sorted_labels = torch.matmul(
            permutation_matrix, labels[..., None].to(permutation_matrix)
        ).squeeze(-1)
        pred_sorted_labels = pred_sorted_labels.masked_fill(mask, 0)
        return pred_sorted_labels


class NeuralNDCG(NeuralLoss):
    def __init__(
        self,
        temperature: float = 1,
        reduction: Literal["mean", "sum"] | None = "mean",
        scale_gains: bool = True,
    ):
        super().__init__(temperature, reduction)
        self.scale_gains = scale_gains

    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        subtopics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = (labels == PAD_VALUE) | (logits == PAD_VALUE)
        pred_sorted_labels = self.get_sorted_labels(logits, labels, mask)
        ranks = torch.arange(logits.shape[-1], device=logits.device) + 1
        ranks = ranks[None, :].expand_as(pred_sorted_labels)
        ndcg = get_ndcg(ranks, labels, k=None, scale_gains=self.scale_gains)
        loss = 1 - ndcg
        return self.aggregate(loss)


class NeuralMRR(NeuralLoss):
    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        subtopics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = (labels == PAD_VALUE) | (logits == PAD_VALUE)
        labels = labels.clamp(None, 1)
        pred_sorted_labels = self.get_sorted_labels(logits, labels, mask)
        ranks = torch.arange(logits.shape[-1], device=logits.device) + 1
        reciprocal_ranks = 1 / ranks
        mrr = reciprocal_ranks * pred_sorted_labels
        loss = 1 - mrr
        return self.aggregate(loss)


class LocalizedContrastive(LossFunc):
    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        subtopics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = (labels == PAD_VALUE) | (logits == PAD_VALUE)
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        labels = labels.argmax(dim=1)
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        return self.aggregate(loss)
