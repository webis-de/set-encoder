import numpy as np
import torch
from lightning_ir.loss.loss import ApproxLossFunction

################
# ERR-IA@k & nERR-IA@k
################


def torch_rankwise_err_ia(
    sorted_q_doc_rele_mat, max_label=None, k=10, point=True, device="cpu"
):
    assert max_label is not None  # it is either query-level or corpus-level
    num_subtopics = sorted_q_doc_rele_mat.size(0)
    valid_max_cutoff = sorted_q_doc_rele_mat.size(1)
    cutoff = min(valid_max_cutoff, k)

    target_q_doc_rele_mat = sorted_q_doc_rele_mat[:, 0:cutoff]

    t2 = torch.tensor([2.0], dtype=torch.float, device=device)
    satis_pros = (torch.pow(t2, target_q_doc_rele_mat) - 1.0) / torch.pow(t2, max_label)
    unsatis_pros = torch.ones_like(target_q_doc_rele_mat, device=device) - satis_pros
    cum_unsatis_pros = torch.cumprod(unsatis_pros, dim=1)
    cascad_unsatis_pros = torch.ones_like(cum_unsatis_pros, device=device)
    cascad_unsatis_pros[:, 1:cutoff] = cum_unsatis_pros[:, 0 : cutoff - 1]

    non_zero_inds = torch.nonzero(torch.sum(target_q_doc_rele_mat, dim=1))
    zero_metric_value = False if non_zero_inds.size(0) > 0 else True

    if zero_metric_value:
        return (
            torch.zeros(1, device=device),
            zero_metric_value,
        )  # since no relevant documents within the list
    else:
        pos_rows = non_zero_inds[:, 0]

    reciprocal_ranks = 1.0 / (
        torch.arange(cutoff, dtype=torch.float, device=device).view(1, -1) + 1.0
    )
    expt_satis_ranks = (
        satis_pros[pos_rows, :] * cascad_unsatis_pros[pos_rows, :] * reciprocal_ranks
    )

    if point:  # a specific position
        err_ia = torch.sum(expt_satis_ranks, dim=(1, 0))
        return err_ia / num_subtopics, zero_metric_value
    else:
        rankwise_err_ia = torch.cumsum(expt_satis_ranks, dim=1)
        rankwise_err_ia = torch.sum(rankwise_err_ia, dim=0)
        return rankwise_err_ia / num_subtopics, zero_metric_value


def torch_err_ia_at_k(sorted_q_doc_rele_mat, max_label=None, k=10, device="cpu"):
    err_ia_at_k, _ = torch_rankwise_err_ia(
        sorted_q_doc_rele_mat=sorted_q_doc_rele_mat,
        max_label=max_label,
        k=k,
        point=True,
        device=device,
    )
    return err_ia_at_k


def torch_err_ia_at_ks(sorted_q_doc_rele_mat, max_label=None, ks=None, device="cpu"):
    valid_max_cutoff = sorted_q_doc_rele_mat.size(1)
    need_padding = True if valid_max_cutoff < max(ks) else False
    used_ks = [k for k in ks if k <= valid_max_cutoff] if need_padding else ks
    max_cutoff = max(used_ks)
    inds = torch.from_numpy(np.asarray(used_ks) - 1)

    rankwise_err_ia, zero_metric_value = torch_rankwise_err_ia(
        sorted_q_doc_rele_mat=sorted_q_doc_rele_mat,
        point=False,
        max_label=max_label,
        k=max_cutoff,
        device=device,
    )
    if zero_metric_value:
        return torch.zeros(len(ks), device=device)
    else:
        err_ia_at_ks = rankwise_err_ia[inds]
        if need_padding:
            padded_err_ia_at_ks = torch.zeros(len(ks), device=device)
            padded_err_ia_at_ks[0 : len(used_ks)] = err_ia_at_ks
            return padded_err_ia_at_ks
        else:
            return err_ia_at_ks


def torch_nerr_ia_at_k(
    sys_q_doc_rele_mat, ideal_q_doc_rele_mat, max_label=None, k=10, device="cpu"
):
    valid_max_cutoff = sys_q_doc_rele_mat.size(1)
    cutoff = min(valid_max_cutoff, k)

    sys_err_ia_at_k, zero_metric_value = torch_rankwise_err_ia(
        sorted_q_doc_rele_mat=sys_q_doc_rele_mat,
        point=True,
        max_label=max_label,
        k=cutoff,
        device=device,
    )
    if zero_metric_value:
        return sys_err_ia_at_k
    else:
        ideal_err_ia_at_k, _ = torch_rankwise_err_ia(
            sorted_q_doc_rele_mat=ideal_q_doc_rele_mat,
            max_label=max_label,
            k=cutoff,
            point=True,
            device=device,
        )
        if ideal_err_ia_at_k > 0:
            nerr_ia_at_k = sys_err_ia_at_k / ideal_err_ia_at_k
        else:
            nerr_ia_at_k = torch.tensor([0.0], device=device)

        return nerr_ia_at_k


def torch_nerr_ia_at_ks(
    sys_q_doc_rele_mat, ideal_q_doc_rele_mat, max_label=None, ks=None, device="cpu"
):
    valid_max_cutoff = sys_q_doc_rele_mat.size(1)
    need_padding = True if valid_max_cutoff < max(ks) else False
    used_ks = [k for k in ks if k <= valid_max_cutoff] if need_padding else ks
    max_cutoff = max(used_ks)
    inds = torch.from_numpy(np.asarray(used_ks) - 1)

    sys_rankwise_err_ia, zero_metric_value = torch_rankwise_err_ia(
        sorted_q_doc_rele_mat=sys_q_doc_rele_mat,
        point=False,
        max_label=max_label,
        k=max_cutoff,
        device=device,
    )
    if zero_metric_value:
        return sys_rankwise_err_ia
    else:
        ideal_rankwise_err_ia, _ = torch_rankwise_err_ia(
            sorted_q_doc_rele_mat=ideal_q_doc_rele_mat,
            max_label=max_label,
            k=max_cutoff,
            point=False,
            device=device,
        )
        rankwise_nerr_ia = sys_rankwise_err_ia / ideal_rankwise_err_ia

        if torch.count_nonzero(ideal_rankwise_err_ia) < max_cutoff:
            zero_mask = ideal_rankwise_err_ia <= 0
            rankwise_nerr_ia[zero_mask] = 0.0

        nerr_ia_at_ks = rankwise_nerr_ia[inds]
        if need_padding:
            padded_nerr_ia_at_ks = torch.zeros(len(ks), device=device)
            padded_nerr_ia_at_ks[0 : len(used_ks)] = nerr_ia_at_ks
            return padded_nerr_ia_at_ks
        else:
            return nerr_ia_at_ks


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
            targets[torch.arange(batch_size), idcs] = 0
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
