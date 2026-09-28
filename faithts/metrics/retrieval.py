"""Retrieval-quality metrics with a RELEVANCE LABEL that is itself automatically verifiable
(no manual annotation, no subjective 'looks similar'): a neighbor is relevant to a query if the
neighbor's OWN future-window facts match the query's ground-truth future-window facts on the
majority of the 5 verifiable facts (direction, peak, volatility, magnitude, level_shift)."""
import torch
import torch.nn.functional as F
from ..explain.facts import N_LABELS


def relevance(query_facts, neighbor_facts, min_match=3):
    """query_facts (B,5), neighbor_facts (B,k,5) -> binary relevance (B,k)."""
    match = (neighbor_facts == query_facts[:, None, :]).sum(-1)
    return (match >= min_match).float()


def recall_at_k(rel, k=None):
    k = k or rel.shape[1]
    return rel[:, :k].max(1).values.mean().item()


def precision_at_k(rel, k=None):
    k = k or rel.shape[1]
    return rel[:, :k].mean().item()


def ndcg_at_k(rel, k=None):
    """rel already rank-ordered by similarity (as returned by RetrievalDB.search)."""
    k = k or rel.shape[1]
    rel = rel[:, :k]
    discount = 1.0 / torch.log2(torch.arange(2, k + 2, device=rel.device).float())
    dcg = (rel * discount[None]).sum(1)
    ideal = torch.sort(rel, dim=1, descending=True).values
    idcg = (ideal * discount[None]).sum(1).clamp_min(1e-6)
    return (dcg / idcg).mean().item()


def dtw_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Plain O(LM) DTW between two 1D sequences (numpy/torch, no extra deps)."""
    a, b = a.detach().cpu().numpy(), b.detach().cpu().numpy()
    L, M = len(a), len(b)
    D = torch.full((L + 1, M + 1), float("inf"))
    D[0, 0] = 0
    import numpy as np
    Dn = D.numpy()
    for i in range(1, L + 1):
        for j in range(1, M + 1):
            cost = abs(a[i - 1] - b[j - 1])
            Dn[i, j] = cost + min(Dn[i - 1, j], Dn[i, j - 1], Dn[i - 1, j - 1])
    return float(Dn[L, M])


def mean_dtw(query_ctx, neighbor_ctx):
    """query_ctx (B,L), neighbor_ctx (B,k,L) -> mean DTW of top-1 neighbor per query (subsampled
    internally by the caller for speed; DTW is O(L^2) per pair)."""
    B, k, L = neighbor_ctx.shape
    vals = [dtw_distance(query_ctx[b], neighbor_ctx[b, 0]) for b in range(B)]
    return sum(vals) / len(vals)


def retrieval_report(query_facts, neighbor_facts, k_list=(1, 3, 5)):
    rel = relevance(query_facts, neighbor_facts)
    out = {}
    for k in k_list:
        if k > rel.shape[1]: continue
        out[f"Recall@{k}"] = recall_at_k(rel, k)
        out[f"Precision@{k}"] = precision_at_k(rel, k)
        out[f"NDCG@{k}"] = ndcg_at_k(rel, k)
    return out
