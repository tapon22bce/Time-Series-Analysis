"""Sample-based probabilistic metrics (no closed form needed since sampling is generic DDPM)."""
import torch


def crps_from_samples(samples, true):
    """samples (S,B,H), true (B,H) -> scalar CRPS (energy-score-consistent empirical estimator,
    Gneiting & Raftery 2007 eq. 21)."""
    S = samples.shape[0]
    term1 = (samples - true[None]).abs().mean(0)                       # (B,H)
    if S > 1:
        diffs = (samples[:, None] - samples[None, :]).abs()             # (S,S,B,H)
        term2 = diffs.mean(dim=(0, 1)) * 0.5
    else:
        term2 = torch.zeros_like(term1)
    return (term1 - term2).mean().item()


def pinball_loss(samples, true, q):
    """Quantile loss at level q in (0,1) using the empirical quantile of `samples` (S,B,H)."""
    qhat = torch.quantile(samples, q, dim=0)
    diff = true - qhat
    return torch.maximum(q * diff, (q - 1) * diff).mean().item()


def quantile_coverage(samples, true, q_lo=0.1, q_hi=0.9):
    lo = torch.quantile(samples, q_lo, dim=0)
    hi = torch.quantile(samples, q_hi, dim=0)
    return ((true >= lo) & (true <= hi)).float().mean().item()


def all_prob_metrics(samples, true, quantiles=(0.1, 0.5, 0.9)):
    out = dict(CRPS=crps_from_samples(samples, true),
               Coverage80=quantile_coverage(samples, true, 0.1, 0.9))
    for q in quantiles:
        out[f"Pinball@{q}"] = pinball_loss(samples, true, q)
    return out
