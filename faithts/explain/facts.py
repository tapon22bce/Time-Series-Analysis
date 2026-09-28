"""Deterministic, verifiable 'fact sheet' of a forecast window.

All facts are computed in *context-normalised* units (context mean 0 / std 1), so they are
comparable across channels, datasets and between forecasts, neighbours and ground truth.
They are the ground truth for the explanation benchmark (no LLM judge needed).
"""
import torch

FACT_NAMES = ["direction", "peak", "volatility", "magnitude", "level_shift"]
FACT_LABELS = {
    "direction": ["down", "flat", "up"],
    "peak": ["early", "mid", "late"],
    "volatility": ["lower", "similar", "higher"],
    "magnitude": ["small", "medium", "large"],
    "level_shift": ["no", "yes"],
}
N_LABELS = [len(FACT_LABELS[n]) for n in FACT_NAMES]
THR = dict(direction=0.5, vol_lo=0.8, vol_hi=1.25, mag_lo=1.5, mag_hi=3.0, shift=1.0)


@torch.no_grad()
def extract_facts(ctx_n: torch.Tensor, fut_n: torch.Tensor) -> torch.Tensor:
    """ctx_n (B,L), fut_n (B,H) -> long (B,5). Works for ground truth, neighbours, forecasts."""
    B, L = ctx_n.shape
    H = fut_n.shape[1]
    tail_c = ctx_n[:, -max(L // 4, 1):].mean(1)
    tail_f = fut_n[:, -max(H // 4, 1):].mean(1)
    d = tail_f - tail_c
    direction = torch.where(d > THR["direction"], 2, torch.where(d < -THR["direction"], 0, 1))
    peak = torch.clamp((fut_n.argmax(1) * 3) // H, 0, 2)
    dc = ctx_n.diff(dim=1).std(1) + 1e-6
    df = fut_n.diff(dim=1).std(1)
    r = df / dc
    vol = torch.where(r > THR["vol_hi"], 2, torch.where(r < THR["vol_lo"], 0, 1))
    rng = fut_n.max(1).values - fut_n.min(1).values
    mag = torch.where(rng > THR["mag_hi"], 2, torch.where(rng < THR["mag_lo"], 0, 1))
    shift = (fut_n.mean(1).abs() > THR["shift"]).long()
    return torch.stack([direction, peak, vol, mag, shift], 1).long()


def facts_to_dict(row):
    return {n: FACT_LABELS[n][int(row[i])] for i, n in enumerate(FACT_NAMES)}


def dict_to_facts(d):
    out = []
    for n in FACT_NAMES:
        v = str(d.get(n, "")).strip().lower()
        out.append(FACT_LABELS[n].index(v) if v in FACT_LABELS[n] else -1)
    return torch.tensor(out).long()


def facts_sentence(row):
    d = facts_to_dict(row)
    return (f"{d['direction']} trend, peak {d['peak']}, {d['volatility']} volatility, "
            f"{d['magnitude']} range, level shift: {d['level_shift']}")
