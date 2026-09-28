"""Bias-variance / conditional-shift analysis of retrieval-conditioned diffusion.

Surrogate model (documented derivation in docs/THEORY.md): treat the forecast as
    yhat = f(ctx) + W * (r - f(ctx)) + noise,   r = weighted retrieval signal
so W (~ the learned `gate`) trades off bias from a possibly-shifted retrieval signal against
variance reduction from averaging over neighbors. If neighbors are drawn from the true
conditional (no corruption), Bias^2 decreases and Variance decreases as W increases -> MSE
strictly improves with more retrieval. If a fraction `p` of neighbors are corrupted (conditional
shift), Bias^2 grows like (p * shift)^2 * W^2 while Variance still shrinks like 1/(1+kW) -- so the
optimal W* shrinks as p or shift grow. This file (a) runs that exact controlled experiment on real
model instances so the empirical curves can be plotted against the surrogate's prediction, and
(b) compares a FIXED-weight retrieval policy (RATD-style ablation, `weighter.sparse_k=k`, gate
fixed to 1) against our ADAPTIVE gate under the same corruption sweep -- this comparison is what
RATD-style papers do not report.
"""
from dataclasses import dataclass
import copy
import numpy as np, torch


def corrupt_database(db, frac, shift_scale, seed=0):
    """Return a COPY of the retrieval DB with `frac` of its stored futures replaced by
    context+random-walk continuations shifted by `shift_scale` std -- i.e. plausible-looking but
    systematically wrong neighbors (the conditional-shift stress test)."""
    rng = torch.Generator().manual_seed(seed)
    db2 = copy.copy(db)
    fut = db.w.fut_n.clone()
    n = fut.shape[0]
    n_bad = int(frac * n)
    bad_idx = torch.randperm(n, generator=rng)[:n_bad]
    noise = torch.randn(n_bad, fut.shape[1], generator=rng) * 0.3
    fut[bad_idx] = fut[bad_idx] + shift_scale + noise
    w2 = db.w.__class__(**{f: (fut if f == "fut_n" else getattr(db.w, f)) for f in db.w.__dataclass_fields__})
    db2.w = w2
    return db2


@torch.no_grad()
def bias_variance_sweep(db_builder, weighter, net_sample_fn, test_windows, fracs, shift_scale=2.0,
                         n_repeats=8, n_queries=40, k=5, fixed_weight_gate=None, seed=0):
    """db_builder() -> fresh RetrievalDB (so each corruption level starts from a clean copy).
    Returns a list of dicts: {frac, bias2, variance, mse, mean_gate}."""
    g = torch.Generator().manual_seed(seed)
    qidx = torch.randperm(len(test_windows), generator=g)[: min(n_queries, len(test_windows))]
    out = []
    for frac in fracs:
        db = corrupt_database(db_builder(), frac, shift_scale, seed=seed)
        preds = torch.zeros(n_repeats, len(qidx), test_windows.fut_n.shape[1])
        gates = []
        for r in range(n_repeats):
            for j, qi in enumerate(qidx.tolist()):
                ctx, chan, tf = test_windows.ctx_n[qi][None], test_windows.chan[qi][None], test_windows.tfeat[qi][None]
                idx, sim = db.search(ctx, chan, k=k)
                nb = db.gather(idx)
                if fixed_weight_gate is not None:
                    w = torch.softmax(sim, 1) * fixed_weight_gate
                    gate = torch.full((1, 1), fixed_weight_gate)
                else:
                    w, gate = weighter(sim, ctx, nb["fut_n"])
                pred = net_sample_fn(ctx, nb["fut_n"], w, tf)[0]
                preds[r, j] = pred
                if r == 0: gates.append(float(gate.mean()))
        true = test_windows.fut_n[qidx]
        mean_pred = preds.mean(0)
        bias2 = (mean_pred - true).pow(2).mean().item()
        variance = preds.var(0, unbiased=(n_repeats > 1)).mean().item()
        mse = (preds - true[None]).pow(2).mean().item()
        out.append(dict(frac=frac, bias2=bias2, variance=variance, mse=mse, mean_gate=float(np.mean(gates))))
    return out


def theoretical_curve(fracs, shift_scale=2.0, base_var=1.0, k_neighbors=5, W_grid=None):
    """Closed-form surrogate prediction (see docstring) for sanity-checking the empirical sweep:
    for each corruption fraction, returns the analytically OPTIMAL W* = argmin MSE(W) and the
    resulting bias^2, variance, mse under the linear-Gaussian surrogate. This is a *toy* model to
    build intuition and to report alongside the empirical numbers, not a claim that real diffusion
    forecasts are linear-Gaussian."""
    W_grid = W_grid if W_grid is not None else np.linspace(0, 1, 101)
    out = []
    for p in fracs:
        bias2 = (p * shift_scale * W_grid) ** 2
        var = base_var / (1 + (k_neighbors - 1) * W_grid)          # averaging benefit of W>0
        mse = bias2 + var
        i = int(np.argmin(mse))
        out.append(dict(frac=p, W_star=float(W_grid[i]), bias2=float(bias2[i]), variance=float(var[i]),
                        mse=float(mse[i])))
    return out
