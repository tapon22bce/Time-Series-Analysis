"""Synthetic multi-regime series for unit tests / smoke tests (retrieval genuinely helps here)."""
import numpy as np, pandas as pd
from .windows import TSData


def make_retrievable_synthetic(n_windows=6000, C=1, seed=0, period=24, L=96, H=24, K=40, ctx_noise=0.25,
                                fut_noise=0.35):
    """A synthetic process where retrieval is PROVABLY useful, built directly as a set of
    (context, future) windows rather than a single long series (avoids the confound in
    `make_synthetic`, where the active regime is often already visible in the context window
    itself). Construction: draw K fixed length-(L+H) "templates" (smooth random curves, several
    superposed sinusoids with random phase/freq/amplitude). Each window independently samples a
    template id k~Uniform(K), then adds INDEPENDENT noise to the context part and the future part.
    Because the noise is independent, a query's context does not fully disambiguate which template
    generated it, nor does it reveal the exact future noise realisation -- but if the retrieval
    database stores its OWN (differently-noised) instances of the same templates, a matching
    context (found via kNN in embedding space) points to a future sample from the SAME template,
    which is a materially better prior than the plain model's best strategy of regressing to a
    noise-blurred posterior mixture over all K templates. This is the standard argument for why
    retrieval helps in RAG-style setups (Guu et al. 2020; Khandelwal et al. 2020) transplanted into
    a controlled time-series testbed. Returns a TSData wrapper that packs windows end-to-end so the
    existing `make_windows` pipeline (splits, standardisation) needs no changes."""
    rng = np.random.RandomState(seed)
    t = np.arange(L + H)
    templates = np.zeros((K, L + H), dtype=np.float32)
    for k in range(K):
        n_comp = rng.randint(2, 5)
        sig = np.zeros(L + H)
        for _ in range(n_comp):
            freq = rng.uniform(1, 6) / (L + H)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.4, 1.2)
            sig += amp * np.sin(2 * np.pi * freq * t + phase)
        templates[k] = sig
    tem_id = rng.randint(0, K, n_windows)
    ctx = templates[tem_id, :L] + ctx_noise * rng.randn(n_windows, L)
    fut = templates[tem_id, L:] + fut_noise * rng.randn(n_windows, H)
    series = np.concatenate([ctx, fut], axis=1).reshape(-1)[:, None].astype(np.float32)  # (n_windows*(L+H), 1)
    times = pd.date_range("2016-01-01", periods=len(series), freq="h")
    n_tot = len(series)
    d = TSData("retrievable_synth", series, times, ["s0"], 60.0,
              (int(n_tot * .7), int(n_tot * .8), n_tot), period)
    d.meta = dict(templates=templates, template_id=tem_id, L=L, H=H, K=K)
    return d


def make_synthetic(T=6000, C=3, seed=0, period=24):
    """Multi-regime series (recurring regime shapes) -- kept for realism/DTW-style sanity checks.
    NOTE: because regimes last much longer than a typical context window, the active regime is
    often already visible in the context itself, so this dataset does NOT strongly require
    retrieval; use `make_retrievable_synthetic` for the controlled "retrieval helps" experiments."""
    rng = np.random.RandomState(seed)
    t = np.arange(T)
    out = []
    for c in range(C):
        base = np.sin(2 * np.pi * t / period + c) + 0.5 * np.sin(2 * np.pi * t / (period * 7))
        # regime switches every ~2-4 days with regime-specific shape (recurring -> retrievable)
        reg = np.zeros(T); i = 0
        shapes = [lambda x: 1.5 * np.sin(2 * np.pi * x / period * 2), lambda x: 2.0 * (x % period) / period - 1,
                  lambda x: 0.0 * x, lambda x: -1.5 * np.sin(2 * np.pi * x / period)]
        while i < T:
            ln = rng.randint(2 * period, 4 * period); r = rng.randint(4)
            reg[i:i + ln] = shapes[r](np.arange(min(ln, T - i)))[:len(reg[i:i + ln])]; i += ln
        out.append(base + reg + 0.15 * rng.randn(T))
    v = np.stack(out, 1).astype(np.float32)
    times = pd.date_range("2016-01-01", periods=T, freq="h")
    return TSData("synthetic", v, times, [f"s{c}" for c in range(C)], 60.0,
                  (int(T * .7), int(T * .8), T), period)
