"""Windowing with chronological splits, train-only retrieval databases and leakage guards."""
from dataclasses import dataclass, field
import numpy as np, pandas as pd, torch
from ..explain.facts import extract_facts

MIN_STD = 0.05  # floor for context std (in globally-standardised units)


@dataclass
class TSData:
    name: str
    values: np.ndarray            # (T, C) raw
    times: pd.DatetimeIndex       # (T,)
    channels: list
    freq_minutes: float
    split_ends: tuple             # (n_train, n_val, n_test) end indices (exclusive)
    period: int = 24
    z: np.ndarray = None          # globally standardised with TRAIN stats only
    mu: np.ndarray = None
    sd: np.ndarray = None

    def __post_init__(self):
        n_tr = self.split_ends[0]
        self.mu = np.nanmean(self.values[:n_tr], 0)
        self.sd = np.nanstd(self.values[:n_tr], 0) + 1e-6
        self.z = ((self.values - self.mu) / self.sd).astype(np.float32)
        self.z = np.nan_to_num(self.z, nan=0.0)


@dataclass
class Windows:
    ctx_n: torch.Tensor     # (N,L) instance-normalised context
    fut_n: torch.Tensor     # (N,H) future in context-normalised units
    mean: torch.Tensor      # (N,) context mean in global-z units
    std: torch.Tensor       # (N,)
    chan: torch.Tensor      # (N,)
    origin: torch.Tensor    # (N,) index of first future step
    tfeat: torch.Tensor     # (N,3) hour-of-day, day-of-week, day-of-year fractions
    facts: torch.Tensor     # (N,5)
    time_ns: torch.Tensor   # (N,) origin timestamp (int64 ns)

    def __len__(self): return self.ctx_n.shape[0]

    def sel(self, idx):
        idx = torch.as_tensor(idx)
        return Windows(*[getattr(self, f)[idx] for f in self.__dataclass_fields__])

    @property
    def fut_z(self):  # future in global-z units
        return self.fut_n * self.std[:, None] + self.mean[:, None]


def _time_feats(times: pd.DatetimeIndex):
    h = (times.hour.values + times.minute.values / 60.0) / 24.0
    dw = times.dayofweek.values / 7.0
    dy = (times.dayofyear.values - 1) / 366.0
    return np.stack([h, dw, dy], 1).astype(np.float32)


def make_windows(data: TSData, split: str, L: int, H: int, stride: int = 1,
                 max_n: int = None, seed: int = 0, channels=None) -> Windows:
    n_tr, n_va, n_te = data.split_ends
    if split == "train":   lo, hi = L, n_tr                 # future must end <= n_tr
    elif split == "val":   lo, hi = n_tr, n_va
    elif split == "test":  lo, hi = n_va, n_te
    else: raise ValueError(split)
    origins = np.arange(lo, hi - H + 1, stride)
    C = data.z.shape[1]
    chans = np.arange(C) if channels is None else np.asarray(channels)
    oo, cc = np.meshgrid(origins, chans, indexing="ij")
    oo, cc = oo.reshape(-1), cc.reshape(-1)
    if max_n is not None and len(oo) > max_n:
        rng = np.random.RandomState(seed)
        keep = np.sort(rng.choice(len(oo), max_n, replace=False))
        oo, cc = oo[keep], cc[keep]
    idx = oo[:, None] + np.arange(-L, H)[None, :]
    w = data.z[idx, cc[:, None]]                        # (N, L+H)
    ctx, fut = torch.from_numpy(w[:, :L]), torch.from_numpy(w[:, L:])
    mean = ctx.mean(1)
    std = ctx.std(1).clamp_min(MIN_STD)
    ctx_n = ((ctx - mean[:, None]) / std[:, None]).clamp(-10, 10)
    fut_n = ((fut - mean[:, None]) / std[:, None]).clamp(-10, 10)
    tf = torch.from_numpy(_time_feats(data.times[oo]))
    return Windows(ctx_n, fut_n, mean, std, torch.from_numpy(cc).long(), torch.from_numpy(oo).long(),
                   tf, extract_facts(ctx_n, fut_n), torch.from_numpy(data.times[oo].asi8.copy()))


def make_block_windows(data: TSData, split: str, L: int, H: int, max_n: int = None, seed: int = 0):
    """Windower for datasets PRE-PACKED into non-overlapping length-(L+H) blocks (currently only
    `retrievable_synth`, see synthetic.py). Unlike `make_windows`, this never lets a context or
    future span two different blocks/templates: block i in [0, n_windows) is fully assigned to
    exactly one split by index (chronological 70/10/20), and each block contributes exactly one
    (context, future) pair. This sidesteps the generic split-boundary convention in `make_windows`
    (val/test origins start at the raw split index, not split index + L), which is correct for
    ordinary continuous series but silently straddles block boundaries for packed data."""
    LH = L + H
    n_windows = data.z.shape[0] // LH
    n_tr = int(n_windows * 0.7); n_va = int(n_windows * 0.8)
    ranges = dict(train=(0, n_tr), val=(n_tr, n_va), test=(n_va, n_windows))
    lo, hi = ranges[split]
    block_ids = np.arange(lo, hi)
    if max_n is not None and len(block_ids) > max_n:
        rng = np.random.RandomState(seed)
        block_ids = np.sort(rng.choice(block_ids, max_n, replace=False))
    origins = block_ids * LH + L                     # first-future-step index, one per block
    idx = origins[:, None] + np.arange(-L, H)[None, :]
    w = data.z[idx, 0]                                 # single channel (C=1 for this dataset)
    ctx, fut = torch.from_numpy(w[:, :L]), torch.from_numpy(w[:, L:])
    mean = ctx.mean(1)
    std = ctx.std(1).clamp_min(MIN_STD)
    ctx_n = ((ctx - mean[:, None]) / std[:, None]).clamp(-10, 10)
    fut_n = ((fut - mean[:, None]) / std[:, None]).clamp(-10, 10)
    tf = torch.from_numpy(_time_feats(data.times[origins]))
    chan = torch.zeros(len(origins), dtype=torch.long)
    return Windows(ctx_n, fut_n, mean, std, chan, torch.from_numpy(origins).long(), tf,
                   extract_facts(ctx_n, fut_n), torch.from_numpy(data.times[origins].asi8.copy()))
