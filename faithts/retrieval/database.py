"""Train-only retrieval database. Embeddings = shape (z-scored, low-pass) + stat features,
so retrieval is well-defined without pretraining and is reproducible/inspectable (important for
a faithfulness benchmark: we need to know exactly why a neighbor was retrieved)."""
import torch
import torch.nn.functional as F
from ..data.windows import Windows


def embed(ctx_n: torch.Tensor, n_low=8) -> torch.Tensor:
    """(N,L) -> embedding used for kNN. Combines three signals:
      1. the normalised RAW shape itself (dominant term) -- cosine similarity on this is exactly
         Pearson correlation of the two z-normalised subsequences, the standard shape-matching
         measure for time-series retrieval (Rakthanmanon et al. 2012); crucially this is
         PHASE-SENSITIVE, unlike FFT magnitude alone, which is required to tell apart two windows
         with similar frequency content but different alignment/shape.
      2. low-frequency FFT magnitude (phase-invariant, robust to small misalignment/periodicity)
      3. a handful of scale/trend summary stats (kept small so they do not dominate the L2 norm).
    Each block is independently L2-normalised, weighted, then the concatenation is normalised
    again so the returned embedding always has unit norm."""
    N, L = ctx_n.shape
    shape = F.normalize(ctx_n, dim=1) * 2.0
    fft = torch.fft.rfft(ctx_n, dim=1)
    mag = fft.abs()[:, : n_low + 1]
    mag = F.normalize(mag, dim=1) * 1.0
    d1 = ctx_n.diff(dim=1)
    stats = torch.stack([
        ctx_n.mean(1), ctx_n.std(1), ctx_n[:, -1] - ctx_n[:, 0],
        d1.std(1), ctx_n.max(1).values, ctx_n.min(1).values,
    ], 1)
    stats = F.normalize(stats, dim=1) * 0.5
    return F.normalize(torch.cat([shape, mag, stats], 1), dim=1)


class RetrievalDB:
    """Holds embeddings + payload (ctx_n, fut_n, facts, chan, time) for a TRAIN split only."""

    def __init__(self, train_windows: Windows, exclude_self=True):
        self.w = train_windows
        self.emb = F.normalize(embed(self.w.ctx_n), dim=1)
        self.exclude_self = exclude_self

    def __len__(self): return len(self.w)

    def to(self, device):
        self.emb = self.emb.to(device); self.w = self.w.sel(torch.arange(len(self.w))).__class__(
            *[getattr(self.w, f).to(device) for f in self.w.__dataclass_fields__]); return self

    @torch.no_grad()
    def search(self, query_ctx_n: torch.Tensor, query_chan: torch.Tensor, k: int,
               same_channel_only=True, exclude_time_ns: torch.Tensor = None):
        """Brute-force cosine kNN, batched. Returns idx (B,k), sim (B,k) in [-1,1]."""
        q = F.normalize(embed(query_ctx_n), dim=1)
        sims = q @ self.emb.T                                  # (B, N_db)
        if same_channel_only:
            mask = self.w.chan[None, :] != query_chan[:, None]
            sims = sims.masked_fill(mask, -2.0)
        if exclude_time_ns is not None:
            same_t = self.w.time_ns[None, :] == exclude_time_ns[:, None]
            sims = sims.masked_fill(same_t, -2.0)
        sim, idx = torch.topk(sims, k, dim=1)
        return idx, sim

    def gather(self, idx: torch.Tensor):
        """idx (B,k) -> dict of (B,k,...) neighbor tensors."""
        return dict(ctx_n=self.w.ctx_n[idx], fut_n=self.w.fut_n[idx], facts=self.w.facts[idx],
                    chan=self.w.chan[idx], time_ns=self.w.time_ns[idx])
