"""Baselines. All operate in the same context-normalised (ctx_n -> fut_n) space as the main model
so metrics are directly comparable. `fit`/`predict` share a scikit-learn-like interface; torch
baselines additionally expose `.parameters()` for a standard training loop (see scripts/train.py)."""
import torch, torch.nn as nn


class NaiveLast(nn.Module):
    """yhat_h = ctx[-1] for all h (in normalised units, last context value is exactly 0 -> this is
    the trivial 'predict the mean' baseline after our instance normalisation)."""
    def forward(self, ctx_n, **kw): return ctx_n[:, -1:].expand(-1, self._H)
    def __init__(self, H): super().__init__(); self._H = H


class SeasonalNaive(nn.Module):
    """yhat_h = ctx[-period + (h mod period)] (needs raw, non-instance-normalised context; here we
    reuse ctx_n since seasonality survives affine normalisation)."""
    def __init__(self, H, period): super().__init__(); self.H, self.period = H, period
    def forward(self, ctx_n, **kw):
        L = ctx_n.shape[1]
        idx = [L - self.period + (h % self.period) for h in range(self.H)]
        idx = [i if 0 <= i < L else L - 1 for i in idx]
        return ctx_n[:, idx]


class RidgeLinear(nn.Module):
    """Closed-form-trainable linear map ctx_n(L) -> fut_n(H); trained with plain SGD here for a
    unified training loop, equivalent at convergence to ridge regression with the given L2 weight."""
    def __init__(self, L, H): super().__init__(); self.lin = nn.Linear(L, H)
    def forward(self, ctx_n, **kw): return self.lin(ctx_n)


class DLinearLite(nn.Module):
    """Trend/seasonal decomposition + two linear heads (Zeng et al. 2023, DLinear), a strong,
    cheap point-forecasting baseline that beats many Transformers on ETT-scale data."""
    def __init__(self, L, H, kernel=25):
        super().__init__()
        self.kernel = kernel
        self.pad = kernel // 2
        self.avg = nn.AvgPool1d(kernel, stride=1, padding=self.pad)
        self.seasonal = nn.Linear(L, H)
        self.trend = nn.Linear(L, H)

    def forward(self, ctx_n, **kw):
        x = ctx_n.unsqueeze(1)
        trend = self.avg(x).squeeze(1)[:, : ctx_n.shape[1]]
        seasonal = ctx_n - trend
        return self.seasonal(seasonal) + self.trend(trend)


class TinyPatchTransformer(nn.Module):
    """Small patch-based Transformer encoder (PatchTST-style) -> linear head. No retrieval, no
    diffusion: the point-forecasting ceiling we compare the diffusion models' MEAN prediction
    against, to check the generative machinery is not sacrificing point accuracy."""
    def __init__(self, L, H, patch=16, d=64, heads=4, depth=2, tfeat_dim=3):
        super().__init__()
        assert L % patch == 0, "context length must be a multiple of patch size"
        self.n_patches, self.patch, self.d = L // patch, patch, d
        self.embed = nn.Linear(patch, d)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, depth)
        self.tfeat = nn.Linear(tfeat_dim, d)
        self.head = nn.Linear(d * self.n_patches + d, H)

    def forward(self, ctx_n, tfeat=None, **kw):
        B, L = ctx_n.shape
        x = ctx_n.view(B, self.n_patches, self.patch)
        h = self.enc(self.embed(x) + self.pos)
        tf = self.tfeat(tfeat) if tfeat is not None else torch.zeros(B, self.d, device=ctx_n.device)
        return self.head(torch.cat([h.reshape(B, -1), tf], -1))


def build_baseline(name, L, H, period=24, tfeat_dim=3):
    return {
        "naive_last": lambda: NaiveLast(H),
        "seasonal_naive": lambda: SeasonalNaive(H, period),
        "ridge_linear": lambda: RidgeLinear(L, H),
        "dlinear": lambda: DLinearLite(L, H),
        "patch_transformer": lambda: TinyPatchTransformer(L, H, tfeat_dim=tfeat_dim),
    }[name]()
