"""Score network for the conditional diffusion model.

Conditioning is split into three EXPLICIT streams so the retrieval contribution is never hidden
inside an opaque cross-attention block:
  1. context stream  : encoding of the (clean) observed window
  2. retrieval stream: gate * sum_i w_i * value(neighbor_i)   <-- w_i is exactly RetrievalWeighter's output
  3. time-feature stream: hour/day/year-of-year sinusoids for the forecast horizon
All three are combined by FiLM (per-layer scale/shift), so we can zero out stream 2 exactly to get
the "no-retrieval" counterfactual used by the faithfulness benchmark, and the mixture weight of
stream 2 (i.e. gate) is a legible, reportable number.
"""
import math
import torch, torch.nn as nn


def sinusoidal_embedding(t: torch.Tensor, dim: int):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    ang = t.float()[:, None] * freqs[None, :]
    return torch.cat([ang.sin(), ang.cos()], 1)


class FiLMBlock(nn.Module):
    def __init__(self, d, d_cond):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.SiLU(), nn.Linear(d * 2, d))
        self.to_film = nn.Linear(d_cond, 2 * d)

    def forward(self, x, cond):
        scale, shift = self.to_film(cond).chunk(2, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        return x + self.ff(h)


class RetrievalConditionedScoreNet(nn.Module):
    """Predicts eps for a (B,H) future window given: noisy x_t, t, context (B,L), neighbor futures
    (B,k,H), explicit weights w (B,k) with sum(w)<=1, and forecast-horizon time features (B,3)."""

    def __init__(self, H=24, L=96, d=128, depth=4, k=5, tfeat_dim=3, use_retrieval=True):
        super().__init__()
        self.H, self.d, self.use_retrieval = H, d, use_retrieval
        self.x_in = nn.Linear(H, d)
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.ctx_enc = nn.Sequential(nn.Linear(L, d), nn.SiLU(), nn.Linear(d, d))
        self.nb_val = nn.Sequential(nn.Linear(H, d), nn.SiLU(), nn.Linear(d, d))   # value(neighbor)
        self.retr_proj = nn.Linear(d, d)
        self.time_enc = nn.Sequential(nn.Linear(tfeat_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.cond_mix = nn.Linear(3 * d, d)
        self.blocks = nn.ModuleList([FiLMBlock(d, d) for _ in range(depth)])
        self.out = nn.Linear(d, H)

    def retrieval_stream(self, nb_fut_n: torch.Tensor, w: torch.Tensor):
        """nb_fut_n (B,k,H), w (B,k) -> (B,d) conditioning vector. Zeroing `w` disables retrieval
        exactly, giving the counterfactual used for faithfulness intervention tests."""
        val = self.nb_val(nb_fut_n)                     # (B,k,d)
        mixed = torch.einsum("bk,bkd->bd", w, val)       # explicit weighted sum
        return self.retr_proj(mixed)

    def forward(self, x_t, t, ctx_n, nb_fut_n, w, tfeat):
        B = x_t.shape[0]
        h = self.x_in(x_t)
        temb = self.t_mlp(sinusoidal_embedding(t, self.d))
        cctx = self.ctx_enc(ctx_n)
        cretr = self.retrieval_stream(nb_fut_n, w) if self.use_retrieval else torch.zeros(B, self.d, device=x_t.device)
        ctime = self.time_enc(tfeat)
        cond = self.cond_mix(torch.cat([cctx + temb, cretr, ctime], -1))
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out(h)
