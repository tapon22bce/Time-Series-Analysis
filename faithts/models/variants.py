"""End-to-end model variants used across the ablation table (docs/BENCHMARK.md Table 2):

  point_*          : deterministic baselines from baselines.py (no sampling / no CRPS)
  diffusion_plain  : GaussianDiffusion with retrieval DISABLED (use_retrieval=False) - unconditional
  diffusion_fixed  : retrieval ENABLED but gate fixed to 1 and weights = plain softmax(sim/tau)
                     (this reproduces the RATD-style "always trust retrieval" design)
  diffusion_faithts: retrieval ENABLED with the learned adaptive gate (RetrievalWeighter) - OURS
Each variant exposes a uniform .forecast(ctx_n, nb_fut_n, w, tfeat, n_samples) -> (S,B,H) API so
scripts/evaluate.py does not need to special-case any of them.
"""
from dataclasses import dataclass
import torch
from .backbone import RetrievalConditionedScoreNet
from .diffusion import GaussianDiffusion
from ..retrieval.weighting import RetrievalWeighter


class DiffusionVariant:
    def __init__(self, L, H, k, d=64, depth=4, T=200, use_retrieval=True, adaptive_gate=True,
                 sparse_k=None, learn_tau=True):
        self.net = RetrievalConditionedScoreNet(H=H, L=L, d=d, depth=depth, k=k, use_retrieval=use_retrieval)
        self.diffusion = GaussianDiffusion(self.net, T=T)
        self.use_retrieval, self.adaptive_gate = use_retrieval, adaptive_gate
        self.weighter = RetrievalWeighter(sparse_k=sparse_k, learn_tau=learn_tau) if use_retrieval else None

    def parameters(self):
        ps = list(self.diffusion.parameters())
        if self.weighter is not None: ps += list(self.weighter.parameters())
        return ps

    def train(self): self.diffusion.train(); (self.weighter and self.weighter.train())
    def eval(self): self.diffusion.eval(); (self.weighter and self.weighter.eval())
    def to(self, device): self.diffusion.to(device); (self.weighter and self.weighter.to(device)); return self

    def weights(self, sim, ctx_n, nb_fut_n=None):
        if not self.use_retrieval:
            return torch.zeros(sim.shape[0], sim.shape[1], device=sim.device), torch.zeros(sim.shape[0], 1, device=sim.device)
        if self.adaptive_gate:
            return self.weighter(sim, ctx_n, nb_fut_n)
        # fixed-weight ablation (RATD-style): gate hard-wired to 1, no learned trust
        soft = torch.softmax(sim / 1.0, dim=1)
        return soft, torch.ones(sim.shape[0], 1, device=sim.device)

    def loss(self, ctx_n, fut_n, nb_fut_n, sim, tfeat, cf_drop=0.1):
        w, gate = self.weights(sim, ctx_n, nb_fut_n)
        return self.diffusion.loss(fut_n, ctx_n, nb_fut_n, w, tfeat, cf_drop=cf_drop), w, gate

    @torch.no_grad()
    def forecast(self, ctx_n, nb_fut_n, sim, tfeat, n_samples=1, steps=50, guidance_scale=0.0,
                 critic=None, critic_every=0, n_particles=1):
        w, gate = self.weights(sim, ctx_n, nb_fut_n)
        outs = []
        for _ in range(n_samples):
            outs.append(self.diffusion.sample(ctx_n, nb_fut_n, w, tfeat, steps=steps,
                                               guidance_scale=guidance_scale, critic=critic,
                                               critic_every=critic_every, n_particles=n_particles))
        return torch.stack(outs, 0), w, gate


def build_variant(name, L, H, k, **kw):
    presets = {
        "diffusion_plain":  dict(use_retrieval=False),
        "diffusion_fixed":  dict(use_retrieval=True, adaptive_gate=False),
        "diffusion_faithts": dict(use_retrieval=True, adaptive_gate=True, sparse_k=kw.pop("sparse_k", None)),
    }
    if name not in presets: raise KeyError(name)
    p = dict(presets[name]); p.update(kw)
    return DiffusionVariant(L, H, k, **p)
