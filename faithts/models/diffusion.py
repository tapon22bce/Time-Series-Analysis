"""Minimal, well-tested Gaussian DDPM with cosine schedule + DDIM sampling + a training-free
critic-guided resampling hook (no gradients through the critic, so any black-box VLM/LLM works)."""
import math
import torch, torch.nn as nn, torch.nn.functional as F


def cosine_betas(T, s=0.008):
    t = torch.linspace(0, T, T + 1)
    f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    ac = f / f[0]
    betas = (1 - ac[1:] / ac[:-1]).clamp(max=0.999)
    return betas


class GaussianDiffusion(nn.Module):
    def __init__(self, net, T=200, clip_x0=8.0):
        super().__init__()
        self.net, self.T = net, T
        self.clip_x0 = clip_x0   # static thresholding (Ho et al. 2020 Sec 3.3): stabilises DDIM at
                                  # high-noise steps / few sampling steps / early (undertrained) nets.
                                  # Matches the [-10,10] clamp used when building normalised windows.
        betas = cosine_betas(T)
        alphas = 1 - betas
        ac = torch.cumprod(alphas, 0)
        ac_prev = F.pad(ac[:-1], (1, 0), value=1.0)
        for k, v in dict(betas=betas, alphas=alphas, ac=ac, ac_prev=ac_prev,
                          sqrt_ac=ac.sqrt(), sqrt_1m_ac=(1 - ac).sqrt()).items():
            self.register_buffer(k, v)

    def q_sample(self, x0, t, noise):
        return self.sqrt_ac[t][:, None] * x0 + self.sqrt_1m_ac[t][:, None] * noise

    def loss(self, x0, ctx_n, nb_fut_n, w, tfeat, cf_drop=0.1):
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        if cf_drop > 0:  # classifier-free-style dropout of the retrieval stream (both context/retrieval)
            drop = (torch.rand(B, device=x0.device) < cf_drop)[:, None]
            w = torch.where(drop, torch.zeros_like(w), w)
        eps_hat = self.net(x_t, t, ctx_n, nb_fut_n, w, tfeat)
        return F.mse_loss(eps_hat, noise)

    @torch.no_grad()
    def eps_to_x0(self, x_t, t, eps):
        x0 = (x_t - self.sqrt_1m_ac[t][:, None] * eps) / self.sqrt_ac[t][:, None]
        if self.clip_x0 is not None:
            x0 = x0.clamp(-self.clip_x0, self.clip_x0)
        return x0

    @torch.no_grad()
    def ddim_step(self, x_t, t, t_prev, eps, eta=0.0):
        x0 = self.eps_to_x0(x_t, t, eps)
        ac_t, ac_p = self.ac[t][:, None], (self.ac[t_prev][:, None] if t_prev[0] >= 0 else torch.ones_like(self.ac[t][:, None]))
        sigma = eta * ((1 - ac_p) / (1 - ac_t) * (1 - ac_t / ac_p)).clamp(min=0).sqrt()
        dir_xt = (1 - ac_p - sigma ** 2).clamp(min=0).sqrt() * eps
        x_prev = ac_p.sqrt() * x0 + dir_xt + sigma * torch.randn_like(x_t)
        return x_prev, x0

    @torch.no_grad()
    def sample(self, ctx_n, nb_fut_n, w, tfeat, steps=50, guidance_scale=0.0, eta=1.0,
               critic=None, critic_every=0, n_particles=1, critic_kwargs=None, H=None,
               critic_temp=1.0, return_all=False):
        """Sample forecasts for a batch of B queries.

        guidance_scale>0: classifier-free guidance on the retrieval stream (w vs 0).

        critic: callable(x0(N,H), ctx_n(N,L), **critic_kwargs) -> scores(N,), no_grad, black-box
        (a VLM or LLM plausibility scorer). If given, we run an SMC-style particle filter: expand
        each of the B queries into `n_particles` i.i.d. trajectories, and every `critic_every`
        steps resample particles within each query's group ~ softmax(critic_score/critic_temp).
        The critic NEVER enters the autograd graph and NEVER backprops into the score network;
        this is the training-free / decoding-time analogue of RLHF-style reward guidance.
        `eta`>0 keeps DDIM stochastic, which is required for resampling to have any effect."""
        B = ctx_n.shape[0]
        H = H or self.net.H
        device = ctx_n.device
        K = max(1, n_particles) if critic is not None else 1
        rep = lambda z: z.repeat_interleave(K, dim=0)
        ctx_r, nb_r, w_r, tf_r = rep(ctx_n), rep(nb_fut_n), rep(w), rep(tfeat)
        N = B * K
        group = torch.arange(B, device=device).repeat_interleave(K)   # which query each particle belongs to

        ts = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)
        ts = torch.cat([ts, torch.tensor([-1], device=device)])
        x_t = torch.randn(N, H, device=device)
        history = []
        for i in range(steps):
            t = ts[i].expand(N); t_prev = ts[i + 1].expand(N)
            eps_c = self.net(x_t, t, ctx_r, nb_r, w_r, tf_r)
            if guidance_scale > 0:
                eps_u = self.net(x_t, t, ctx_r, nb_r, torch.zeros_like(w_r), tf_r)
                eps = eps_u + guidance_scale * (eps_c - eps_u)
            else:
                eps = eps_c
            x_t, x0 = self.ddim_step(x_t, t, t_prev, eps, eta=eta)

            if critic is not None and critic_every > 0 and K > 1 and (i + 1) % critic_every == 0:
                scores = critic(x0, ctx_r, **(critic_kwargs or {}))        # (N,)
                new_x, new_hist = torch.empty_like(x_t), []
                for b in range(B):
                    sel = (group == b).nonzero(as_tuple=True)[0]
                    logits = (scores[sel] / max(critic_temp, 1e-4))
                    probs = torch.softmax(logits, 0)
                    picks = sel[torch.multinomial(probs, K, replacement=True)]
                    new_x[sel] = x_t[picks]
                    if return_all: new_hist.append((sel.tolist(), scores[sel].tolist(), probs.tolist()))
                x_t = new_x
                if return_all: history.append(dict(step=i, resample=new_hist))
        if critic is not None and K > 1:
            final_scores = critic(x_t, ctx_r, **(critic_kwargs or {}))
            best = torch.stack([sel[final_scores[sel].argmax()] for b in range(B)
                                 for sel in [(group == b).nonzero(as_tuple=True)[0]]])
            out = x_t[best]
        else:
            out = x_t[::K] if K > 1 else x_t
        return (out, history) if return_all else out
