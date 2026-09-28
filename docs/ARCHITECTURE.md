# Architecture

## Data flow (one training/inference step)
```
raw dataset (csv/npz) --loaders.py--> TSData (global z-score, train-only stats)
                        --windows.py--> Windows(ctx_n, fut_n, chan, tfeat, facts, time_ns)
train split -----------> RetrievalDB (embeddings + payload; TRAIN ONLY, see leakage note below)

query window --embed()--> cosine kNN against RetrievalDB --> (idx, sim)  [faithts/retrieval/database.py]
(idx, sim) + ctx_n + nb_fut_n --RetrievalWeighter--> (w, gate)            [faithts/retrieval/weighting.py]
(x_t, t, ctx_n, nb_fut_n, w, tfeat) --RetrievalConditionedScoreNet--> eps_hat  [faithts/models/backbone.py]
eps_hat --GaussianDiffusion (DDPM/DDIM)--> forecast sample(s)             [faithts/models/diffusion.py]
forecast + w + gate + neighbor facts --Explainer--> Explanation(text, facts, neighbor_attrib)
```

## Why the conditioning is split into three explicit streams (not one attention block)
`RetrievalConditionedScoreNet.forward` builds three separate conditioning vectors -- context,
retrieval, forecast-horizon time features -- and only THEN mixes them with a small linear layer
before FiLM-modulating the residual blocks. The retrieval vector is built as
```python
mixed = torch.einsum("bk,bkd->bd", w, value(neighbor_futures))   # w is RetrievalWeighter's output
```
i.e. a literal weighted sum using the SAME `w` that the explainer reports. This is the mechanism
required by the faithfulness contribution: because `w` is a first-class, externally-visible tensor
(not folded into an opaque multi-head attention weight matrix), we can (a) zero it to get an exact
"no-retrieval" counterfactual for classifier-free guidance and for `diffusion_plain`, (b) mask a
single neighbor's contribution and renormalise for the intervention benchmark, and (c) hand the
identical numbers to `RuleExplainer`/`QwenVLMExplainer` so a claim like "neighbor #2, weight 0.41"
is checkable against the actual forward pass, not a post-hoc rationalisation.

## The adaptive gate needs to see the retrieved SET, not just the query
An earlier version of `RetrievalWeighter` predicted the gate from context features alone. Trained
this way, the gate cannot learn a real "distrust this neighborhood" policy, because the context is
identical regardless of which (possibly corrupted) neighbors were retrieved -- there is no training
signal linking context to retrieval quality. `RetrievalWeighter.set_feats` adds retrieved-set
statistics (mean/min/std similarity, and cross-neighbor future disagreement) as additional gate
inputs, and `scripts/train.py`'s `corrupt_aug_frac` injects occasional bad neighbors *at their
original (high) similarity score* during training, so the gate is forced to use disagreement,
not similarity, to detect them. This is documented as a design correction in the commit history
because it materially changes the empirical results (see `docs/RESULTS_SANDBOX.md`).

## Diffusion core
Standard cosine-schedule Gaussian DDPM (`GaussianDiffusion`) with DDIM sampling. Two things are not
textbook-default and are load-bearing:
- **Static thresholding of the x0 estimate** (`clip_x0`, default ±8 in context-normalised units).
  Without it, few-step DDIM with a small/undertrained score network diverges (we hit this in
  practice, see git history / the debugging note in this repo's build log): early high-noise steps
  produce enormous x0 estimates via division by a near-zero `sqrt(ac_t)`, which then compound
  through subsequent steps. This is standard practice (Ho et al. 2020 sec. 3.3; every major DDPM
  codebase enables an equivalent option) but is easy to omit and easy to miss until you look at
  actual numbers, which is why it's called out here.
- **Particle-filter critic guidance** (`critic_every`, `n_particles` in `sample()`): expands each
  query into `n_particles` i.i.d. DDIM trajectories and resamples them within-query every
  `critic_every` steps, proportional to `softmax(critic(x0)/temp)`. The critic is called under
  `torch.no_grad()` and never differentiated -- this is the "no end-to-end backprop through the
  VLM" requirement, implemented as SMC/Feynman-Kac resampling rather than reward backprop.

## Model variants (`faithts/models/variants.py`)
| name | retrieval | gate | purpose |
|---|---|---|---|
| `diffusion_plain` | off | n/a | unconditional diffusion; isolates what retrieval adds |
| `diffusion_fixed` | on | hard-wired to 1, plain softmax(sim) weights | RATD-style "always trust" ablation |
| `diffusion_faithts` | on | learned, per-query | ours |

All three share one `GaussianDiffusion`/`RetrievalConditionedScoreNet` implementation so
differences in the results table are attributable to the gating policy, not to incidental
architecture changes between an "ours" and a "baseline" codepath.
