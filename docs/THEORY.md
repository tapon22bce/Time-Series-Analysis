# Bias-variance / conditional-shift analysis

`faithts/theory/analysis.py`, driven by `scripts/run_bias_variance.py`.

## The gap this fills
RATD-style retrieval-conditioned diffusion papers report that retrieval helps on average, but do
not analyse what happens when the retrieval database is *partially wrong* (a realistic scenario:
mislabeled logs, a database built from a different regime, adversarial or out-of-distribution
queries). This is exactly the situation an ADAPTIVE trust mechanism is supposed to handle better
than a fixed-weight one, so we built the sweep needed to check that claim rather than asserting it.

## Linear-Gaussian surrogate (a toy model to build intuition, not a claim about real diffusion)
Model the forecast as
```
yhat = f(ctx) + W * (r - f(ctx)) + noise,   r = retrieval signal, W in [0, 1] the retrieval trust
```
If a fraction `p` of the retrieval database is corrupted with mean shift `s` (in context-normalised
units), then approximately
```
Bias(W)^2 ≈ (p * s * W)^2                      (grows with trust AND corruption)
Var(W)    ≈ base_var / (1 + (k-1) * W)          (shrinks with trust, via neighbor averaging)
MSE(W) = Bias(W)^2 + Var(W)
```
`theoretical_curve()` grid-searches `W* = argmin_W MSE(W)` for a range of `p`. The qualitative
prediction we test empirically: **W\* should shrink monotonically as corruption `p` grows** --- e.g.
for `k=5`, `s=2.0`: `W*(p=0)=1.00 -> W*(p=0.2)=0.76 -> W*(p=0.4)=0.43 -> W*(p=0.6)=0.29`.

## Empirical protocol (`bias_variance_sweep`)
For a grid of corruption fractions, build a corrupted copy of the retrieval database
(`corrupt_database`: replace a fraction of stored futures with `future + shift + noise`, keeping
their original -- now misleading -- similarity scores), then for each of `n_queries` test windows
draw `n_repeats` independent forecast samples and decompose the resulting `(n_repeats, n_queries,
H)` tensor into `Bias^2 = mean((mean_over_repeats - true)^2)` and `Variance = var_over_repeats`,
compared between the **adaptive** gate and a **fixed** (gate hard-wired to 1, plain softmax weights
-- the RATD-style ablation) variant sharing the same trained score network family.

## What we actually found in this sandbox (`docs/RESULTS_SANDBOX.md` has the full table)
On the `retrievable_synth` testbed, trained with `corrupt_aug_frac=0.35` (training-time exposure to
occasional bad neighbors -- see `docs/ARCHITECTURE.md`'s note on why the gate needs this to have any
training signal at all), sweeping test-time database corruption from 0% to 100%:
- **fixed**: MSE degrades from 0.320 to 0.393 (**+22.5%** relative)
- **adaptive**: MSE degrades from 0.322 to 0.364 (**+13.0%** relative)

The adaptive gate is *slightly worse* at zero corruption (0.322 vs 0.320, ~0.6%) and *clearly
better* under heavy corruption, consistent with the theoretical prediction that trust should shrink
as corruption grows -- even though the reported *population-mean* gate barely moves (0.50 -> 0.50)
across the sweep. That last point is worth stating precisely: the mean gate is not doing the
adapting; the *per-query* gate must be (some queries get pushed toward 0, others toward 1, and the
average stays flat), which we did not additionally verify by looking at the per-query gate
distribution in this pass -- a natural follow-up (`gate.std()` broken out by whether that specific
query's retrieved neighbor was corrupted) that the code supports but this sandbox run did not
produce as a separate artifact.
