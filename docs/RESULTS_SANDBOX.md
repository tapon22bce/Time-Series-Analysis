# Results actually produced in this sandbox -- scope and honest numbers

**Environment**: 1 CPU core, 3 GB RAM, no GPU (verified via `nproc`/`free`/`nvidia-smi` at build
time). Every number below was computed by running the code in this repository in that environment;
none are placeholders or literature numbers dressed up as ours. Every run uses tiny models
(`d_model` 24-64, depth 2-3), short training (15-40 epochs), and subsampled data (hundreds to a few
thousand windows). **This is a plumbing-and-plausibility check, not a publication-scale result.**
`docs/REPRODUCE.md` lists the settings a real submission needs (bigger models, full data, more
epochs, a GPU, several seeds, significance testing) -- none of that was run here.

## 1. Does retrieval-conditioning help at all, when the database is trustworthy?
Dataset: `retrievable_synth` (`configs/demo_synth_clean.yaml`; see `faithts/data/synthetic.py` for
why this controlled testbed exists and `tests/test_data.py::test_pipeline_retrieval_recovers_template`
for the oracle check confirming retrieval genuinely identifies the right template 100% of the time
before any model training happens).

| model | MAE | MSE |
|---|---|---|
| naive_last | 1.209 | 2.568 |
| seasonal_naive | 1.250 | 2.432 |
| ridge_linear | 0.598 | 0.595 |
| dlinear | 0.598 | 0.596 |
| patch_transformer (no retrieval) | 0.332 | 0.186 |
| **diffusion_plain** (no retrieval) | 0.366 | 0.223 |
| **diffusion_fixed** (retrieval, gate=1) | 0.349 | 0.202 |
| **diffusion_faithts** (retrieval, learned gate, mean_gate=0.55) | 0.349 | 0.202 |

Retrieval-conditioned diffusion (fixed or adaptive) beats unconditional diffusion by ~9% MSE here,
and the adaptive gate matches the always-trust ablation almost exactly (0.2024 vs 0.2019) while
using only ~55% average trust -- i.e. no accuracy is given up for the added flexibility, on this
controlled testbed. `patch_transformer`, a strong parametric point-forecaster, is competitive with
retrieval-conditioned diffusion here; with only K=40 fixed templates and ~4200 training windows, a
sufficiently expressive network can partly memorise the template-to-continuation mapping directly.
This is an expected and informative limit case, not a failure: retrieval's comparative advantage
should be largest exactly where parametric memorisation is hardest (many/rare/evolving patterns,
limited capacity, or distribution shift), which is what section 3 tests directly.

## 2. Does it hold on real data? (ETTh1, small scale)
`configs/etth1_demo.yaml`: 15 epochs, 8000/800/800 train/val/test windows (real ETTh1 series, not
synthetic), `d_model=64`.

| model | MAE | MSE |
|---|---|---|
| naive_last | 1.040 | 2.005 |
| seasonal_naive | 0.750 | 1.225 |
| dlinear | 0.670 | 0.936 |
| patch_transformer | 0.693 | 0.964 |
| diffusion_plain | 0.884 | 1.382 |
| diffusion_fixed | 0.922 | 1.488 |
| diffusion_faithts | 0.971 | 1.631 |

**Honest negative result**: at this tiny scale, diffusion trails the strong linear/Transformer point
baselines by a wide margin (well documented for diffusion time-series forecasters more broadly --
they typically need substantially more training than point forecasters to reach comparable MSE),
and retrieval-conditioning does not help here -- it is mildly *harmful*, with `diffusion_plain <
diffusion_fixed < diffusion_faithts` in error. Retrieval quality itself is decent in isolation
(`NDCG@5 = 0.786`, computed independently of which diffusion variant is scored), so the gap is most
plausibly a training-budget problem (15 epochs is not enough for the conditioning pathway to be
learned well) rather than a retrieval-quality problem, but this sandbox cannot rule out other
explanations, and does not claim to. Do not read this as "retrieval doesn't work on real data" --
read it as "15 epochs on a CPU is not enough to show whether it does," and re-run with
`docs/REPRODUCE.md`'s settings before drawing conclusions for a paper.

## 3. Robustness under retrieval-database corruption (bias-variance / conditional-shift)
`configs/demo_synth_robust.yaml` (retrievable_synth, trained with `corrupt_aug_frac=0.35` so the
gate has a training signal for "distrust disagreeing neighbors" -- see `docs/ARCHITECTURE.md`),
evaluated with `scripts/run_bias_variance.py` sweeping TEST-TIME database corruption fraction from
0.0 to 1.0 (a fraction of stored futures replaced with `future + shift(2.5) + noise`, at their
original -- now misleading -- similarity scores), 5 repeats x 40 queries per point:

| corruption frac | adaptive MSE | fixed (RATD-style) MSE |
|---|---|---|
| 0.00 | 0.322 | 0.320 |
| 0.15 | 0.339 | 0.349 |
| 0.30 | 0.339 | 0.335 |
| 0.50 | 0.363 | 0.338 |
| 0.70 | 0.344 | 0.359 |
| 1.00 | 0.364 | 0.393 |

Relative MSE degradation from 0% to 100% corruption: **adaptive +13.0% vs fixed +22.5%**. The
adaptive gate gives up ~0.6% accuracy when the database is clean and recovers roughly 40% of the
degradation the fixed-weight ablation suffers when the database is maximally corrupted. This is the
core empirical claim of the "adaptive retrieval" contribution, produced honestly at small scale;
it should be re-run with more repeats/queries and a significance test before being reported as a
paper result (n=40 queries x 5 repeats is a plausibility check, not a well-powered experiment).

## 4. Faithfulness-by-intervention benchmark
Run on the same robust-trained `diffusion_faithts` checkpoint, 150 test queries, `RuleExplainer`
(deterministic; `QwenVLMExplainer` requires a GPU and was not run here, see `docs/VLM.md`):

| metric | value |
|---|---|
| mean_top_weight | 0.167 |
| sensitivity_corr_remove | -0.151 |
| sensitivity_corr_swap | -0.176 |
| explanation_consistency_remove | 1.000 |
| explanation_consistency_swap | 1.000 |
| adversary_robustness_gap | 0.485 |

`explanation_consistency_*` is 1.0 by construction for `RuleExplainer` (it derives its text
directly from the exposed weights, so a removed citation always disappears from the regenerated
text) -- this checks the *plumbing*, not whether a learned/VLM explainer would stay faithful, which
is the more interesting question and needs `docs/VLM.md`'s GPU setup. The negative
`sensitivity_corr_*` values are a genuine, if unflattering, finding on this specific dataset: because
`retrievable_synth`'s top-k neighbors are close substitutes for one another (same template,
independent noise), removing any single one barely moves the forecast regardless of its reported
weight, so weight-magnitude does not predict forecast-sensitivity here. `adversary_robustness_gap`
(0.485 > 0) does show the intended qualitative effect: an honestly-scored adversarial replacement
ends up trusted far less than simply removing a neighbor and renormalising over the rest. Testing
the sensitivity-correlation metric on a real, heterogeneous-neighbor dataset (ETTh1) is listed in
`docs/REPRODUCE.md` as the natural next step; it was not completed in this pass due to sandbox time
budget.
