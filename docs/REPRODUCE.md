# Reproducing at publication scale

Settings actually used in this CPU sandbox (see `configs/demo_synth_clean.yaml`,
`configs/demo_synth_robust.yaml`, `configs/etth1_demo.yaml`) are far below what a paper needs.
Suggested GPU-scale settings, keeping the same config schema:

| field | sandbox | suggested for a paper |
|---|---|---|
| `d_model` | 24-64 | 256-512 |
| `depth` | 2-3 | 6-8, or swap `RetrievalConditionedScoreNet`'s FiLM-MLP blocks for a small Transformer |
| `T` (diffusion steps) | 50-100 | 200-1000 (match TimeGrad/CSDI/TSDiff conventions per dataset) |
| `sample_steps` | 8-20 | 50-100 DDIM steps, or switch to DPM-Solver++ for fewer steps at similar quality |
| `epochs` | 3-40 | until validation CRPS plateaus (typically 100-300 for ETT-scale data) |
| `batch_size` | 64-128 | 256-512 if VRAM allows |
| `max_train`/`max_eval` | 1.5k-8k | null (use the full split) |
| seeds | 1 | >=3, report mean +/- std |

## Baselines to add before submission
- Foundation models zero/few-shot: Chronos, TimesFM, Moirai (all have public weights; wrap them
  behind `faithts/models/baselines.py`'s `build_baseline` interface -- `forward(ctx_n, tfeat=None)`).
- Pure diffusion time-series forecasters without retrieval: CSDI, TimeGrad, TSDiff (useful to
  separate "does diffusion help" from "does retrieval help").
- RATD itself, TS-RAG, Time-VLM, LDM4TS, MAS4TS, VLM4TS (the closest prior art; see
  `docs/POSITIONING.md`) -- at minimum cite and ideally re-run their released code on the same
  splits.
- An LLM given the numbers as text (not an image) as a control for the VLM explainer/critic, to
  isolate whether the VISUAL modality is doing any work beyond what a text-only model could do from
  the same numbers.

## Statistical rigour
- Every table in `docs/RESULTS_SANDBOX.md` is a single seed, single run. Re-run with >=3 seeds and
  report confidence intervals (bootstrap over the test set is simplest, given `faithts/metrics/`
  already operates on batched tensors).
- The bias-variance sweep (`scripts/run_bias_variance.py`) used `n_repeats=5, n_queries=40`; a paper
  needs both increased substantially (n_repeats >= 20 to get a stable variance estimate; n_queries
  as large as compute allows) and a significance test on the adaptive-vs-fixed MSE gap at each
  corruption level (e.g. a paired bootstrap over queries, since both variants are evaluated on the
  same queries).
- The faithfulness benchmark's `n_queries=150` should also be increased, and
  `sensitivity_corr_remove/_swap` should be re-run on a real, heterogeneous-neighbor dataset (ETTh1)
  in addition to the synthetic testbed, per the note in `docs/RESULTS_SANDBOX.md` section 4.

## Full dataset coverage
`docs/DATASETS.md` lists exactly which of the eleven requested sources are verified, which are
written-from-spec but unverified (need the actual file placed and a quick shape/column sanity
check), and which have no confirmed public bulk-download schema and need a manual CSV export.
Complete the "manual"/"unverified" rows there before running the full eleven-dataset sweep.
