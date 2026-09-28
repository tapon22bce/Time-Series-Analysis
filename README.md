# FaithTS: Faithful Retrieval-Grounded Diffusion Forecasting

Code accompanying a research programme built from four reviewer-facing requirements (see
`docs/POSITIONING.md` for how each maps onto prior work):

1. **Faithful retrieval-grounded forecasting and explanation** — retrieved neighbors condition the
   diffusion model through an *explicit, exposed* weight vector `w` (not a hidden attention block);
   the same `w` is handed to the explainer. `faithts/retrieval/weighting.py`, `faithts/models/backbone.py`.
2. **Faithfulness-by-intervention benchmark** — swap/remove the top-weighted neighbor and check that
   both the forecast and the explanation move consistently. `faithts/bench/faithfulness.py`.
3. **Critic-guided diffusion sampling** — a black-box (VLM/LLM/heuristic) plausibility scorer
   reweights samples at decode time via SMC-style particle resampling; it is never in the autograd
   graph. `faithts/models/diffusion.py` (`GaussianDiffusion.sample`), `faithts/guidance/critic.py`.
4. **Adaptive retrieval + bias-variance/conditional-shift analysis** — a learned gate decides how
   much to trust retrieval per query, using signals from the retrieved set itself (not just the
   query), and we measure how its behaviour compares to a fixed-trust ablation as the retrieval
   database is synthetically corrupted. `faithts/retrieval/weighting.py`, `faithts/theory/analysis.py`.

**Read `docs/RESULTS_SANDBOX.md` first.** This repository was built and unit-/smoke-tested inside a
1-CPU-core, no-GPU, 3 GB-RAM sandbox. Every number in that file was actually computed there (never
fabricated), but every run is small-scale and short (tens of epochs on subsampled data) and is a
correctness/plausibility demonstration, not a claim of SOTA. `docs/REPRODUCE.md` gives the
GPU-scale settings needed for publication-grade numbers.

## Layout
```
faithts/
  data/        loaders + windowing + leakage-safe splits + synthetic testbeds
  retrieval/   embedding, kNN database, explicit weighted conditioning (the adaptive gate)
  models/      score network, DDPM/DDIM core, point baselines, unified variant wrapper
  explain/     verifiable fact sheets, rule-based + Qwen-VLM explainers, chart renderer
  guidance/    critics used only at sampling time (heuristic / Qwen-text / Qwen-VLM)
  metrics/     point, probabilistic (CRPS/pinball), retrieval, explanation-accuracy metrics
  bench/       the faithfulness-by-intervention benchmark
  theory/      bias-variance / conditional-shift sweep + the linear-Gaussian surrogate
scripts/       train.py, evaluate.py, run_faithfulness_bench.py, run_bias_variance.py,
               aggregate_results.py, download_data.py
configs/       base.yaml, smoke.yaml (CPU sanity config), configs/datasets/*.yaml
docs/          ARCHITECTURE, BENCHMARK, THEORY, DATASETS, VLM, RESULTS_SANDBOX, REPRODUCE, POSITIONING
tests/         pytest suite (12 tests, all runnable on CPU, `python -m pytest tests/`)
results/       everything this sandbox actually produced (checkpoints + eval/benchmark json)
```

## Quickstart (CPU sandbox scale)
```bash
pip install -r requirements.txt
python scripts/download_data.py --names ETTh1 ETTh2 ETTm1        # scriptable datasets only
python scripts/train.py    --config configs/smoke.yaml --out results/smoke_faithts
python scripts/evaluate.py --config configs/smoke.yaml --ckpt_dir results/smoke_faithts
python scripts/run_faithfulness_bench.py --config configs/smoke.yaml --ckpt_dir results/smoke_faithts --out results/smoke_faithts/faithfulness
python -m pytest tests/ -q
```
For the actual demonstration runs referenced in `docs/RESULTS_SANDBOX.md`, use
`configs/demo_synth_clean.yaml`, `configs/demo_synth_robust.yaml`, `configs/etth1_demo.yaml`.

## VLM (Qwen) integration
The explainer, the sampling-time critic, and the chart renderer are fully coded
(`faithts/explain/explainer.py:QwenVLMExplainer`, `faithts/guidance/critic.py:QwenVLMCritic`), but
require a GPU and are **not instantiated in this CPU sandbox**. See `docs/VLM.md` for the exact
swap-in steps -- no other code changes are needed; both classes are duck-typed to match the
`RuleExplainer` / `HeuristicCritic` interfaces already used throughout.
