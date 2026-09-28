# Swapping in the real Qwen VLM

Nothing in `faithts/models`, `faithts/retrieval`, `faithts/metrics`, or `faithts/bench` needs to
change. Three classes are fully coded against `transformers` but not instantiated in this CPU
sandbox (no GPU, and a 7B-parameter VLM does not fit in 3 GB RAM):

- `faithts.explain.explainer.QwenVLMExplainer` (default `Qwen/Qwen2.5-VL-7B-Instruct`)
- `faithts.guidance.critic.QwenVLMCritic` (image-based plausibility critic, same checkpoint)
- `faithts.guidance.critic.QwenTextCritic` (text-only, cheaper alternative for large
  `critic_every * n_particles` sweeps; default `Qwen/Qwen2.5-7B-Instruct`)

## Requirements
- A GPU with >=16 GB VRAM for bf16, or use 4/8-bit loading (`load_in_4bit=True` via `bitsandbytes`,
  a two-line change inside each class's `_lazy_load`).
- `pip install transformers>=4.49 accelerate` (already in `requirements.txt`; `qwen-vl-utils` is
  optional but recommended by Qwen's model card for more robust image preprocessing).

## Swapping the explainer
```python
from faithts.explain.explainer import QwenVLMExplainer
explainer = QwenVLMExplainer()   # instead of RuleExplainer()
```
Everywhere in this codebase an explainer is used generically as `explainer.explain(ctx_n, fut_n,
gate, w, neighbor_facts, neighbor_sim, ...)` and returns an `Explanation(facts, text,
neighbor_attrib)` -- `scripts/run_faithfulness_bench.py` and `faithts/bench/faithfulness.py` do not
need to change at all.

## Swapping the critic
```python
from faithts.guidance.critic import QwenVLMCritic
critic = QwenVLMCritic()
samples, w, gate = variant.forecast(ctx_n, nb_fut_n, sim, tfeat, n_samples=1, steps=50,
                                     critic=critic, critic_every=10, n_particles=4)
```
`critic_every` and `n_particles` trade compute for guidance strength: with a 7B VLM critic, expect
roughly `n_particles x (steps / critic_every)` forward passes through the VLM per forecast, so keep
`n_particles` small (4-8) and `critic_every` large (e.g. every 10th of 50 DDIM steps) for a first
GPU run, and profile before scaling up.

## Re-running the sandbox's VLM-shaped gaps on a GPU
1. `docs/RESULTS_SANDBOX.md` section 4's `explanation_consistency_*` was only checked against
   `RuleExplainer`. Re-run `scripts/run_faithfulness_bench.py` with `QwenVLMExplainer` swapped in
   (a small edit to that script's `explainer = RuleExplainer()` line) to check whether a *learned*
   explainer stays faithful under intervention -- this is the more interesting version of the test
   and is exactly what this repo could not run without a GPU.
2. Compare `HeuristicCritic` vs `QwenTextCritic` vs `QwenVLMCritic` guidance on the same checkpoint
   (`--critic` flag would need to be added to `scripts/evaluate.py`; currently only exercised via
   the smoke test in this repo's build log, not wired into the CLI -- a small, mechanical addition).
