# The Faithfulness-by-Intervention Benchmark

`faithts/bench/faithfulness.py`, driven by `scripts/run_faithfulness_bench.py`.

## Motivation
A retrieval-augmented forecaster can *claim* to rely on a retrieved neighbor (e.g. via an attention
weight or a natural-language citation) without the forecast actually depending on it. Conversely, an
explainer can state a plausible-sounding reason that has nothing to do with what the numerical model
did. Existing RAG-for-time-series work (RATD, TS-RAG) reports retrieval-quality metrics (NDCG,
recall) and forecast-quality metrics (MSE, CRPS) but not whether the two are causally linked, or
whether an attached explanation is faithful to the mechanism that produced the forecast. That gap is
what this benchmark targets, and it is dataset-agnostic (works for any (RetrievalDB,
RetrievalWeighter, forecaster, explainer) satisfying the interfaces in `faithts/bench/faithfulness.py`),
so it is intended as a reusable artifact, not just an evaluation script for this one paper.

## Protocol
For each test query:
1. Retrieve `k` neighbors, compute the exposed weight vector `w` (summing to the learned gate) and
   the resulting forecast `f0` and explanation `ex0` (which cites specific neighbor indices with
   specific weights -- see `RuleExplainer`/`QwenVLMExplainer`).
2. **REMOVE** intervention: mask the top-weighted neighbor's similarity to -2 (i.e. remove it from
   consideration), recompute `w_r`/`gate_r`, forecast `f_r`, explanation `ex_r`.
3. **SWAP** intervention: replace the top-weighted neighbor's *future* with a randomly-drawn other
   database entry's future, but keep its similarity to the query HONEST (recompute the true cosine
   similarity of the query to that random entry's embedding -- typically low, an adversarial/hard
   negative). Recompute `w_s`/`gate_s`, forecast `f_s`, explanation `ex_s`.

## Reported metrics (`summarize()`)
- **`sensitivity_corr_remove` / `_swap`**: Pearson correlation, across the test set, between the
  removed/swapped neighbor's *original* weight and the resulting forecast RMS change. A faithful
  model should show this correlation clearly positive (high-weight neighbors matter more when
  perturbed); near-zero or negative values mean the exposed weight does not predict actual model
  behaviour -- i.e. the weight is decorative. **Caveat we found empirically**: on datasets where
  top-k neighbors are close substitutes of one another (our `retrievable_synth` testbed by
  construction), this correlation is legitimately weak even for a well-behaved model, because
  removing any one of several near-duplicate neighbors barely changes the forecast. The metric is
  most discriminating on datasets with heterogeneous, non-substitutable neighbors (e.g. real ETT
  data); see `docs/RESULTS_SANDBOX.md`.
- **`explanation_consistency_remove` / `_swap`**: for `RuleExplainer`, whether the removed/swapped
  neighbor's citation disappears from the regenerated explanation (checkable by construction, since
  the rule explainer derives its text from the same `w`). For `QwenVLMExplainer`, the analogous
  check is whether the VLM's stated `neighbor_reliance` changes; this requires a GPU run (see
  `docs/VLM.md`) and is not exercised in this CPU sandbox.
- **`adversary_robustness_gap`**: how much less an adversarial (low-similarity) replacement is
  trusted, compared to simply removing the neighbor and renormalising over the rest. Positive values
  indicate the gate treats "an obviously bad match" differently from "one fewer (still-good)
  match", which is the behaviour you want from an adaptive (vs. fixed-softmax) weighting scheme.

## What would make this a stronger "datasets & benchmarks" submission
- Human-verified relevance judgments on a real dataset (currently relevance for the retrieval
  metrics, `faithts/metrics/retrieval.py`, is defined automatically from fact-sheet agreement --
  reproducible and leakage-free, but not human-validated).
- A second explainer backbone (beyond Qwen) to check the benchmark's conclusions are not
  VLM-specific.
- Confidence intervals via bootstrap over `n_queries` (currently point estimates only).
