# Positioning against prior art

(Context: this codebase exists because a preceding review of a "RAG + Diffusion + VLM for time
series" concept found the three-way combination alone insufficiently novel for an A*
venue -- Time-VLM already combines retrieval + vision + text for forecasting; RATD is
retrieval+diffusion; LDM4TS is vision+diffusion+text. The four contributions this repo implements
were chosen specifically to differ from each of these along a mechanism, not just a
combination-of-modalities, axis.)

| prior work | what it does | what FaithTS adds beyond it |
|---|---|---|
| RATD (retrieval-augmented diffusion, NeurIPS) | retrieved references guide denoising via an attention mechanism (RMA) | weights are an explicit, externally-exposed tensor (not internal to an attention block), a learned per-query trust GATE (not a fixed attention softmax), and a faithfulness-by-intervention benchmark + bias-variance/conditional-shift analysis under database corruption -- none of which RATD reports |
| TS-RAG | retrieval blended into frozen foundation models | no diffusion, no adaptive gate, no faithfulness testing |
| Time-VLM | retrieval + vision + text jointly, for forecasting accuracy | no explicit exposed weights, no intervention-based faithfulness check, no critic-guided sampling |
| LDM4TS | vision-conditioned latent diffusion | no retrieval component, no explainability angle |
| MAS4TS / VLM4TS | VLM reads rendered plots for anchors/anomaly verification | no retrieval-diffusion coupling; used here as the inspiration for the critic/explainer's chart-reading design, not a competitor on the same axis |

## What is genuinely new here
1. The **explicit-weight faithfulness mechanism + intervention benchmark** (contribution 1): to our
   knowledge no retrieval-augmented time-series diffusion paper tests whether its conditioning
   weights are causally load-bearing, as opposed to just reporting that retrieval improves an
   aggregate metric.
2. **Training-free, particle-filter critic guidance with a VLM/LLM reward** (contribution 2): this
   is the correct way to use a large multimodal model as a decoding-time plausibility check without
   the (impractical, as the earlier review noted) claim of differentiating through the VLM.
3. **A gate that must use retrieved-SET statistics, not just the query, to detect misleading
   neighbors, trained with corruption augmentation, and evaluated with a bias-variance sweep against
   a fixed-weight ablation** (contribution 3): this is the piece RATD-style papers do not report,
   and the sandbox run in `docs/RESULTS_SANDBOX.md` shows it behaving as the linear-Gaussian
   surrogate predicts (smaller MSE degradation under corruption than a fixed-trust policy).
4. **A reusable, dataset-agnostic benchmark artifact** (contribution 4): `faithts/bench/faithfulness.py`
   only depends on four duck-typed interfaces and could be dropped into a different retrieval-
   diffusion codebase with minimal glue, which is the bar for a datasets/benchmarks-track
   submission rather than a single paper's internal ablation.

## Remaining novelty risk to flag honestly
- The score-network architecture itself (FiLM-conditioned residual MLP blocks) is not novel; the
  contribution is the faithfulness/adaptivity/evaluation machinery around a fairly standard
  conditional-diffusion backbone. A reviewer could reasonably ask for the mechanism to be tested
  with a stronger backbone (e.g. a Transformer score network) to confirm the findings are not an
  artifact of a weak backbone -- `docs/REPRODUCE.md` flags this.
- All quantitative evidence in this repo is small-scale (see `docs/RESULTS_SANDBOX.md`). The
  qualitative direction of every result (adaptive >= fixed under corruption, retrieval-conditioning
  helps when neighbors are genuinely informative and training is sufficient) is consistent with the
  theoretical prediction, which is the strongest claim that should be made before GPU-scale
  reproduction.
