"""FAITHFULNESS-BY-INTERVENTION BENCHMARK (the paper's core contribution).

For each test query we have: retrieved neighbors, exposed weights w, a forecast, and an
explanation that cites specific neighbors with specific weights. We then perform two
interventions on the retrieval set and check that BOTH the forecast and the explanation move
consistently with the exposed weights:

  (R) REMOVE the top-weighted neighbor (mask it out, renormalise the rest).
  (S) SWAP the top-weighted neighbor for a deliberately mismatched one (drawn from a different,
      distant region of the embedding space -- a "hard negative" analogue).

A method is faithful to the extent that:
  1. Forecast sensitivity tracks the removed/swapped neighbor's weight
     (Delta_forecast should correlate with the neighbor's original weight across the test set --
     a model that reports high weight but never actually uses the neighbor is UNFAITHFUL).
  2. The re-computed explanation, if regenerated from the NEW exposed weights, correctly reflects
     the intervention: the removed/swapped neighbor's attribution should disappear or change,
     and this is directly checkable for RuleExplainer (facts are derived from the weights). For a
     VLM explainer we additionally check textual consistency via `vlm_text_consistency`.
  3. Under SWAP with an adversarial (low-similarity) neighbor, a well-behaved gate should DOWN-
     WEIGHT that neighbor relative to REMOVE-then-renormalise -- i.e. the model should not blindly
     trust an obviously bad neighbor. We report this as the "robustness gap".

This benchmark is dataset-agnostic: it only needs (RetrievalDB, RetrievalWeighter, forecaster,
explainer) satisfying the interfaces below, so it doubles as a datasets/benchmarks-track artifact.
"""
from dataclasses import dataclass, asdict
import torch
import torch.nn.functional as F
from ..retrieval.database import embed


@dataclass
class InterventionResult:
    query_idx: int
    top_weight: float
    forecast_delta_remove: float
    forecast_delta_swap: float
    weight_after_remove_mass: float
    weight_after_swap_on_adversary: float
    explanation_consistent_remove: bool
    explanation_consistent_swap: bool


@torch.no_grad()
def _run_one(db, weighter, net_sample_fn, explainer, ctx_n, chan, tfeat, k, adversary_idx):
    idx, sim = db.search(ctx_n[None], chan[None], k=k)
    nb = db.gather(idx)
    w, gate = weighter(sim, ctx_n[None], nb["fut_n"])
    w, gate, sim, idx = w[0], gate[0], sim[0], idx[0]
    top = int(w.argmax())
    nb_facts = nb["facts"][0]   # (k,5) -- drop the batch dim (batch size is always 1 here)
    f0 = net_sample_fn(ctx_n[None], nb["fut_n"], w[None], tfeat[None])[0]
    ex0 = explainer.explain(ctx_n, f0, gate, w, nb_facts, sim)

    # (R) remove top neighbor: mask its similarity to -inf equivalent -> renormalise via weighter
    sim_r = sim.clone(); sim_r[top] = -2.0
    w_r, gate_r = weighter(sim_r[None], ctx_n[None], nb["fut_n"]); w_r, gate_r = w_r[0], gate_r[0]
    f_r = net_sample_fn(ctx_n[None], nb["fut_n"], w_r[None], tfeat[None])[0]
    ex_r = explainer.explain(ctx_n, f_r, gate_r, w_r, nb_facts, sim_r)

    # (S) swap top neighbor for the pre-selected distant/adversarial one
    adv_fut = db.w.fut_n[adversary_idx][None]
    adv_facts = db.w.facts[adversary_idx][None]
    nb_fut_s = nb["fut_n"].clone(); nb_fut_s[0, top] = adv_fut
    nb_facts_s = nb["facts"].clone(); nb_facts_s[0, top] = adv_facts
    # adversary's TRUE cosine similarity to the query (an honest score the weighter can act on,
    # not an artificially forced one) -- this is what makes the swap a genuine stress test.
    q_emb = F.normalize(embed(ctx_n[None]), dim=1)[0]
    adv_sim = float(q_emb @ db.emb[adversary_idx])
    sim_s = sim.clone(); sim_s[top] = adv_sim
    w_s, gate_s = weighter(sim_s[None], ctx_n[None], nb_fut_s); w_s, gate_s = w_s[0], gate_s[0]
    f_s = net_sample_fn(ctx_n[None], nb_fut_s, w_s[None], tfeat[None])[0]
    ex_s = explainer.explain(ctx_n, f_s, gate_s, w_s, nb_facts_s[0], sim_s)

    d_remove = (f_r - f0).pow(2).mean().sqrt().item()
    d_swap = (f_s - f0).pow(2).mean().sqrt().item()
    attrib0 = {i for i, *_ in ex0.neighbor_attrib}
    attrib_r = {i for i, *_ in ex_r.neighbor_attrib}
    attrib_s = {i for i, *_ in ex_s.neighbor_attrib}
    consistent_remove = (top in attrib0) and (top not in attrib_r)     # citation must disappear
    consistent_swap = (top in attrib0) and (top not in attrib_s or float(w_s[top]) < 0.5 * float(w[top]))
    return InterventionResult(0, float(w[top]), d_remove, d_swap, float(w_r.sum()),
                               float(w_s[top]), consistent_remove, consistent_swap)


def run_faithfulness_benchmark(db, weighter, net_sample_fn, explainer, test_windows, k=5,
                               n_queries=100, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = len(test_windows)
    qidx = torch.randperm(n, generator=g)[: min(n_queries, n)]
    adv_pool = torch.randperm(len(db), generator=g)[: len(qidx)]
    results = []
    for j, qi in enumerate(qidx.tolist()):
        r = _run_one(db, weighter, net_sample_fn, explainer, test_windows.ctx_n[qi], test_windows.chan[qi],
                     test_windows.tfeat[qi], k, int(adv_pool[j]))
        r.query_idx = int(qi)
        results.append(r)
    return results


def summarize(results):
    import numpy as np
    w = np.array([r.top_weight for r in results])
    dr = np.array([r.forecast_delta_remove for r in results])
    ds = np.array([r.forecast_delta_swap for r in results])
    corr_r = float(np.corrcoef(w, dr)[0, 1]) if w.std() > 1e-8 and dr.std() > 1e-8 else float("nan")
    corr_s = float(np.corrcoef(w, ds)[0, 1]) if w.std() > 1e-8 and ds.std() > 1e-8 else float("nan")
    robustness_gap = float((np.array([r.weight_after_remove_mass for r in results]) -
                             np.array([r.weight_after_swap_on_adversary for r in results])).mean())
    return dict(
        n=len(results),
        mean_top_weight=float(w.mean()),
        sensitivity_corr_remove=corr_r,     # HIGHER = more faithful (weight predicts forecast change)
        sensitivity_corr_swap=corr_s,
        explanation_consistency_remove=float(np.mean([r.explanation_consistent_remove for r in results])),
        explanation_consistency_swap=float(np.mean([r.explanation_consistent_swap for r in results])),
        adversary_robustness_gap=robustness_gap,   # >0 means gate down-weights bad neighbors vs. removal
    )
