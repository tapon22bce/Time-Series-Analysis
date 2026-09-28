"""Explanation-quality metrics scored against the VERIFIABLE fact sheet (faithts.explain.facts),
not against free-text references (no ROUGE/BLEU: they do not correlate with factual correctness,
see docs/BENCHMARK.md). Works identically for RuleExplainer and QwenVLMExplainer output."""
import torch
from ..explain.facts import FACT_NAMES


def fact_accuracy(pred_facts: torch.Tensor, true_facts: torch.Tensor):
    """pred_facts/true_facts: (B,5) long, -1 = unparseable/abstained -> counted wrong."""
    valid = (pred_facts >= 0)
    correct = (pred_facts == true_facts) & valid
    per_fact = {n: correct[:, i].float().mean().item() for i, n in enumerate(FACT_NAMES)}
    per_fact["overall"] = correct.float().mean().item()
    per_fact["coverage"] = valid.float().mean().item()   # fraction of facts the explainer dared to state
    return per_fact
