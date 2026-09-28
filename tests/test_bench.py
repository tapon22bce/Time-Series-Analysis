import torch
from faithts.explain.facts import extract_facts, facts_to_dict, dict_to_facts
from faithts.metrics.explain import fact_accuracy
from faithts.metrics.retrieval import relevance, ndcg_at_k
from faithts.theory.analysis import theoretical_curve


def test_facts_roundtrip():
    ctx = torch.randn(5, 96); fut = torch.randn(5, 24)
    f = extract_facts(ctx, fut)
    for i in range(5):
        d = facts_to_dict(f[i])
        back = dict_to_facts(d)
        assert torch.equal(back, f[i])


def test_fact_accuracy_perfect_when_equal():
    f = torch.randint(0, 3, (10, 5))
    acc = fact_accuracy(f, f)
    assert acc["overall"] == 1.0
    assert acc["coverage"] == 1.0


def test_ndcg_perfect_ranking():
    rel = torch.tensor([[1., 1., 0., 0.]])
    assert abs(ndcg_at_k(rel, 4) - 1.0) < 1e-6


def test_theoretical_curve_shrinks_with_corruption():
    curve = theoretical_curve([0.0, 0.3, 0.6])
    ws = [c["W_star"] for c in curve]
    assert ws[0] >= ws[1] >= ws[2], "optimal retrieval trust should shrink as corruption grows"
