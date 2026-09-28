import torch
from faithts.models.backbone import RetrievalConditionedScoreNet
from faithts.models.diffusion import GaussianDiffusion
from faithts.retrieval.weighting import RetrievalWeighter
from faithts.models.variants import build_variant
from faithts.models.baselines import build_baseline


def test_score_net_shapes():
    net = RetrievalConditionedScoreNet(H=24, L=96, d=16, depth=2, k=5)
    out = net(torch.randn(4, 24), torch.randint(0, 50, (4,)), torch.randn(4, 96),
              torch.randn(4, 5, 24), torch.rand(4, 5), torch.rand(4, 3))
    assert out.shape == (4, 24)


def test_diffusion_zero_weight_disables_retrieval():
    net = RetrievalConditionedScoreNet(H=24, L=96, d=16, depth=2, k=5)
    diff = GaussianDiffusion(net, T=50)
    ctx = torch.randn(2, 96); nb = torch.randn(2, 5, 24); tf = torch.rand(2, 3)
    w_on = torch.rand(2, 5); w_on = w_on / w_on.sum(1, keepdim=True)
    torch.manual_seed(0)
    with torch.no_grad():
        s1 = diff.sample(ctx, nb, w_on, tf, steps=5)
    torch.manual_seed(0)
    with torch.no_grad():
        s2 = diff.sample(ctx, nb, torch.zeros_like(w_on), tf, steps=5)
    assert not torch.allclose(s1, s2), "retrieval on/off should give different samples"


def test_weighter_sums_to_gate():
    w = RetrievalWeighter(sparse_k=3)
    sim = torch.rand(8, 5)
    ctx = torch.randn(8, 96); nbf = torch.randn(8, 5, 24)
    weights, gate = w(sim, ctx, nbf)
    assert torch.allclose(weights.sum(1, keepdim=True), gate, atol=1e-5)
    assert (weights >= 0).all() and (gate >= 0).all() and (gate <= 1).all()


def test_variants_have_expected_gate_behavior():
    ctx = torch.randn(3, 96); nbf = torch.randn(3, 5, 24); tf = torch.rand(3, 3)
    sim = torch.rand(3, 5) * 0.5 + 0.5
    for name, expect in [("diffusion_plain", 0.0), ("diffusion_fixed", 1.0)]:
        v = build_variant(name, 96, 24, k=5, d=8, depth=1, T=20)
        w, gate = v.weights(sim, ctx, nbf)
        assert torch.allclose(gate, torch.full_like(gate, expect), atol=1e-4)


def test_baselines_run():
    ctx = torch.randn(4, 96); tf = torch.randn(4, 3)
    for name in ["naive_last", "seasonal_naive", "ridge_linear", "dlinear", "patch_transformer"]:
        m = build_baseline(name, 96, 24)
        out = m(ctx, tfeat=tf)
        assert out.shape == (4, 24)
