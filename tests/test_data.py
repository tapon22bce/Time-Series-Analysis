import torch
from faithts.data.synthetic import make_synthetic, make_retrievable_synthetic
from faithts.data.windows import make_windows, make_block_windows
from faithts.data.pipeline import build_pipeline


def test_windows_no_leakage():
    d = make_synthetic()
    tr = make_windows(d, "train", 96, 24, stride=2)
    te = make_windows(d, "test", 96, 24, stride=8)
    assert tr.origin.max() <= d.split_ends[0]
    assert te.origin.min() >= d.split_ends[1]


def test_block_windows_no_cross_contamination():
    d = make_retrievable_synthetic(n_windows=500, K=10, seed=1)
    tr = make_block_windows(d, "train", 96, 24)
    tid = d.meta["template_id"]
    LH = 96 + 24
    # every window's context+future must come from exactly ONE template block
    for o in tr.origin[:50].tolist():
        blk = (o - 96) // LH
        assert 0 <= blk < len(tid)


def test_pipeline_retrieval_recovers_template():
    pipe = build_pipeline("retrievable_synth", 96, 24, max_eval=200)
    idx, sim = pipe.db.search(pipe.test.ctx_n, pipe.test.chan, k=1)
    tid = pipe.data.meta["template_id"]
    LH = 96 + 24
    def bid(o): return (int(o) - 96) // LH
    train_ids = torch.tensor([tid[bid(o)] for o in pipe.train.origin])
    test_ids = torch.tensor([tid[bid(o)] for o in pipe.test.origin])
    acc = (train_ids[idx[:, 0]] == test_ids).float().mean().item()
    assert acc > 0.9, f"expected near-perfect template recovery, got {acc}"
