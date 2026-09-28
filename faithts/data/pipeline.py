"""Glue: dataset -> (train/val/test Windows, RetrievalDB built from TRAIN ONLY)."""
from dataclasses import dataclass
from .registry import load_dataset
from .windows import make_windows, make_block_windows
from ..retrieval.database import RetrievalDB


@dataclass
class Pipeline:
    data: object
    train: object
    val: object
    test: object
    db: RetrievalDB


def build_pipeline(name, L, H, stride_train=1, stride_eval=4, max_train=None, max_eval=None,
                   root="data/raw", seed=0):
    d = load_dataset(name, root=root)
    if name == "retrievable_synth":
        assert (L, H) == (d.meta["L"], d.meta["H"]), (
            f"retrievable_synth was generated with L={d.meta['L']},H={d.meta['H']}; "
            f"the pipeline config asked for L={L},H={H}. Regenerate with matching L/H or fix the config.")
        tr = make_block_windows(d, "train", L, H, max_n=max_train, seed=seed)
        va = make_block_windows(d, "val", L, H, max_n=max_eval, seed=seed)
        te = make_block_windows(d, "test", L, H, max_n=max_eval, seed=seed)
        db = RetrievalDB(tr)
        return Pipeline(d, tr, va, te, db)
    tr = make_windows(d, "train", L, H, stride=stride_train, max_n=max_train, seed=seed)
    va = make_windows(d, "val", L, H, stride=stride_eval, max_n=max_eval, seed=seed)
    te = make_windows(d, "test", L, H, stride=stride_eval, max_n=max_eval, seed=seed)
    db = RetrievalDB(tr)
    return Pipeline(d, tr, va, te, db)
