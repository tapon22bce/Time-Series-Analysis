#!/usr/bin/env python3
"""Run the intervention-based faithfulness benchmark (faithts.bench.faithfulness) on a trained
diffusion variant and save per-query results + a summary.

Usage:
  python scripts/run_faithfulness_bench.py --config configs/smoke.yaml \
      --ckpt_dir results/smoke_faithts --out results/smoke_faithts/faithfulness
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from faithts.utils import load_cfg, parse_overrides, seed_all, get_device, save_json
from faithts.data.pipeline import build_pipeline
from faithts.models.variants import build_variant
from faithts.explain.explainer import RuleExplainer
from faithts.bench.faithfulness import run_faithfulness_benchmark, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_queries", type=int, default=200)
    ap.add_argument("--sample_steps", type=int, default=None)
    a = ap.parse_args()
    cfg = load_cfg(a.config, parse_overrides(a.set))
    seed_all(cfg.get("seed", 0))
    device = cfg.get("device") or get_device()
    os.makedirs(a.out, exist_ok=True)

    pipe = build_pipeline(cfg["dataset"], cfg["L"], cfg["horizon"], cfg.get("stride_train", 1),
                          cfg.get("stride_eval", 4), cfg.get("max_train"), cfg.get("max_eval"),
                          root=cfg.get("data_root", "data/raw"), seed=cfg.get("seed", 0))
    variant = build_variant(cfg["model"], cfg["L"], cfg["horizon"], cfg["k_neighbors"], d=cfg.get("d_model", 64),
                            depth=cfg.get("depth", 4), T=cfg.get("T", 200), sparse_k=cfg.get("sparse_k"))
    variant.diffusion.load_state_dict(torch.load(os.path.join(a.ckpt_dir, "diffusion.pt"), map_location="cpu"))
    if variant.weighter is not None and os.path.exists(os.path.join(a.ckpt_dir, "weighter.pt")):
        variant.weighter.load_state_dict(torch.load(os.path.join(a.ckpt_dir, "weighter.pt"), map_location="cpu"))
    variant.to(device); variant.eval()
    db = pipe.db.to(device) if device != "cpu" else pipe.db

    steps = a.sample_steps or cfg.get("sample_steps", 50)

    def net_sample_fn(ctx_n, nb_fut_n, w, tfeat):
        with torch.no_grad():
            return variant.diffusion.sample(ctx_n, nb_fut_n, w, tfeat, steps=steps)

    explainer = RuleExplainer()
    results = run_faithfulness_benchmark(db, variant.weighter, net_sample_fn, explainer, pipe.test,
                                         k=cfg["k_neighbors"], n_queries=a.n_queries, seed=cfg.get("seed", 0))
    summary = summarize(results)
    print(json.dumps(summary, indent=2))
    save_json(summary, os.path.join(a.out, "summary.json"))
    save_json([r.__dict__ for r in results], os.path.join(a.out, "per_query.json"))


if __name__ == "__main__":
    main()
