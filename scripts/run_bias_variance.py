#!/usr/bin/env python3
"""Run the conditional-shift bias-variance sweep (faithts.theory.analysis) comparing the adaptive
gate (ours) against a fixed-weight retrieval policy (RATD-style ablation) as neighbor corruption
increases, and save both the empirical curves and the theoretical surrogate prediction.

Usage:
  python scripts/run_bias_variance.py --config configs/smoke.yaml \
      --ckpt_dir results/smoke_faithts --out results/smoke_faithts/bias_variance
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from faithts.utils import load_cfg, parse_overrides, seed_all, get_device, save_json
from faithts.data.pipeline import build_pipeline
from faithts.models.variants import build_variant
from faithts.theory.analysis import bias_variance_sweep, theoretical_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--ckpt_dir", required=True, help="checkpoint dir for the ADAPTIVE (diffusion_faithts) model")
    ap.add_argument("--ckpt_dir_fixed", default=None, help="checkpoint dir for the diffusion_fixed ablation "
                    "(RATD-style, gate hard-wired to 1); omit to skip that comparison curve")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fracs", nargs="*", type=float, default=[0.0, 0.1, 0.2, 0.4, 0.6])
    ap.add_argument("--shift_scale", type=float, default=2.0)
    ap.add_argument("--n_repeats", type=int, default=6)
    ap.add_argument("--n_queries", type=int, default=30)
    a = ap.parse_args()
    cfg = load_cfg(a.config, parse_overrides(a.set))
    seed_all(cfg.get("seed", 0))
    device = cfg.get("device") or get_device()
    os.makedirs(a.out, exist_ok=True)

    pipe = build_pipeline(cfg["dataset"], cfg["L"], cfg["horizon"], cfg.get("stride_train", 1),
                          cfg.get("stride_eval", 4), cfg.get("max_train"), cfg.get("max_eval"),
                          root=cfg.get("data_root", "data/raw"), seed=cfg.get("seed", 0))

    def load_variant(model_name, ckpt_dir):
        v = build_variant(model_name, cfg["L"], cfg["horizon"], cfg["k_neighbors"], d=cfg.get("d_model", 64),
                          depth=cfg.get("depth", 4), T=cfg.get("T", 200), sparse_k=cfg.get("sparse_k"))
        v.diffusion.load_state_dict(torch.load(os.path.join(ckpt_dir, "diffusion.pt"), map_location="cpu"))
        if v.weighter is not None and os.path.exists(os.path.join(ckpt_dir, "weighter.pt")):
            v.weighter.load_state_dict(torch.load(os.path.join(ckpt_dir, "weighter.pt"), map_location="cpu"))
        v.to(device); v.eval()
        return v

    steps = cfg.get("sample_steps", 50)

    def make_sample_fn(variant):
        def f(ctx_n, nb_fut_n, w, tfeat):
            with torch.no_grad():
                return variant.diffusion.sample(ctx_n, nb_fut_n, w, tfeat, steps=steps)
        return f

    out = {}
    # OURS: adaptive gate
    v_ada = load_variant("diffusion_faithts", a.ckpt_dir)
    out["adaptive"] = bias_variance_sweep(lambda: pipe.db, v_ada.weighter, make_sample_fn(v_ada), pipe.test,
                                          a.fracs, a.shift_scale, a.n_repeats, a.n_queries, cfg["k_neighbors"])
    # ABLATION: fixed-weight retrieval (gate hard-wired to 1), RATD-style
    if a.ckpt_dir_fixed:
        v_fix = load_variant("diffusion_fixed", a.ckpt_dir_fixed)
        out["fixed"] = bias_variance_sweep(lambda: pipe.db, None, make_sample_fn(v_fix), pipe.test,
                                           a.fracs, a.shift_scale, a.n_repeats, a.n_queries, cfg["k_neighbors"],
                                           fixed_weight_gate=1.0)
    else:
        print("[info] --ckpt_dir_fixed not given; skipping the fixed-weight (RATD-style) comparison curve.")
    out["theoretical_surrogate"] = theoretical_curve(a.fracs, a.shift_scale, k_neighbors=cfg["k_neighbors"])

    print(json.dumps(out, indent=2))
    save_json(out, os.path.join(a.out, "bias_variance.json"))


if __name__ == "__main__":
    main()
