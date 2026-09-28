#!/usr/bin/env python3
"""Evaluate one trained model (point or diffusion) on the test split: point metrics always; CRPS/
pinball/coverage + retrieval-quality metrics for diffusion variants."""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from faithts.utils import load_cfg, parse_overrides, seed_all, get_device, save_json
from faithts.data.pipeline import build_pipeline
from faithts.models.baselines import build_baseline
from faithts.models.variants import build_variant, DiffusionVariant
from faithts.metrics.point import all_point_metrics
from faithts.metrics.prob import all_prob_metrics
from faithts.metrics.retrieval import retrieval_report
from scripts.train import POINT_MODELS


def eval_point(model, pipe, device, batch=256):
    model.to(device).eval()
    preds, trues = [], []
    with torch.no_grad():
        for i in range(0, len(pipe.test), batch):
            w = pipe.test.sel(torch.arange(i, min(i + batch, len(pipe.test))))
            ctx, fut, tf = w.ctx_n.to(device), w.fut_n.to(device), w.tfeat.to(device)
            preds.append(model(ctx, tfeat=tf).cpu()); trues.append(fut.cpu())
    pred, true = torch.cat(preds), torch.cat(trues)
    return all_point_metrics(pred, true), {}


def eval_diffusion(variant: DiffusionVariant, pipe, cfg, device, n_samples=8, batch=64):
    variant.to(device); variant.eval()
    db = pipe.db.to(device) if device != "cpu" else pipe.db
    k = cfg["k_neighbors"]
    all_samples, trues, gates, rel_reports = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(pipe.test), batch):
            idx = torch.arange(i, min(i + batch, len(pipe.test)))
            w = pipe.test.sel(idx)
            ctx, fut, tf, chan = w.ctx_n.to(device), w.fut_n.to(device), w.tfeat.to(device), w.chan.to(device)
            ridx, sim = db.search(ctx, chan, k=k)
            nb = db.gather(ridx)
            rel_reports.append(retrieval_report(w.facts.to(device), nb["facts"]))
            samples, wgt, gate = variant.forecast(ctx, nb["fut_n"], sim, tf, n_samples=n_samples,
                                                    steps=cfg.get("sample_steps", 50))
            all_samples.append(samples.cpu()); trues.append(fut.cpu()); gates.append(gate.cpu())
    samples = torch.cat(all_samples, 1)     # (S, N, H)
    true = torch.cat(trues, 0)
    gate = torch.cat(gates, 0)
    point = all_point_metrics(samples.mean(0), true)
    prob = all_prob_metrics(samples, true)
    retr = {kk: sum(r.get(kk, 0) for r in rel_reports) / len(rel_reports) for kk in rel_reports[0]} if rel_reports else {}
    extra = dict(**prob, **retr, mean_gate=float(gate.mean()))
    return point, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=8)
    a = ap.parse_args()
    cfg = load_cfg(a.config, parse_overrides(a.set))
    seed_all(cfg.get("seed", 0))
    device = cfg.get("device") or get_device()

    pipe = build_pipeline(cfg["dataset"], cfg["L"], cfg["horizon"], cfg.get("stride_train", 1),
                          cfg.get("stride_eval", 4), cfg.get("max_train"), cfg.get("max_eval"),
                          root=cfg.get("data_root", "data/raw"), seed=cfg.get("seed", 0))
    name = cfg["model"]
    if name in POINT_MODELS:
        model = build_baseline(name, cfg["L"], cfg["horizon"], period=pipe.data.period)
        sd_path = os.path.join(a.ckpt_dir, "model.pt")
        if os.path.exists(sd_path):
            sd = torch.load(sd_path, map_location="cpu")
            if sd: model.load_state_dict(sd)
        point, extra = eval_point(model, pipe, device)
    else:
        variant = build_variant(name, cfg["L"], cfg["horizon"], cfg["k_neighbors"], d=cfg.get("d_model", 64),
                                depth=cfg.get("depth", 4), T=cfg.get("T", 200), sparse_k=cfg.get("sparse_k"))
        variant.diffusion.load_state_dict(torch.load(os.path.join(a.ckpt_dir, "diffusion.pt"), map_location="cpu"))
        if variant.weighter is not None and os.path.exists(os.path.join(a.ckpt_dir, "weighter.pt")):
            variant.weighter.load_state_dict(torch.load(os.path.join(a.ckpt_dir, "weighter.pt"), map_location="cpu"))
        point, extra = eval_diffusion(variant, pipe, cfg, device, n_samples=a.n_samples)

    result = dict(dataset=cfg["dataset"], model=name, horizon=cfg["horizon"], **point, **extra)
    print(json.dumps(result, indent=2))
    save_json(result, os.path.join(a.ckpt_dir, "eval.json"))


if __name__ == "__main__":
    main()
