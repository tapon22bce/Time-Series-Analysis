#!/usr/bin/env python3
"""Train one model variant (a diffusion variant OR a point baseline) on one dataset.

Usage:
  python scripts/train.py --config configs/base.yaml \
      --set dataset=ETTh1 model=diffusion_faithts horizon=24 --out results/ETTh1_faithts
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import TensorDataset, DataLoader

from faithts.utils import load_cfg, parse_overrides, seed_all, get_device, save_json
from faithts.data.pipeline import build_pipeline
from faithts.models.baselines import build_baseline
from faithts.models.variants import build_variant

POINT_MODELS = {"naive_last", "seasonal_naive", "ridge_linear", "dlinear", "patch_transformer"}


def make_loader(pipe, cfg, split):
    w = getattr(pipe, split)
    idx = torch.arange(len(w))
    return DataLoader(TensorDataset(idx), batch_size=cfg["batch_size"], shuffle=(split == "train"),
                       drop_last=(split == "train"))


def train_point(model, pipe, cfg, device):
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"]) if list(model.parameters()) else None
    loader = make_loader(pipe, cfg, "train")
    model.to(device)
    for ep in range(cfg["epochs"]):
        tot, n = 0.0, 0
        for (idx,) in loader:
            w = pipe.train.sel(idx)
            ctx, fut, tf = w.ctx_n.to(device), w.fut_n.to(device), w.tfeat.to(device)
            pred = model(ctx, tfeat=tf)
            loss = torch.nn.functional.mse_loss(pred, fut)
            if opt is not None:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx); n += len(idx)
        if opt is None: break  # naive/seasonal-naive have no parameters
        print(f"  epoch {ep+1}/{cfg['epochs']}  train_mse={tot/n:.4f}")
    return model


def train_diffusion(variant, pipe, cfg, device):
    opt = torch.optim.Adam(variant.parameters(), lr=cfg["lr"])
    loader = make_loader(pipe, cfg, "train")
    variant.to(device); variant.train()
    db = pipe.db.to(device) if device != "cpu" else pipe.db
    k = cfg["k_neighbors"]
    caf = cfg.get("corrupt_aug_frac", 0.0)          # P(corrupt a random neighbor's future in this batch)
    cas = cfg.get("corrupt_aug_shift", 2.0)          # magnitude of the injected shift
    for ep in range(cfg["epochs"]):
        tot, n = 0.0, 0
        for (idx,) in loader:
            w = pipe.train.sel(idx)
            ctx, fut, tf, chan, tns = w.ctx_n.to(device), w.fut_n.to(device), w.tfeat.to(device), w.chan.to(device), w.time_ns.to(device)
            ridx, sim = db.search(ctx, chan, k=k, exclude_time_ns=tns)
            nb = db.gather(ridx)
            nb_fut = nb["fut_n"]
            if caf > 0:
                # TRAINING-TIME CORRUPTION AUGMENTATION: this is what lets the adaptive gate learn
                # a general "distrust disagreeing/misleading neighbors" policy instead of only ever
                # seeing a clean database. The corrupted neighbor keeps its ORIGINAL similarity
                # score (a plausible-looking match with a wrong outcome) so the gate is FORCED to
                # use retrieval-set disagreement (RetrievalWeighter.set_feats), not similarity
                # alone, to detect it -- exactly the failure mode probed by the bias-variance sweep.
                B = ctx.shape[0]
                hit = torch.rand(B, device=device) < caf
                slot = torch.randint(0, k, (B,), device=device)
                nb_fut = nb_fut.clone()
                shift = torch.randn(B, nb_fut.shape[-1], device=device) * 0.3 + cas
                corrupted = nb_fut[torch.arange(B, device=device), slot] + shift
                nb_fut[torch.arange(B, device=device)[hit], slot[hit]] = corrupted[hit]
            loss, wgt, gate = variant.loss(ctx, fut, nb_fut, sim, tf, cf_drop=cfg.get("cf_drop", 0.1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(variant.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx); n += len(idx)
        print(f"  epoch {ep+1}/{cfg['epochs']}  train_diffusion_loss={tot/n:.4f}  mean_gate={float(gate.detach().mean()):.3f}")
    return variant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_cfg(a.config, parse_overrides(a.set))
    seed_all(cfg.get("seed", 0))
    device = cfg.get("device") or get_device()
    os.makedirs(a.out, exist_ok=True)

    t0 = time.time()
    pipe = build_pipeline(cfg["dataset"], cfg["L"], cfg["horizon"], cfg.get("stride_train", 1),
                          cfg.get("stride_eval", 4), cfg.get("max_train"), cfg.get("max_eval"),
                          root=cfg.get("data_root", "data/raw"), seed=cfg.get("seed", 0))
    print(f"[{cfg['dataset']}] train/val/test windows: {len(pipe.train)}/{len(pipe.val)}/{len(pipe.test)}  "
          f"(db size = {len(pipe.db)})  loaded in {time.time()-t0:.1f}s")

    name = cfg["model"]
    if name in POINT_MODELS:
        model = build_baseline(name, cfg["L"], cfg["horizon"], period=pipe.data.period)
        model = train_point(model, pipe, cfg, device)
        torch.save(model.state_dict() if list(model.parameters()) else {}, os.path.join(a.out, "model.pt"))
    else:
        variant = build_variant(name, cfg["L"], cfg["horizon"], cfg["k_neighbors"], d=cfg.get("d_model", 64),
                                depth=cfg.get("depth", 4), T=cfg.get("T", 200), sparse_k=cfg.get("sparse_k"))
        variant = train_diffusion(variant, pipe, cfg, device)
        torch.save(variant.diffusion.state_dict(), os.path.join(a.out, "diffusion.pt"))
        if variant.weighter is not None:
            torch.save(variant.weighter.state_dict(), os.path.join(a.out, "weighter.pt"))
    save_json(cfg, os.path.join(a.out, "config_used.json"))
    print(f"done in {time.time()-t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
