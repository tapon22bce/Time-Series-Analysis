import os, random, json, copy
import numpy as np, torch, yaml

def seed_all(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def deep_update(a, b):
    a = copy.deepcopy(a)
    for k, v in (b or {}).items():
        a[k] = deep_update(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a

def load_cfg(path, overrides=None):
    with open(path) as f: cfg = yaml.safe_load(f)
    return deep_update(cfg, overrides or {})

def parse_overrides(items):
    """--set a.b=1 c=foo  ->  nested dict"""
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try: v = yaml.safe_load(v)
        except Exception: pass
        d = out
        ks = k.split(".")
        for kk in ks[:-1]: d = d.setdefault(kk, {})
        d[ks[-1]] = v
    return out

def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
