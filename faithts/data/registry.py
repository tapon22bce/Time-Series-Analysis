import os, yaml
from .loaders import KINDS
from .synthetic import make_synthetic, make_retrievable_synthetic

_CFG = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "datasets.yaml")


def load_dataset(name, root="data/raw", registry=_CFG, **kw):
    if name == "synthetic": return make_synthetic()
    if name == "retrievable_synth":
        # NOTE: this dataset is pre-packed into non-overlapping (L+H)-length blocks (see
        # make_retrievable_synthetic docstring). The calling pipeline MUST use L/H matching the
        # generator's defaults (96/24) and stride_train=stride_eval=L+H=120, or windows will
        # straddle template boundaries and the controlled construction breaks. build_pipeline()
        # enforces this automatically when it detects this dataset name.
        return make_retrievable_synthetic(**kw)
    with open(registry) as f: reg = yaml.safe_load(f)
    if name not in reg: raise KeyError(f"{name} not in {registry}; known: {list(reg)}")
    spec = dict(reg[name]); spec["name"] = name
    sub = spec.get("dir", name)
    return KINDS[spec["kind"]](os.path.join(root, sub), spec)
