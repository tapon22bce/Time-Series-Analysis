"""Download what is scriptable; print manual instructions for the rest.
   python scripts/download_data.py --names ETTh1 ETTh2 Electricity"""
import argparse, os, urllib.request, zipfile, yaml
ap = argparse.ArgumentParser(); ap.add_argument("--names", nargs="+", required=True); ap.add_argument("--root", default="data/raw")
a = ap.parse_args()
reg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "datasets.yaml")))
for n in a.names:
    s = reg[n]; d = os.path.join(a.root, s.get("dir", n)); os.makedirs(d, exist_ok=True)
    if s.get("manual") or "url" not in s:
        print(f"[{n}] MANUAL: place file(s) in {d}/ (see docs/DATASETS.md)"); continue
    fn = os.path.join(d, os.path.basename(s["url"]))
    if not os.path.exists(fn): print(f"[{n}] downloading {s['url']}"); urllib.request.urlretrieve(s["url"], fn)
    if s.get("unzip"): zipfile.ZipFile(fn).extractall(d)
    print(f"[{n}] ok -> {d}")
