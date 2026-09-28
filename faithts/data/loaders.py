"""Loaders for the datasets you listed. Status per loader is documented in docs/DATASETS.md:
  verified-in-sandbox : ETT (downloaded + parsed here)
  written-from-spec   : Electricity(UCI), PEMS(npz), Weather(MPI/Autoformer csv), StateAir csv
  generic/manual      : epmap water, nocrew weather, sxcoal, stats.gov.cn, SAPFLUXNET (you must place files
                        in data/raw/<name>/ and edit configs/datasets.yaml columns)
"""
import glob, os
import numpy as np, pandas as pd
from .windows import TSData


def _mk(name, df, freq_min, split, period):
    df = df.sort_index()
    df = df[~df.index.duplicated()]
    df = df.astype("float32").interpolate(limit_direction="both")
    T = len(df)
    if split == "etth":   n_tr, n_va, n_te = 12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24
    elif split == "ettm": n_tr, n_va, n_te = 12 * 30 * 96, 16 * 30 * 96, 20 * 30 * 96
    elif split == "622":  n_tr, n_va, n_te = int(T * .6), int(T * .8), T
    else:                 n_tr, n_va, n_te = int(T * .7), int(T * .8), T   # "712"
    n_te = min(n_te, T)
    return TSData(name, df.values, pd.DatetimeIndex(df.index), list(df.columns), freq_min, (n_tr, n_va, n_te), period)


def _resample(df, rule):
    return df.resample(rule).mean() if rule else df


def load_csv_generic(path, date_col="date", cols=None, dayfirst=False, resample=None, **kw):
    df = pd.read_csv(path, **kw)
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=dayfirst)
    df = df.set_index(date_col)
    df = df[cols] if cols else df.select_dtypes("number")
    return _resample(df, resample)


def load_ett(root, spec):
    df = load_csv_generic(os.path.join(root, spec["file"]), "date")
    return _mk(spec["name"], df, spec["freq_min"], spec["split"], spec["period"])


def load_electricity(root, spec):
    p = os.path.join(root, "LD2011_2014.txt")
    df = pd.read_csv(p, sep=";", decimal=",", index_col=0, parse_dates=True)
    df = df.iloc[:, : spec.get("max_channels", 32)]
    df = df.resample("1h").sum()
    df = df.loc[df.index >= "2012-01-01"]
    return _mk(spec["name"], df, 60, spec["split"], 24)


def load_pems(root, spec):
    arr = np.load(os.path.join(root, spec["file"]))["data"][:, :, 0]      # flow
    arr = arr[:, : spec.get("max_channels", 64)]
    idx = pd.date_range(spec["start"], periods=arr.shape[0], freq="5min")
    df = pd.DataFrame(arr, index=idx, columns=[f"n{i}" for i in range(arr.shape[1])])
    df = _resample(df, spec.get("resample"))
    return _mk(spec["name"], df, 60 if spec.get("resample") == "1h" else 5, spec["split"], 24 if spec.get("resample") == "1h" else 288)


def load_weather(root, spec):
    p = os.path.join(root, spec["file"])
    df = pd.read_csv(p, encoding="latin1")
    dc = "date" if "date" in df.columns else "Date Time"
    df[dc] = pd.to_datetime(df[dc], dayfirst=(dc != "date"))
    df = df.set_index(dc).select_dtypes("number")
    df = df.iloc[:, : spec.get("max_channels", 21)]
    df = _resample(df, spec.get("resample", "1h"))
    return _mk(spec["name"], df, 60, spec["split"], 24)


def load_stateair(root, spec):
    fs = sorted(glob.glob(os.path.join(root, spec.get("glob", "*.csv"))))
    if not fs: raise FileNotFoundError(f"put StateAir hourly csv files in {root}")
    dfs = []
    for f in fs:
        d = pd.read_csv(f)
        vc = next(c for c in ["Value", "NowCast Conc.", "Raw Conc."] if c in d.columns)
        dc = next(c for c in ["Date (LST)", "Date", "date"] if c in d.columns)
        d = d.assign(**{dc: pd.to_datetime(d[dc])}).set_index(dc)[[vc]].rename(columns={vc: "PM2.5"})
        d[d <= -900] = np.nan
        dfs.append(d)
    return _mk(spec["name"], pd.concat(dfs), 60, spec["split"], 24)


def load_sapfluxnet(root, spec):
    """One SAPFLUXNET site: <root>/<site>_sapf_data.csv (TIMESTAMP + plant columns). Written from the
    Zenodo description; column names UNVERIFIED -> edit if your extraction differs."""
    p = os.path.join(root, spec["file"])
    d = pd.read_csv(p)
    tc = next(c for c in d.columns if c.lower().startswith("timestamp"))
    d[tc] = pd.to_datetime(d[tc]); d = d.set_index(tc).select_dtypes("number")
    d = d.loc[:, d.notna().mean() > 0.8].iloc[:, : spec.get("max_channels", 16)]
    return _mk(spec["name"], _resample(d, spec.get("resample", "1h")), 60, spec["split"], 24)


def load_generic(root, spec):
    df = load_csv_generic(os.path.join(root, spec["file"]), spec.get("date_col", "date"), spec.get("cols"),
                          spec.get("dayfirst", False), spec.get("resample"))
    return _mk(spec["name"], df, spec.get("freq_min", 60), spec.get("split", "712"), spec.get("period", 24))


KINDS = dict(ett=load_ett, electricity=load_electricity, pems=load_pems, weather=load_weather,
             stateair=load_stateair, sapfluxnet=load_sapfluxnet, generic=load_generic)
