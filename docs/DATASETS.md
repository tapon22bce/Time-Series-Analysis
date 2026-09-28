# Dataset status

| dataset | status | notes |
|---|---|---|
| ETTh1/h2/m1/m2 | **verified in this sandbox** | downloaded + loaded + windowed + trained on here |
| Electricity (UCI 321) | written from spec, not run here | zip download configured; `LD2011_2014.txt`, `;`-sep, comma-decimal, resampled hourly from 15-min |
| PEMS04/08 (STSGCN) | written from spec, not run here | STSGCN repo documents `.npz` files with shape `(T, N, 3)` (flow/occupancy/speed); we use channel 0 (flow). **You must obtain the `.npz` yourself** (STSGCN's repo links Google Drive/Baidu Netdisk, not a stable direct URL our `download_data.py` could hit) and place it at `data/raw/PEMS/pems04.npz` |
| Weather (MPI Jena) | written from spec, not run here | `bgc-jena.mpg.de/wetter` publishes per-year zips; the commonly-used pre-merged `weather.csv` (as in Autoformer/Informer repos) is easier -- place at `data/raw/weather/weather.csv` |
| StateAir Beijing PM2.5 | written from spec, not run here | `stateair.net` publishes one CSV per site-year; place all of them in `data/raw/stateair/` |
| SAPFLUXNET (Zenodo 3971689) | written from spec, schema **unverified** | this is a 3.2 GB multi-site sap-flow database (fetched and inspected at build time -- it is a Zenodo record host page, not a single file); extract ONE site's `*_sapf_data.csv` and place it at `data/raw/sapfluxnet/<file>`, then edit `configs/datasets.yaml`'s `SAPFLUX_SITE` entry's `file:` field |
| epmap water (`wat.epmap.org`) | **no verified public schema found** | this looks like a live monitoring dashboard, not a bulk-download endpoint; export what you can from its UI to a `date,value[,value2,...]` CSV and point `EPMapWater.file` at it |
| nocrew weather (`weather.nocrew.org`) | **no verified public schema found** | same as above: export to CSV, edit `NocrewWeather.file`/`date_col`/`cols` |
| sxcoal (`www.sxcoal.com`) | **no verified public schema found** | a commercial coal-price data provider; likely needs a paid API/account, not a public bulk file. Export to CSV and edit `SXCoal.file` |
| stats.gov.cn (NBS China) | **no verified public schema found** | NBS publishes many different tables via its own query interface; pick the specific series you want, export to CSV, edit `NBSChina.file` |

For the three "no verified public schema" sources, `faithts/data/loaders.py:load_generic` will work
with ANY csv that has a date column plus one or more numeric columns -- there is nothing dataset-
specific to write once you have the export; only `configs/datasets.yaml`'s `date_col`/`cols`/
`freq_min`/`period` need editing to match.

## How a new dataset gets wired in
1. Add an entry to `configs/datasets.yaml` (`kind` picks a loader in `faithts/data/loaders.py`;
   `generic` covers any single/multi-column CSV with a date column).
2. `python scripts/download_data.py --names <YourName>` (or place the file manually per the table
   above).
3. `python scripts/train.py --config configs/base.yaml --set dataset=<YourName> --out results/x`.

No changes to `faithts/data/windows.py`, `retrieval/`, `models/`, or `metrics/` are needed --
everything downstream of `load_dataset()` is dataset-agnostic.
