"""Render context + forecast (+ neighbors) as a chart PNG, for the VLM explainer/critic."""
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def render_chart(ctx_n, fut_n, neighbors_fut=None, weights=None, title=None, dpi=100):
    """ctx_n, fut_n: 1D array-like. neighbors_fut: list of 1D arrays. weights: matching list."""
    fig, ax = plt.subplots(figsize=(6, 3), dpi=dpi)
    L, H = len(ctx_n), len(fut_n)
    ax.plot(range(L), ctx_n, color="black", lw=1.3, label="context")
    ax.plot(range(L, L + H), fut_n, color="tab:blue", lw=1.8, label="forecast")
    ax.axvline(L, color="gray", ls="--", lw=0.7)
    if neighbors_fut is not None:
        for i, nb in enumerate(neighbors_fut):
            wgt = weights[i] if weights is not None else 1.0
            if wgt <= 1e-3: continue
            ax.plot(range(L, L + H), nb, color="tab:orange", lw=0.8 + 2.5 * wgt, alpha=min(1.0, 0.3 + wgt))
    ax.legend(fontsize=7, loc="upper left"); ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=9)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png"); plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
