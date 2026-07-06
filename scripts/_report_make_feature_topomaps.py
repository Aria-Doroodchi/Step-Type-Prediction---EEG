# -*- coding: utf-8 -*-
"""Matrix of example feature topographies (single trials) for the supervisor report.
8 panels: amplitude, theta/alpha/beta/gamma power (scalp topomaps), and 2 source
eLORETA maps (glass-brain). Each panel captioned with feature, participant,
condition (straight/diagonal) and time bin. Replaces the informativeness topomap.
"""
import re, io
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import mne
from mne.channels import make_standard_montage
from nilearn import plotting

mne.set_log_level("ERROR")
ROOT = Path(r"C:/Users/Ali D/Documents/ML")
FEAT = ROOT / "data/features"
OUT = ROOT / "outputs/reports/supervisor_2026-06-25/figures"
SD = str(ROOT)  # fsaverage lives at ./fsaverage
plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 200, "font.family": "DejaVu Sans"})

BIN_S = 0.0625
BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
BAND_HZ = {"Delta": "0.5–4 Hz", "Theta": "4–8 Hz", "Alpha": "8–13 Hz", "Beta": "13–30 Hz", "Gamma": "30–40 Hz"}
COND_DISP = {"One": "straight", "Two": "diagonal"}
FEAT_COL = {"amp": "#1f77b4", "band": "#e08a1e", "src": "#2f7d4f"}
montage = make_standard_montage("standard_1005")

def tstr(k):
    return f"{k*BIN_S:.2f}–{(k+1)*BIN_S:.2f} s"

def scalp_info(chans):
    info = mne.create_info(list(chans), sfreq=1.0, ch_types="eeg")
    info.set_montage(montage)
    return info

# --- source ROI centroids in MNI (Destrieux / aparc.a2009s) ---
print("computing Destrieux ROI centroids ...")
labels = [l for l in mne.read_labels_from_annot("fsaverage", "aparc.a2009s", subjects_dir=SD)
          if "unknown" not in l.name.lower() and "?" not in l.name]
ROI_MNI = {}
for l in labels:
    v = l.center_of_mass(subject="fsaverage", subjects_dir=SD)
    ROI_MNI[l.name] = mne.vertex_to_mni(v, hemis=0 if l.hemi == "lh" else 1,
                                        subject="fsaverage", subjects_dir=SD)

def load_row(pid, cond, ep, cols):
    f = FEAT / f"{pid}_{cond}_features_t0p0-2p0_b0p0625.parquet"
    cols = [c for c in cols if c]  # filter
    df = pd.read_parquet(f, columns=cols)
    ep = min(ep, len(df) - 1)
    return df.iloc[ep]

def amp_vec(pid, cond, ep, k):
    chans = [re.match(r"amp_w0p0625_(.+)_mean_bin_0$", c).group(1)
             for c in pd.read_parquet(FEAT / f"{pid}_{cond}_features_t0p0-2p0_b0p0625.parquet", columns=None).columns
             if re.match(r"amp_w0p0625_(.+)_mean_bin_0$", c)]
    chans = [c for c in chans if c in montage.ch_names]
    cols = [f"amp_w0p0625_{c}_mean_bin_{k}" for c in chans]
    row = load_row(pid, cond, ep, cols)
    return chans, np.array([row[c] for c in cols], float)

def band_vec(pid, cond, ep, k, band):
    fcols = pd.read_parquet(FEAT / f"{pid}_{cond}_features_t0p0-2p0_b0p0625.parquet", columns=None).columns
    chans = [re.match(rf"(.+)_{band}_bin_0$", c).group(1) for c in fcols if re.match(rf"(.+)_{band}_bin_0$", c)]
    chans = [c for c in chans if c in montage.ch_names]
    cols = [f"{c}_{band}_bin_{k}" for c in chans]
    row = load_row(pid, cond, ep, cols)
    return chans, np.array([row[c] for c in cols], float)

def src_vec(pid, cond, ep, k):
    rois = [r for r in ROI_MNI]
    cols = [f"{r}_bin_{k}" for r in rois]
    fcols = set(pd.read_parquet(FEAT / f"{pid}_{cond}_features_t0p0-2p0_b0p0625.parquet", columns=None).columns)
    keep = [(r, c) for r, c in zip(rois, cols) if c in fcols]
    row = load_row(pid, cond, ep, [c for _, c in keep])
    coords = np.array([ROI_MNI[r] for r, _ in keep])
    vals = np.array([row[c] for _, c in keep], float)
    return coords, vals

# --- panels (4 rows x 2 cols, row-major) ---
PANELS = [
    dict(kind="amp",  pid="P12", cond="One", ep=0, k=3),
    dict(kind="src",  pid="P12", cond="One", ep=0, k=3),
    dict(kind="band", band="Theta", pid="P13", cond="One", ep=4, k=5),
    dict(kind="band", band="Alpha", pid="P25", cond="Two", ep=7, k=12),
    dict(kind="amp",  pid="P15", cond="Two", ep=2, k=8),
    dict(kind="src",  pid="P08", cond="Two", ep=6, k=10),
    dict(kind="band", band="Beta", pid="P19", cond="One", ep=3, k=6),
    dict(kind="band", band="Gamma", pid="P30", cond="Two", ep=5, k=20),
]

fig, axes = plt.subplots(4, 2, figsize=(7.6, 10.6))
axes = axes.ravel()

def feat_label(p):
    if p["kind"] == "amp": return "Amplitude (CSD)"
    if p["kind"] == "src": return "Source (eLORETA)"
    return f"{p['band']} power ({BAND_HZ[p['band']]})"

def render_glassbrain(coords, vals):
    """Render a top-down glass-brain to a standalone image array (consistent sizing)."""
    m = np.max(np.abs(vals)) or 1.0
    f2 = plt.figure(figsize=(3.2, 2.4))
    plotting.plot_markers(vals, coords, node_size=30, display_mode="z", figure=f2,
                          node_cmap="RdBu_r", node_vmin=-m, node_vmax=m, colorbar=False, annotate=False)
    buf = io.BytesIO(); f2.savefig(buf, format="png", dpi=160, bbox_inches="tight", transparent=True)
    plt.close(f2); buf.seek(0)
    return mpimg.imread(buf)

for ax, p in zip(axes, PANELS):
    col = FEAT_COL[p["kind"]]
    if p["kind"] == "src":
        coords, vals = src_vec(p["pid"], p["cond"], p["ep"], p["k"])
        ax.imshow(render_glassbrain(coords, vals)); ax.axis("off")
    else:
        if p["kind"] == "amp":
            chans, vals = amp_vec(p["pid"], p["cond"], p["ep"], p["k"]); cmap = "RdBu_r"
        else:
            chans, vals = band_vec(p["pid"], p["cond"], p["ep"], p["k"], p["band"]); cmap = "YlOrRd"
        info = scalp_info(chans)
        if cmap == "RdBu_r":
            m = np.max(np.abs(vals)) or 1.0; vmin, vmax = -m, m
        else:
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
        mne.viz.plot_topomap(vals, info, axes=ax, show=False, cmap=cmap,
                             vlim=(vmin, vmax), contours=2, sensors=True, outlines="head")
    # captions (3 short lines so adjacent columns don't collide); lower for src (brain fills axis)
    y0 = -0.12 if p["kind"] == "src" else -0.09
    ax.text(0.5, y0, feat_label(p), transform=ax.transAxes, ha="center", va="top",
            fontsize=11, fontweight="bold", color=col)
    ax.text(0.5, y0 - 0.12, f"{p['pid']} · {COND_DISP[p['cond']]} · trial {p['ep']+1}",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color="#333")
    ax.text(0.5, y0 - 0.22, f"bin {p['k']} · {tstr(p['k'])}",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color="#333")

fig.suptitle("Example feature topographies (single trials)", fontsize=15, fontweight="bold", y=0.99)
fig.subplots_adjust(left=0.03, right=0.95, top=0.95, bottom=0.03, wspace=0.30, hspace=0.95)
fig.savefig(OUT / "fig_feature_topomaps.png", bbox_inches="tight")
plt.close(fig)
print("written ->", OUT / "fig_feature_topomaps.png")
