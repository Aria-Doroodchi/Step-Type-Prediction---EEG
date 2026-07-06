"""Generate data-driven figures for the supervisor report from real parquet features.
Figures: (1) grand-average CNV waveform One vs Two @ Cz, (2) binning schematic,
(3) PSD band power One vs Two @ Cz, (4) per-participant AUC bar chart.
Run with .venv (py3.14) which has pandas/pyarrow/matplotlib.
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(r"C:/Users/Ali D/Documents/ML")
FEAT = ROOT / "data" / "features"
OUT = ROOT / "outputs" / "reports" / "supervisor_2026-06-25" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RUN = ROOT / "outputs" / "runs" / "xgb_full_full_cnv_20260612_093700" / "final_statistics"

# ---- style ----
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "font.family": "DejaVu Sans",
})
C_ONE = "#1f77b4"   # straight
C_TWO = "#d62728"   # diagonal
BIN_S = 0.0625
CH = "Cz"
BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
# 30-subject cohort from the headline run
COHORT = ("P01 P02 P03 P05 P06 P07 P08 P10 P11 P12 P13 P14 P15 P16 P18 P19 P21 "
          "P23 P24 P25 P26 P27 P28 P29 P30 P31 P33 P35 P37 P39").split()

def amp_cols(nbin=32):
    return [f"amp_w0p0625_{CH}_mean_bin_{k}" for k in range(nbin)]

def psd_cols(nbin=32):
    return [f"{CH}_{b}_bin_{k}" for b in BANDS for k in range(nbin)]

def load_cond(cond, wanted):
    """Return per-participant mean vector (averaged over that subject's epochs)."""
    rows = []
    for pid in COHORT:
        f = FEAT / f"{pid}_{cond}_features_t0p0-2p0_b0p0625.parquet"
        if not f.exists():
            continue
        avail = pd.read_parquet(f, columns=None).columns
        cols = [c for c in wanted if c in set(avail)]
        if not cols:
            continue
        df = pd.read_parquet(f, columns=cols)
        rows.append(df[cols].mean(axis=0))
    M = pd.DataFrame(rows)
    return M  # rows=participants, cols=features

# ================= FIG 1: CNV waveform =================
acols = amp_cols(32)
one = load_cond("One", acols)
two = load_cond("Two", acols)
t = (np.arange(32) + 0.5) * BIN_S
m1, s1 = one.mean(0).values, one.sem(0).values
m2, s2 = two.mean(0).values, two.sem(0).values

fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(t, m1, color=C_ONE, lw=2.4, label="Straight (One)")
ax.fill_between(t, m1 - s1, m1 + s1, color=C_ONE, alpha=0.18)
ax.plot(t, m2, color=C_TWO, lw=2.4, label="Diagonal (Two)")
ax.fill_between(t, m2 - s2, m2 + s2, color=C_TWO, alpha=0.18)
ax.axhline(0, color="#888", lw=0.8, ls=":")
ax.set_xlabel("Time after direction cue (s)")
ax.set_ylabel(f"Mean signal at {CH} (a.u.)")
ax.set_title(f"Average brain response at the vertex ({CH}): straight vs diagonal")
ax.legend(frameon=False, loc="best")
ax.set_xlim(0, 2.0)
ax.text(0.01, -0.22, "Grand average over 30 participants; shaded band = standard error.",
        transform=ax.transAxes, fontsize=9, color="#555")
fig.tight_layout()
fig.savefig(OUT / "fig_cnv_waveform.png", bbox_inches="tight")
plt.close(fig)
print("[1] CNV waveform written")

# ================= FIG 2: binning schematic =================
sig = m1.copy()  # use the straight-condition grand average as the 'continuous' signal
coarse_w = 0.25  # 0.25 s coarse bins -> every 4 fine bins
nb = int(2.0 / coarse_w)
fig, ax = plt.subplots(figsize=(8, 4.6))
# faux-continuous interpolation
tt = np.linspace(0, 2.0, 400)
ss = np.interp(tt, t, sig)
ax.plot(tt, ss, color="#444", lw=2.0, label="Continuous CNV signal")
cmap = plt.cm.viridis(np.linspace(0.15, 0.85, nb))
for b in range(nb):
    lo, hi = b * coarse_w, (b + 1) * coarse_w
    mask = (t >= lo) & (t < hi)
    val = sig[mask].mean()
    ax.add_patch(Rectangle((lo, min(0, val) if val < 0 else 0), coarse_w, abs(val),
                            color=cmap[b], alpha=0.28, ec="none"))
    ax.hlines(val, lo, hi, color=cmap[b], lw=3)
    ax.axvline(lo, color="#bbb", lw=0.8, ls="--")
    ax.text((lo + hi) / 2, ax.get_ylim()[1] if False else val, "", ha="center")
ax.axvline(2.0, color="#bbb", lw=0.8, ls="--")
ax.set_xlabel("Time after direction cue (s)")
ax.set_ylabel(f"Signal at {CH} (a.u.)")
ax.set_title("From signal to features: averaging the wave inside time bins")
ax.plot([], [], color="#6a51a3", lw=3, label="One feature = mean per bin")
ax.legend(frameon=False, loc="best")
ax.set_xlim(0, 2.0)
ax.text(0.01, -0.22, "Each coloured bar is one numeric feature (the average signal in a 0.25 s slice). "
        "The real model uses finer 0.0625 s slices.",
        transform=ax.transAxes, fontsize=9, color="#555")
fig.tight_layout()
fig.savefig(OUT / "fig_binning.png", bbox_inches="tight")
plt.close(fig)
print("[2] binning schematic written")

# ================= FIG 3: PSD band power =================
pcols = psd_cols(32)
one_p = load_cond("One", pcols)
two_p = load_cond("Two", pcols)
def band_means(M):
    out = {}
    for b in BANDS:
        cs = [c for c in M.columns if f"_{b}_bin_" in c]
        out[b] = M[cs].mean(axis=1)  # per-participant mean across bins
    return pd.DataFrame(out)
b1 = band_means(one_p); b2 = band_means(two_p)
x = np.arange(len(BANDS)); w = 0.38
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.bar(x - w/2, b1.mean().values, w, yerr=b1.sem().values, capsize=3,
       color=C_ONE, label="Straight (One)")
ax.bar(x + w/2, b2.mean().values, w, yerr=b2.sem().values, capsize=3,
       color=C_TWO, label="Diagonal (Two)")
ax.set_xticks(x); ax.set_xticklabels([f"{b}\n{r}" for b, r in zip(
    BANDS, ["0.5–4 Hz", "4–8 Hz", "8–13 Hz", "13–30 Hz", "30–40 Hz"])])
ax.set_ylabel(f"Average power at {CH} (a.u.)")
ax.set_title("Rhythm strength by frequency band: straight vs diagonal")
ax.legend(frameon=False)
ax.set_yscale("log")
ax.text(0.01, -0.28, "Grand average over 30 participants; error bars = standard error. Log scale.",
        transform=ax.transAxes, fontsize=9, color="#555")
fig.tight_layout()
fig.savefig(OUT / "fig_psd_bands.png", bbox_inches="tight")
plt.close(fig)
print("[3] PSD band power written")

# ================= FIG 4: per-participant AUC =================
ps = pd.read_csv(RUN / "participant_summary.csv")
ps = ps.sort_values("auc").reset_index(drop=True)
mean_auc = ps["auc"].mean()
colors = [C_TWO if a < 0.5 else ("#7fb069" if a >= 0.7 else "#f0a202") for a in ps["auc"]]
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.bar(ps["participant_id"], ps["auc"], color=colors, edgecolor="white", lw=0.5)
ax.axhline(0.5, color="#333", lw=1.3, ls="--")
ax.text(0.2, 0.505, "chance (0.50)", fontsize=9, color="#333", va="bottom")
ax.axhline(mean_auc, color="#1b3a6b", lw=1.6)
ax.text(len(ps) - 0.5, mean_auc + 0.006, f"cohort mean {mean_auc:.3f}",
        fontsize=9.5, color="#1b3a6b", ha="right", va="bottom", weight="bold")
ax.set_ylim(0.4, 1.02)
ax.set_ylabel("Accuracy of detection (AUC)")
ax.set_title("How well the model separates straight vs diagonal — per participant")
ax.set_xticklabels(ps["participant_id"], rotation=90, fontsize=8)
ax.text(0.01, -0.30, "AUC 0.5 = coin flip, 1.0 = perfect. Each bar is one participant's own model "
        "(latest full-cohort run, 30 participants).",
        transform=ax.transAxes, fontsize=9, color="#555")
fig.tight_layout()
fig.savefig(OUT / "fig_auc_bars.png", bbox_inches="tight")
plt.close(fig)
print("[4] AUC bars written; cohort mean AUC =", round(mean_auc, 4))
print("DONE ->", OUT)
