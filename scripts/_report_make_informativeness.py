# -*- coding: utf-8 -*-
"""Feature-informativeness figures from the run's selection-frequency data.
 fig_feat_informativeness.png : (a) selection mass by feature family, (b) over time
 fig_feat_topomap.png         : scalp topography of per-channel informativeness
Informativeness = how consistently a feature is retained by the selection pipeline
across the 150 folds / 30 participants (fold_frequency in feature_selection_frequency.csv).
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"C:/Users/Ali D/Documents/ML")
RUN = ROOT / "outputs/runs/xgb_full_full_cnv_20260612_093700"
FREQ = RUN / "feature_recovery/feature_selection_frequency.csv"
PARQ = ROOT / "data/features/P12_One_features_t0p0-2p0_b0p0625.parquet"
OUT = ROOT / "outputs/reports/supervisor_2026-06-25/figures"
plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 200, "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 13, "axes.titleweight": "bold", "font.family": "DejaVu Sans"})

BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
BIN_S = 0.0625
df = pd.read_csv(FREQ)

def parse(f):
    if f.startswith("slope_"):
        m = re.match(r"slope_(.+)_bin_(\d+)", f); return ("Slope", m.group(1), None, int(m.group(2)))
    if f.startswith("amp_w"):
        m = re.match(r"amp_w[0-9p]+_(.+?)_(mean|std|min|max)_bin_(\d+)", f); return ("Amplitude", m.group(1), m.group(2), int(m.group(3)))
    m = re.match(r"(.+?)_(" + "|".join(BANDS) + r")_bin_(\d+)", f)
    if m: return ("Power", m.group(1), m.group(2), int(m.group(3)))
    m = re.match(r"(.+)_bin_(\d+)", f)
    if m: return ("Source", m.group(1), None, int(m.group(2)))
    return ("Other", f, None, None)

df[["family", "chan", "band", "bin"]] = df["feature"].apply(lambda f: pd.Series(parse(f)))
df["time"] = df["bin"] * BIN_S

# ============ FIGURE A: family + time ============
fig, (axf, axt) = plt.subplots(1, 2, figsize=(10.2, 4.3))

# (a) selection mass by family (sum of fold_frequency); show Amplitude & Source as 0 (considered, not retained)
fam_order = ["Slope", "Power", "Amplitude", "Source"]
fam_color = {"Slope": "#6a51a3", "Power": "#e08a1e", "Amplitude": "#1f77b4", "Source": "#2f7d4f"}
mass = df.groupby("family")["fold_frequency"].sum()
counts = df.groupby("family")["feature"].size()
vals = [mass.get(f, 0.0) for f in fam_order]
ncnt = [int(counts.get(f, 0)) for f in fam_order]
y = np.arange(len(fam_order))[::-1]
axf.barh(y, vals, color=[fam_color[f] for f in fam_order], edgecolor="white")
for yi, v, n in zip(y, vals, ncnt):
    axf.text(v + 1.5, yi, f"{n} features" if n else "none retained", va="center", fontsize=9, color="#444")
axf.set_yticks(y); axf.set_yticklabels(fam_order)
axf.set_xlabel("Total informativeness (summed selection frequency)")
axf.set_title("(a)  Which feature types carry the signal")
axf.set_xlim(0, max(vals) * 1.35)

# (b) informativeness over time (summed fold_frequency per bin)
tmass = df.groupby("bin")["fold_frequency"].sum().reindex(range(32), fill_value=0.0)
t = (np.arange(32) + 0.5) * BIN_S
axt.fill_between(t, tmass.values, color="#6a51a3", alpha=0.30)
axt.plot(t, tmass.values, color="#4b2e83", lw=2.2)
peak = t[np.argmax(tmass.values)]
axt.axvspan(0.0625, 0.375, color="#e08a1e", alpha=0.10)
axt.set_xlabel("Time after direction cue (s)")
axt.set_ylabel("Informativeness (summed selection freq.)")
axt.set_title("(b)  When in the window the signal lives")
axt.set_xlim(0, 2.0)
axt.annotate(f"peak ≈ {peak:.2f} s\n(early foreperiod)", xy=(peak, tmass.max()),
             xytext=(0.65, tmass.max() * 0.82), fontsize=9.5, color="#a85a00",
             arrowprops=dict(arrowstyle="->", color="#a85a00", lw=1.3))
fig.suptitle("Feature informativeness — how often each feature is retained across the 150 folds",
             fontsize=13.5, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig_feat_informativeness.png", bbox_inches="tight")
plt.close(fig)
print("[A] family+time written")

# ============ FIGURE B: scalp topography ============
import mne
mne.set_log_level("ERROR")
# full recording channel set from a parquet (slope_<chan>_bin_0)
cols = pd.read_parquet(PARQ).columns
all_chans = sorted({re.match(r"slope_(.+)_bin_0$", c).group(1) for c in cols if re.match(r"slope_(.+)_bin_0$", c)})
ch_mass = df.groupby("chan")["fold_frequency"].sum()
montage = mne.channels.make_standard_montage("standard_1005")
keep = [c for c in all_chans if c in montage.ch_names]
vals = np.array([ch_mass.get(c, 0.0) for c in keep])
info = mne.create_info(keep, sfreq=1.0, ch_types="eeg"); info.set_montage(montage)

fig, ax = plt.subplots(figsize=(5.6, 5.2))
im, _ = mne.viz.plot_topomap(vals, info, axes=ax, show=False, cmap="YlOrRd",
                             contours=4, sensors=True, outlines="head", sphere=None)
cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.04)
cb.set_label("informativeness (summed selection freq.)", fontsize=9)
ax.set_title("Where on the scalp the informative features sit\n(brighter = more consistently selected)",
             fontsize=12.5, fontweight="bold")
fig.text(0.5, 0.02, "Strongest over central & fronto-central sensorimotor sites (C, FC, CP), right-lateralised (TP8, C6/C4).",
         ha="center", fontsize=8.3, color="#555")
fig.tight_layout()
fig.savefig(OUT / "fig_feat_topomap.png", bbox_inches="tight")
plt.close(fig)
print("[B] topomap written; top channels:", ch_mass.sort_values(ascending=False).head(6).round(1).to_dict())
