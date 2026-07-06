# -*- coding: utf-8 -*-
"""Confusion-matrix figure for the supervisor report, from the same full-cohort run."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(r"C:/Users/Ali D/Documents/ML")
PRED = ROOT / "outputs/runs/xgb_full_full_cnv_20260612_093700/final_statistics/predictions.csv"
OUT = ROOT / "outputs/reports/supervisor_2026-06-25/figures"
plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 200, "font.size": 12,
                     "font.family": "DejaVu Sans"})

df = pd.read_csv(PRED)
cm = pd.crosstab(df.y_true, df.y_pred).reindex(index=[0, 1], columns=[0, 1]).values
labels = ["Straight\n(One)", "Diagonal\n(Two)"]
row_tot = cm.sum(axis=1, keepdims=True)
pct = cm / row_tot * 100  # row-normalised (recall)

GREEN, RED = "#2f7d4f", "#c0392b"
fig, ax = plt.subplots(figsize=(6.4, 5.4))
for i in range(2):
    for j in range(2):
        correct = (i == j)
        base = GREEN if correct else RED
        alpha = 0.18 + 0.62 * (pct[i, j] / 100.0)
        ax.add_patch(Rectangle((j, 1 - i), 1, 1, facecolor=base, alpha=alpha, edgecolor="white", lw=3))
        tcol = "white" if alpha > 0.55 else "#1f2933"
        ax.text(j + 0.5, 1 - i + 0.60, f"{cm[i, j]:,}", ha="center", va="center",
                fontsize=23, fontweight="bold", color=tcol)
        tag = "correctly identified" if correct else "missed"
        ax.text(j + 0.5, 1 - i + 0.30, f"{pct[i, j]:.0f}% {tag}", ha="center", va="center",
                fontsize=10.5, color=tcol)

ax.set_xlim(0, 2); ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(labels, fontsize=11.5)
ax.set_yticks([1.5, 0.5]); ax.set_yticklabels(labels, fontsize=11.5)
ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
ax.set_xlabel("What the model predicted", fontsize=12.5, fontweight="bold", labelpad=10)
ax.set_ylabel("What the step actually was", fontsize=12.5, fontweight="bold", labelpad=10)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.set_title("Confusion matrix — all 2,384 predictions pooled\n(latest full-cohort run, 30 participants)",
             fontsize=13, fontweight="bold", pad=34)
# green = correct (the diagonal), red = mistakes (off-diagonal)
ax.text(1.0, -0.18, "Green diagonal = correct calls   ·   Red off-diagonal = mistakes   ·   "
        "percentages are out of each actual class",
        ha="center", va="top", fontsize=9, color="#555", transform=ax.transData)
fig.tight_layout()
fig.savefig(OUT / "fig_confusion.png", bbox_inches="tight")
plt.close(fig)
print("confusion matrix written ->", OUT / "fig_confusion.png")
print("cells [TN,FP,FN,TP] =", cm.ravel().tolist(), "overall acc %.4f" % ((cm[0,0]+cm[1,1])/cm.sum()))
