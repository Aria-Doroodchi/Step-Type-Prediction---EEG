"""Generate the figures embedded in docs/MODELS.md.

Two kinds of figure:
  * REAL  — plotted from this repo's screening summaries and the CNN/EEGNet
            occlusion diagnostics under outputs/diagnostics/.
  * SYNTH — illustrative tuning curves on *made-up* data, used to show what a
            given hyperparameter does in the abstract (clearly labelled as such).

Run:  .venv/Scripts/python.exe docs/make_models_figs.py
Output: docs/models_figs/*.png
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "models_figs"
OUT.mkdir(parents=True, exist_ok=True)

# ---- house style -----------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.autolayout": True,
})
# Per-model colours, reused everywhere.
C = {
    "logistic": "#4C72B0",
    "svm": "#DD8452",
    "xgb": "#55A868",
    "riemannian": "#C44E52",
    "cnn": "#8172B3",
    "eegnet": "#937860",
}
SYNTH = "#B07AA1"  # accent for made-up illustrative curves


def save(fig, name):
    p = OUT / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p.relative_to(ROOT))


# ===========================================================================
# REAL DATA (transcribed from outputs/screening/*.md — committed in-repo)
# ===========================================================================

# Diagnostic 1 — mean test AUC +/- 95% CI, early cohort n=8, Express tier.
D1 = {
    "logistic":   (0.4822, 0.0268),
    "svm":        (0.4905, 0.0266),
    "riemannian": (0.5088, 0.0212),
    "xgb":        (0.5777, 0.0402),
}

# Window effect — same binning recipe (rich_mean_0125), full 20-participant
# cohort, late vs full CNV window.
WINDOW = {
    #            late,    full
    "logistic":   (0.4539, 0.6499),
    "svm":        (0.5267, 0.6224),
    "xgb":        (0.5679, 0.6548),
    "riemannian": (0.5316, 0.5066),
}

# Per-participant AUC matrix (early cohort n=8, Express tier).
PMAT = pd.DataFrame(
    {
        "logistic":   [0.4331, 0.5094, 0.5406, 0.3670, 0.5500, 0.4656, 0.5484, 0.4437],
        "svm":        [0.4293, 0.4750, 0.4992, 0.4809, 0.5328, 0.5172, 0.5062, 0.4836],
        "riemannian": [0.6071, 0.3628, 0.5038, 0.6205, 0.4603, 0.4263, 0.6106, 0.4791],
        "xgb":        [0.4918, 0.3828, 0.5859, 0.6036, 0.6094, 0.5312, 0.8984, 0.5188],
    },
    index=["P08", "P11", "P19", "P23", "P24", "P25", "P30", "P39"],
)


def fig_real_auc_ci():
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    order = ["logistic", "svm", "riemannian", "xgb"]
    xs = np.arange(len(order))
    means = [D1[m][0] for m in order]
    cis = [D1[m][1] for m in order]
    ax.bar(xs, means, yerr=cis, capsize=6,
           color=[C[m] for m in order], edgecolor="black", linewidth=0.6)
    ax.axhline(0.5, ls="--", color="0.4", lw=1.2)
    ax.text(len(order) - 0.5, 0.505, "chance (0.50)", ha="right", va="bottom",
            color="0.4", fontsize=9)
    for x, m, ci in zip(xs, means, cis):
        ax.text(x, m + ci + 0.006, f"{m:.3f}", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(order)
    ax.set_ylim(0.40, 0.66)
    ax.set_ylabel("Mean test ROC-AUC")
    ax.set_title("REAL · Mean test AUC ± 95% CI (late-CNV, n=8)")
    save(fig, "real_auc_ci.png")


def fig_real_window_effect():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    order = ["logistic", "svm", "xgb", "riemannian"]
    xs = np.arange(len(order))
    w = 0.38
    late = [WINDOW[m][0] for m in order]
    full = [WINDOW[m][1] for m in order]
    ax.bar(xs - w / 2, late, w, label="late CNV (1.0–2.0 s)",
           color="#BBBBBB", edgecolor="black", linewidth=0.5)
    ax.bar(xs + w / 2, full, w, label="full CNV (0.0–2.0 s)",
           color=[C[m] for m in order], edgecolor="black", linewidth=0.5)
    for x, v in zip(xs - w / 2, late):
        ax.text(x, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
    for x, v in zip(xs + w / 2, full):
        ax.text(x, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(0.5, ls="--", color="0.4", lw=1.1)
    ax.set_xticks(xs)
    ax.set_xticklabels(order)
    ax.set_ylim(0.40, 0.72)
    ax.set_ylabel("Mean test ROC-AUC")
    ax.set_title("REAL · Window effect: full CNV beats the primary late-CNV window")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "real_window_effect.png")


def fig_real_participant_heatmap():
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    data = PMAT.values
    im = ax.imshow(data, cmap="RdYlGn", vmin=0.35, vmax=0.90, aspect="auto")
    ax.set_xticks(range(PMAT.shape[1]))
    ax.set_xticklabels(PMAT.columns, rotation=30, ha="right")
    ax.set_yticks(range(PMAT.shape[0]))
    ax.set_yticklabels(PMAT.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="black", fontsize=9)
    ax.set_title("REAL · Per-participant AUC by model (n=8)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("ROC-AUC")
    ax.grid(False)
    save(fig, "real_participant_heatmap.png")


def fig_real_eegnet_vs_cnn():
    """Baseline AUC per participant: full-window EEGNet vs late-window CNN."""
    cnn = pd.read_csv(ROOT / "outputs/diagnostics/cnn_tensor_diagnostics/"
                      "cnn_cohort_starter/participant_summary.csv")
    eeg = pd.read_csv(ROOT / "outputs/diagnostics/eegnet_tensor_diagnostics/"
                      "eegnet_cohort_starter/participant_summary.csv")
    m = pd.merge(cnn[["participant", "baseline_auc"]],
                 eeg[["participant", "baseline_auc"]],
                 on="participant", suffixes=("_cnn", "_eegnet"))
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    xs = np.arange(len(m))
    w = 0.4
    ax.bar(xs - w / 2, m["baseline_auc_cnn"], w, label="CNN (late CNV 1–2 s)",
           color=C["cnn"], edgecolor="black", linewidth=0.4)
    ax.bar(xs + w / 2, m["baseline_auc_eegnet"], w, label="EEGNet (full CNV 0–2 s)",
           color=C["eegnet"], edgecolor="black", linewidth=0.4)
    ax.axhline(0.5, ls="--", color="0.4", lw=1.1)
    ax.set_xticks(xs)
    ax.set_xticklabels(m["participant"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Baseline ROC-AUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("REAL · Hybrid-neural baseline AUC per participant")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    save(fig, "real_eegnet_vs_cnn.png")


def fig_real_time_occlusion():
    """CNN time-occlusion: which 250 ms window, when removed, costs the most AUC."""
    df = pd.read_csv(ROOT / "outputs/diagnostics/cnn_tensor_diagnostics/"
                     "cnn_cohort_starter/time_occlusion.csv")
    # Drop the 1-sample tail window (2.000-2.001 s); keep the four 250 ms bins.
    df = df[df["sample_stop"] - df["sample_start"] > 2]
    df["win"] = df["time_start_s"].round(2).astype(str) + "–" + \
        df["time_stop_s"].round(2).astype(str) + " s"
    g = df.groupby("win", sort=False)["delta_auc"].agg(["mean", "sem"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    xs = np.arange(len(g))
    ax.bar(xs, g["mean"], yerr=g["sem"], capsize=5,
           color=C["cnn"], edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(g["win"], rotation=20, ha="right")
    ax.set_ylabel("Mean AUC drop when occluded")
    ax.set_title("REAL · CNN time-occlusion (cohort mean ± SEM)")
    ax.text(0.99, 0.96, "higher = more informative window",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="0.4")
    save(fig, "real_time_occlusion.png")


# ===========================================================================
# SYNTHETIC illustrative tuning curves (made-up data; labelled SYNTH)
# ===========================================================================

def fig_synth_xgb_depth():
    depth = np.array([1, 2, 3, 4, 6, 8, 12, 16])
    train = 0.62 + 0.36 * (1 - np.exp(-depth / 2.2))
    val = 0.78 - 0.018 * (depth - 4) ** 2 / 4
    val = np.clip(val, 0.5, 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(depth, train, "o-", color="#999999", label="train AUC")
    ax.plot(depth, val, "o-", color=C["xgb"], label="validation AUC")
    best = depth[np.argmax(val)]
    ax.axvline(best, ls="--", color=C["xgb"], lw=1)
    ax.text(best + 0.3, 0.55, f"sweet spot\n(max_depth≈{best})",
            color=C["xgb"], fontsize=9)
    ax.fill_between(depth, val, train, color="red", alpha=0.06)
    ax.annotate("overfitting gap", xy=(12, 0.8), xytext=(8.5, 0.66),
                fontsize=9, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
    ax.set_xlabel("max_depth")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("SYNTH · XGBoost tree depth = bias–variance dial")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "synth_xgb_depth.png")


def fig_synth_xgb_lr():
    trees = np.arange(1, 1001)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for lr, col in [(0.30, "#C44E52"), (0.10, "#DD8452"),
                    (0.03, C["xgb"]), (0.01, "#4C72B0")]:
        val = 0.5 + 0.28 * (1 - np.exp(-lr * trees / 1.5))
        # high LR overshoots into overfit territory after a while
        val = val - 0.10 * lr * np.clip(trees - 60 / (lr / 0.03), 0, None) / 1000
        ax.plot(trees, np.clip(val, 0.5, 1), color=col, label=f"learning_rate={lr}")
    ax.set_xlabel("n_estimators (boosting rounds)")
    ax.set_ylabel("validation ROC-AUC")
    ax.set_title("SYNTH · learning_rate × n_estimators trade-off")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_xlim(0, 1000)
    save(fig, "synth_xgb_lr.png")


def fig_synth_svm_cgamma():
    C_vals = np.array([0.1, 1, 10, 100])
    g_vals = np.array([0.001, 0.01, 0.1, 1.0])
    rng = np.random.default_rng(0)
    # smooth ridge: best around C=10, gamma=0.01, decaying away + small noise
    grid = np.zeros((len(g_vals), len(C_vals)))
    for i, g in enumerate(g_vals):
        for j, c in enumerate(C_vals):
            d = (np.log10(c) - 1) ** 2 + (np.log10(g) + 2) ** 2
            grid[i, j] = 0.78 - 0.07 * d + rng.normal(0, 0.004)
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    im = ax.imshow(grid, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(C_vals)))
    ax.set_xticklabels(C_vals)
    ax.set_yticks(range(len(g_vals)))
    ax.set_yticklabels(g_vals)
    ax.set_xlabel("C (regularization strength)")
    ax.set_ylabel("gamma (RBF kernel width)")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    color="white", fontsize=8)
    ax.set_title("SYNTH · SVM validation accuracy over C×gamma")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="val accuracy")
    ax.grid(False)
    save(fig, "synth_svm_cgamma.png")


def fig_synth_logreg_path():
    Cs = np.logspace(-3, 2, 40)
    rng = np.random.default_rng(1)
    betas = rng.normal(0, 1, 6) * np.array([1.4, -1.1, 0.8, -0.6, 0.4, -0.3])
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for k, b in enumerate(betas):
        coef = b * Cs / (Cs + 0.5)  # shrink toward 0 as C->0
        ax.plot(Cs, coef, label=f"feature {k+1}")
    ax.set_xscale("log")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("C  (low = strong L2 shrinkage →,  high = weak →)")
    ax.set_ylabel("coefficient value")
    ax.set_title("SYNTH · Logistic regression regularization path")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    save(fig, "synth_logreg_path.png")


def fig_synth_dropout():
    drop = np.linspace(0.0, 0.7, 15)
    train = 0.98 - 0.18 * drop
    val = 0.62 + 0.45 * drop - 0.85 * drop ** 2
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(drop, train, "o-", color="#999999", label="train acc")
    ax.plot(drop, val, "o-", color=C["cnn"], label="validation acc")
    best = drop[np.argmax(val)]
    ax.axvline(best, ls="--", color=C["cnn"], lw=1)
    ax.text(best + 0.02, 0.6, f"best dropout≈{best:.2f}", color=C["cnn"], fontsize=9)
    ax.set_xlabel("dropout rate")
    ax.set_ylabel("accuracy")
    ax.set_title("SYNTH · Dropout closes the train–val gap (LSTM / CNN / EEGNet)")
    ax.legend(frameon=False, loc="lower center")
    save(fig, "synth_dropout.png")


def fig_synth_riemannian_shrinkage():
    shr = np.linspace(0, 1, 40)
    auc = 0.52 + 0.06 * np.sqrt(np.clip(shr, 0, 1)) - 0.05 * shr ** 2
    cond = 50 * np.exp(-4 * shr) + 2  # condition number drops with shrinkage
    fig, ax1 = plt.subplots(figsize=(6.6, 4.0))
    l1, = ax1.plot(shr, auc, color=C["riemannian"], label="held-out AUC")
    ax1.set_xlabel("covariance shrinkage intensity")
    ax1.set_ylabel("ROC-AUC", color=C["riemannian"])
    ax1.tick_params(axis="y", labelcolor=C["riemannian"])
    ax2 = ax1.twinx()
    ax2.grid(False)
    l2, = ax2.plot(shr, cond, color="#4C72B0", ls="--",
                   label="covariance condition number")
    ax2.set_ylabel("condition number (lower = stabler)", color="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.set_title("SYNTH · Riemannian: shrinkage stabilizes the covariance")
    ax1.legend(handles=[l1, l2], frameon=False, loc="center right", fontsize=9)
    save(fig, "synth_riemannian_shrinkage.png")


def fig_synth_cv_schedule():
    """Illustrate the feature-selection funnel (counts are nominal)."""
    stages = ["raw\nfeatures", "corr\ndrop", "ANOVA\nk-best",
              "stability\nselection", "gain\nprune", "SHAP\nprune"]
    counts = [2400, 1600, 500, 150, 90, 70]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    xs = np.arange(len(stages))
    ax.bar(xs, counts, color=plt.cm.Greens(np.linspace(0.4, 0.9, len(stages))),
           edgecolor="black", linewidth=0.5)
    for x, c in zip(xs, counts):
        ax.text(x, c + 30, f"{c}", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(stages)
    ax.set_ylabel("# features kept (nominal)")
    ax.set_title("SYNTH · In-fold feature-selection funnel (XGB path)")
    save(fig, "synth_feature_funnel.png")


if __name__ == "__main__":
    fig_real_auc_ci()
    fig_real_window_effect()
    fig_real_participant_heatmap()
    fig_real_eegnet_vs_cnn()
    fig_real_time_occlusion()
    fig_synth_xgb_depth()
    fig_synth_xgb_lr()
    fig_synth_svm_cgamma()
    fig_synth_logreg_path()
    fig_synth_dropout()
    fig_synth_riemannian_shrinkage()
    fig_synth_cv_schedule()
    print("\nAll figures written to", OUT.relative_to(ROOT))
