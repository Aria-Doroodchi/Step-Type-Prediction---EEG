"""Diagnostic figures for the 3-state classifier.

* 3×3 confusion heatmap (row-normalized recall)
* per-participant macro-OVR AUC + accuracy vs chance (0.5 / 0.333)
* per-condition grand-average vertex ERP (the 2 s window)
* grand-average vertex SEP per condition
* SEP-vs-window ablation bars (from several runs' rollups)

All figures land under ``outputs/state_module/figs/``. Functions degrade
gracefully if inputs are missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

from ..io import epochs_path, sep_epochs_path, outputs_root, ensure_dir
from ..logging_utils import get_logger
from ..models.evaluate import CLASS_NAMES

log = get_logger(__name__)
mne.set_log_level("ERROR")

VERTEX = ["Cz", "C1", "C2", "FCz", "CPz"]


def figs_dir(cfg: dict) -> Path:
    return ensure_dir(outputs_root(cfg) / "state_module" / "figs")


def confusion_heatmap(metrics: pd.DataFrame, out: Path, title: str = "") -> None:
    cm = np.zeros((3, 3), dtype=float)
    for i, ti in enumerate(CLASS_NAMES):
        for j, tj in enumerate(CLASS_NAMES):
            col = f"cm_{ti}_{tj}"
            if col in metrics:
                cm[i, j] = metrics[col].sum()
    if cm.sum() == 0:
        log.warning("confusion_heatmap: no cm_* columns; skipping.")
        return
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j])})",
                    ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_title(title or "3-class confusion (row-normalized recall)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log.info("wrote %s", out)


def per_participant_bars(metrics: pd.DataFrame, out: Path) -> None:
    if "participant_id" not in metrics:
        return
    g = metrics.groupby("participant_id").agg(
        auc=("macro_auc", "mean"), acc=("overall_accuracy", "mean")).reset_index()
    g = g.sort_values("auc")
    x = np.arange(len(g))
    fig, ax = plt.subplots(2, 1, figsize=(max(7, len(g) * 0.45), 7), sharex=True)
    ax[0].bar(x, g["auc"], color="C0")
    ax[0].axhline(0.5, color="r", ls=":", label="chance (0.5)")
    ax[0].set_ylabel("macro-OVR AUC")
    ax[0].set_ylim(0, 1)
    ax[0].legend(loc="lower right")
    ax[1].bar(x, g["acc"], color="C2")
    ax[1].axhline(1 / 3, color="r", ls=":", label="chance (0.333)")
    ax[1].set_ylabel("overall accuracy")
    ax[1].set_ylim(0, 1)
    ax[1].legend(loc="lower right")
    ax[1].set_xticks(x, g["participant_id"], rotation=90)
    fig.suptitle("Per-participant 3-class performance vs chance")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log.info("wrote %s", out)


def _grand_average(cfg: dict, pids: list[str], path_fn, picks) -> tuple | None:
    """Return (times, {cond: mean_waveform}) averaging vertex channels across pids."""
    acc: dict[str, list] = {c: [] for c in CLASS_NAMES}
    times = None
    for pid in pids:
        for cond in CLASS_NAMES:
            p = path_fn(cfg, pid, cond)
            if not Path(p).exists():
                continue
            try:
                ep = mne.read_epochs(str(p), preload=True, verbose="ERROR")
            except Exception:
                continue
            chans = [c for c in picks if c in ep.ch_names]
            if not chans:
                continue
            data = ep.get_data(picks=chans).mean(axis=1).mean(axis=0) * 1e6
            acc[cond].append((ep.times, data))
            times = ep.times
    if times is None:
        return None
    out = {}
    for cond, lst in acc.items():
        if lst:
            grid = times
            stacked = [np.interp(grid, t, d) for t, d in lst]
            out[cond] = np.mean(stacked, axis=0)
    return times, out


def condition_erp(cfg: dict, pids: list[str], out: Path) -> None:
    res = _grand_average(cfg, pids, epochs_path, VERTEX)
    if res is None:
        log.warning("condition_erp: no epochs; skipping.")
        return
    times, waves = res
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cond, w in waves.items():
        ax.plot(times, w, label=cond, lw=1.8)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("vertex CSD (a.u.)")
    ax.set_title("Per-condition grand-average vertex window ERP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log.info("wrote %s", out)


def sep_grand_average(cfg: dict, pids: list[str], out: Path) -> None:
    res = _grand_average(cfg, pids, sep_epochs_path, VERTEX)
    if res is None:
        log.warning("sep_grand_average: no SEP epochs; skipping.")
        return
    times, waves = res
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cond, w in waves.items():
        ax.plot(times * 1000, w, label=cond, lw=1.8)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.axvspan(15, 130, color="0.85", alpha=0.4, label="SEP window")
    ax.set_xlabel("ms from true stim")
    ax.set_ylabel("vertex µV")
    ax.set_title("Grand-average vertex foot-SEP per condition")
    ax.set_xlim(-50, 200)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log.info("wrote %s", out)


def ablation_bars(rollups: dict[str, pd.DataFrame], out: Path) -> None:
    """rollups: {label -> rollup_df}. Plots macro-AUC mean ± ci per variant."""
    labels, means, errs = [], [], []
    for label, rdf in rollups.items():
        if rdf is None or rdf.empty or "macro_auc_mean" not in rdf:
            continue
        labels.append(label)
        means.append(float(rdf["macro_auc_mean"].iloc[0]))
        errs.append(float(rdf.get("macro_auc_ci95", pd.Series([0])).iloc[0]))
    if not labels:
        return
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.3), 4.5))
    ax.bar(x, means, yerr=errs, capsize=4, color="C0")
    ax.axhline(0.5, color="r", ls=":", label="chance (0.5)")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("cohort macro-OVR AUC")
    ax.set_ylim(0, 1)
    ax.set_title("SEP-vs-window ablation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log.info("wrote %s", out)


def build_report(cfg: dict, metrics: pd.DataFrame, pids: list[str],
                 tag: str = "baseline") -> None:
    d = figs_dir(cfg)
    confusion_heatmap(metrics, d / f"confusion_{tag}.png",
                      title=f"3-class confusion ({tag})")
    per_participant_bars(metrics, d / f"per_participant_{tag}.png")
    condition_erp(cfg, pids, d / "condition_window_erp.png")
    sep_grand_average(cfg, pids, d / "sep_grandaverage.png")
