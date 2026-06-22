"""Task 2 stage 2: SEP evoked plots + t-tests (P2P, P50, N90).

Reads cached per-participant evokeds (cache_06_evokeds.py), measures the SEP
components per participant at Cz, and runs paired t-tests across participants.

Part A  (condition-level):   standing / straight(4 pooled) / diagonal(4 pooled)
Part B1 (within-path order):  straight 1v2v3v4 ;  diagonal 1v2v3v4
Part B2 (matched order):      straight-N vs diagonal-N  (N=1..4)

Components measured per participant from the cell's averaged SEP at Cz:
  P50  = max amplitude in P50_WIN   (positive vertex component)
  N90  = min amplitude in N90_WIN   (negative vertex component)
  P2P  = P50 - N90                  (peak-to-peak)
Cells with < MIN_TRIALS kept epochs are set NaN; t-tests use pairwise-complete
participants (paired / within-subject design).
"""
from __future__ import annotations

import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EV = Path("outputs/stim_module/evokeds")
OUT = Path("outputs/stim_module")
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

CHANNEL = "Cz"
ROI = ["Cz", "C1", "C2", "FCz", "CPz"]
# Windows bracket the grand-average vertex components (avg-ref): a small early
# positive (P50, ~40 ms) and the dominant negative trough (N90, ~70-90 ms).
P50_WIN = (25, 65)
N90_WIN = (65, 115)
MIN_TRIALS = 15
COND_CELLS = ["standing", "straight", "diagonal"]
ORDER_CELLS = [f"{p}_{k}" for p in ("straight", "diagonal") for k in (1, 2, 3, 4)]

COND_COLORS = {"standing": "0.35", "straight": "C0", "diagonal": "C3"}


def load_all():
    recs = {}
    for fp in sorted(EV.glob("P*.npz")):
        d = np.load(fp, allow_pickle=True)
        ch = list(d["ch_names"])
        counts = d["counts"][0]
        grid = d["grid_ms"]
        cells = {}
        for key in d.files:
            if key.startswith("data__"):
                cells[key[6:]] = d[key]
        recs[fp.stem] = {"ch": ch, "counts": counts, "grid": grid,
                         "cells": cells, "meta": d["meta"][0]}
    return recs


def trace(rec, cell, channel=CHANNEL):
    """Return (grid_ms, µV trace) for a channel (or ROI mean) or None."""
    if cell not in rec["cells"] or rec["counts"].get(cell, 0) < MIN_TRIALS:
        return None
    data = rec["cells"][cell]
    chs = channel if isinstance(channel, list) else [channel]
    idx = [rec["ch"].index(c) for c in chs if c in rec["ch"]]
    if not idx:
        return None
    return rec["grid"], data[idx].mean(axis=0) * 1e6


def measure(grid, y):
    p50m = (grid >= P50_WIN[0]) & (grid <= P50_WIN[1])
    n90m = (grid >= N90_WIN[0]) & (grid <= N90_WIN[1])
    p50 = float(np.max(y[p50m])); p50_lat = float(grid[p50m][np.argmax(y[p50m])])
    n90 = float(np.min(y[n90m])); n90_lat = float(grid[n90m][np.argmin(y[n90m])])
    return {"p50": p50, "p50_lat": p50_lat, "n90": n90, "n90_lat": n90_lat,
            "p2p": p50 - n90}


def build_measures(recs, channel=CHANNEL):
    rows = []
    for pid, rec in recs.items():
        for cell in COND_CELLS + ORDER_CELLS:
            tr = trace(rec, cell, channel)
            if tr is None:
                continue
            m = measure(*tr)
            m.update(pid=pid, cell=cell, n=int(rec["counts"].get(cell, 0)))
            rows.append(m)
    return pd.DataFrame(rows)


def paired_t(df, cell_a, cell_b, measure_col):
    w = df.pivot_table(index="pid", columns="cell", values=measure_col)
    if cell_a not in w or cell_b not in w:
        return None
    sub = w[[cell_a, cell_b]].dropna()
    if len(sub) < 3:
        return None
    a, b = sub[cell_a].values, sub[cell_b].values
    t, p = stats.ttest_rel(a, b)
    d = (a - b)
    # paired Cohen's d (dz)
    dz = d.mean() / (d.std(ddof=1) + 1e-20)
    return {"a": cell_a, "b": cell_b, "measure": measure_col, "n": len(sub),
            "mean_a": round(a.mean(), 3), "mean_b": round(b.mean(), 3),
            "mean_diff": round(d.mean(), 3), "t": round(float(t), 3),
            "df": len(sub) - 1, "p": float(p), "dz": round(float(dz), 3)}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values for a family."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        prev = max(prev, min(val, 1.0))
        adj[idx] = prev
    return adj


def run_family(df, pairs, label):
    rows = []
    for a, b in pairs:
        for mc in ("p2p", "p50", "n90"):
            r = paired_t(df, a, b, mc)
            if r:
                rows.append(r)
    fam = pd.DataFrame(rows)
    if len(fam):
        # Holm-correct within each measure family
        fam["p_holm"] = np.nan
        for mc in fam["measure"].unique():
            m = fam["measure"] == mc
            fam.loc[m, "p_holm"] = holm(fam.loc[m, "p"].values)
        fam["p"] = fam["p"].round(4); fam["p_holm"] = fam["p_holm"].round(4)
        fam["sig"] = np.where(fam["p_holm"] < 0.05, "*", "")
    fam.to_csv(OUT / f"ttests_{label}.csv", index=False)
    return fam


# ----------------------- plotting -----------------------
def ga_trace(recs, cell, channel=CHANNEL):
    ys, n = [], 0
    grid = None
    for rec in recs.values():
        tr = trace(rec, cell, channel)
        if tr is None:
            continue
        grid, y = tr
        ys.append(y); n += 1
    if not ys:
        return None, None, None, 0
    Y = np.vstack(ys)
    return grid, Y.mean(0), Y.std(0) / np.sqrt(len(Y)), n


def plot_overlay(recs, cells, colors, title, fname, channel=CHANNEL):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cell in cells:
        grid, mu, sem, n = ga_trace(recs, cell, channel)
        if grid is None:
            continue
        c = colors.get(cell, None)
        ax.plot(grid, mu, color=c, lw=2, label=f"{cell} (n={n})")
        ax.fill_between(grid, mu - sem, mu + sem, color=c, alpha=0.2)
    ax.axvspan(*P50_WIN, color="green", alpha=0.06)
    ax.axvspan(*N90_WIN, color="purple", alpha=0.06)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("ms from true stim"); ax.set_ylabel(f"{channel} µV")
    ax.set_title(title); ax.legend(fontsize=8)
    ax.invert_yaxis()   # SEP convention: positive down
    fig.tight_layout(); fig.savefig(FIGS / fname, dpi=120); plt.close(fig)


def main():
    recs = load_all()
    print(f"loaded {len(recs)} participants:", sorted(recs))
    df = build_measures(recs, CHANNEL)
    df.to_csv(OUT / "sep_measures_per_participant.csv", index=False)
    print(f"measures: {len(df)} rows over {df.pid.nunique()} participants")

    # ---- Part A figure + t-tests ----
    plot_overlay(recs, COND_CELLS, COND_COLORS,
                 f"Part A — condition SEPs at {CHANNEL} (grand-avg ±SEM)",
                 "sep_partA_conditions.png")
    famA = run_family(df, list(combinations(COND_CELLS, 2)), "partA")

    # ---- Part B figures ----
    order_colors = {f"straight_{k}": plt.cm.Blues(0.4 + 0.15 * k) for k in (1, 2, 3, 4)}
    order_colors.update({f"diagonal_{k}": plt.cm.Reds(0.4 + 0.15 * k) for k in (1, 2, 3, 4)})
    plot_overlay(recs, [f"straight_{k}" for k in (1, 2, 3, 4)], order_colors,
                 f"Part B — straight stepping by stim order at {CHANNEL}",
                 "sep_partB_straight_order.png")
    plot_overlay(recs, [f"diagonal_{k}" for k in (1, 2, 3, 4)], order_colors,
                 f"Part B — diagonal stepping by stim order at {CHANNEL}",
                 "sep_partB_diagonal_order.png")
    # matched-order 4-panel
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    for k, ax in zip((1, 2, 3, 4), axes):
        for path, c in (("straight", "C0"), ("diagonal", "C3")):
            grid, mu, sem, n = ga_trace(recs, f"{path}_{k}")
            if grid is None:
                continue
            ax.plot(grid, mu, color=c, lw=2, label=f"{path} (n={n})")
            ax.fill_between(grid, mu - sem, mu + sem, color=c, alpha=0.2)
        ax.axvline(0, color="k", ls=":", lw=0.8); ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"stim #{k}"); ax.set_xlabel("ms from true stim")
        ax.invert_yaxis(); ax.legend(fontsize=8)
    axes[0].set_ylabel(f"{CHANNEL} µV")
    fig.suptitle("Part B — matched stim order: straight vs diagonal")
    fig.tight_layout(); fig.savefig(FIGS / "sep_partB_matched_order.png", dpi=120)
    plt.close(fig)

    # within-path order pairs
    within_pairs = ([(f"straight_{i}", f"straight_{j}") for i, j in combinations((1, 2, 3, 4), 2)]
                    + [(f"diagonal_{i}", f"diagonal_{j}") for i, j in combinations((1, 2, 3, 4), 2)])
    famB1 = run_family(df, within_pairs, "partB_within")
    matched_pairs = [(f"straight_{k}", f"diagonal_{k}") for k in (1, 2, 3, 4)]
    famB2 = run_family(df, matched_pairs, "partB_matched")

    # ---- console summary ----
    def show(fam, name):
        print(f"\n=== {name} (paired t-tests, Cz; p_holm within each measure) ===")
        if not len(fam):
            print("  (no data)"); return
        for _, r in fam.iterrows():
            print(f"  {r['a']:>11} vs {r['b']:<11} {r['measure']:>4}: "
                  f"diff={r['mean_diff']:+.2f}µV t({r['df']})={r['t']:+.2f} "
                  f"p={r['p']:.4f} p_holm={r['p_holm']:.4f} dz={r['dz']:+.2f} "
                  f"n={r['n']} {r['sig']}")
    show(famA, "PART A: standing/straight/diagonal")
    show(famB2, "PART B2: matched order straight vs diagonal")
    show(famB1, "PART B1: within-path order")
    print("\nwrote sep_measures_per_participant.csv, ttests_partA/partB_within/"
          "partB_matched.csv, and 4 figures.")


if __name__ == "__main__":
    main()
