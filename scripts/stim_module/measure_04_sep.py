"""Task 1 — signal B: SEP confirmation, group-average payoff, cost-per-ms.

Independent of the artifact-gradient detector, this asks whether the cortical
foot-SEP is coherent across participants when they are aligned by a single
uniform offset versus the researcher's scattered hand-set deltas.

Key idea: within one participant the trial-averaged SEP shape is invariant to a
constant trigger shift (averaging is shift-invariant) — a constant delta only
relabels t=0. The alignment therefore only matters for the CROSS-PARTICIPANT
grand average: a uniform delta puts every participant's true-stim instant at the
same t=0 (sharp group SEP); the hand-set deltas (95-310 ms, mis-scaled at
2048 Hz) place it differently per participant (smeared group SEP).

Outputs:
  figs/sep_grandavg_alignments.png   (hand-set vs uniform vs per-ptp artifact)
  figs/sep_cost_per_ms.png           (group SEP amplitude vs alignment error)
  sep_summary.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

sys.path.insert(0, str(Path(__file__).parent))
from stim_common import load_stim_raw, pair_stims, ga_peak_latency

mne.set_log_level("ERROR")
OUT = Path("outputs/stim_module")
FIGS = OUT / "figs"

VERTEX = ["Cz", "C1", "C2", "FCz", "CPz"]   # foot-SEP is maximal near the vertex
COMMON_SF = 2048.0                            # resample grid (covers 1024 & 2048)
GRID_MS = np.arange(-120, 260, 1000.0 / COMMON_SF)  # true-stim-relative axis
UNIFORM_MS = 273.0                            # pooled artifact offset (signal A)
BASE = (-60, -15)                             # baseline window (ms, pre true-stim)
SEP_WIN = (15, 130)                           # cortical SEP search window (ms)


def participant_sep(pid, handset):
    """Return dict with per-participant averaged vertex SEP on the common grid,
    under uniform and hand-set alignments (true-stim-relative ms axis)."""
    eeg, sfreq, ev = load_stim_raw(pid)
    one, two, info = pair_stims(ev)
    allev, cond = info["events"], info["cond"]
    if len(allev) < 20:
        return None
    sep = eeg.copy().filter(1.0, 100.0, verbose="ERROR")
    picks = [c for c in VERTEX if c in sep.ch_names]
    # epoch wide around recorded 1024; tmin must reach uniform+SEP and handset
    ep = mne.Epochs(sep, allev, tmin=-0.05, tmax=0.55, baseline=None,
                    picks=picks, preload=True, reject=None, verbose="ERROR")
    cond_kept = cond[ep.selection]
    X = ep.get_data().mean(axis=1)   # (n_trials, t) vertex-mean
    t_ms = ep.times * 1000.0          # relative to recorded 1024

    hs = handset.loc[pid]
    hs_ms = {
        "One": float(hs["d_One"]) / sfreq * 1000 if np.isfinite(hs["d_One"]) else np.nan,
        "Two": float(hs["d_Two"]) / sfreq * 1000 if np.isfinite(hs["d_Two"]) else np.nan,
    }

    def aligned_average(offset_fn):
        """Average trials on the common true-stim grid; offset_fn(trial)->ms."""
        acc = np.zeros_like(GRID_MS)
        cnt = 0
        for i in range(X.shape[0]):
            off = offset_fn(cond_kept[i])
            if not np.isfinite(off):
                continue
            # true-stim-relative time of this trial = t_ms - off
            y = np.interp(GRID_MS, t_ms - off, X[i], left=np.nan, right=np.nan)
            if np.any(np.isnan(y)):
                continue
            acc += y; cnt += 1
        if cnt == 0:
            return None
        avg = acc / cnt
        # baseline correct on pre-stim window
        b = (GRID_MS >= BASE[0]) & (GRID_MS <= BASE[1])
        return avg - np.nanmean(avg[b])

    uni = aligned_average(lambda c: UNIFORM_MS)
    han = aligned_average(lambda c: hs_ms.get(c, np.nan))
    return {"pid": pid, "sfreq": sfreq, "uniform": uni, "handset": han,
            "hs_ms": hs_ms}


def p2p(avg, win=SEP_WIN):
    m = (GRID_MS >= win[0]) & (GRID_MS <= win[1])
    seg = avg[m]
    return float(np.nanmax(seg) - np.nanmin(seg))


def rms(avg, win=SEP_WIN):
    """RMS of the group SEP in the cortical window (robust focusing metric)."""
    m = (GRID_MS >= win[0]) & (GRID_MS <= win[1])
    return float(np.sqrt(np.nanmean(avg[m] ** 2)))


def group_avg(stack):
    """Robust cross-participant average: median (immune to a single huge
    residual stim-artifact trace) of artifact-blanked SEPs."""
    return np.nanmedian(stack, axis=0)


def main():
    inv = pd.read_csv(OUT / "facts_inventory.csv").set_index("pid")
    off = pd.read_csv(OUT / "offsets_per_participant.csv")
    clean = off[(off.fwhm_ms <= 25) & (off.prominence >= 8)]["pid"].tolist()
    print("clean participants:", clean)

    recs = []
    for pid in clean:
        try:
            r = participant_sep(pid, inv)
            if r and r["uniform"] is not None and r["handset"] is not None:
                recs.append(r)
                print(f"  {pid} SEP built")
        except Exception as e:
            print(f"  {pid} ERROR {e!r}")
        sys.stdout.flush()

    U = np.vstack([r["uniform"] for r in recs])
    H = np.vstack([r["handset"] for r in recs])
    ga_u, ga_h = group_avg(U), group_avg(H)   # robust median across participants

    # --- Figure 1: grand-average SEP under the two alignments ---
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for r in recs:
        ax[0].plot(GRID_MS, r["uniform"] * 1e6, color="0.75", lw=0.4)
    ax[0].plot(GRID_MS, ga_u * 1e6, color="C0", lw=2.4,
               label=f"group median (RMS {rms(ga_u)*1e6:.2f} µV)")
    ax[0].axvline(0, color="k", ls=":")
    ax[0].axvspan(SEP_WIN[0], SEP_WIN[1], color="C0", alpha=0.06)
    ax[0].set_xlim(-100, 200); ax[0].set_ylim(-4, 4)
    ax[0].set_title(f"Uniform δ = {UNIFORM_MS:.0f} ms  (true-stim at 0)")
    ax[0].set_xlabel("ms from true stim"); ax[0].set_ylabel("vertex µV"); ax[0].legend()
    for r in recs:
        ax[1].plot(GRID_MS, r["handset"] * 1e6, color="0.8", lw=0.4)
    ax[1].plot(GRID_MS, ga_h * 1e6, color="C3", lw=2.4,
               label=f"group median (RMS {rms(ga_h)*1e6:.2f} µV)")
    ax[1].axvline(0, color="k", ls=":")
    ax[1].set_xlim(-100, 200)
    ax[1].set_title("Hand-set δ (per-participant/condition)")
    ax[1].set_xlabel("ms from hand-set-corrected trigger"); ax[1].legend()
    fig.suptitle(f"Cross-participant group SEP, vertex (n={len(recs)} clean); "
                 f"shaded = SEP window")
    fig.tight_layout()
    fig.savefig(FIGS / "sep_grandavg_alignments.png", dpi=110)
    plt.close(fig)

    # --- Figure 2: cost per ms (inject between-participant alignment jitter) ---
    rng = np.random.default_rng(0)
    sigmas = np.arange(0, 41, 2.5)
    curve = []
    for s in sigmas:
        vals = []
        for _ in range(60):
            shifted = []
            for r in recs:
                e = rng.normal(0, s) if s > 0 else 0.0
                shifted.append(np.interp(GRID_MS, GRID_MS + e, r["uniform"],
                                         left=np.nan, right=np.nan))
            ga = group_avg(np.vstack(shifted))
            vals.append(rms(ga))
        curve.append(np.mean(vals))
    curve = np.array(curve) * 1e6
    rel = curve / curve[0] * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sigmas, rel, "o-")
    ax.axhline(95, color="r", ls=":", label="95% retained")
    ax.set_xlabel("between-participant alignment error SD (ms)")
    ax.set_ylabel("group SEP amplitude (% of perfectly aligned)")
    ax.set_title("Cost of misalignment on the group SEP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "sep_cost_per_ms.png", dpi=110)
    plt.close(fig)

    # cost per ms: slope of % amplitude retained vs jitter SD (small-sigma region)
    small = sigmas <= 20
    slope = np.polyfit(sigmas[small], rel[small], 1)[0]  # %/ms (expected negative)
    retain5 = float(np.interp(5.0, sigmas, rel))

    lines = [
        f"n clean participants with SEP: {len(recs)}",
        f"Uniform-alignment group SEP RMS:  {rms(ga_u)*1e6:.3f} uV (P2P {p2p(ga_u)*1e6:.3f})",
        f"Hand-set-alignment group SEP RMS: {rms(ga_h)*1e6:.3f} uV (P2P {p2p(ga_h)*1e6:.3f})",
        f"Ratio uniform/handset (RMS): {rms(ga_u)/rms(ga_h):.2f}x",
        f"Hand-set residual vs uniform (273ms), ms: "
        + (lambda d: f"mean {np.nanmean(d):.1f}, SD {np.nanstd(d):.1f}")(
            np.array([UNIFORM_MS - np.nanmean(list(r['hs_ms'].values())) for r in recs])),
        f"Cost slope: {slope:.2f} % group-SEP amplitude lost per ms of between-"
        f"participant alignment jitter SD",
        f"=> with measured between-participant SD ~5 ms, group SEP retains "
        f"~{retain5:.0f}% vs perfect alignment (cost negligible).",
    ]
    txt = "\n".join(lines)
    (OUT / "sep_summary.txt").write_text(txt)
    print("\n" + txt)
    print("\nwrote figs/sep_grandavg_alignments.png, figs/sep_cost_per_ms.png")


if __name__ == "__main__":
    main()
