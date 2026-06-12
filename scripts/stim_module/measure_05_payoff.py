"""Task 1 — payoff demonstration + deliverable summary figures.

The cortical foot-SEP is too weak to serve as a clean group metric here, so the
honest, high-SNR demonstration that a uniform offset beats the hand-set deltas
is at the level of the stim ARTIFACT (the ground truth): align every clean
participant's artifact-saliency by (a) the uniform offset and (b) the hand-set
delta, average across participants, and compare the coherence of the group peak.

Also emits:
  figs/offset_distribution.png   — per-participant offset histogram (+ hand-set)
  figs/optimum_vs_handset.png    — data-driven optimum vs researcher hand-set
  figs/artifact_coherence.png    — group artifact peak: uniform vs hand-set
  figs/coherence_vs_jitter.png   — group peak amplitude vs alignment-error SD
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
from stim_common import load_stim_raw, pair_stims, ga_peak_latency, ga_saliency

mne.set_log_level("ERROR")
OUT = Path("outputs/stim_module")
FIGS = OUT / "figs"

COMMON_SF = 2048.0
GRID_MS = np.arange(-150, 200, 1000.0 / COMMON_SF)
UNIFORM_MS = 273.0
TMIN, TMAX = -0.10, 0.45


def participant_saliency(pid):
    """Per-participant trial-averaged artifact saliency on the recorded-1024 axis."""
    eeg, sfreq, ev = load_stim_raw(pid)
    one, two, info = pair_stims(ev)
    allev, cond = info["events"], info["cond"]
    bb = eeg.copy().filter(1, None, verbose="ERROR")
    ep = mne.Epochs(bb, allev, tmin=TMIN, tmax=TMAX, baseline=None,
                    preload=True, reject=None, verbose="ERROR")
    sal = ga_saliency(ep.get_data())          # (n, t-1), median across ch
    ga = sal.mean(axis=0)
    t_ms = ep.times[1:] * 1000.0
    # normalise so each participant contributes equally to the coherence avg
    ga = (ga - np.median(ga)) / (np.max(ga) - np.median(ga) + 1e-20)
    cond_kept = cond[ep.selection]
    return ga, t_ms, sfreq


def main():
    inv = pd.read_csv(OUT / "facts_inventory.csv").set_index("pid")
    off = pd.read_csv(OUT / "offsets_per_participant.csv")
    clean = off[(off.fwhm_ms <= 25) & (off.prominence >= 8)].copy()
    clean_ids = clean.pid.tolist()
    print("clean:", clean_ids)

    # ---- summary fig A: offset distribution ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(clean.offset_ms, bins=np.arange(255, 285, 1.5), color="C0",
            alpha=0.85, label=f"data-driven (clean n={len(clean)})")
    hs_all = pd.concat([inv["d_One"] / inv["sfreq"] * 1000,
                        inv["d_Two"] / inv["sfreq"] * 1000]).dropna()
    ax.hist(hs_all.dropna(), bins=np.arange(80, 330, 12), color="C3",
            alpha=0.4, label="hand-set One/Two (all)")
    ax.axvline(clean.offset_ms.mean(), color="C0", ls="--",
               label=f"uniform {clean.offset_ms.mean():.1f} ms (SD {clean.offset_ms.std():.1f})")
    ax.set_xlabel("trigger→true-stim offset (ms)"); ax.set_ylabel("count")
    ax.set_title("Offset distribution: tight data-driven vs scattered hand-set")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGS / "offset_distribution.png", dpi=110)
    plt.close(fig)

    # ---- summary fig B: optimum vs hand-set scatter ----
    fig, ax = plt.subplots(figsize=(6.5, 6))
    m = off[(off.fwhm_ms <= 25) & (off.prominence >= 8)]
    ax.errorbar(m.handset_One_ms, m.offset_ms,
                yerr=[m.offset_ms - m.ci_lo, m.ci_hi - m.offset_ms],
                fmt="o", color="C0", label="One hand-set", alpha=0.8)
    ax.scatter(m.handset_Two_ms, m.offset_ms, marker="s", color="C1",
               label="Two hand-set", alpha=0.8)
    lim = [80, 330]
    ax.plot(lim, lim, "k:", label="y=x")
    ax.axhline(m.offset_ms.mean(), color="C0", ls="--",
               label=f"uniform {m.offset_ms.mean():.0f} ms")
    ax.set_xlabel("researcher hand-set delta (ms)")
    ax.set_ylabel("data-driven artifact offset (ms)")
    ax.set_title("Data-driven optimum vs hand-set delta")
    ax.set_xlim(lim); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "optimum_vs_handset.png", dpi=110)
    plt.close(fig)

    # ---- artifact-coherence payoff ----
    recs = []
    for pid in clean_ids:
        try:
            ga, t_ms, sf = participant_saliency(pid)
            hs = inv.loc[pid]
            hs_one = float(hs["d_One"]) / sf * 1000 if np.isfinite(hs["d_One"]) else np.nan
            hs_two = float(hs["d_Two"]) / sf * 1000 if np.isfinite(hs["d_Two"]) else np.nan
            hs_ms = np.nanmean([hs_one, hs_two])
            recs.append({"pid": pid, "ga": ga, "t_ms": t_ms, "hs_ms": hs_ms})
            print("  coherence:", pid)
        except Exception as e:
            print("  ERR", pid, repr(e))
        sys.stdout.flush()

    def stack(offset_key):
        rows = []
        for r in recs:
            off_ms = UNIFORM_MS if offset_key == "uniform" else r["hs_ms"]
            if not np.isfinite(off_ms):
                continue
            y = np.interp(GRID_MS, r["t_ms"] - off_ms, r["ga"],
                          left=np.nan, right=np.nan)
            rows.append(y)
        return np.vstack(rows)

    Su, Sh = stack("uniform"), stack("handset")
    gu, gh = np.nanmean(Su, axis=0), np.nanmean(Sh, axis=0)

    def peak_fwhm(g):
        w = (GRID_MS >= -60) & (GRID_MS <= 60)
        seg = g[w]; pk = np.nanmax(seg); base = np.nanmedian(g)
        above = np.where(seg >= base + 0.5 * (pk - base))[0]
        fw = GRID_MS[w][above[-1]] - GRID_MS[w][above[0]] if len(above) else np.nan
        return pk, fw

    pu, fu = peak_fwhm(gu)
    ph, fh = peak_fwhm(gh)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(GRID_MS, gu, color="C0", lw=2.3,
            label=f"uniform δ: peak {pu:.2f}, FWHM {fu:.0f} ms")
    ax.plot(GRID_MS, gh, color="C3", lw=2.3,
            label=f"hand-set δ: peak {ph:.2f}, FWHM {fh:.0f} ms")
    ax.axvline(0, color="k", ls=":")
    ax.set_xlim(-120, 150)
    ax.set_xlabel("ms from alignment point"); ax.set_ylabel("normalised artifact saliency")
    ax.set_title(f"Group artifact coherence (n={Su.shape[0]} clean): "
                 f"uniform {pu/ph:.2f}× sharper than hand-set")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "artifact_coherence.png", dpi=110)
    plt.close(fig)

    # ---- cost of misalignment, by component width ----
    # Averaging N responses each mis-shifted by ~N(0,sigma) scales a Gaussian
    # component of std s_p by s_p/sqrt(s_p^2 + sigma^2). The razor-sharp artifact
    # (FWHM ~13 ms) is very jitter-sensitive; a real cortical SEP component (tens
    # of ms wide) is not. This is the practically relevant "cost per ms".
    sigmas = np.arange(0, 51, 1.0)
    fwhms = {"stim artifact (FWHM 13 ms)": 13,
             "early cortical SEP (FWHM 30 ms)": 30,
             "broad SEP/CNV (FWHM 60 ms)": 60}
    meas_sd = clean.offset_ms.std()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label, fwhm in fwhms.items():
        s_p = fwhm / 2.3548
        retain = s_p / np.sqrt(s_p ** 2 + sigmas ** 2) * 100
        ax.plot(sigmas, retain, lw=2, label=label)
        ax.plot(meas_sd, np.interp(meas_sd, sigmas, retain), "ko")
    ax.axvline(meas_sd, color="g", ls="--",
               label=f"measured between-ptp SD ({meas_sd:.1f} ms)")
    ax.axhline(95, color="r", ls=":", label="95% retained")
    ax.set_xlabel("between-participant alignment-error SD (ms)")
    ax.set_ylabel("averaged-component amplitude (% retained)")
    ax.set_title("Cost of misalignment depends on component width")
    ax.set_ylim(0, 102); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "coherence_vs_jitter.png", dpi=110)
    plt.close(fig)
    s_p30 = 30 / 2.3548
    sep_retain = s_p30 / np.sqrt(s_p30 ** 2 + meas_sd ** 2) * 100
    slope = 0.0  # reported via retention below

    lines = [
        f"clean n={len(clean)}; uniform offset mean={clean.offset_ms.mean():.2f} ms "
        f"SD={clean.offset_ms.std():.2f}; onset mean={clean.onset_ms.mean():.2f} ms",
        f"group artifact peak uniform/handset = {pu/ph:.2f}x; "
        f"FWHM uniform={fu:.0f} ms vs hand-set={fh:.0f} ms",
        f"hand-set residual vs uniform: mean "
        f"{np.nanmean([UNIFORM_MS - r['hs_ms'] for r in recs]):.1f} ms, "
        f"SD {np.nanstd([UNIFORM_MS - r['hs_ms'] for r in recs]):.1f} ms",
        f"cost: at measured between-ptp SD {meas_sd:.1f} ms, an early cortical "
        f"SEP component (FWHM 30 ms) retains {sep_retain:.1f}% of its amplitude "
        f"vs perfect alignment (negligible). Required precision is ~+-10 ms.",
    ]
    (OUT / "payoff_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print("\nwrote 4 figures + payoff_summary.txt")


if __name__ == "__main__":
    main()
