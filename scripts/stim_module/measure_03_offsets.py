"""Task 1 core measurement: per-participant trigger->artifact offset.

Primary estimator = grand-average artifact-saliency peak latency with a
bootstrap CI over trials (robust for a sharp, stimulus-locked artifact). Also
reports the GA peak FWHM (jitter proxy) and the per-condition (One/Two) offsets,
and compares to the researcher's hand-set delta.

Outputs:
  outputs/stim_module/offsets_per_participant.csv
  outputs/stim_module/figs/offset_<pid>.png
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
from stim_common import (load_stim_raw, pair_stims, ga_peak_latency)

mne.set_log_level("ERROR")
OUT = Path("outputs/stim_module")
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

SEARCH_MS = (120, 340)
TMIN, TMAX = -0.10, 0.45


def analyze(pid, handset, make_fig=True):
    eeg, sfreq, ev = load_stim_raw(pid)
    one, two, info = pair_stims(ev)
    allev = info["events"]
    cond = info["cond"]
    if len(allev) < 10:
        return None

    bb = eeg.copy().filter(1, None, verbose="ERROR")
    ep = mne.Epochs(bb, allev, tmin=TMIN, tmax=TMAX, baseline=None,
                    preload=True, reject=None, verbose="ERROR")
    cond_kept = cond[ep.selection]
    X = ep.get_data()

    res = ga_peak_latency(X, ep.times, search_ms=SEARCH_MS)
    resO = ga_peak_latency(X[cond_kept == "One"], ep.times, search_ms=SEARCH_MS)
    resT = ga_peak_latency(X[cond_kept == "Two"], ep.times, search_ms=SEARCH_MS)

    hs = handset.loc[pid] if pid in handset.index else None
    hs_one = float(hs["d_One"]) if hs is not None and np.isfinite(hs["d_One"]) else np.nan
    hs_two = float(hs["d_Two"]) if hs is not None and np.isfinite(hs["d_Two"]) else np.nan
    hs_ctrl = float(hs["d_Control"]) if hs is not None and np.isfinite(hs["d_Control"]) else np.nan
    hs_one_ms = hs_one / sfreq * 1000 if np.isfinite(hs_one) else np.nan
    hs_two_ms = hs_two / sfreq * 1000 if np.isfinite(hs_two) else np.nan

    summ = {
        "pid": pid, "sfreq": sfreq, "n_trials": len(X),
        "offset_ms": round(res["peak_ms"], 2),
        "onset_ms": round(res["onset_ms"], 2),
        "ci_lo": round(res["ci_lo"], 2), "ci_hi": round(res["ci_hi"], 2),
        "boot_sd_ms": round(res["boot_sd"], 2),
        "fwhm_ms": round(res["fwhm_ms"], 2),
        "prominence": round(res["prominence"], 1),
        "offset_samp": round(res["peak_ms"] / 1000 * sfreq, 1),
        "onset_samp": round(res["onset_ms"] / 1000 * sfreq, 1),
        "offset_ms_One": round(resO["peak_ms"], 2),
        "offset_ms_Two": round(resT["peak_ms"], 2),
        "handset_One_samp": hs_one, "handset_Two_samp": hs_two,
        "handset_Control_samp": hs_ctrl,
        "handset_One_ms": round(hs_one_ms, 1) if np.isfinite(hs_one_ms) else np.nan,
        "handset_Two_ms": round(hs_two_ms, 1) if np.isfinite(hs_two_ms) else np.nan,
        "resid_One_ms": round(res["peak_ms"] - hs_one_ms, 1) if np.isfinite(hs_one_ms) else np.nan,
        "resid_Two_ms": round(res["peak_ms"] - hs_two_ms, 1) if np.isfinite(hs_two_ms) else np.nan,
    }

    if make_fig:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        ax[0].plot(res["ga_times_ms"], res["ga"] * 1e6, color="k", lw=1)
        ax[0].axvspan(res["ci_lo"], res["ci_hi"], color="r", alpha=0.25)
        ax[0].axvline(res["peak_ms"], color="r", ls="--",
                      label=f"data {res['peak_ms']:.0f} ms (FWHM {res['fwhm_ms']:.0f})")
        if np.isfinite(hs_one_ms):
            ax[0].axvline(hs_one_ms, color="g", ls=":", label=f"handset One {hs_one_ms:.0f}")
        if np.isfinite(hs_two_ms):
            ax[0].axvline(hs_two_ms, color="m", ls=":", label=f"handset Two {hs_two_ms:.0f}")
        ax[0].set_xlim(-50, 400)
        ax[0].set_title(f"{pid} ({sfreq:.0f} Hz) artifact-saliency GA")
        ax[0].set_xlabel("ms from 1024 trigger"); ax[0].legend(fontsize=8)
        # One vs Two saliency overlay
        ax[1].plot(resO["ga_times_ms"], resO["ga"] * 1e6, label=f"One {resO['peak_ms']:.0f} ms")
        ax[1].plot(resT["ga_times_ms"], resT["ga"] * 1e6, label=f"Two {resT['peak_ms']:.0f} ms")
        ax[1].set_xlim(100, 360)
        ax[1].set_title("One vs Two artifact saliency"); ax[1].legend(fontsize=8)
        ax[1].set_xlabel("ms from 1024 trigger")
        fig.tight_layout()
        fig.savefig(FIGS / f"offset_{pid}.png", dpi=100)
        plt.close(fig)

    print(f"{pid}: {sfreq:.0f}Hz n={len(X)} peak={res['peak_ms']:.1f} onset={res['onset_ms']:.1f}ms "
          f"[{res['ci_lo']:.1f},{res['ci_hi']:.1f}] FWHM={res['fwhm_ms']:.0f} prom={res['prominence']:.0f} "
          f"One/Two={resO['peak_ms']:.0f}/{resT['peak_ms']:.0f} "
          f"handset={summ['handset_One_ms']}/{summ['handset_Two_ms']}ms")
    sys.stdout.flush()
    return summ


def main():
    handset = pd.read_csv(OUT / "facts_inventory.csv").set_index("pid")
    pids = sys.argv[1:]
    if not pids:
        inv = pd.read_csv(OUT / "facts_inventory.csv")
        pids = inv[(inv.has_bdf) & (inv.n_1024 >= 50)]["pid"].tolist()
    rows = []
    for pid in pids:
        try:
            su = analyze(pid, handset)
            if su:
                rows.append(su)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"{pid}: ERROR {e!r}")
    if not rows:
        return
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "offsets_per_participant.csv", index=False)
    # "Clean" = sharp, detectable artifact (FWHM small AND high prominence).
    clean = summ[(summ.fwhm_ms <= 25) & (summ.prominence >= 8)]
    print("\n=== POOLED (data-driven artifact offset) ===")
    print(f"all n={len(summ)}; clean n={len(clean)} "
          f"(excluded: {sorted(set(summ.pid)-set(clean.pid))})")
    for label, d in (("PEAK", "offset_ms"), ("ONSET", "onset_ms")):
        m = clean[d]
        print(f"  [{label}] clean mean={m.mean():.2f} ms  sd(between)={m.std():.2f} ms  "
              f"median={m.median():.2f}  range=[{m.min():.1f},{m.max():.1f}]")
    print(f"  median within-ptp boot_sd={clean['boot_sd_ms'].median():.2f} ms  "
          f"median FWHM={clean['fwhm_ms'].median():.1f} ms")
    print("wrote", OUT / "offsets_per_participant.csv")


if __name__ == "__main__":
    main()
