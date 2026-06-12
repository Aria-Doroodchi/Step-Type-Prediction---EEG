"""Task 1 — fact-finding pass 2: decode higher trigger codes + locate stim artifact.

For a couple of participants (one 1024 Hz, one 2048 Hz):
  1. Print every trigger code with counts and the timing of 2048/4096/2144
     relative to the surrounding 256/512/1024 events (is 2048 the audio onset?).
  2. Epoch the *raw* (lightly filtered) signal around each paired 1024 event over
     a wide window and (a) build the Cz SEP, (b) per-trial detect the stim
     artifact as the peak across-channel activity, and histogram the
     trigger->artifact latency.

Saves figures under outputs/stim_module/figs/ and prints stats.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

mne.set_log_level("ERROR")

RAW_ROOT = Path(
    "C:/Users/Ali D/OneDrive - The University of Western Ontario/MSc/Thesis/Data/Participants"
)
FIGS = Path("outputs/stim_module/figs")
FIGS.mkdir(parents=True, exist_ok=True)

MAPPING = {
    'A1': 'Fp1', 'A2': 'AF7', 'A3': 'AF3', 'A4': 'F1', 'A5': 'F3', 'A6': 'F5',
    'A7': 'F7', 'A8': 'FT7', 'A9': 'FC5', 'A10': 'FC3', 'A11': 'FC1', 'A12': 'C1',
    'A13': 'C3', 'A14': 'C5', 'A15': 'T7', 'A16': 'TP7', 'A17': 'CP5', 'A18': 'CP3',
    'A19': 'CP1', 'A20': 'P1', 'A21': 'P3', 'A22': 'P5', 'A23': 'P7', 'A24': 'P9',
    'A25': 'PO7', 'A26': 'PO3', 'A27': 'O1', 'A28': 'Iz', 'A29': 'Oz', 'A30': 'POz',
    'A31': 'Pz', 'A32': 'CPz', 'B1': 'Fpz', 'B2': 'Fp2', 'B3': 'AF8', 'B4': 'AF4',
    'B5': 'AFz', 'B6': 'Fz', 'B7': 'F2', 'B8': 'F4', 'B9': 'F6', 'B10': 'F8',
    'B11': 'FT8', 'B12': 'FC6', 'B13': 'FC4', 'B14': 'FC2', 'B15': 'FCz', 'B16': 'Cz',
    'B17': 'C2', 'B18': 'C4', 'B19': 'C6', 'B20': 'T8', 'B21': 'TP8', 'B22': 'CP6',
    'B23': 'CP4', 'B24': 'CP2', 'B25': 'P2', 'B26': 'P4', 'B27': 'P6', 'B28': 'P8',
    'B29': 'P10', 'B30': 'PO8', 'B31': 'PO4', 'B32': 'O2', 'Status': 'Stim',
}
EEG64 = [f"A{i}" for i in range(1, 33)] + [f"B{i}" for i in range(1, 33)]


def pair_stims(events):
    one, two = [], []
    cond, count = None, 0
    for row in events:
        c = row[2]
        if c == 256:
            cond, count = "One", 0
        elif c == 512:
            cond, count = "Two", 0
        elif c == 1024 and row[1] == 0:
            count += 1
            (one if cond == "One" else two if cond == "Two" else []).append(row)
            if count >= 4:
                cond = None
    return (np.array(one) if one else np.empty((0, 3), int),
            np.array(two) if two else np.empty((0, 3), int))


def decode_codes(ev, sfreq):
    codes = ev[:, 2]
    print("  code counts:")
    for c in sorted(set(int(x) for x in codes)):
        if c < 8000:
            print(f"    {c:5d}: {int(np.sum(codes == c))}")
    # For each 2048 and 4096, find nearest preceding 256/512 and 1024 (in ms)
    for target in (2048, 4096, 2144):
        idx = np.where(codes == target)[0]
        if len(idx) == 0:
            continue
        diffs_prompt, diffs_1024 = [], []
        for i in idx:
            s = ev[i, 0]
            prompts = ev[(codes == 256) | (codes == 512)]
            stims = ev[codes == 1024]
            pre_p = prompts[prompts[:, 0] <= s]
            pre_s = stims[stims[:, 0] <= s]
            if len(pre_p):
                diffs_prompt.append((s - pre_p[-1, 0]) / sfreq * 1000)
            if len(pre_s):
                diffs_1024.append((s - pre_s[-1, 0]) / sfreq * 1000)
        dp = np.array(diffs_prompt)
        ds = np.array(diffs_1024)
        print(f"  code {target}: n={len(idx)}  "
              f"ms-after-prompt: med={np.median(dp):.1f} [{dp.min():.0f},{dp.max():.0f}]  "
              f"ms-after-1024: med={np.median(ds):.1f} [{ds.min():.0f},{ds.max():.0f}]")


def analyze(pid):
    print(f"\n========== {pid} ==========")
    bdf = RAW_ROOT / pid / f"{pid}_Stim.bdf"
    raw = mne.io.read_raw_bdf(bdf, preload=True, verbose="ERROR")
    sfreq = raw.info["sfreq"]
    print(f"  sfreq={sfreq}  n_times={raw.n_times}  ch={len(raw.ch_names)}")

    ev = mne.find_events(raw, stim_channel="Status", initial_event=True,
                         min_duration=0.002, shortest_event=0.002,
                         consecutive=True, output="onset", verbose="ERROR")
    decode_codes(ev, sfreq)

    one, two = pair_stims(ev)
    print(f"  paired one_stim={len(one)} two_stim={len(two)}")

    # Build an EEG-only raw with montage for Cz SEP.
    present = [c for c in EEG64 if c in raw.ch_names] + ["Status"]
    eeg = raw.copy().pick(present)
    eeg.rename_channels({k: v for k, v in MAPPING.items() if k in eeg.ch_names})
    eeg.set_montage(mne.channels.make_standard_montage("biosemi64"), on_missing="warn")
    eeg_data = eeg.copy().pick("eeg")

    allstim = np.concatenate([one, two]) if len(one) and len(two) else (one if len(one) else two)

    # --- Artifact detection on broadband (1-300 Hz) data ---
    bb = eeg_data.copy().filter(1, None, verbose="ERROR")
    # wide epochs, no baseline, no rejection
    ep = mne.Epochs(bb, allstim, tmin=-0.10, tmax=0.45, baseline=None,
                    preload=True, reject=None, verbose="ERROR")
    X = ep.get_data()  # (n, ch, t)
    times = ep.times
    # across-channel mean abs gradient at each time => artifact saliency
    grad = np.abs(np.diff(X, axis=2)).mean(axis=1)  # (n, t-1)
    gt = times[1:]
    # search the artifact only after the trigger (0..400 ms)
    win = (gt >= 0.0) & (gt <= 0.40)
    peak_idx = np.argmax(grad[:, win], axis=1)
    peak_lat_ms = gt[win][peak_idx] * 1000.0
    # also peak abs amplitude latency
    absamp = np.abs(X).mean(axis=1)
    pk2 = np.argmax(absamp[:, (times >= 0) & (times <= 0.40)], axis=1)
    amp_lat_ms = times[(times >= 0) & (times <= 0.40)][pk2] * 1000.0

    print(f"  trigger->artifact (grad peak) latency ms: "
          f"med={np.median(peak_lat_ms):.1f} mean={peak_lat_ms.mean():.1f} "
          f"sd={peak_lat_ms.std():.1f} IQR=[{np.percentile(peak_lat_ms,25):.0f},"
          f"{np.percentile(peak_lat_ms,75):.0f}]")
    print(f"  trigger->artifact (amp peak)  latency ms: "
          f"med={np.median(amp_lat_ms):.1f} sd={amp_lat_ms.std():.1f}")

    # --- Figures ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    # (0,0) grand-avg butterfly broadband
    ga = X.mean(axis=0)
    axes[0, 0].plot(times * 1000, ga.T * 1e6, lw=0.4, alpha=0.6)
    axes[0, 0].axvline(0, color="k", ls=":")
    axes[0, 0].axvline(np.median(peak_lat_ms), color="r", ls="--",
                       label=f"med artifact {np.median(peak_lat_ms):.0f} ms")
    axes[0, 0].set_title(f"{pid} broadband GA butterfly (all ch)")
    axes[0, 0].set_xlabel("ms from 1024 trigger"); axes[0, 0].legend()
    # (0,1) across-channel grad saliency GA
    axes[0, 1].plot(gt * 1000, grad.mean(axis=0) * 1e6)
    axes[0, 1].axvline(0, color="k", ls=":")
    axes[0, 1].axvline(np.median(peak_lat_ms), color="r", ls="--")
    axes[0, 1].set_title("mean |gradient| across ch (artifact saliency)")
    axes[0, 1].set_xlabel("ms from 1024 trigger")
    # (1,0) per-trial artifact latency histogram
    axes[1, 0].hist(peak_lat_ms, bins=40)
    axes[1, 0].axvline(np.median(peak_lat_ms), color="r", ls="--")
    axes[1, 0].set_title("per-trial trigger->artifact latency (ms)")
    axes[1, 0].set_xlabel("ms")
    # (1,1) Cz SEP aligned to hand-corrected vs raw trigger
    cz = "Cz" if "Cz" in ep.ch_names else ep.ch_names[0]
    czi = ep.ch_names.index(cz)
    axes[1, 1].plot(times * 1000, X[:, czi, :].mean(0) * 1e6, label=f"{cz} SEP (raw trig)")
    axes[1, 1].axvline(0, color="k", ls=":")
    axes[1, 1].set_title(f"{cz} SEP locked to raw 1024")
    axes[1, 1].set_xlabel("ms from 1024 trigger"); axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(FIGS / f"deepdive_{pid}.png", dpi=110)
    plt.close(fig)
    print(f"  saved {FIGS / f'deepdive_{pid}.png'}")
    sys.stdout.flush()
    return peak_lat_ms


if __name__ == "__main__":
    pids = sys.argv[1:] or ["P25", "P02"]
    for pid in pids:
        analyze(pid)
