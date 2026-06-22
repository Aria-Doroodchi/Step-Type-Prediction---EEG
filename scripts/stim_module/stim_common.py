"""Shared helpers for the stim-module Task-1 calibration analysis.

Loads a Pxx_Stim.bdf, attributes each 1024 stim trigger to its preceding
256/512 prompt (the explicit pairing the task requires), and detects the true
stim-artifact onset per trial via a per-participant matched filter.

Nothing here mutates the main pipeline; it is a standalone analysis layer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import mne

mne.set_log_level("ERROR")

RAW_ROOT = Path(
    "C:/Users/Ali D/OneDrive - The University of Western Ontario/MSc/Thesis/Data/Participants"
)

# A/B BioSemi label -> 10-20 name (from the per-participant Pxx_Stim.py scripts).
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

CODE_ONE, CODE_TWO, CODE_STIM = 256, 512, 1024


def pair_stims(events, max_per_prompt=4):
    """Bind each 1024 to its preceding 256/512 prompt (explicit walk).

    Mirrors the per-participant scripts: a 256 opens a One block, 512 a Two
    block, and the following 1024 events (prev-value column == 0) are bound to
    that block up to ``max_per_prompt`` stims, after which the block closes.

    Returns (one, two, info) where one/two are (n,3) event arrays and info holds
    per-event condition + within-prompt index for downstream grouping.
    """
    one, two = [], []
    cond, count = None, 0
    rows, conds, orders = [], [], []
    for row in events:
        c = row[2]
        if c == CODE_ONE:
            cond, count = "One", 0
        elif c == CODE_TWO:
            cond, count = "Two", 0
        elif c == CODE_STIM and row[1] == 0:
            count += 1
            if cond == "One":
                one.append(row); rows.append(row); conds.append("One"); orders.append(count)
            elif cond == "Two":
                two.append(row); rows.append(row); conds.append("Two"); orders.append(count)
            if count >= max_per_prompt:
                cond = None
    one = np.array(one) if one else np.empty((0, 3), int)
    two = np.array(two) if two else np.empty((0, 3), int)
    info = {
        "events": np.array(rows) if rows else np.empty((0, 3), int),
        "cond": np.array(conds),
        "order": np.array(orders, int),
    }
    return one, two, info


def load_stim_raw(pid, picks_eeg=True):
    """Return (raw_eeg_with_montage, sfreq). EEG channels renamed, montage set."""
    bdf = RAW_ROOT / pid / f"{pid}_Stim.bdf"
    raw = mne.io.read_raw_bdf(bdf, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    ev = mne.find_events(raw, stim_channel="Status", initial_event=True,
                         min_duration=0.002, shortest_event=0.002,
                         consecutive=True, output="onset", verbose="ERROR")
    if picks_eeg:
        montage = mne.channels.make_standard_montage("biosemi64")
        names_1020 = set(montage.ch_names)
        ab_present = [c for c in EEG64 if c in raw.ch_names]
        renamed_present = [c for c in raw.ch_names if c in names_1020]
        if len(renamed_present) >= 32:
            # File already carries 10-20 names (e.g. P09).
            eeg = raw.copy().pick(renamed_present)
        elif len(ab_present) >= 32:
            eeg = raw.copy().pick(ab_present)
            eeg.rename_channels({k: v for k, v in MAPPING.items() if k in eeg.ch_names})
        else:
            raise ValueError(f"{pid}: cannot identify EEG channels "
                             f"(have {raw.ch_names[:6]}...)")
        eeg.set_montage(montage, on_missing="ignore")
        return eeg, sfreq, ev
    return raw, sfreq, ev


def detect_artifact_latency(epochs_data, times, search_ms=(80, 360),
                            template_iters=2):
    """Per-trial trigger->artifact latency (ms) via matched-filter xcorr.

    Pipeline:
      1. saliency(t) = mean over channels of |d/dt signal|  (sharp edges of the
         electrical-stim artifact dominate this).
      2. coarse per-trial peak in the search window -> initial latencies.
      3. build a template from the median-aligned saliency; cross-correlate each
         trial against it for a refined, low-jitter latency. Iterate the template.

    Returns dict with per-trial latency_ms, a quality score (peak prominence),
    and the template.
    """
    # saliency
    sal = np.abs(np.diff(epochs_data, axis=2)).mean(axis=1)  # (n, t-1)
    st = times[1:]
    win = (st * 1000 >= search_ms[0]) & (st * 1000 <= search_ms[1])
    widx = np.where(win)[0]
    sal_w = sal[:, widx]

    # coarse peaks
    coarse = np.argmax(sal_w, axis=1)
    coarse_lat = st[widx][coarse] * 1000.0

    # prominence quality: peak height vs MAD of the window
    med = np.median(sal_w, axis=1, keepdims=True)
    mad = np.median(np.abs(sal_w - med), axis=1) + 1e-20
    prom = (sal_w.max(axis=1) - med.squeeze(1)) / mad

    # template cross-correlation refinement
    n = sal.shape[0]
    half = int(round(0.040 * (1 / (st[1] - st[0]))))  # +-40 ms window for template
    lat = coarse.copy()
    template = None
    for _ in range(template_iters):
        # build template by averaging windows centred on current latencies
        stack = []
        for i in range(n):
            c = widx[lat[i]]
            lo, hi = c - half, c + half
            if lo < 0 or hi >= sal.shape[1]:
                continue
            stack.append(sal[i, lo:hi])
        if len(stack) < 5:
            break
        template = np.mean(stack, axis=0)
        template = template - template.mean()
        tnorm = np.linalg.norm(template) + 1e-20
        # cross-correlate each trial's window against the template
        new_lat = []
        for i in range(n):
            seg = sal_w[i]
            seg0 = seg - seg.mean()
            xc = np.correlate(seg0, template, mode="same")
            new_lat.append(int(np.argmax(xc)))
        lat = np.array(new_lat)
    refined_lat = st[widx][lat] * 1000.0

    return {
        "coarse_ms": coarse_lat,
        "latency_ms": refined_lat,
        "prominence": prom,
        "saliency_ga": sal.mean(axis=0),
        "saliency_times_ms": st * 1000.0,
        "template": template,
    }


def _parabolic_peak(y, i):
    """Sub-sample peak position around integer index i via parabolic fit."""
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    a, b, c = y[i - 1], y[i], y[i + 1]
    denom = (a - 2 * b + c)
    if denom == 0:
        return float(i)
    return i + 0.5 * (a - c) / denom


def ga_saliency(epochs_data, robust=True):
    """Across-channel |gradient| saliency per time sample, per trial.

    The electrical-stim artifact deflects many channels *synchronously*, so a
    median across channels is high only when many channels move together (the
    artifact) and stays low for asynchronous single-channel noise. This is far
    more selective than the mean, which a single noisy channel inflates
    everywhere. Returns (n_trials, t-1).
    """
    g = np.abs(np.diff(epochs_data, axis=2))  # (n, ch, t-1)
    return np.median(g, axis=1) if robust else g.mean(axis=1)


def ga_peak_latency(epochs_data, times, search_ms=(120, 340), n_boot=2000, seed=0):
    """Grand-average artifact-onset latency (ms) with a bootstrap CI over trials.

    The electrical-stim artifact is sharp and stimulus-locked, so the trial
    average of the across-channel gradient saliency has a clean peak at the true
    onset even when single trials are noisy. Bootstrap resamples trials, rebuilds
    the GA, and re-locates the peak (parabolic sub-sample) to get a CI.

    Also returns the GA peak width (FWHM, ms) as a jitter proxy: a sharp peak
    implies low trial-to-trial timing jitter (a constant offset), a broad peak
    implies real jitter.
    """
    sal = ga_saliency(epochs_data)                 # (n, t-1)
    st = times[1:] * 1000.0                          # ms
    win = (st >= search_ms[0]) & (st <= search_ms[1])
    widx = np.where(win)[0]

    def peak_ms(sal_mean):
        seg = sal_mean[widx]
        j = int(np.argmax(seg))
        sub = _parabolic_peak(seg, j)
        # map sub-sample index back to ms
        i0 = widx[0] + sub
        lo = int(np.floor(i0)); frac = i0 - lo
        if lo + 1 < len(st):
            return st[lo] * (1 - frac) + st[lo + 1] * frac
        return st[lo]

    def onset_ms(sal_mean, frac=0.5):
        """Artifact onset = left half-max crossing before the peak (true stim)."""
        seg = sal_mean[widx]
        base = np.median(seg)
        j = int(np.argmax(seg))
        thr = base + frac * (seg[j] - base)
        i = j
        while i > 0 and seg[i] >= thr:
            i -= 1
        # linear interpolation between i and i+1 for sub-sample onset
        if seg[i + 1] != seg[i]:
            f = (thr - seg[i]) / (seg[i + 1] - seg[i])
        else:
            f = 0.0
        idx = widx[0] + i + f
        lo = int(np.floor(idx)); fr = idx - lo
        return st[lo] * (1 - fr) + st[min(lo + 1, len(st) - 1)] * fr

    ga = sal.mean(axis=0)
    point = peak_ms(ga)
    onset = onset_ms(ga)

    # FWHM of the GA peak (within window), baseline = window median
    seg = ga[widx]
    base = np.median(seg)
    pk = seg.max()
    half = base + 0.5 * (pk - base)
    above = np.where(seg >= half)[0]
    fwhm_ms = (st[widx][above[-1]] - st[widx][above[0]]) if len(above) else np.nan
    # prominence: peak height over window MAD (artifact detectability)
    mad = np.median(np.abs(seg - base)) + 1e-20
    prominence = (pk - base) / mad

    rng = np.random.default_rng(seed)
    n = sal.shape[0]
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(peak_ms(sal[idx].mean(axis=0)))
    return {
        "peak_ms": float(point),
        "onset_ms": float(onset),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "boot_sd": float(np.std(boots)),
        "fwhm_ms": float(fwhm_ms),
        "prominence": float(prominence),
        "ga": ga, "ga_times_ms": st,
    }


def bootstrap_ci(x, n_boot=2000, seed=0, stat=np.median):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = [stat(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    return float(stat(x)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
