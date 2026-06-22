"""Uniform automated SEP preprocessing for the stim module (Task 2).

One consistent pipeline applied to every participant/task (NO CSD — it would
distort the P50/N90 SEP morphology):
  pick 64 EEG -> montage -> notch (line + harmonics) -> auto bad-channel
  detect+interpolate (pyprep) -> average reference -> ICA (picard, fit on a
  1-100 Hz downsampled copy, auto-labelled by ICLabel, ocular/muscle/etc.
  excluded) -> 0.1-36 Hz bandpass -> epoch around the trigger (with the Task-1
  calibrated offset applied) -> fixed-threshold trial rejection -> average.

Returns per-cell averaged SEPs (all 64 channels) on a common time grid so the
stats/plot stage never re-reads the raw files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import mne
from mne.preprocessing import ICA

import sys
sys.path.insert(0, str(Path(__file__).parent))
from stim_common import (load_stim_raw, pair_stims, RAW_ROOT, EEG64, MAPPING,
                         ga_peak_latency)

mne.set_log_level("ERROR")

# --- common analysis grid (true-stim-relative), 1024 Hz ---
GRID_SF = 1024.0
TMIN, TMAX = -0.10, 0.30
GRID_MS = np.arange(TMIN * 1000, TMAX * 1000 + 1e-6, 1000.0 / GRID_SF)
BASELINE = (-0.05, 0.0)
UNIFORM_OFFSET_S = 0.273          # Task-1 calibrated stim-task offset
REJECT = dict(eeg=150e-6)         # fixed automated peak-to-peak rejection
ICA_FIT_SF = 256.0                # downsample for fast, artifact-faithful ICA
ICLABEL_ARTIFACTS = {"eye blink", "muscle artifact", "heart beat",
                     "line noise", "channel noise"}
ICLABEL_PROB = 0.80


def _load_raw_eeg(path):
    raw = mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    ev = mne.find_events(raw, stim_channel="Status", initial_event=True,
                         min_duration=0.002, shortest_event=0.002,
                         consecutive=True, output="onset", verbose="ERROR")
    montage = mne.channels.make_standard_montage("biosemi64")
    names_1020 = set(montage.ch_names)
    ab = [c for c in EEG64 if c in raw.ch_names]
    renamed = [c for c in raw.ch_names if c in names_1020]
    if len(renamed) >= 32:
        eeg = raw.copy().pick(renamed)
    else:
        eeg = raw.copy().pick(ab)
        eeg.rename_channels({k: v for k, v in MAPPING.items() if k in eeg.ch_names})
    eeg.set_montage(montage, on_missing="ignore")
    return eeg, sfreq, ev


def _preprocess(eeg, sfreq):
    """Notch -> bad-channel interpolate -> avg ref -> ICA(ICLabel) -> 0.1-36."""
    # line noise + harmonics up to below Nyquist
    nyq = sfreq / 2.0
    freqs = [f for f in range(60, int(nyq), 60)]
    eeg.notch_filter(freqs=freqs, verbose="ERROR")

    # automated bad channels (pyprep), interpolate
    try:
        from pyprep.find_noisy_channels import NoisyChannels
        nd = NoisyChannels(eeg.copy().filter(1.0, 100.0, verbose="ERROR"),
                           random_state=42)
        nd.find_all_bads(ransac=False)
        bads = nd.get_bads()
        eeg.info["bads"] = bads
        if bads:
            eeg.interpolate_bads(reset_bads=True, verbose="ERROR")
    except Exception:
        bads = []

    eeg.set_eeg_reference("average", projection=False, verbose="ERROR")

    # ICA on a 1-100 Hz, downsampled copy; label with ICLabel; exclude artifacts
    n_excl = 0
    try:
        from mne_icalabel import label_components
        ica_raw = eeg.copy().filter(1.0, 100.0, verbose="ERROR")
        if sfreq > ICA_FIT_SF:
            ica_raw.resample(ICA_FIT_SF, verbose="ERROR")
        n_comp = min(25, len(mne.pick_types(eeg.info, eeg=True)) - len(eeg.info["bads"]) - 1)
        ica = ICA(n_components=n_comp, method="infomax",
                  fit_params=dict(extended=True), random_state=97, max_iter="auto")
        ica.fit(ica_raw, verbose="ERROR")
        labels = label_components(ica_raw, ica, method="iclabel")
        excl = [i for i, (lab, prob) in
                enumerate(zip(labels["labels"], labels["y_pred_proba"]))
                if lab in ICLABEL_ARTIFACTS and prob >= ICLABEL_PROB]
        ica.exclude = excl
        n_excl = len(excl)
        ica.apply(eeg, verbose="ERROR")
    except Exception as e:
        print(f"    ICA skipped: {e!r}")

    eeg.filter(0.1, 36.0, verbose="ERROR")
    return eeg, len(bads), n_excl


def _epoch_average(eeg, events, sfreq, offset_s):
    """Apply sample offset, epoch, reject, return (n_kept, 64xT avg on GRID)."""
    if len(events) == 0:
        return 0, None, None
    ev = events.copy()
    ev[:, 0] = ev[:, 0] + int(round(offset_s * sfreq))
    ep = mne.Epochs(eeg, ev, tmin=TMIN, tmax=TMAX, baseline=BASELINE,
                    preload=True, reject=REJECT, verbose="ERROR")
    if len(ep) == 0:
        return 0, None, ep.ch_names
    avg = ep.average()
    t_ms = avg.times * 1000.0
    data = np.vstack([np.interp(GRID_MS, t_ms, avg.data[c])
                      for c in range(avg.data.shape[0])])  # (64, len(GRID))
    return len(ep), data, avg.ch_names


def process_participant(pid):
    """Return dict of cell -> {n, data(64xT), ch_names} for stim + standing."""
    out = {"pid": pid, "cells": {}, "ch_names": None, "grid_ms": GRID_MS,
           "meta": {}}

    # ---------- STIM task (straight=One, diagonal=Two) ----------
    stim_path = RAW_ROOT / pid / f"{pid}_Stim.bdf"
    if stim_path.exists():
        eeg, sfreq, ev = _load_raw_eeg(stim_path)
        eeg, nbad, nexcl = _preprocess(eeg, sfreq)
        out["meta"]["stim_sfreq"] = sfreq
        out["meta"]["stim_bads"] = nbad
        out["meta"]["stim_ica_excl"] = nexcl
        one, two, info = pair_stims(ev)
        cond, order = info["cond"], info["order"]
        evs = info["events"]
        # pooled
        for name, mask in (("straight", cond == "One"), ("diagonal", cond == "Two")):
            n, data, ch = _epoch_average(eeg, evs[mask], sfreq, UNIFORM_OFFSET_S)
            out["cells"][name] = {"n": n, "data": data}
            if ch and out["ch_names"] is None:
                out["ch_names"] = ch
        # by order
        for path_name, cval in (("straight", "One"), ("diagonal", "Two")):
            for k in (1, 2, 3, 4):
                mask = (cond == cval) & (order == k)
                n, data, ch = _epoch_average(eeg, evs[mask], sfreq, UNIFORM_OFFSET_S)
                out["cells"][f"{path_name}_{k}"] = {"n": n, "data": data}

    # ---------- STANDING task (pooled), re-measured offset ----------
    stand_path = RAW_ROOT / pid / f"{pid}_Standing.bdf"
    if stand_path.exists():
        eeg, sfreq, ev = _load_raw_eeg(stand_path)
        # re-measure standing artifact offset from a broadband copy
        st_off_s = UNIFORM_OFFSET_S
        st_off_ms, st_fwhm = np.nan, np.nan
        try:
            bb = eeg.copy().filter(1, None, verbose="ERROR")
            stim_ev = ev[(ev[:, 2] == 1024) & (ev[:, 1] == 0)]
            if len(stim_ev) >= 20:
                epb = mne.Epochs(bb, stim_ev, tmin=-0.10, tmax=0.45,
                                 baseline=None, preload=True, reject=None,
                                 verbose="ERROR")
                res = ga_peak_latency(epb.get_data(), epb.times, search_ms=(120, 340))
                st_fwhm = res["fwhm_ms"]
                if res["fwhm_ms"] <= 25:           # clean artifact -> use it
                    st_off_ms = res["peak_ms"]
                    st_off_s = st_off_ms / 1000.0
        except Exception as e:
            print(f"    standing offset measure failed: {e!r}")
        eeg, nbad, nexcl = _preprocess(eeg, sfreq)
        out["meta"]["stand_sfreq"] = sfreq
        out["meta"]["stand_offset_ms"] = st_off_ms
        out["meta"]["stand_offset_fwhm"] = st_fwhm
        out["meta"]["stand_offset_used_s"] = st_off_s
        out["meta"]["stand_bads"] = nbad
        out["meta"]["stand_ica_excl"] = nexcl
        stim_ev = ev[(ev[:, 2] == 1024) & (ev[:, 1] == 0)]
        n, data, ch = _epoch_average(eeg, stim_ev, sfreq, st_off_s)
        out["cells"]["standing"] = {"n": n, "data": data}
        if ch and out["ch_names"] is None:
            out["ch_names"] = ch

    return out
