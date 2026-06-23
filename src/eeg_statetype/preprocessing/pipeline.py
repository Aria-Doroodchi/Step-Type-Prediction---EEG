"""Per-participant state-task preprocessing orchestrator.

For each source recording (Stim → straight/diagonal; Standing → standing):
  load+assemble → ZapLine → PyPREP bads/interp → ASR → provisional CAR →
  dual-filter Picard ICA + ICLabel → undo CAR → (events) →
    (a) SEP branch:    avg-ref copy → per-e-stim SEP epochs  (state_sep_epochs/)
    (b) source branch: orig-ref     → 2 s analysis epochs    (state_source_epochs/)
    (c) window branch: CSD + e-stim-artifact blanking → 2 s analysis epochs →
                       AutoReject                              (state_epochs/)

Mirrors the CNV pipeline (reusing its bads/asr/ica/filter/reference/reject
primitives); forks only the two-file / 3-condition orchestration, the e-stim
SEP/attribution outputs, and the artifact-blanking of window features.
"""

from __future__ import annotations

import gc
from dataclasses import replace
from html import escape

import numpy as np
import mne

from eeg_steptype.preprocessing import asr as _asr
from eeg_steptype.preprocessing import bads as _bads
from eeg_steptype.preprocessing import filter as _filter
from eeg_steptype.preprocessing import ica as _ica
from eeg_steptype.preprocessing import reference as _ref
from eeg_steptype.preprocessing.reject import reject_epochs

from ..config import apply_participant_override
from ..io import (
    epochs_path,
    source_epochs_path,
    sep_epochs_path,
    qc_report_path,
    ensure_dir,
)
from ..logging_utils import get_logger
from .events_state import find_state_events
from .epoching import build_analysis_epochs, build_sep_epochs
from .load import load_state_raw, source_file_present


log = get_logger(__name__)

# condition -> source recording
SOURCE_FILE = {"standing": "Standing", "straight": "Stim", "diagonal": "Stim"}


def run(participant_id: str, cfg: dict, *, force: bool = False) -> None:
    cfg = apply_participant_override(cfg, participant_id)
    conditions = list(cfg["conditions"])

    files_needed: dict[str, list[str]] = {}
    for cond in conditions:
        files_needed.setdefault(SOURCE_FILE[cond], []).append(cond)

    if not force and _all_outputs_exist(cfg, participant_id, conditions):
        log.info("[%s] outputs already exist — skipping (use --force).", participant_id)
        return

    log.info("=" * 70)
    log.info("[%s] state preprocessing (conditions=%s)", participant_id, conditions)
    log.info("=" * 70)

    qc: dict = {"bads": {}, "ica": {}, "conditions": {}}

    for source_file, conds in files_needed.items():
        if not source_file_present(cfg, participant_id, source_file):
            log.warning("[%s] %s recording missing — skipping conditions %s.",
                        participant_id, source_file, conds)
            continue
        _process_source_file(participant_id, cfg, source_file, conds, qc, force=force)

    _write_qc_report(cfg, participant_id, qc)
    log.info("[%s] done.", participant_id)


def _process_source_file(participant_id, cfg, source_file, conds, qc, *, force):
    log.info("[%s] --- source recording: %s (conditions %s) ---",
             participant_id, source_file, conds)

    # 1. Load + assemble, 2. line noise
    raw = load_state_raw(cfg, participant_id, source_file)
    sfreq = float(raw.info["sfreq"])
    raw = _filter.apply_line_noise_removal(raw, cfg)

    # 3. Bad channels + interpolation
    bads, bad_summary = _bads.detect_bads(raw, cfg, return_summary=True)
    raw = _bads.apply_bads(raw, bads)
    interpolated = list(raw.info["bads"])
    if interpolated:
        log.info("[%s/%s] interpolating bads: %s", participant_id, source_file, interpolated)
        raw.interpolate_bads(reset_bads=True)
    bad_summary = replace(bad_summary, interpolated=interpolated)
    qc["bads"][source_file] = bad_summary

    # 4. ASR, 5. provisional CAR
    raw = _asr.apply_asr(raw, cfg)
    raw, car_state = _ref.apply_car(raw)

    # 6. Dual-filter ICA (fit on 1 Hz copy, apply to 0.1-40 analysis copy).
    # Optional: downsample ONLY the ICA-fit copy for speed (the unmixing matrix
    # is sfreq-agnostic and is applied to the full-res analysis copy). Mirrors
    # the stim module (ICA_FIT_SF=256). Off by default to stay CNV-faithful.
    analysis = _filter.make_analysis_copy(raw, cfg)
    ica_train = _filter.make_ica_training_copy(raw, cfg)
    fit_sf = cfg.get("preprocessing", {}).get("ica", {}).get("fit_resample_sfreq")
    if fit_sf and float(ica_train.info["sfreq"]) > float(fit_sf):
        log.info("[%s/%s] downsampling ICA-fit copy %.0f -> %.0f Hz for speed",
                 participant_id, source_file, ica_train.info["sfreq"], float(fit_sf))
        ica_train.resample(float(fit_sf), verbose=False)
    ica = _ica.fit_ica(ica_train, cfg)
    excluded = _ica.auto_exclude(ica, ica_train, cfg)
    qc["ica"][source_file] = {"n_excluded": len(excluded), "excluded": excluded}
    del ica_train
    gc.collect()
    analysis = _ica.apply_ica(ica, analysis, excluded)
    del ica
    gc.collect()

    # 7. Restore pre-CAR signal -> continuous ICA-cleaned non-CSD (orig ref)
    raw = _ref.undo_car(analysis, car_state)
    del analysis
    gc.collect()

    # 8. Events + e-stim attribution (273 ms offset per this file's sfreq)
    events = find_state_events(raw, cfg, source_file, participant_id)

    # 9. SEP branch — avg-ref copy, per-e-stim SEP epochs (NOT blanked)
    sep_raw = raw.copy().set_eeg_reference("average", projection=False, verbose=False)
    for cond in conds:
        ev = events.get(cond)
        if ev is None:
            continue
        sep_ep = build_sep_epochs(sep_raw, ev["estim"], ev["parent"], cfg, cond)
        if sep_ep is not None and len(sep_ep):
            out = sep_epochs_path(cfg, participant_id, cond)
            ensure_dir(out.parent)
            sep_ep.save(str(out), overwrite=True)
            qc.setdefault("conditions", {}).setdefault(cond, {})["sep_kept"] = len(sep_ep)
    del sep_raw
    gc.collect()

    # 10. Source branch — orig-ref analysis epochs (no reject), for source loc
    for cond in conds:
        ev = events.get(cond)
        if ev is None or len(ev["analysis"]) == 0:
            continue
        src_ep = build_analysis_epochs(raw, ev["analysis"], ev["parent"], cfg, cond)
        if len(src_ep):
            out = source_epochs_path(cfg, participant_id, cond)
            ensure_dir(out.parent)
            src_ep.save(str(out), overwrite=True)

    # 11. Window branch — CSD + e-stim-artifact blanking, AutoReject
    raw = _ref.apply_csd(raw, cfg)
    scfg = cfg["features"]["sep"]
    if bool(scfg.get("blank_window_features", True)):
        all_estim = np.concatenate(
            [events[c]["estim"][:, 0] for c in conds if len(events[c]["estim"])]
        ) if any(len(events[c]["estim"]) for c in conds) else np.empty(0, int)
        n_blanked = _blank_estim_artifacts(
            raw, all_estim, float(scfg.get("blank_halfwidth_ms", 12.0)))
        log.info("[%s/%s] blanked %d e-stim artifact intervals (±%.0f ms) "
                 "from window-feature data", participant_id, source_file,
                 n_blanked, float(scfg.get("blank_halfwidth_ms", 12.0)))

    for cond in conds:
        ev = events.get(cond)
        if ev is None or len(ev["analysis"]) == 0:
            log.warning("[%s] no analysis events for %s; skipping.", participant_id, cond)
            continue
        epochs = build_analysis_epochs(raw, ev["analysis"], ev["parent"], cfg, cond)
        n_pre = len(epochs)
        epochs = reject_epochs(epochs, cfg, cond)
        out = epochs_path(cfg, participant_id, cond)
        ensure_dir(out.parent)
        epochs.save(str(out), overwrite=True)
        c = qc.setdefault("conditions", {}).setdefault(cond, {})
        c.update({
            "sfreq": sfreq,
            "n_pre_reject": n_pre,
            "n_kept": len(epochs),
            "n_estim_total": int(len(ev["estim"])),
            "mean_estim_per_epoch": (float(len(ev["estim"]) / n_pre) if n_pre else 0.0),
        })
        log.info("[%s] wrote %s (%d/%d epochs kept)",
                 participant_id, out.name, len(epochs), n_pre)

    del raw
    gc.collect()


def _blank_estim_artifacts(raw: mne.io.BaseRaw, estim_samples: np.ndarray,
                           halfwidth_ms: float) -> int:
    """Linearly interpolate EEG across ±halfwidth around each e-stim sample.

    Removes the electrical-stim artifact (and thus its rhythm) from the window
    feature data so the model cannot separate standing↔stepping on the artifact
    cadence alone (validity trap #2). Operates in place on ``raw._data``.
    """
    if len(estim_samples) == 0:
        return 0
    sf = float(raw.info["sfreq"])
    hw = max(1, int(round(halfwidth_ms / 1000.0 * sf)))
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    data = raw._data
    first = int(raw.first_samp)
    nt = int(raw.n_times)
    n = 0
    for s_abs in np.sort(np.unique(estim_samples.astype(int))):
        i = int(s_abs - first)
        x0, x1 = i - hw - 1, i + hw + 1
        if x0 < 0 or x1 >= nt:
            continue
        ramp = np.linspace(0.0, 1.0, x1 - x0 + 1)[None, :]
        y0 = data[picks, x0][:, None]
        y1 = data[picks, x1][:, None]
        data[np.ix_(picks, np.arange(x0, x1 + 1))] = y0 + (y1 - y0) * ramp
        n += 1
    return n


def _all_outputs_exist(cfg, participant_id, conditions) -> bool:
    for cond in conditions:
        if not source_file_present(cfg, participant_id, SOURCE_FILE[cond]):
            continue  # missing source file -> can't produce; don't block skip
        for builder in (epochs_path, source_epochs_path):
            if not builder(cfg, participant_id, cond).exists():
                return False
    return True


def _write_qc_report(cfg: dict, participant_id: str, qc: dict) -> None:
    if not cfg.get("preprocessing", {}).get("qc_report", True):
        return
    out = qc_report_path(cfg, participant_id)
    ensure_dir(out.parent)

    bad_rows = "".join(
        f"<tr><th>{escape(sf)}</th><td>{len(s.final)}</td>"
        f"<td>{escape(', '.join(s.final) or 'None')}</td>"
        f"<td>{qc['ica'].get(sf, {}).get('n_excluded', '?')}</td></tr>"
        for sf, s in qc.get("bads", {}).items()
    )
    cond_rows = "".join(
        f"<tr><th>{escape(cond)}</th>"
        f"<td>{c.get('sfreq', '?')}</td>"
        f"<td>{c.get('n_pre_reject', '?')}</td>"
        f"<td>{c.get('n_kept', '?')}</td>"
        f"<td>{c.get('n_estim_total', '?')}</td>"
        f"<td>{c.get('mean_estim_per_epoch', 0):.2f}</td>"
        f"<td>{c.get('sep_kept', '?')}</td></tr>"
        for cond, c in qc.get("conditions", {}).items()
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{escape(participant_id)} state QC</title>
<style>body{{font-family:Arial;margin:2rem}}table{{border-collapse:collapse;margin-bottom:1.5rem}}
th,td{{border:1px solid #ccc;padding:.4rem .6rem;text-align:left}}th{{background:#f3f4f6}}</style>
</head><body>
<h1>{escape(participant_id)} — state preprocessing QC</h1>
<h2>Bad channels / ICA (per source recording)</h2>
<table><tr><th>Recording</th><th>#bads</th><th>bads</th><th>ICA excl</th></tr>{bad_rows}</table>
<h2>Epochs per condition</h2>
<table><tr><th>Condition</th><th>sfreq</th><th>pre-reject</th><th>kept</th>
<th>e-stims</th><th>e-stim/epoch</th><th>SEP epochs</th></tr>{cond_rows}</table>
</body></html>"""
    out.write_text(html, encoding="utf-8")
    log.info("[%s] wrote QC report: %s", participant_id, out)
