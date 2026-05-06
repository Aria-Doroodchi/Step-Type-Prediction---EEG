"""Per-participant preprocessing orchestrator.

Reads raw → assemble → ZapLine → bads/interpolation → ASR → provisional CAR
→ 1 Hz ICA-training copy + 0.1-40 Hz analysis copy → fit/label ICA on the
training copy → apply unmixing to the analysis copy → undo CAR → CSD → events
→ epochs
(per condition) → reject → save .fif

Outputs to ``data/interim/epochs/{pid}_CNV_{One,Two}-epo.fif``. Skips the
per-participant work if both outputs already exist (idempotent).
"""

from __future__ import annotations

import gc
from dataclasses import replace
from html import escape

from ..config import apply_participant_override
from ..io import epochs_path, ensure_dir, qc_report_path, source_epochs_path
from ..logging_utils import get_logger
from . import asr as _asr
from . import bads as _bads
from . import filter as _filter
from . import ica as _ica
from . import reference as _ref
from .epoching import build_epochs
from .events import find_step_events
from .load import load_raw
from .reject import reject_epochs


log = get_logger(__name__)


def run(participant_id: str, cfg: dict, *, force: bool = False) -> None:
    """Run the full preprocessing pipeline for one participant."""
    cfg = apply_participant_override(cfg, participant_id)

    out_one = epochs_path(cfg, participant_id, "One")
    out_two = epochs_path(cfg, participant_id, "Two")
    if (not force) and out_one.exists() and out_two.exists():
        log.info("[%s] outputs already exist — skipping (use --force to override)",
                 participant_id)
        return

    log.info("=" * 70)
    log.info("[%s] preprocessing", participant_id)
    log.info("=" * 70)

    # 1. Load + assemble raw
    raw = load_raw(cfg, participant_id)

    # 2. Line-noise removal before bad-channel detection and ICA
    raw = _filter.apply_line_noise_removal(raw, cfg)

    # 3. Bad channels + interpolation
    bads, bad_summary = _bads.detect_bads(raw, cfg, return_summary=True)
    raw = _bads.apply_bads(raw, bads)
    interpolated = list(raw.info["bads"])
    if interpolated:
        log.info("[%s] interpolating bad channels: %s", participant_id, interpolated)
        raw.interpolate_bads(reset_bads=True)
    else:
        log.info("[%s] no bad channels to interpolate", participant_id)
    bad_summary = replace(bad_summary, interpolated=interpolated)
    _write_bad_channel_qc_report(cfg, participant_id, bad_summary)

    # 4. Remove transient bursts before ICA sees the data.
    raw = _asr.apply_asr(raw, cfg)

    # 5. Provisional CAR for dual-filter ICA.
    raw, car_state = _ref.apply_car(raw)

    # 6. Dual-filter ICA: fit/label on 1 Hz HP full data, apply to the
    # 0.1-40 Hz analysis copy so slow CNV content is preserved.
    analysis = _filter.make_analysis_copy(raw, cfg)
    ica_train = _filter.make_ica_training_copy(raw, cfg)
    ica = _ica.fit_ica(ica_train, cfg)
    excluded = _ica.auto_exclude(ica, ica_train, cfg)
    del ica_train
    gc.collect()
    analysis = _ica.apply_ica(ica, analysis, excluded)
    del ica
    gc.collect()

    # 7. Restore pre-CAR signal, then apply analysis reference.
    raw = _ref.undo_car(analysis, car_state)
    del analysis
    gc.collect()

    # 8. Events are shared by source-compatible EEG epochs and final CSD epochs.
    events = find_step_events(raw, cfg)

    # 9. Save EEG-potential epochs for source localization before CSD changes
    # channel types to CSD. These files are intentionally not the final
    # analysis epochs; they exist so MNE forward modeling still sees EEG data.
    for cond in cfg["conditions"]:
        if len(events[cond]) == 0:
            continue
        source_epochs = build_epochs(raw, events[cond], cfg, cond)
        out = source_epochs_path(cfg, participant_id, cond)
        ensure_dir(out.parent)
        source_epochs.save(str(out), overwrite=True)
        log.info("[%s] wrote source epochs %s (%d epochs)", participant_id, out, len(source_epochs))

    # 10. Final analysis reference. Epoch .fif files saved below are CSD data.
    raw = _ref.apply_csd(raw, cfg)

    # 11. CSD epochs (per condition) → reject → save for electrode features
    for cond in cfg["conditions"]:
        if len(events[cond]) == 0:
            log.warning("[%s] no events for condition %s; skipping.", participant_id, cond)
            continue
        epochs = build_epochs(raw, events[cond], cfg, cond)
        epochs = reject_epochs(epochs, cfg, cond)

        out = epochs_path(cfg, participant_id, cond)
        ensure_dir(out.parent)
        epochs.save(str(out), overwrite=True)
        log.info("[%s] wrote %s (%d epochs)", participant_id, out, len(epochs))

    log.info("[%s] done.", participant_id)


def _write_bad_channel_qc_report(
    cfg: dict,
    participant_id: str,
    summary: _bads.BadChannelSummary,
) -> None:
    """Write a small bad-channel QC report for one preprocessing run."""
    if not cfg.get("preprocessing", {}).get("qc_report", True):
        return

    out = qc_report_path(cfg, participant_id)
    ensure_dir(out.parent)
    rows = [
        ("PyPREP detected", summary.pyprep),
        ("Override added", summary.override),
        ("Final marked bad", summary.final),
        ("Interpolated", summary.interpolated),
    ]
    table_rows = "\n".join(
        "<tr>"
        f"<th>{escape(label)}</th>"
        f"<td>{len(channels)}</td>"
        f"<td>{escape(', '.join(channels) if channels else 'None')}</td>"
        "</tr>"
        for label, channels in rows
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(participant_id)} preprocessing QC</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; line-height: 1.4; }}
    table {{ border-collapse: collapse; min-width: 42rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.5rem 0.65rem; text-align: left; }}
    th {{ background: #f3f4f6; width: 12rem; }}
    caption {{ font-weight: 700; margin-bottom: 0.75rem; text-align: left; }}
  </style>
</head>
<body>
  <h1>{escape(participant_id)} Preprocessing QC</h1>
  <table>
    <caption>Bad-channel detection and interpolation summary</caption>
    <thead>
      <tr><th>Source</th><th>Count</th><th>Channels</th></tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
</body>
</html>
"""
    out.write_text(html, encoding="utf-8")
    log.info("[%s] wrote bad-channel QC report: %s", participant_id, out)
