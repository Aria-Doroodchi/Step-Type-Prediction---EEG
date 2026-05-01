"""Per-participant preprocessing orchestrator.

Reads raw → assemble → bads → reference → notch → ICA fit/exclude/apply
→ bandpass → events → epochs (per condition) → reject → save .fif

Outputs to ``data/interim/epochs/{pid}_CNV_{One,Two}-epo.fif``. Skips the
per-participant work if both outputs already exist (idempotent).
"""

from __future__ import annotations

from pathlib import Path

from ..config import apply_participant_override
from ..io import epochs_path, ensure_dir
from ..logging_utils import get_logger
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

    # 2. Bad channels + interpolation
    bads = _bads.detect_bads(raw, cfg)
    raw = _bads.apply_bads(raw, bads)

    # 3. Average reference
    raw = _ref.apply_average_reference(raw)

    # 4. Notch filter
    raw = _filter.apply_notch(raw, cfg)

    # 5. ICA fit + auto-classify + apply
    ica = _ica.fit_ica(raw, cfg)
    excluded = _ica.auto_exclude(ica, raw, cfg)
    raw = _ica.apply_ica(ica, raw, excluded)

    # 6. Final bandpass
    raw = _filter.apply_bandpass(raw, cfg)

    # 7. Events → epochs (per condition) → reject → save
    events = find_step_events(raw, cfg)
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
