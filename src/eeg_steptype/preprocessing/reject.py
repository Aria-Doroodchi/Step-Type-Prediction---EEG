"""Automated epoch rejection.

Default: ``autoreject.AutoReject`` (cross-validated per-channel thresholds).
Fallback: a single voltage threshold per condition. Per-participant overrides
can supply explicit thresholds (preserved from the original hand-tuned
``One_rejection_criteria`` / ``Two_rejection_criteria`` values).

Override schema:

    preprocessing:
      reject:
        method: autoreject | threshold
        thresholds:
          One: 48e-6
          Two: 51.5e-6
        manual_drop:        # rare; e.g. P17 had hand-picked epoch indices
          One: []
          Two: []
"""

from __future__ import annotations

import numpy as np
import mne

from ..logging_utils import get_logger


log = get_logger(__name__)


def reject_epochs(
    epochs: mne.Epochs,
    cfg: dict,
    condition: str,
) -> mne.Epochs:
    """Drop bad epochs and return the cleaned Epochs object."""
    rcfg = cfg["preprocessing"]["reject"]
    method = rcfg.get("method", "autoreject")

    if method == "autoreject":
        epochs = _autoreject(epochs)
    elif method == "threshold":
        thr = (rcfg.get("thresholds") or {}).get(condition,
                                                 rcfg.get("fallback_eeg_threshold"))
        epochs = _threshold(epochs, thr)
    else:
        raise ValueError(f"Unknown reject.method: {method}")

    # Hand-curated epoch indices to drop (rare cases, e.g. P17).
    manual = (rcfg.get("manual_drop") or {}).get(condition) or []
    if manual:
        log.info("Dropping manual epoch indices for %s: %s", condition, manual)
        epochs.drop(np.asarray(manual, dtype=int), reason="manual_drop")

    log.info("Post-rejection %s: %d epochs", condition, len(epochs))
    return epochs


# ---------------------------------------------------------------------------
def _autoreject(epochs: mne.Epochs) -> mne.Epochs:
    try:
        from autoreject import AutoReject
    except Exception as exc:                            # noqa: BLE001
        log.warning(
            "autoreject unavailable (%s); falling back to no automated rejection.",
            exc,
        )
        return epochs
    try:
        ar = AutoReject(random_state=42, verbose=False)
        return ar.fit_transform(epochs)
    except Exception as exc:                            # noqa: BLE001
        log.warning("autoreject failed: %s; returning epochs unchanged.", exc)
        return epochs


def _threshold(epochs: mne.Epochs, thr: float | None) -> mne.Epochs:
    if thr is None:
        return epochs
    log.info("Voltage-threshold rejection: eeg=%g V", thr)
    epochs.drop_bad(reject=dict(eeg=float(thr)))
    return epochs
