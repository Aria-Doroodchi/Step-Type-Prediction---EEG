"""Automated epoch repair/rejection.

Default: local ``autoreject.AutoReject`` with CV-tuned per-channel thresholds,
epoch-level sensor interpolation, and consensus-based trial dropping. This
replaces unconditional amplitude-based trial removal so scarce trials are
repaired when possible instead of discarded.

Override schema:

    preprocessing:
      reject:
        method: autoreject
        n_interpolate: [1, 4, 8, 16]
        consensus: [0.2, 0.4, 0.6, 0.8]
        cv: 10
        random_state: 42
        manual_drop:        # rare; e.g. P17 had hand-picked epoch indices
          One: []
          Two: []
"""

from __future__ import annotations

import numpy as np
import mne

from ..logging_utils import get_logger
from ..resources import resolve_n_jobs


log = get_logger(__name__)


def reject_epochs(
    epochs: mne.Epochs,
    cfg: dict,
    condition: str,
) -> mne.Epochs:
    """Repair/drop bad epochs and return the cleaned Epochs object."""
    rcfg = cfg["preprocessing"]["reject"]
    method = rcfg.get("method", "autoreject")

    if method == "autoreject":
        epochs = _autoreject(epochs, rcfg, condition, cfg=cfg)
    elif method == "threshold":
        log.warning(
            "Legacy voltage-threshold rejection requested for %s; "
            "prefer preprocessing.reject.method=autoreject for local repair.",
            condition,
        )
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
def _autoreject(
    epochs: mne.Epochs,
    rcfg: dict,
    condition: str,
    *,
    cfg: dict | None = None,
) -> mne.Epochs:
    try:
        from autoreject import AutoReject
    except Exception as exc:                            # noqa: BLE001
        log.warning(
            "autoreject unavailable (%s); falling back to no automated rejection.",
            exc,
        )
        return epochs
    try:
        n_interpolate = rcfg.get("n_interpolate", rcfg.get("n_interpolates"))
        consensus = rcfg.get("consensus")
        cv = int(rcfg.get("cv", 10))
        cv = max(2, min(cv, len(epochs)))
        ar = AutoReject(
            n_interpolate=n_interpolate,
            consensus=consensus,
            cv=cv,
            random_state=rcfg.get("random_state", 42),
            n_jobs=resolve_n_jobs(cfg or {}, rcfg.get("n_jobs"), default=1),
            verbose=False,
        )
        cleaned, reject_log = ar.fit_transform(epochs, return_log=True)
        dropped = int(np.sum(reject_log.bad_epochs))
        repaired = int(np.sum(np.any(reject_log.labels == 2, axis=1)))
        log.info(
            "AutoReject local %s: n_interpolate=%s, consensus=%s, cv=%d, "
            "repaired_epochs=%d, dropped_epochs=%d",
            condition, ar.n_interpolate_, ar.consensus_, cv, repaired, dropped,
        )
        return cleaned
    except Exception as exc:                            # noqa: BLE001
        log.warning("autoreject failed: %s; returning epochs unchanged.", exc)
        return epochs


def _threshold(epochs: mne.Epochs, thr: float | None) -> mne.Epochs:
    if thr is None:
        return epochs
    log.info("Voltage-threshold rejection: eeg=%g V", thr)
    epochs.drop_bad(reject=dict(eeg=float(thr)))
    return epochs
