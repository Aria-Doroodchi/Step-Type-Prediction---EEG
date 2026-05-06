"""Referencing helpers for ICA and analysis.

The preprocessing sequence uses a provisional common-average reference (CAR)
for ICA, restores the pre-CAR common signal, then applies current source
density (CSD) before epoching. Saved epochs are therefore CSD-transformed, not
CAR-referenced.
"""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
from mne.preprocessing import compute_current_source_density

from ..logging_utils import get_logger


log = get_logger(__name__)


@dataclass(frozen=True)
class CARState:
    picks: np.ndarray
    common_signal: np.ndarray


def apply_car(raw: mne.io.BaseRaw) -> tuple[mne.io.BaseRaw, CARState]:
    """Apply a provisional CAR projection and keep what is needed to undo it."""
    _require_preloaded(raw)
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(picks) == 0:
        log.warning("No EEG channels found; skipping provisional CAR")
        return raw, CARState(picks=picks, common_signal=np.empty((0, raw.n_times)))

    common_signal = raw.get_data(picks=picks).mean(axis=0, keepdims=True)
    raw._data[picks, :] -= common_signal
    log.info("Applied provisional CAR projection for ICA (%d EEG channels)", len(picks))
    return raw, CARState(picks=picks, common_signal=common_signal)


def undo_car(raw: mne.io.BaseRaw, state: CARState) -> mne.io.BaseRaw:
    """Restore the common signal removed by ``apply_car``."""
    _require_preloaded(raw)
    if len(state.picks) == 0:
        return raw
    raw._data[state.picks, :] += state.common_signal
    log.info("Restored pre-CAR common signal before CSD")
    return raw


def apply_csd(raw: mne.io.BaseRaw, cfg: dict) -> mne.io.BaseRaw:
    """Apply spherical-spline current source density before epoching."""
    csd_cfg = cfg.get("preprocessing", {}).get("reference", {}).get("csd", {})
    lambda2 = float(csd_cfg.get("lambda2", 1e-5))
    stiffness = int(csd_cfg.get("stiffness", 4))
    log.info("Applying CSD reference: lambda2=%s, stiffness=%s", lambda2, stiffness)
    return compute_current_source_density(
        raw,
        lambda2=lambda2,
        stiffness=stiffness,
        copy=False,
    )


def apply_average_reference(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Backward-compatible alias for the provisional CAR step."""
    raw, _ = apply_car(raw)
    return raw


def _require_preloaded(raw: mne.io.BaseRaw) -> None:
    if not raw.preload:
        raise RuntimeError("Referencing operations require raw data to be preloaded")
