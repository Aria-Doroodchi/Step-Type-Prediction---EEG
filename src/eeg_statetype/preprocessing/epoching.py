"""Build state-task epochs.

Two epoch types per condition:
* **analysis epochs** — the 2 s windows ([-0.1, +2.0], baseline (None, 0)) used
  for the window feature blocks (CSD branch) and source localization (avg-ref
  branch). Carry metadata ``condition``/``orig_index``/``n_estim``.
* **SEP epochs** — short windows ([-0.1, +0.3], baseline (-0.05, 0)) around each
  offset-corrected e-stim (avg-ref branch), with metadata ``parent_epoch``
  linking each to the analysis epoch (original event index) it belongs to.

``orig_index`` / ``parent_epoch`` are both in original-event-index space, which is
exactly what ``epochs.selection`` reports, so the SEP block aligns with the
window blocks even after AutoReject drops some analysis epochs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import mne

from ..logging_utils import get_logger
from .events_state import ONSET_CODE


log = get_logger(__name__)


def build_analysis_epochs(
    raw: mne.io.BaseRaw,
    analysis_events: np.ndarray,
    parent: np.ndarray,
    cfg: dict,
    condition: str,
) -> mne.Epochs:
    """Build the 2 s analysis epochs with per-epoch metadata."""
    ep = cfg["preprocessing"]["epoch"]
    tmin, tmax = float(ep["tmin"]), float(ep["tmax"])
    baseline = tuple(ep.get("baseline", [None, 0.0]))

    n = len(analysis_events)
    counts = _estim_counts(parent, n)
    meta = pd.DataFrame({
        "condition": condition,
        "orig_index": np.arange(n, dtype=int),
        "n_estim": counts,
    })

    epochs = mne.Epochs(
        raw, analysis_events, event_id={str(ONSET_CODE): ONSET_CODE},
        tmin=tmin, tmax=tmax, baseline=baseline, metadata=meta,
        preload=True, reject=None, verbose=False,
    )
    log.info("[%s] analysis epochs: n=%d (range [%.2f, %.2f]s, ~%.2f e-stim/epoch)",
             condition, len(epochs), tmin, tmax,
             float(np.mean(counts)) if n else 0.0)
    return epochs


def build_sep_epochs(
    raw_avgref: mne.io.BaseRaw,
    estim_events: np.ndarray,
    parent: np.ndarray,
    cfg: dict,
    condition: str,
) -> mne.Epochs | None:
    """Build per-e-stim SEP epochs (avg-ref) with ``parent_epoch`` metadata."""
    if len(estim_events) == 0:
        log.warning("[%s] no e-stims; no SEP epochs.", condition)
        return None
    scfg = cfg["features"]["sep"]
    stim_code = int(cfg["state_events"]["stim"])
    tmin, tmax = float(scfg["tmin"]), float(scfg["tmax"])
    baseline = tuple(scfg.get("baseline", [-0.05, 0.0]))
    reject = {"eeg": float(scfg.get("reject_uv", 150.0)) * 1e-6}

    meta = pd.DataFrame({"condition": condition, "parent_epoch": parent.astype(int)})
    epochs = mne.Epochs(
        raw_avgref, estim_events, event_id={str(stim_code): stim_code},
        tmin=tmin, tmax=tmax, baseline=baseline, metadata=meta,
        preload=True, reject=reject, verbose=False,
    )
    log.info("[%s] SEP epochs: %d/%d kept (reject %.0f µV p2p), %d parent epochs covered",
             condition, len(epochs), len(estim_events),
             float(scfg.get("reject_uv", 150.0)),
             int(epochs.metadata["parent_epoch"].nunique()) if len(epochs) else 0)
    return epochs


def _estim_counts(parent: np.ndarray, n: int) -> np.ndarray:
    counts = np.zeros(n, dtype=int)
    if parent is not None and len(parent):
        idx, c = np.unique(parent, return_counts=True)
        ok = (idx >= 0) & (idx < n)
        counts[idx[ok]] = c[ok]
    return counts
