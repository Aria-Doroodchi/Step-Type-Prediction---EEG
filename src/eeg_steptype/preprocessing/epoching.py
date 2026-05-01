"""Build per-condition epochs from filtered raw + paired response events."""

from __future__ import annotations

import numpy as np
import mne

from ..logging_utils import get_logger


log = get_logger(__name__)


def build_epochs(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    cfg: dict,
    condition: str,
) -> mne.Epochs:
    """Create epochs around the given event onsets.

    ``condition`` is just a label that ends up in the event_id dict and the
    Epochs metadata; the upstream `events.py` already filters by condition.
    """
    ep_cfg = cfg["preprocessing"]["epoch"]
    tmin = float(ep_cfg["tmin"])
    tmax = float(ep_cfg["tmax"])
    baseline = tuple(ep_cfg.get("baseline", [None, 0.0]))
    # YAML serialises None as null which becomes Python None already.

    response_code = int(cfg["events"]["response"])
    event_id = {str(response_code): response_code}

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        reject=None,            # rejection happens in reject.py via autoreject
        verbose=False,
    )
    log.info(
        "Built %s epochs: n=%d, range=[%.2fs, %.2fs]",
        condition, len(epochs), tmin, tmax,
    )
    return epochs
