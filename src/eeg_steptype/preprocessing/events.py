"""Find condition-paired response events.

Original logic from the per-participant scripts:

    For each row in the event stream:
        if row[2] == ONE_CODE   →  remember it as the active "One" condition
        elif row[2] == TWO_CODE →  remember it as the active "Two" condition
        elif row[2] == RESPONSE_CODE and row[1] == 0:
            attach this response to whatever condition was last active,
            then clear the active condition.

Codes default to One=256, Two=512, response=96 and are configurable via
``cfg.events``.
"""

from __future__ import annotations

import numpy as np
import mne

from ..logging_utils import get_logger


log = get_logger(__name__)


def find_step_events(raw: mne.io.BaseRaw, cfg: dict) -> dict[str, np.ndarray]:
    """Return {'One': ndarray, 'Two': ndarray, 'all': ndarray}.

    Each ndarray has shape (n_events, 3) and is suitable for `mne.Epochs`.
    """
    events_cfg = cfg["events"]
    code_one = int(events_cfg["one"])
    code_two = int(events_cfg["two"])
    code_resp = int(events_cfg["response"])

    all_events = mne.find_events(
        raw,
        stim_channel="Stim",
        initial_event=True,
        verbose=False,
        min_duration=0.002,
        shortest_event=0.002,
        consecutive=True,
        output="onset",
    )

    one_resp = _pair(all_events, condition_code=code_one,  response_code=code_resp)
    two_resp = _pair(all_events, condition_code=code_two,  response_code=code_resp)

    one_trigs = all_events[all_events[:, 2] == code_one]
    two_trigs = all_events[all_events[:, 2] == code_two]
    combined = np.concatenate(
        [a for a in (one_resp, two_resp, one_trigs, two_trigs) if len(a) > 0],
        axis=0,
    ) if any(len(a) for a in (one_resp, two_resp, one_trigs, two_trigs)) else \
        np.empty((0, 3), dtype=int)

    log.info("Events found: One→resp=%d, Two→resp=%d", len(one_resp), len(two_resp))
    return {"One": one_resp, "Two": two_resp, "all": combined}


def _pair(events: np.ndarray, condition_code: int, response_code: int) -> np.ndarray:
    """Return response events that follow a condition trigger of the given code."""
    out = []
    active = False
    for row in events:
        if row[2] == condition_code:
            active = True
        elif row[2] == response_code and row[1] == 0 and active:
            out.append(row)
            active = False
    return np.array(out, dtype=int) if out else np.empty((0, 3), dtype=int)
