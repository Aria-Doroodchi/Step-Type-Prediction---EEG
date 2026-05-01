"""Average reference (projection)."""

from __future__ import annotations

import mne

from ..logging_utils import get_logger


log = get_logger(__name__)


def apply_average_reference(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    log.info("Setting EEG reference: average (projection)")
    raw.set_eeg_reference(ref_channels="average", projection=True).apply_proj()
    return raw
