"""Notch and bandpass filtering.

Filter values come from ``cfg.preprocessing.filter`` and can be overridden
per participant (e.g. P10 final highpass=30 Hz, P11 lowpass=0.2 Hz).
"""

from __future__ import annotations

import mne

from ..logging_utils import get_logger


log = get_logger(__name__)


def apply_notch(raw: mne.io.BaseRaw, cfg: dict) -> mne.io.BaseRaw:
    freqs = cfg["preprocessing"]["filter"]["notch"]
    if freqs:
        log.info("Notch filter at %s Hz", freqs)
        raw.notch_filter(freqs)
    return raw


def apply_bandpass(raw: mne.io.BaseRaw, cfg: dict) -> mne.io.BaseRaw:
    l, h = cfg["preprocessing"]["filter"]["bandpass"]
    log.info("Bandpass %s–%s Hz", l, h)
    raw.filter(l_freq=l, h_freq=h)
    return raw
