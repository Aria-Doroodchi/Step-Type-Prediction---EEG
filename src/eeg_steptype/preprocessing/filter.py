"""Line-noise and bandpass filtering.

Filter values come from ``cfg.preprocessing.filter``. Per-participant filter
tuning is available only when full participant override mode is enabled.
"""

from __future__ import annotations

import mne
import numpy as np

from ..logging_utils import get_logger


log = get_logger(__name__)


def apply_line_noise_removal(raw: mne.io.BaseRaw, cfg: dict) -> mne.io.BaseRaw:
    """Remove line noise with ZapLine from EEG channels only."""
    line_cfg = cfg.get("preprocessing", {}).get("line_noise", {})
    method = line_cfg.get("method", "zapline")
    if method in (None, "none"):
        log.info("Line-noise removal disabled")
        return raw
    if method != "zapline":
        raise ValueError(f"Unsupported line-noise removal method: {method!r}")

    freq = float(line_cfg.get("freq", 60))
    nremove = int(line_cfg.get("nremove", 1))
    if nremove <= 0:
        log.info("ZapLine skipped because nremove=%d", nremove)
        return raw
    if not raw.preload:
        raise RuntimeError("ZapLine requires raw data to be preloaded")

    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(picks) == 0:
        log.warning("No EEG channels found; skipping ZapLine")
        return raw

    log.info(
        "ZapLine line-noise removal: freq=%s Hz, nremove=%d, channels=%d",
        freq, nremove, len(picks),
    )
    data = raw.get_data(picks=picks).T[:, :, np.newaxis]
    cleaned, _artifact = _dss_line(
        data,
        fline=freq,
        sfreq=float(raw.info["sfreq"]),
        nremove=nremove,
    )
    raw._data[picks, :] = cleaned[:, :, 0].T
    return raw


def apply_notch(raw: mne.io.BaseRaw, cfg: dict) -> mne.io.BaseRaw:
    """Backward-compatible alias; notch filtering was replaced by ZapLine."""
    return apply_line_noise_removal(raw, cfg)


def apply_bandpass(raw: mne.io.BaseRaw, cfg: dict) -> mne.io.BaseRaw:
    l, h = cfg["preprocessing"]["filter"]["bandpass"]
    log.info("Bandpass %s–%s Hz", l, h)
    raw.filter(l_freq=l, h_freq=h)
    return raw


def make_analysis_copy(raw: mne.io.BaseRaw, cfg: dict) -> mne.io.BaseRaw:
    """Return the 0.1-40 Hz analysis copy ICA will clean."""
    log.info("Creating analysis copy for ICA application")
    return apply_bandpass(raw.copy(), cfg)


def make_ica_training_copy(raw: mne.io.BaseRaw, cfg: dict) -> mne.io.BaseRaw:
    """Return the full-duration high-pass copy used to fit ICA."""
    hp = float(cfg["preprocessing"]["filter"].get("ica_highpass", 1.0))
    log.info("Creating ICA training copy: high-pass %.2f Hz over full data", hp)
    train = raw.copy()
    train.filter(l_freq=hp, h_freq=None)
    return train


def _dss_line(data: np.ndarray, *, fline: float, sfreq: float, nremove: int):
    try:
        from meegkit.dss import dss_line
    except Exception as exc:                            # noqa: BLE001
        raise RuntimeError(
            "ZapLine requires the 'meegkit' package. Install it with "
            "`pip install -e .` or `pip install meegkit`."
        ) from exc
    return dss_line(data, fline=fline, sfreq=sfreq, nremove=nremove)
