"""Artifact Subspace Reconstruction (ASR) for transient burst removal."""

from __future__ import annotations

import mne

from ..logging_utils import get_logger


log = get_logger(__name__)


def apply_asr(raw: mne.io.BaseRaw, cfg: dict) -> mne.io.BaseRaw:
    """Apply ASR to EEG channels in-place.

    The config key ``preprocessing.asr.k`` maps to meegkit ASR's ``cutoff``.
    """
    asr_cfg = cfg.get("preprocessing", {}).get("asr", {})
    if not asr_cfg.get("enabled", True):
        log.info("ASR disabled")
        return raw
    if not raw.preload:
        raise RuntimeError("ASR requires raw data to be preloaded")

    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(picks) == 0:
        log.warning("No EEG channels found; skipping ASR")
        return raw

    k = float(asr_cfg.get("k", asr_cfg.get("cutoff", 20)))
    log.info("Applying ASR burst correction: k=%s, channels=%d", k, len(picks))

    cleaner = _make_asr(
        sfreq=float(raw.info["sfreq"]),
        cutoff=k,
        method=asr_cfg.get("method", "euclid"),
        estimator=asr_cfg.get("estimator", "scm"),
    )
    data = raw.get_data(picks=picks)
    cleaner.fit(data)
    raw._data[picks, :] = cleaner.transform(data)
    return raw


def _make_asr(*, sfreq: float, cutoff: float, method: str, estimator: str):
    try:
        from meegkit.asr import ASR
    except Exception as exc:                            # noqa: BLE001
        raise RuntimeError(
            "ASR requires the 'meegkit' package. Install it with "
            "`pip install -e .` or `pip install meegkit`."
        ) from exc
    return ASR(sfreq=sfreq, cutoff=cutoff, method=method, estimator=estimator)
