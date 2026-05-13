"""Artifact Subspace Reconstruction (ASR) for transient burst removal."""

from __future__ import annotations

import mne
import numpy as np

from ..logging_utils import get_logger


log = get_logger(__name__)


# ---------------------------------------------------------------------------
# meegkit 0.1.9 + numpy 2.x compatibility shim
# ---------------------------------------------------------------------------
def _patch_meegkit_for_numpy2() -> None:
    """Make ``meegkit.utils.asr.fit_eeg_distribution`` numpy-2.x safe.

    In ``fit_eeg_distribution`` (meegkit 0.1.9):

        alpha = (opt_lu[1] - opt_lu[0]) / np.diff(opt_bounds)
        mu    = opt_lu[0] - opt_bounds[0] * alpha

    ``np.diff(opt_bounds)`` on a length-2 input returns a *1-element array*,
    not a scalar, so ``alpha`` and ``mu`` come back as 1-element arrays. Under
    numpy < 2 the caller's ``mu[ichan] = ...`` assignment auto-extracted the
    scalar; under numpy >= 2 the same assignment raises
    ``ValueError: setting an array element with a sequence`` (with an
    underlying ``TypeError: only 0-dimensional arrays can be converted to
    Python scalars``).

    This wrapper coerces each return value to a Python float when it is a
    1-element sequence, leaving genuine multi-element outputs untouched. The
    patch is idempotent and patches both binding sites used by
    ``asr_calibrate`` (the function lives in ``meegkit.utils.asr`` and is
    re-imported into ``meegkit.asr``'s namespace).
    """
    import meegkit.utils.asr as _utils
    import meegkit.asr as _top

    original = getattr(_utils, "fit_eeg_distribution", None)
    if original is None or getattr(original, "_numpy2_patched", False):
        return  # nothing to patch or already done

    def _to_scalar(value):
        arr = np.asarray(value)
        if arr.size == 1:
            return float(arr.item())
        return value

    def patched(*args, **kwargs):
        mu, sig, alpha, beta = original(*args, **kwargs)
        return _to_scalar(mu), _to_scalar(sig), _to_scalar(alpha), _to_scalar(beta)

    patched._numpy2_patched = True            # type: ignore[attr-defined]
    patched.__wrapped__ = original            # type: ignore[attr-defined]

    # asr_calibrate references the symbol as bound in meegkit.asr at import
    # time, so we need to replace both bindings.
    _utils.fit_eeg_distribution = patched
    if getattr(_top, "fit_eeg_distribution", None) is original:
        _top.fit_eeg_distribution = patched

    log.debug("Patched meegkit.fit_eeg_distribution for numpy>=2 compatibility")


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
    _patch_meegkit_for_numpy2()
    return ASR(sfreq=sfreq, cutoff=cutoff, method=method, estimator=estimator)
