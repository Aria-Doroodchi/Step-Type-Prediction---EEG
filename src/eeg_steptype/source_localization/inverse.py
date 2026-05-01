"""Noise covariance + inverse operator.

Both are participant-level (not epoch-level). Computed once and reused for
every epoch in the LORETA loop.
"""

from __future__ import annotations

import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse

from ..logging_utils import get_logger


log = get_logger(__name__)


def compute_noise_cov(epochs: mne.BaseEpochs) -> mne.Covariance:
    """Pre-stimulus baseline covariance over all epochs at once.

    The original code recomputed this per epoch, which doesn't add information
    and burns CPU.
    """
    log.info("Computing noise covariance over %d epochs …", len(epochs))
    return mne.compute_covariance(epochs, tmin=-0.1, tmax=0.0)


def build_inverse(
    info: mne.Info,
    fwd: mne.Forward,
    noise_cov: mne.Covariance,
) -> mne.minimum_norm.InverseOperator:
    log.info("Building inverse operator …")
    return make_inverse_operator(info, fwd, noise_cov)


def apply_to_evoked(
    evoked: mne.Evoked,
    inv_op: mne.minimum_norm.InverseOperator,
    cfg: dict,
) -> mne.SourceEstimate:
    sl = cfg["source_localization"]
    snr = float(sl.get("snr", 2.0))
    lambda2 = 1.0 / snr ** 2
    return apply_inverse(evoked, inv_op, lambda2, method=sl.get("method", "eLORETA"))
