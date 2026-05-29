"""Shape-decomposition (basis-expansion) features for the late-CNV window.

Instead of collapsing each channel's time course into per-bin means, these
helpers describe the *shape* of the trace with a handful of basis coefficients.
For a slow cortical potential like the CNV this preserves within-bin structure
(slope, curvature) that mean-binning discards, while *reducing* dimensionality.

Three bases are provided:

1. ``polynomial_basis_features`` — orthogonal (Legendre/Chebyshev) polynomial
   coefficients. Fixed, per-epoch, **leakage-free** (each epoch is fit on its
   own samples only), so this is computed at feature-extraction time as an
   opt-in ``features.blocks`` entry. Coefficient k carries an interpretable
   meaning: c0 = overall level, c1 = linear slope (the CNV ramp), c2 = curvature.

2. ``bspline_basis_features`` — least-squares B-spline coefficients over a
   clamped knot vector. Also fixed and per-epoch (leakage-free); better than a
   low-order polynomial when the discriminative feature is *localized* in time.

3. ``FunctionalPCABasis`` — data-driven functional PCA. The eigen-basis is
   *learned from the data*, so to avoid leakage it MUST be fit on the training
   fold only. It is therefore implemented as a scikit-learn transformer (not an
   extraction-time block) and wired into the per-fold pipeline in
   ``models.train``. It operates on the cached per-bin amplitude columns, which
   act as the discretised time course for each channel.

The numpy core of each method is split into an ``_*_coeffs`` helper that takes
plain ``(n_epochs, n_channels, n_times)`` arrays, so the math is unit-testable
without constructing an MNE ``Epochs`` object. ``mne`` is imported lazily inside
the public wrappers for the same reason.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from numpy.polynomial import chebyshev as _cheb
from numpy.polynomial import legendre as _leg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

from ..logging_utils import get_logger


log = get_logger(__name__)

SUPPORTED_POLY_KINDS = ("legendre", "chebyshev")


# ---------------------------------------------------------------------------
# Numpy cores (MNE-free, unit-testable)
# ---------------------------------------------------------------------------
def _normalise_time(times: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map sample times onto ``[lo, hi]`` (degenerate windows map to ``lo``)."""
    t = np.asarray(times, dtype=float)
    span = t[-1] - t[0]
    if span <= 0:
        return np.full_like(t, lo)
    return lo + (hi - lo) * (t - t[0]) / span


def _poly_design(times: np.ndarray, *, degree: int, kind: str) -> np.ndarray:
    """Return the (n_times, degree+1) orthogonal-polynomial design matrix."""
    if kind not in SUPPORTED_POLY_KINDS:
        raise ValueError(f"poly kind must be one of {SUPPORTED_POLY_KINDS}; got {kind!r}")
    x = _normalise_time(times, -1.0, 1.0)
    if kind == "legendre":
        return _leg.legvander(x, degree)
    return _cheb.chebvander(x, degree)


def _bspline_design(times: np.ndarray, *, n_knots: int, degree: int) -> np.ndarray:
    """Return the (n_times, n_basis) clamped B-spline design matrix.

    ``n_basis = degree + 1 + n_knots`` (n_knots = number of *interior* knots).
    """
    from scipy.interpolate import BSpline

    x = _normalise_time(times, 0.0, 1.0)
    x = np.clip(x, 0.0, 1.0)
    interior = np.linspace(0.0, 1.0, n_knots + 2)[1:-1]
    knots = np.concatenate((np.zeros(degree + 1), interior, np.ones(degree + 1)))
    return np.asarray(BSpline.design_matrix(x, knots, degree).todense())


def _project(data: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Least-squares projection of each channel time course onto ``design``.

    ``data`` is ``(n_epochs, n_channels, n_times)``; ``design`` is
    ``(n_times, n_basis)``. Returns ``(n_epochs, n_channels, n_basis)`` of
    least-squares coefficients (shared design ⇒ one pseudo-inverse for all).
    """
    n_times = design.shape[0]
    if data.shape[-1] != n_times:
        raise ValueError(
            f"time axis mismatch: data has {data.shape[-1]} samples, "
            f"design expects {n_times}"
        )
    if design.shape[1] > n_times:
        raise ValueError(
            f"basis size ({design.shape[1]}) exceeds available time samples "
            f"({n_times}); reduce degree / n_knots or widen the window"
        )
    pinv = np.linalg.pinv(design)                 # (n_basis, n_times)
    return np.einsum("ect,kt->eck", data, pinv)   # (n_epochs, n_channels, n_basis)


def _poly_coeffs(data, times, *, degree: int, kind: str = "legendre") -> np.ndarray:
    return _project(np.asarray(data, dtype=float), _poly_design(times, degree=degree, kind=kind))


def _bspline_coeffs(data, times, *, n_knots: int, degree: int) -> np.ndarray:
    return _project(np.asarray(data, dtype=float), _bspline_design(times, n_knots=n_knots, degree=degree))


def _coeffs_to_frame(
    coeffs: np.ndarray,
    *,
    ch_names: list[str],
    prefix: str,
    epoch_index: np.ndarray,
) -> pd.DataFrame:
    """Flatten ``(n_epochs, n_channels, n_basis)`` into wide ``{prefix}_{ch}_{k}``."""
    n_epochs, n_channels, n_basis = coeffs.shape
    if n_channels != len(ch_names):
        raise ValueError("channel count does not match ch_names")
    out = {"epoch": np.asarray(epoch_index)}
    for c, ch in enumerate(ch_names):
        for k in range(n_basis):
            out[f"{prefix}_{ch}_{k}"] = coeffs[:, c, k]
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Public extraction-time blocks (leakage-free)
# ---------------------------------------------------------------------------
def polynomial_basis_features(
    epochs,
    *,
    degree: int,
    ch_names: list[str],
    kind: str = "legendre",
) -> pd.DataFrame:
    """Per-epoch orthogonal-polynomial coefficients → wide ``poly_{ch}_{k}``."""
    data = epochs.get_data(picks=ch_names)
    coeffs = _poly_coeffs(data, epochs.times, degree=degree, kind=kind)
    log.info("[basis] polynomial (%s, degree=%d): %d coeff(s)/channel",
             kind, degree, coeffs.shape[-1])
    return _coeffs_to_frame(coeffs, ch_names=ch_names, prefix="poly",
                            epoch_index=epochs.selection)


def bspline_basis_features(
    epochs,
    *,
    n_knots: int,
    degree: int,
    ch_names: list[str],
) -> pd.DataFrame:
    """Per-epoch least-squares B-spline coefficients → wide ``bspl_{ch}_{k}``."""
    data = epochs.get_data(picks=ch_names)
    coeffs = _bspline_coeffs(data, epochs.times, n_knots=n_knots, degree=degree)
    log.info("[basis] b-spline (degree=%d, interior knots=%d): %d coeff(s)/channel",
             degree, n_knots, coeffs.shape[-1])
    return _coeffs_to_frame(coeffs, ch_names=ch_names, prefix="bspl",
                            epoch_index=epochs.selection)


# ---------------------------------------------------------------------------
# Functional PCA — data-driven, leakage-safe (fit inside the CV fold)
# ---------------------------------------------------------------------------
# Amplitude time-course columns this transformer consumes. Matches the rich
# ``amp_w{width}_{ch}_mean_bin_{b}`` form and the legacy ``{ch}_bin_{b}`` form;
# source-space labels (containing '-') are excluded so we only fit fPCA on
# electrode amplitude trajectories.
_RICH_AMP = re.compile(r"^amp_w[^_]+_(?P<ch>.+?)_mean_bin_(?P<b>-?\d+)$")
_LEGACY_AMP = re.compile(r"^(?P<ch>[^-]+?)_bin_(?P<b>-?\d+)$")


def _amplitude_timecourse_columns(columns) -> dict[str, list[tuple[int, str]]]:
    """Group amplitude bin columns by channel, ordered by bin index.

    Returns ``{channel: [(bin_index, column_name), ...sorted...]}``.
    """
    by_channel: dict[str, list[tuple[int, str]]] = {}
    for col in columns:
        m = _RICH_AMP.match(col) or _LEGACY_AMP.match(col)
        if not m:
            continue
        ch = m.group("ch")
        by_channel.setdefault(ch, []).append((int(m.group("b")), col))
    for ch in by_channel:
        by_channel[ch].sort(key=lambda bc: bc[0])
    return by_channel


class FunctionalPCABasis(BaseEstimator, TransformerMixin):
    """Functional-PCA transformer over per-channel amplitude time courses.

    Treats each channel's ordered amplitude bins as a discretised time course,
    fits a PCA per channel on the training rows, and replaces those bin columns
    with the top-``n_components`` projection scores (``fpca_{ch}_{k}``). Columns
    that are not amplitude time courses are passed through unchanged when
    ``passthrough=True``.

    Being data-driven, it must be ``fit`` on training data only — wire it into
    the per-fold pipeline, never at extraction time.
    """

    def __init__(self, n_components: int = 4, *, passthrough: bool = True,
                 random_state: int = 1):
        self.n_components = n_components
        self.passthrough = passthrough
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None):
        groups = _amplitude_timecourse_columns(X.columns)
        self.channel_columns_: dict[str, list[str]] = {}
        self.pca_: dict[str, PCA] = {}
        for ch, ordered in groups.items():
            cols = [c for _, c in ordered]
            if len(cols) < 2:
                continue
            k = min(int(self.n_components), len(cols))
            pca = PCA(n_components=k, random_state=self.random_state)
            pca.fit(X[cols].to_numpy(dtype=float))
            self.channel_columns_[ch] = cols
            self.pca_[ch] = pca
        consumed = {c for cols in self.channel_columns_.values() for c in cols}
        self.consumed_columns_ = consumed
        self.passthrough_columns_ = [c for c in X.columns if c not in consumed]
        if not self.pca_:
            log.warning("[fpca] no amplitude time-course columns found; "
                        "transform is a pass-through.")
        else:
            log.info("[fpca] fit on %d channel(s), %d component(s) each",
                     len(self.pca_), int(self.n_components))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frames = []
        if self.passthrough and self.passthrough_columns_:
            frames.append(X[self.passthrough_columns_].reset_index(drop=True))
        for ch, pca in self.pca_.items():
            cols = self.channel_columns_[ch]
            scores = pca.transform(X[cols].to_numpy(dtype=float))
            frames.append(pd.DataFrame(
                scores,
                columns=[f"fpca_{ch}_{k}" for k in range(scores.shape[1])],
            ))
        if not frames:
            return X.reset_index(drop=True)
        return pd.concat(frames, axis=1)
