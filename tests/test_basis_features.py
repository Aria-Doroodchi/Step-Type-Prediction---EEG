"""Unit tests for the shape-decomposition feature helpers.

These exercise the numpy cores and the FunctionalPCABasis transformer with
synthetic data, so they run without MNE / a real dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eeg_steptype.features.basis import (
    FunctionalPCABasis,
    _bspline_coeffs,
    _bspline_design,
    _poly_coeffs,
    _poly_design,
)


@pytest.fixture
def times():
    # 1.0–2.0 s window sampled at ~256 Hz -> matches the late-CNV window.
    return np.linspace(1.0, 2.0, 257)


def test_poly_design_shape_and_constant(times):
    design = _poly_design(times, degree=4, kind="legendre")
    assert design.shape == (len(times), 5)
    # First Legendre column is the constant 1.
    np.testing.assert_allclose(design[:, 0], 1.0)


def test_polynomial_recovers_linear_ramp(times):
    # A pure downward ramp should be captured almost entirely by c0 (level)
    # and c1 (slope); higher-order coefficients must be ~0.
    slope, offset = -8.0, 0.5
    ramp = offset + slope * (times - times[0])
    data = ramp[None, None, :]  # (1 epoch, 1 channel, T)
    coeffs = _poly_coeffs(data, times, degree=4, kind="legendre")[0, 0]
    assert abs(coeffs[2]) < 1e-6
    assert abs(coeffs[3]) < 1e-6
    # Reconstruction is essentially exact for a degree-1 signal.
    recon = _poly_design(times, degree=4, kind="legendre") @ coeffs
    assert np.corrcoef(recon, ramp)[0, 1] > 0.999999


def test_polynomial_reconstruction_smooth_signal(times):
    # A smooth quadratic-plus-sine CNV-like trace: degree-4 should reconstruct
    # it with high fidelity (far better than its 8 bin-means would).
    x = (times - times[0])
    trace = -6 * x + 3 * x**2 + 0.4 * np.sin(2 * np.pi * x)
    data = trace[None, None, :]
    coeffs = _poly_coeffs(data, times, degree=4, kind="legendre")[0, 0]
    recon = _poly_design(times, degree=4, kind="legendre") @ coeffs
    rel_err = np.linalg.norm(recon - trace) / np.linalg.norm(trace)
    assert rel_err < 0.05


def test_bspline_design_basis_count(times):
    design = _bspline_design(times, n_knots=3, degree=3)
    # n_basis = degree + 1 + n_knots
    assert design.shape == (len(times), 3 + 1 + 3)
    # B-spline basis functions form a partition of unity (sum to 1 per sample).
    np.testing.assert_allclose(design.sum(axis=1), 1.0, atol=1e-8)


def test_bspline_reconstruction(times):
    x = (times - times[0])
    trace = -6 * x + 3 * x**2
    data = trace[None, None, :]
    coeffs = _bspline_coeffs(data, times, n_knots=4, degree=3)[0, 0]
    recon = _bspline_design(times, n_knots=4, degree=3) @ coeffs
    rel_err = np.linalg.norm(recon - trace) / np.linalg.norm(trace)
    assert rel_err < 0.02


def test_project_rejects_oversized_basis(times):
    short = np.linspace(0, 1, 3)
    data = np.zeros((1, 1, 3))
    with pytest.raises(ValueError):
        _bspline_coeffs(data, short, n_knots=10, degree=3)


def test_multi_epoch_multi_channel_shapes(times):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((12, 5, len(times)))
    coeffs = _poly_coeffs(data, times, degree=3, kind="chebyshev")
    assert coeffs.shape == (12, 5, 4)


def _make_lowrank_amplitude_frame(n=60, n_bins=8, n_ch=3, seed=0):
    """Synthetic frame of `{ch}_bin_{b}` columns from a rank-1 time course."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_bins)
    mode = np.exp(-t)  # shared temporal shape
    cols = {}
    for c in range(n_ch):
        amp = rng.standard_normal(n)  # per-epoch loading on the shared mode
        signal = np.outer(amp, mode) + 0.01 * rng.standard_normal((n, n_bins))
        for b in range(n_bins):
            cols[f"Ch{c}_bin_{b}"] = signal[:, b]
    cols["psd_extra"] = rng.standard_normal(n)  # passthrough, not a time course
    return pd.DataFrame(cols)


def test_fpca_reduces_and_passes_through():
    df = _make_lowrank_amplitude_frame()
    fpca = FunctionalPCABasis(n_components=2).fit(df)
    out = fpca.transform(df)
    # 3 channels x 2 components + 1 passthrough column.
    assert sorted(out.columns) == sorted(
        [f"fpca_Ch{c}_{k}" for c in range(3) for k in range(2)] + ["psd_extra"]
    )
    assert len(out) == len(df)
    # The rank-1 construction means the first component dominates variance.
    assert fpca.pca_["Ch0"].explained_variance_ratio_[0] > 0.95


def test_fpca_passthrough_only_when_no_timecourse():
    df = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0) ** 2})
    fpca = FunctionalPCABasis(n_components=3).fit(df)
    out = fpca.transform(df)
    assert list(out.columns) == ["a", "b"]
    assert not fpca.pca_
