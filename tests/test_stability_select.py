"""Unit tests for complementary-pairs stability selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eeg_steptype.models.feature_selection import stability_select


def _make_data(n=120, n_informative=4, n_noise=40, seed=0):
    rng = np.random.default_rng(seed)
    Xi = rng.standard_normal((n, n_informative))
    w = np.array([2.5, -2.0, 1.8, -1.6])[:n_informative]
    logits = Xi @ w
    y = (logits + 0.3 * rng.standard_normal(n) > 0).astype(int)
    Xn = rng.standard_normal((n, n_noise))
    cols = (
        [f"sig_{i}" for i in range(n_informative)]
        + [f"noise_{j}" for j in range(n_noise)]
    )
    X = pd.DataFrame(np.hstack([Xi, Xn]), columns=cols)
    return X, pd.Series(y)


def test_informative_features_score_higher_than_noise():
    X, y = _make_data()
    kept, prob = stability_select(
        X, y, n_subsamples=40, n_lambda=12, threshold=0.6,
        max_features=None, random_state=0,
    )
    sig_prob = prob[[c for c in prob.index if c.startswith("sig_")]].mean()
    noise_prob = prob[[c for c in prob.index if c.startswith("noise_")]].mean()
    assert sig_prob > noise_prob
    # Most informative features should clear the threshold and be kept.
    assert sum(c.startswith("sig_") for c in kept) >= 3


def test_max_features_cap_respected():
    X, y = _make_data()
    kept, _ = stability_select(
        X, y, n_subsamples=20, n_lambda=8, threshold=0.0,
        max_features=5, min_features=1, random_state=1,
    )
    assert len(kept) == 5


def test_min_features_floor_when_threshold_too_high():
    X, y = _make_data()
    kept, _ = stability_select(
        X, y, n_subsamples=20, n_lambda=8, threshold=1.01,  # impossible
        max_features=None, min_features=7, random_state=2,
    )
    assert len(kept) == 7


def test_constant_columns_dropped():
    X, y = _make_data()
    X = X.copy()
    X["const"] = 3.0
    kept, prob = stability_select(
        X, y, n_subsamples=10, n_lambda=6, threshold=0.0,
        max_features=None, min_features=1, random_state=3,
    )
    assert "const" not in prob.index
    assert "const" not in kept


def test_deterministic_with_seed():
    X, y = _make_data()
    a, _ = stability_select(X, y, n_subsamples=16, n_lambda=6, random_state=7)
    b, _ = stability_select(X, y, n_subsamples=16, n_lambda=6, random_state=7)
    assert a == b
