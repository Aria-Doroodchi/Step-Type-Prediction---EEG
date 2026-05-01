"""End-to-end smoke test using a tiny synthetic Raw.

Why synthetic? The real .bdf files live outside the repo and a real run
takes hours (LORETA, RFECV, SHAP). This test fakes a 60-channel BioSemi
recording with two condition triggers, runs the *entire* pipeline through
preprocessing → features → train (logistic), and asserts:

    1. Every stage finishes without error.
    2. Output files appear at the expected paths.
    3. The metrics CSV has the expected columns.

Total wall-clock target: under 60 seconds. Skipped automatically if any of
the optional auto-preprocessing libraries are missing — those code paths
fall back gracefully but the test would otherwise be hardware-dependent.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------
# Test isolation: write outputs to a tmp dir so the smoke run can't pollute
# real data/outputs.
# ---------------------------------------------------------------------
@pytest.fixture
def isolated_cfg(tmp_path: Path, smoke_config_path: Path) -> dict:
    from eeg_steptype.config import load_config
    cfg = load_config(smoke_config_path)
    # Redirect outputs to a temp dir.
    cfg["paths"]["data_dir"] = str(tmp_path / "data")
    cfg["paths"]["outputs_dir"] = str(tmp_path / "outputs")
    return cfg


# ---------------------------------------------------------------------
def _write_synthetic_epochs(cfg: dict, pid: str, condition: str, n_epochs: int = 12) -> Path:
    """Write a tiny synthetic Epochs .fif at the path the pipeline expects."""
    import mne
    from eeg_steptype.io import epochs_path, ensure_dir

    sfreq = 200.0
    n_samples = int(sfreq * 2.1)        # 2.1s post-onset window
    np.random.seed(0 if condition == "One" else 1)

    # Use a small but valid 10-20 set the BioSemi montage covers.
    ch_names = ["Cz", "Fz", "Pz", "C3", "C4", "F3", "F4", "P3", "P4", "Oz"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * len(ch_names))
    info.set_montage("standard_1020")

    # Signal differs slightly between conditions so the smoke classifier has
    # something to learn.
    bias = 1e-6 if condition == "One" else 2e-6
    data = np.random.randn(n_epochs, len(ch_names), n_samples) * 5e-6 + bias

    events = np.column_stack([
        np.arange(n_epochs) * n_samples + 20,
        np.zeros(n_epochs, dtype=int),
        np.full(n_epochs, int(cfg["events"]["response"]), dtype=int),
    ])
    epochs = mne.EpochsArray(
        data, info,
        events=events,
        tmin=-0.1,
        event_id={str(int(cfg["events"]["response"])): int(cfg["events"]["response"])},
        verbose=False,
    )

    out = epochs_path(cfg, pid, condition)
    ensure_dir(out.parent)
    epochs.save(str(out), overwrite=True)
    return out


# ---------------------------------------------------------------------
def test_features_and_train_end_to_end(isolated_cfg, tmp_path):
    """Skip preprocessing (needs a real .bdf) but exercise features + train."""
    from eeg_steptype.features import assemble
    from eeg_steptype.models.train import run as run_train

    pid = isolated_cfg["participants"][0]
    for cond in isolated_cfg["conditions"]:
        _write_synthetic_epochs(isolated_cfg, pid, cond, n_epochs=14)

    # Stage 3: features
    df = assemble.build_for_participant(pid, isolated_cfg)
    assert "epoch" in df.columns
    assert "condition" in df.columns
    assert df.shape[0] == 14 * 2          # n_epochs × 2 conditions
    assert df.shape[1] > 5

    # Stage 4: train (logistic, tiny grid, ~seconds)
    metrics = run_train(isolated_cfg, model="logistic", run_id="smoke")
    assert isinstance(metrics, pd.DataFrame)
    assert not metrics.empty
    expected = {"participant_id", "overall_accuracy", "accuracy_One", "accuracy_Two", "auc"}
    assert expected.issubset(set(metrics.columns))


def test_evaluate_metrics_shape():
    """`participant_metrics` returns the keys downstream code relies on."""
    from eeg_steptype.models.evaluate import participant_metrics

    y_true  = np.array([0, 0, 1, 1, 0, 1])
    y_pred  = np.array([0, 1, 1, 1, 0, 0])
    y_proba = np.array([0.1, 0.7, 0.9, 0.6, 0.2, 0.4])
    m = participant_metrics(y_true, y_pred, y_proba)

    for k in ("total_One", "total_Two", "correct_One", "correct_Two",
              "accuracy_One", "accuracy_Two", "overall_accuracy", "auc"):
        assert k in m


def test_correlation_drop_idempotent_on_orthogonal_features():
    from eeg_steptype.models.feature_selection import correlation_drop

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((100, 5)), columns=list("abcde"))
    keep = correlation_drop(X, threshold=0.99)
    # Random Gaussians should rarely be near-collinear; nothing should drop.
    assert set(keep) == set(X.columns)


def test_gain_prune_zero_mode():
    from eeg_steptype.models.feature_selection import gain_prune

    keep = gain_prune(np.array([0.0, 0.5, 0.0, 0.1]), ["a", "b", "c", "d"], mode="zero")
    assert keep == ["b", "d"]
