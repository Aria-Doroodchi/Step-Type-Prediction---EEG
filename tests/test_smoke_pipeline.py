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


def _synthetic_raw(cfg: dict, participant_id: str, *, n_pairs_per_condition: int = 6):
    """Create a tiny continuous Raw with paired condition/response triggers."""
    import mne

    sfreq = 100.0
    ch_names = ["Cz", "Fz", "Pz", "C3", "C4", "F3", "F4", "P3", "P4", "Oz", "Stim"]
    ch_types = ["eeg"] * 10 + ["stim"]
    duration = n_pairs_per_condition * 2 * 3.0 + 3.0
    n_samples = int(duration * sfreq)
    rng = np.random.default_rng(abs(hash(participant_id)) % (2 ** 32))

    data = rng.normal(0, 2e-6, size=(len(ch_names), n_samples))
    data[-1, :] = 0.0
    events = []
    sample = int(0.5 * sfreq)
    for i in range(n_pairs_per_condition * 2):
        condition = "One" if i % 2 == 0 else "Two"
        condition_code = int(cfg["events"]["one"] if condition == "One" else cfg["events"]["two"])
        response_code = int(cfg["events"]["response"])
        cond_sample = sample
        resp_sample = sample + int(0.5 * sfreq)
        data[-1, cond_sample] = condition_code
        data[-1, resp_sample] = response_code
        events.append((condition, resp_sample))

        # Add a condition-specific late-CNV-scale signal so downstream
        # features/training have nonconstant structure.
        window = slice(resp_sample + int(1.0 * sfreq), resp_sample + int(2.0 * sfreq))
        data[0, window] += (1e-6 if condition == "One" else 2e-6)
        sample += int(3.0 * sfreq)

    info = mne.create_info(ch_names, sfreq=sfreq, ch_types=ch_types)
    info.set_montage("standard_1020", on_missing="ignore")
    return mne.io.RawArray(data, info, verbose=False)


def test_two_participant_full_workflow_smoke(isolated_cfg, monkeypatch):
    """Run preprocess -> source -> features -> train for two synthetic participants."""
    from eeg_steptype.io import epochs_path, features_path, run_dir, source_epochs_path, src_csv_path
    from eeg_steptype.preprocessing import pipeline as preprocess
    from eeg_steptype.source_localization import pipeline as src_loc
    from eeg_steptype.features import assemble
    from eeg_steptype.models.train import run as run_train

    cfg = isolated_cfg
    cfg["participants"] = ["S01", "S02"]
    cfg["participant_overrides"]["mode"] = "none"
    cfg["preprocessing"]["line_noise"]["method"] = "none"
    cfg["preprocessing"]["bads"]["method"] = "manual"
    cfg["preprocessing"]["asr"]["enabled"] = False
    cfg["preprocessing"]["ica"]["n_components"] = 2
    cfg["preprocessing"]["reject"]["method"] = "threshold"
    cfg["features"]["blocks"] = ["amplitude", "slopes", "psd", "src"]
    cfg["features"]["freqs"] = {"fmin": 4.0, "fmax": 8.0, "fstep": 4.0}
    cfg["features"]["freq_bands"] = {"Theta": [4.0, 8.0]}
    cfg["source_localization"]["bin_n"] = cfg["features"]["bin_n"]
    cfg["modeling"]["cv"] = {
        "mode": "repeated_stratified",
        "n_splits": 2,
        "n_repeats": 1,
        "inner_splits": 2,
        "chronological_check": False,
    }

    monkeypatch.setattr(
        preprocess,
        "load_raw",
        lambda cfg, participant_id: _synthetic_raw(cfg, participant_id),
    )
    monkeypatch.setattr(src_loc, "load_labels", lambda cfg: ([object()], ["synthetic_label"]))
    monkeypatch.setattr(src_loc, "build_forward", lambda info, cfg, participant_id=None: {"src": None})
    monkeypatch.setattr(src_loc, "compute_noise_cov", lambda epochs: object())
    monkeypatch.setattr(src_loc, "build_inverse", lambda info, fwd, noise_cov: object())
    monkeypatch.setattr(src_loc, "apply_to_evoked", lambda evoked, inv_op, cfg: evoked)
    monkeypatch.setattr(
        src_loc,
        "extract_label_courses",
        lambda stc, labels, src: np.atleast_2d(stc.data[:1]),
    )
    monkeypatch.setattr(src_loc, "validate_source_assets", lambda cfg: None)

    for pid in cfg["participants"]:
        preprocess.run(pid, cfg, force=True)
        for cond in cfg["conditions"]:
            assert epochs_path(cfg, pid, cond).exists()
            assert source_epochs_path(cfg, pid, cond).exists()

    for pid in cfg["participants"]:
        src_loc.run(pid, cfg, force=True)
        for cond in cfg["conditions"]:
            assert src_csv_path(cfg, pid, cond).exists()

    for pid in cfg["participants"]:
        assemble.run(pid, cfg, force=True)
        for cond in cfg["conditions"]:
            assert features_path(cfg, pid, cond).exists()

    metrics = run_train(cfg, model="logistic", run_id="two_participant_full_smoke")
    assert set(metrics["participant_id"]) == {"S01", "S02"}
    assert {"cv_mode", "fold", "overall_accuracy", "search_method"}.issubset(metrics.columns)
    assert (run_dir(cfg, "two_participant_full_smoke") / "metrics.csv").exists()
    assert (run_dir(cfg, "two_participant_full_smoke") / "rollup.csv").exists()


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
