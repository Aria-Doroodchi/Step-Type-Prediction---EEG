"""Tests for the pooled / partially-pooled training workflows.

Uses tiny synthetic multi-subject epochs (no real data, no preprocessing) to
verify the *data-sharing semantics* of each workflow -- which is what the
inner-vs-outer overfitting-gap experiment hinges on:

  * full pooling    -> trains on every OTHER subject (held-out subject leaks
                       nowhere into training).
  * partial pooling -> trains on the target's own split PLUS all other subjects.
  * per_participant -> trains on the target's own split only.

Kept under a few seconds by using a coarse feature set and the logistic model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

mne = pytest.importorskip("mne")


SUBJECTS = ["S01", "S02", "S03", "S04"]


@pytest.fixture
def pooled_cfg(tmp_path: Path, smoke_config_path: Path) -> dict:
    from eeg_steptype.config import load_config

    cfg = load_config(smoke_config_path)
    cfg["paths"]["data_dir"] = str(tmp_path / "data")
    cfg["paths"]["outputs_dir"] = str(tmp_path / "outputs")
    cfg["participants"] = list(SUBJECTS)
    # Coarse, cheap features; logistic keeps the search trivial.
    cfg["features"]["blocks"] = ["amplitude"]
    cfg["features"]["bin_n"] = 0.5
    cfg["modeling"]["cv"] = {
        "mode": "repeated_stratified", "n_splits": 2, "n_repeats": 1,
        "inner_splits": 2, "chronological_check": False,
    }
    cfg["modeling"]["k_best"] = 30
    cfg["modeling"]["feature_selection"] = {"method": "none"}
    cfg["modeling"]["gain_prune"] = {"enabled": False}
    cfg["modeling"]["shap_prune"] = {"enabled": False}
    return cfg


def _write_synthetic_epochs(cfg: dict, pid: str, condition: str, n_epochs: int, seed: int) -> None:
    from eeg_steptype.io import epochs_path, ensure_dir

    sfreq = 200.0
    n_samples = int(sfreq * 2.1)
    rng = np.random.default_rng(seed)
    ch_names = ["Cz", "Fz", "Pz", "C3", "C4", "F3", "F4", "P3", "P4", "Oz"]
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types=["eeg"] * len(ch_names))
    info.set_montage("standard_1020")

    bias = 1e-6 if condition == "One" else 2e-6
    data = rng.standard_normal((n_epochs, len(ch_names), n_samples)) * 5e-6 + bias
    events = np.column_stack([
        np.arange(n_epochs) * n_samples + 20,
        np.zeros(n_epochs, dtype=int),
        np.full(n_epochs, int(cfg["events"]["response"]), dtype=int),
    ])
    epochs = mne.EpochsArray(
        data, info, events=events, tmin=-0.1,
        event_id={str(int(cfg["events"]["response"])): int(cfg["events"]["response"])},
        verbose=False,
    )
    out = epochs_path(cfg, pid, condition)
    ensure_dir(out.parent)
    epochs.save(str(out), overwrite=True)


@pytest.fixture
def synthetic_cohort(pooled_cfg: dict) -> dict:
    for i, pid in enumerate(SUBJECTS):
        for cond in pooled_cfg["conditions"]:
            _write_synthetic_epochs(pooled_cfg, pid, cond, n_epochs=10, seed=i * 2 + (cond == "Two"))
    return pooled_cfg


# ---------------------------------------------------------------------------
def test_build_pooled_frame_concatenates_subjects(synthetic_cohort):
    from eeg_steptype.models import pooling

    pooled = pooling.build_pooled_frame(synthetic_cohort, SUBJECTS, "logistic")
    # 4 subjects x 2 conditions x 10 epochs.
    assert pooled.shape[0] == 4 * 2 * 10
    assert set(pooled["participant_id"].unique()) == set(SUBJECTS)
    X, y, groups = pooling._prep_xyg(pooled)
    # Identifier columns must not survive into the feature matrix.
    for col in ("participant_id", "condition", "block_id", "epoch"):
        assert col not in X.columns
    assert set(np.unique(groups)) == set(SUBJECTS)
    assert set(y.unique()) <= {0, 1}


def test_full_pooling_holds_out_each_subject(synthetic_cohort):
    from eeg_steptype.models import pooling

    rows = pooling.train_pooled(synthetic_cohort, "logistic", mode="full")
    df = pd.DataFrame(rows)
    # One row per held-out subject; training always uses the other 3 subjects.
    assert len(df) == len(SUBJECTS)
    assert set(df["held_out_participant"]) == set(SUBJECTS)
    assert (df["n_train_subjects"] == len(SUBJECTS) - 1).all()
    assert (df["pooling_mode"] == "full").all()
    assert {"auc", "inner_best_score"}.issubset(df.columns)


def test_partial_pooling_adds_all_subjects_to_training(synthetic_cohort):
    from eeg_steptype.models import pooling

    rows = pooling.train_pooled(synthetic_cohort, "logistic", mode="partial")
    df = pd.DataFrame(rows)
    # One row per (subject, within-subject fold); training sees every subject.
    assert df["held_out_participant"].nunique() == len(SUBJECTS)
    assert (df["n_train_subjects"] == len(SUBJECTS)).all()
    assert len(df) > len(SUBJECTS)  # multiple folds per subject


def test_per_participant_trains_on_one_subject(synthetic_cohort):
    from eeg_steptype.models import pooling

    rows = pooling.train_pooled(synthetic_cohort, "logistic", mode="per_participant")
    df = pd.DataFrame(rows)
    # Baseline: each fold trains on a single subject's own data.
    assert (df["n_train_subjects"] == 1).all()
    # Per_participant and partial share the same test folds (paired comparison).
    partial = pd.DataFrame(pooling.train_pooled(synthetic_cohort, "logistic", mode="partial"))
    assert len(df) == len(partial)


def test_overfit_gap_summary(synthetic_cohort):
    from eeg_steptype.models import pooling

    rows = pooling.train_pooled(synthetic_cohort, "logistic", mode="full")
    gap = pooling.overfit_gap(rows)
    assert gap["n"] == len(SUBJECTS)
    assert gap["n_subjects"] == len(SUBJECTS)
    assert np.isclose(gap["gap"], gap["inner_cv"] - gap["test_auc"])


def test_invalid_mode_rejected(synthetic_cohort):
    from eeg_steptype.models import pooling

    with pytest.raises(ValueError):
        pooling.train_pooled(synthetic_cohort, "logistic", mode="bogus")


def test_pre_kbest_caps_features_before_funnel(synthetic_cohort):
    """modeling.pre_kbest cuts the column count before the correlation drop.

    With selection disabled (method=none) the final feature count is governed by
    the pre-filter, so a small pre_kbest must cap n_features_final at <= K. The
    pre-filter is fit on the train fold only (it reuses select_kbest, the same
    leakage-safe ANOVA used downstream), so this only checks the cap + wiring.
    """
    from eeg_steptype.models import pooling

    base = pd.DataFrame(pooling.train_pooled(synthetic_cohort, "logistic", mode="partial"))
    n_base = int(base["n_features_final"].max())

    synthetic_cohort["modeling"]["pre_kbest"] = 5
    capped = pd.DataFrame(pooling.train_pooled(synthetic_cohort, "logistic", mode="partial"))
    assert (capped["n_features_final"] <= 5).all()
    # The cap must actually bind (the synthetic frame has > 5 raw columns).
    assert n_base > 5


def test_pre_kbest_null_is_noop(synthetic_cohort):
    """Default (pre_kbest unset / null) leaves the funnel unchanged."""
    from eeg_steptype.models import pooling

    rows_default = pd.DataFrame(pooling.train_pooled(synthetic_cohort, "logistic", mode="partial"))
    synthetic_cohort["modeling"]["pre_kbest"] = None
    rows_null = pd.DataFrame(pooling.train_pooled(synthetic_cohort, "logistic", mode="partial"))
    assert rows_default["n_features_final"].tolist() == rows_null["n_features_final"].tolist()
