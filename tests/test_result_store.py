from __future__ import annotations

import json

import pandas as pd
import pytest

from eeg_steptype.result_store import (
    StepwiseResultStore,
    atomic_write_text,
)


def test_stepwise_store_saves_and_resumes_each_stage(tmp_path):
    store = StepwiseResultStore(tmp_path / "run")
    store.start_unit("P01/fold_0", metadata={"model": "xgb"})
    store.save_stage("P01/fold_0", "selected_features", {"features": ["a", "b"]})
    store.save_table(
        "P01/fold_0",
        "search_results",
        pd.DataFrame([{"score": 0.7}]),
    )

    reopened = StepwiseResultStore(tmp_path / "run")
    assert reopened.stage_exists("P01/fold_0", "selected_features")
    assert reopened.load_stage("P01/fold_0", "selected_features") == {
        "features": ["a", "b"]
    }
    assert reopened.status("P01/fold_0")["latest_stage"] == "selected_features"
    assert not reopened.is_complete("P01/fold_0")


def test_complete_unit_saves_predictions_and_rebuilds_aggregate(tmp_path):
    store = StepwiseResultStore(tmp_path / "run")
    for fold, auc in [(0, 0.7), (1, 0.8)]:
        unit_id = f"P01/fold_{fold}"
        store.start_unit(unit_id)
        store.complete_unit(
            unit_id,
            {"participant_id": "P01", "fold": fold, "auc": auc},
            predictions=pd.DataFrame(
                [{"y_true": fold, "y_pred": fold, "y_proba": auc}]
            ),
        )

    aggregate = pd.read_csv(tmp_path / "run" / "metrics.partial.csv")
    assert aggregate["auc"].tolist() == [0.7, 0.8]
    assert store.is_complete("P01/fold_0")
    assert (
        tmp_path
        / "run"
        / "units"
        / "P01"
        / "fold_0"
        / "tables"
        / "predictions.csv"
    ).exists()


def test_failed_unit_is_not_in_aggregate(tmp_path):
    store = StepwiseResultStore(tmp_path / "run")
    store.start_unit("P01/fold_0")
    store.fail_unit("P01/fold_0", RuntimeError("interrupted"))
    store.write_aggregate_metrics()

    assert store.status("P01/fold_0")["status"] == "failed"
    assert store.completed_metrics().empty
    assert pd.read_csv(tmp_path / "run" / "metrics.partial.csv").empty


def test_atomic_write_keeps_existing_file_if_replace_fails(tmp_path, monkeypatch):
    target = tmp_path / "value.txt"
    target.write_text("old", encoding="utf-8")

    def fail_replace(_source, _target):
        raise OSError("simulated interruption")

    monkeypatch.setattr("eeg_steptype.result_store.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        atomic_write_text(target, "new")

    assert target.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob("*.tmp"))


def test_status_files_are_valid_json_after_updates(tmp_path):
    store = StepwiseResultStore(tmp_path / "run")
    store.start_unit("P 01/fold 0")
    store.save_stage("P 01/fold 0", "fit/search", {"candidate": 3})

    status_path = (
        tmp_path / "run" / "units" / "P_01" / "fold_0" / "status.json"
    )
    with status_path.open(encoding="utf-8") as handle:
        status = json.load(handle)
    assert status["latest_stage"] == "fit_search"
