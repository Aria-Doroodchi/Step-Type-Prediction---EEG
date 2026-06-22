"""Run the final post-SHAP search from recovered pooled-feature checkpoints.

This is the short exact recovery path for metrics lost when the original
partial-pooling run was interrupted. It reuses each fold's final 120 selected
features, runs only the final hyperparameter search, saves predictions and
metrics atomically per fold, and resumes completed/search-finished folds.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from eeg_steptype.models import pooling
from eeg_steptype.models import train as train_module
from eeg_steptype.models.evaluate import cohort_rollup, participant_metrics
from eeg_steptype.result_store import (
    StepwiseResultStore,
    atomic_write_csv,
    atomic_write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--feature-recovery-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=5)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def load_features(
    recovery_dir: Path,
    participant: str,
    fold: int,
) -> list[str]:
    path = (
        recovery_dir
        / "checkpoints"
        / participant
        / f"fold_{fold}"
        / "final_features.json"
    )
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    features = value.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError(f"Invalid final feature checkpoint: {path}")
    return [str(feature) for feature in features]


def make_splits(
    cfg: dict,
    groups: np.ndarray,
    y: pd.Series,
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    result: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for participant in cfg["participants"]:
        subject_idx = np.where(groups == participant)[0]
        other_idx = np.where(groups != participant)[0]
        for fold, (train_pos, test_pos) in enumerate(
            pooling._subject_inner_splits(cfg, subject_idx, y)
        ):
            result[(participant, fold)] = (
                np.concatenate([subject_idx[train_pos], other_idx]),
                subject_idx[test_pos],
            )
    return result


def fit_from_saved_params(
    cfg: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
):
    model = train_module.MODEL_FACTORIES["xgb"]["make"](
        cfg,
        scale_pos_weight=train_module._scale_pos_weight(y_train),
    )
    model.set_params(**params)
    model.fit(X_train, y_train)
    return model


def run_fold(
    key: tuple[str, int],
    *,
    cfg: dict,
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    splits: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    recovery_dir: Path,
    store: StepwiseResultStore,
    print_lock: threading.Lock,
) -> dict:
    participant, fold = key
    unit_id = f"{participant}/fold_{fold}"
    if store.is_complete(unit_id):
        status = store.status(unit_id) or {}
        with print_lock:
            print(
                f"[resume] {participant} fold {fold}: complete "
                f"(AUC={status.get('auc', 'saved')})",
                flush=True,
            )
        return {"participant_id": participant, "fold": fold, "resumed": True}

    started = time.monotonic()
    train_idx, test_idx = splits[key]
    features = load_features(recovery_dir, participant, fold)
    missing = [feature for feature in features if feature not in X.columns]
    if missing:
        raise KeyError(
            f"{participant} fold {fold}: {len(missing)} selected features "
            "are absent from the pooled frame"
        )
    X_train = X.iloc[train_idx][features]
    X_test = X.iloc[test_idx][features]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    groups_train = groups[train_idx]

    store.start_unit(
        unit_id,
        metadata={
            "participant_id": participant,
            "fold": fold,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_features": len(features),
        },
    )
    store.save_stage(
        unit_id,
        "selected_features",
        {"count": len(features), "features": features},
    )

    if store.stage_exists(unit_id, "final_search"):
        search_result = store.load_stage(unit_id, "final_search")
        best_params = dict(search_result["best_params"])
        inner_best_score = float(search_result["inner_best_score"])
        model = fit_from_saved_params(cfg, X_train, y_train, best_params)
        search_source = "saved_search_refit"
    else:
        with print_lock:
            print(
                f"[search] {participant} fold {fold}: "
                f"{len(train_idx)} train / {len(test_idx)} test / {len(features)} features",
                flush=True,
            )
        search = train_module._fit_search(
            train_module.MODEL_FACTORIES["xgb"],
            cfg,
            "xgb",
            X_train,
            y_train,
            scale_pos_weight=train_module._scale_pos_weight(y_train),
            groups=groups_train,
        )
        model = search.best_estimator_
        best_params = dict(search.best_params_)
        inner_best_score = float(search.best_score_)
        store.save_stage(
            unit_id,
            "final_search",
            {
                "best_params": best_params,
                "inner_best_score": inner_best_score,
                "search_method": train_module._search_method(cfg, "xgb"),
            },
        )
        search_source = "fresh_search"

    proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )
    pred = (proba >= 0.5).astype(int)
    metrics = participant_metrics(
        np.asarray(y_test),
        np.asarray(pred),
        np.asarray(proba),
    )
    metrics.update(
        {
            "participant_id": participant,
            "held_out_participant": participant,
            "fold": fold,
            "repeat": 0,
            "model": "xgb",
            "pooling_mode": "partial",
            "cv_mode": "pooled_partial",
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_train_subjects": int(pd.unique(groups_train).size),
            "n_features_final": len(features),
            "best_params": str(best_params),
            "inner_best_score": inner_best_score,
            "search_method": train_module._search_method(cfg, "xgb"),
            "prediction_window": cfg.get("_prediction_window", "full_cnv"),
            "window_min_time": float(cfg["features"]["min_time"]),
            "window_max_time": float(cfg["features"]["max_time"]),
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    predictions = pd.DataFrame(
        {
            "participant_id": participant,
            "fold": fold,
            "pooled_row_index": test_idx,
            "y_true": np.asarray(y_test),
            "y_pred": pred,
            "y_proba": proba,
        }
    )
    store.complete_unit(
        unit_id,
        metrics,
        predictions=predictions,
        metadata={
            "auc": metrics["auc"],
            "overall_accuracy": metrics["overall_accuracy"],
            "search_source": search_source,
        },
    )
    with print_lock:
        print(
            f"[complete] {participant} fold {fold}: "
            f"accuracy={metrics['overall_accuracy']:.3f} "
            f"AUC={metrics['auc']:.3f} innerCV={inner_best_score:.3f} "
            f"({metrics['elapsed_seconds'] / 60:.1f} min)",
            flush=True,
        )
    return metrics


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in frame[columns].iterrows()
    ]
    return "\n".join([header, separator, *rows])


def write_final_outputs(store: StepwiseResultStore) -> Path:
    metrics = store.completed_metrics().sort_values(
        ["participant_id", "fold"]
    ).reset_index(drop=True)
    if len(metrics) != 150:
        raise RuntimeError(f"Expected 150 completed folds, found {len(metrics)}")

    predictions = pd.concat(
        [
            pd.read_csv(path)
            for path in sorted(store.units_dir.rglob("tables/predictions.csv"))
        ],
        ignore_index=True,
    ).sort_values(["participant_id", "fold", "pooled_row_index"])

    rollup = cohort_rollup(metrics)
    participant_summary = (
        metrics.groupby("participant_id", sort=False)
        .agg(
            folds=("fold", "count"),
            accuracy=("overall_accuracy", "mean"),
            accuracy_One=("accuracy_One", "mean"),
            accuracy_Two=("accuracy_Two", "mean"),
            auc=("auc", "mean"),
            inner_cv=("inner_best_score", "mean"),
        )
        .reset_index()
    )
    for column in [
        "accuracy",
        "accuracy_One",
        "accuracy_Two",
        "auc",
        "inner_cv",
    ]:
        participant_summary[column] = participant_summary[column].map(
            lambda value: f"{value:.4f}"
        )

    atomic_write_csv(store.root / "metrics.csv", metrics)
    atomic_write_csv(store.root / "predictions.csv", predictions)
    atomic_write_csv(store.root / "rollup.csv", pd.DataFrame([rollup]))
    atomic_write_csv(store.root / "participant_summary.csv", participant_summary)

    tn = int(metrics["correct_One"].sum())
    fp = int(metrics["total_One"].sum() - tn)
    tp = int(metrics["correct_Two"].sum())
    fn = int(metrics["total_Two"].sum() - tp)
    report = f"""# XGBoost Partial-Pooling Final Statistics

## Run Status

- Completed folds: **{len(metrics)} / 150**
- Participants: **{metrics["participant_id"].nunique()} / 30**
- Final selected features per fold: **120**
- Predictions saved: **{len(predictions)}**

## Cohort Results

| Metric | Value |
| --- | ---: |
| Overall accuracy | {rollup["overall_accuracy"]:.4f} |
| Mean fold accuracy | {rollup["overall_accuracy_mean"]:.4f} |
| Fold accuracy SD | {rollup["overall_accuracy_sd"]:.4f} |
| Fold accuracy 95% CI half-width | {rollup["overall_accuracy_ci95"]:.4f} |
| Condition One accuracy | {rollup["accuracy_One"]:.4f} |
| Condition Two accuracy | {rollup["accuracy_Two"]:.4f} |
| Mean held-out AUC | {rollup["auc_mean"]:.4f} |
| Held-out AUC SD | {rollup["auc_sd"]:.4f} |
| Held-out AUC 95% CI half-width | {rollup["auc_ci95"]:.4f} |
| Mean inner-CV score | {metrics["inner_best_score"].mean():.4f} |
| Inner minus held-out AUC gap | {(metrics["inner_best_score"].mean() - metrics["auc"].mean()):.4f} |

## Aggregate Confusion Matrix

| Actual / Predicted | One | Two |
| --- | ---: | ---: |
| One | {tn} | {fp} |
| Two | {fn} | {tp} |

## Per-Participant Summary

{markdown_table(participant_summary, list(participant_summary.columns))}

## Saved Artifacts

- `metrics.csv`: complete fold-level metrics and final hyperparameters.
- `predictions.csv`: held-out labels, predictions, and probabilities.
- `rollup.csv`: cohort aggregate statistics.
- `participant_summary.csv`: participant-level means across five folds.
- `metrics.partial.csv`: atomically refreshed during the run.
- `units/<participant>/fold_<n>/`: resumable per-fold search, prediction, and status checkpoints.
"""
    path = store.root / "final_statistics_report.md"
    atomic_write_text(path, report)
    return path


def main() -> None:
    args = parse_args()
    with args.config.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    threads = max(1, int(args.threads_per_worker))
    cfg["resources"]["n_jobs"] = threads
    cfg["modeling"]["xgb"]["n_jobs"] = threads
    cfg["modeling"]["search"]["n_jobs"] = 1

    print("[setup] loading cached pooled feature frame", flush=True)
    pooled = pooling.build_pooled_frame(
        cfg,
        list(cfg["participants"]),
        "xgb",
        channel_mode="full",
    )
    X, y, groups = pooling._prep_xyg(pooled)
    splits = make_splits(cfg, groups, y)
    for participant, fold in splits:
        features = load_features(args.feature_recovery_dir, participant, fold)
        missing = [feature for feature in features if feature not in X.columns]
        if missing:
            raise KeyError(
                f"{participant} fold {fold}: selected features missing from pooled data"
            )
    print(
        f"[setup] validated {len(splits)} folds and all recovered feature sets",
        flush=True,
    )
    if args.validate_only:
        print("[done] validation only; no searches were run", flush=True)
        return

    store = StepwiseResultStore(args.output_dir)
    print(
        f"[setup] final search with {args.workers} fold workers, "
        f"{threads} XGBoost threads each; results: {args.output_dir}",
        flush=True,
    )
    print_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(
                run_fold,
                key,
                cfg=cfg,
                X=X,
                y=y,
                groups=groups,
                splits=splits,
                recovery_dir=args.feature_recovery_dir,
                store=store,
                print_lock=print_lock,
            ): key
            for key in splits
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                future.result()
            except BaseException as exc:
                store.fail_unit(f"{key[0]}/fold_{key[1]}", exc)
                raise

    report = write_final_outputs(store)
    print(f"[done] final statistics and report written to {report}", flush=True)


if __name__ == "__main__":
    main()
