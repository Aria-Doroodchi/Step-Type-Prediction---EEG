"""Recover final selected features from an interrupted partial-pooling run.

The original run logged the best pre-SHAP XGBoost parameters but did not save
feature names. This script deterministically replays feature selection, reuses
those logged parameters, and writes an atomic checkpoint after every expensive
stage. Re-running the command resumes from the latest valid checkpoint.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from eeg_steptype.models import feature_selection as fs
from eeg_steptype.models import pooling
from eeg_steptype.models import train as train_module
from eeg_steptype.models.normalization import unwrap_classifier


BEST_RE = re.compile(
    r"\[(?P<participant>[^/\]]+)/xgb\] outer fold best CV="
    r"(?P<score>[0-9.]+), params=(?P<params>\{.*\})"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--recovered-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
        help="Threads available to each XGBoost fit/search within a fold worker.",
    )
    parser.add_argument("--only", nargs="*", help="Optional keys such as P39:2")
    return parser.parse_args()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_write_text(path, frame.to_csv(index=False))


def load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_log_text(path: Path) -> str:
    prefix = path.read_bytes()[:2]
    encoding = "utf-16" if prefix in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
    return path.read_text(encoding=encoding, errors="replace")


def parse_logged_params(path: Path) -> dict[tuple[str, int], dict]:
    counters: dict[str, int] = {}
    result: dict[tuple[str, int], dict] = {}
    for line in read_log_text(path).splitlines():
        match = BEST_RE.search(line)
        if not match:
            continue
        participant = match["participant"]
        fold = counters.get(participant, 0)
        counters[participant] = fold + 1
        result[(participant, fold)] = ast.literal_eval(match["params"])
    return result


def load_recovered_params(path: Path) -> dict[tuple[str, int], dict]:
    frame = pd.read_csv(path)
    result: dict[tuple[str, int], dict] = {}
    for row in frame.itertuples(index=False):
        result[(str(row.participant_id), int(row.fold))] = ast.literal_eval(
            str(row.best_params)
        )
    return result


def checkpoint_dir(root: Path, participant: str, fold: int) -> Path:
    return root / "checkpoints" / participant / f"fold_{fold}"


def save_feature_stage(path: Path, features: list[str]) -> None:
    atomic_write_json(path, {"count": len(features), "features": features})


def load_feature_stage(path: Path) -> list[str]:
    value = load_json(path)
    if not isinstance(value, dict) or not isinstance(value.get("features"), list):
        raise ValueError(f"Invalid feature checkpoint: {path}")
    return [str(item) for item in value["features"]]


def make_splits(
    cfg: dict, groups: np.ndarray, y: pd.Series
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    result: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for participant in cfg["participants"]:
        subject_idx = np.where(groups == participant)[0]
        other_idx = np.where(groups != participant)[0]
        for fold, (train_pos, test_pos) in enumerate(
            pooling._subject_inner_splits(cfg, subject_idx, y)
        ):
            train_idx = np.concatenate([subject_idx[train_pos], other_idx])
            test_idx = subject_idx[test_pos]
            result[(participant, fold)] = (train_idx, test_idx)
    return result


def recover_fold(
    key: tuple[str, int],
    *,
    cfg: dict,
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    splits: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    logged_params: dict[tuple[str, int], dict],
    fallback_params: dict[tuple[str, int], dict],
    root: Path,
    print_lock: threading.Lock,
) -> dict:
    participant, fold = key
    fold_dir = checkpoint_dir(root, participant, fold)
    final_path = fold_dir / "final_features.json"
    if final_path.exists():
        final = load_feature_stage(final_path)
        with print_lock:
            print(f"[resume] {participant} fold {fold}: complete ({len(final)} features)", flush=True)
        return {"participant_id": participant, "fold": fold, "features": final}

    started = time.monotonic()
    train_idx, _test_idx = splits[key]
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    stage_meta = {
        "participant_id": participant,
        "fold": fold,
        "n_train": int(len(train_idx)),
        "status": "running",
    }
    atomic_write_json(fold_dir / "status.json", stage_meta)

    corr_path = fold_dir / "01_correlation_features.json"
    if corr_path.exists():
        corr_features = load_feature_stage(corr_path)
    else:
        corr_features = fs.correlation_drop(
            X_train, threshold=float(cfg["modeling"].get("correlation_threshold", 0.9))
        )
        save_feature_stage(corr_path, corr_features)
    X_corr = X_train[corr_features]

    kbest_path = fold_dir / "02_kbest_features.json"
    if kbest_path.exists():
        kbest_features = load_feature_stage(kbest_path)
    else:
        kbest_features = fs.select_kbest(
            X_corr, y_train, k=int(cfg["modeling"].get("k_best", 500))
        )
        save_feature_stage(kbest_path, kbest_features)
    X_kbest = X_corr[kbest_features]

    stability_path = fold_dir / "03_stability_features.json"
    probability_path = fold_dir / "03_stability_probabilities.csv"
    if stability_path.exists() and probability_path.exists():
        stability_features = load_feature_stage(stability_path)
    else:
        scfg = cfg["modeling"]["feature_selection"]["stability"]
        max_features = scfg.get("max_features", 150)
        stability_features, probabilities = fs.stability_select(
            X_kbest,
            y_train,
            n_subsamples=int(scfg.get("n_subsamples", 50)),
            sample_fraction=float(scfg.get("sample_fraction", 0.5)),
            l1_ratio=float(scfg.get("l1_ratio", 0.5)),
            n_lambda=int(scfg.get("n_lambda", 15)),
            threshold=float(scfg.get("threshold", 0.6)),
            max_features=None if max_features is None else int(max_features),
            min_features=int(scfg.get("min_features", 10)),
            random_state=int(cfg["modeling"].get("random_state", 1)),
        )
        save_feature_stage(stability_path, stability_features)
        atomic_write_csv(
            probability_path,
            probabilities.rename("selection_probability")
            .rename_axis("feature")
            .reset_index(),
        )
    X_stability = X_kbest[stability_features]

    params_path = fold_dir / "04_pre_shap_params.json"
    params_source = "original_log"
    if params_path.exists():
        params_payload = load_json(params_path)
        best_params = dict(params_payload["params"])
        params_source = str(params_payload["source"])
    elif key in logged_params:
        best_params = logged_params[key]
        atomic_write_json(
            params_path, {"source": params_source, "params": best_params}
        )
    else:
        # The three folds rerun after the outage lack their original pre-SHAP
        # parameter line. Re-run only this search to preserve exact semantics.
        params_source = "recomputed_search"
        search = train_module._fit_search(
            train_module.MODEL_FACTORIES["xgb"],
            cfg,
            "xgb",
            X_stability,
            y_train,
            scale_pos_weight=train_module._scale_pos_weight(y_train),
            groups=groups[train_idx],
        )
        best_params = dict(search.best_params_)
        atomic_write_json(
            params_path,
            {
                "source": params_source,
                "params": best_params,
                "best_score": float(search.best_score_),
                "prior_final_params_for_reference": fallback_params.get(key),
            },
        )

    model = train_module.MODEL_FACTORIES["xgb"]["make"](
        cfg, scale_pos_weight=train_module._scale_pos_weight(y_train)
    )
    model.set_params(**best_params)
    model.fit(X_stability, y_train)
    classifier = unwrap_classifier(model)

    gain_path = fold_dir / "05_gain_features.json"
    if gain_path.exists():
        gain_features = load_feature_stage(gain_path)
    else:
        gcfg = cfg["modeling"].get("gain_prune", {}) or {}
        gain_features = fs.gain_prune(
            classifier.feature_importances_,
            list(X_stability.columns),
            mode=gcfg.get("mode", "zero"),
            percentile=float(gcfg.get("percentile", 10)),
            absolute=float(gcfg.get("absolute", 0.001)),
        )
        save_feature_stage(gain_path, gain_features)

    # The original log confirms gain pruning retained all 150 features in every
    # fold. If that ever differs here, refitting would be required before SHAP.
    if gain_features != list(X_stability.columns):
        raise RuntimeError(
            f"{participant} fold {fold}: gain selection changed "
            f"({len(gain_features)} != {len(X_stability.columns)}); "
            "exact replay requires a gain-refit path"
        )

    final_features = fs.shap_prune(
        classifier,
        X_stability,
        quantile=float(cfg["modeling"].get("shap_prune_quantile", 0.2)),
    )
    save_feature_stage(final_path, final_features)
    elapsed = time.monotonic() - started
    atomic_write_json(
        fold_dir / "status.json",
        {
            **stage_meta,
            "status": "complete",
            "params_source": params_source,
            "n_correlation": len(corr_features),
            "n_kbest": len(kbest_features),
            "n_stability": len(stability_features),
            "n_gain": len(gain_features),
            "n_final": len(final_features),
            "elapsed_seconds": elapsed,
        },
    )
    with print_lock:
        print(
            f"[complete] {participant} fold {fold}: {len(final_features)} features "
            f"in {elapsed / 60:.1f} min ({params_source})",
            flush=True,
        )
    return {"participant_id": participant, "fold": fold, "features": final_features}


def write_aggregate(root: Path, results: list[dict]) -> None:
    long_rows = [
        {
            "participant_id": row["participant_id"],
            "fold": row["fold"],
            "feature": feature,
        }
        for row in results
        for feature in row["features"]
    ]
    long_frame = pd.DataFrame(long_rows).sort_values(
        ["participant_id", "fold", "feature"]
    )
    atomic_write_csv(root / "selected_features_by_fold.csv", long_frame)

    frequencies = (
        long_frame.groupby("feature")
        .agg(
            selected_folds=("feature", "size"),
            selected_participants=("participant_id", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["selected_folds", "selected_participants", "feature"],
            ascending=[False, False, True],
        )
    )
    frequencies["fold_frequency"] = frequencies["selected_folds"] / len(results)
    atomic_write_csv(root / "feature_selection_frequency.csv", frequencies)
    atomic_write_json(
        root / "summary.json",
        {
            "completed_folds": len(results),
            "expected_folds": 150,
            "unique_features_selected": int(long_frame["feature"].nunique()),
            "features_per_fold_min": int(
                long_frame.groupby(["participant_id", "fold"]).size().min()
            ),
            "features_per_fold_max": int(
                long_frame.groupby(["participant_id", "fold"]).size().max()
            ),
        },
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.config.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    # Bound nested work explicitly: outer concurrency is --workers and each
    # XGBoost fit/search receives --threads-per-worker threads.
    threads_per_worker = max(1, int(args.threads_per_worker))
    cfg["resources"]["n_jobs"] = threads_per_worker
    cfg["modeling"]["xgb"]["n_jobs"] = threads_per_worker
    cfg["modeling"]["search"]["n_jobs"] = 1

    logged_params = parse_logged_params(args.log)
    fallback_params = load_recovered_params(args.recovered_metrics)
    print(
        f"[setup] loaded pre-SHAP parameters for {len(logged_params)} folds; "
        f"{150 - len(logged_params)} folds require parameter search",
        flush=True,
    )
    pooled = pooling.build_pooled_frame(
        cfg, list(cfg["participants"]), "xgb", channel_mode="full"
    )
    X, y, groups = pooling._prep_xyg(pooled)
    splits = make_splits(cfg, groups, y)
    keys = list(splits)
    if args.only:
        requested = {
            (item.split(":", 1)[0], int(item.split(":", 1)[1]))
            for item in args.only
        }
        keys = [key for key in keys if key in requested]
        missing = requested - set(keys)
        if missing:
            raise ValueError(f"Unknown requested folds: {sorted(missing)}")

    print(
        f"[setup] recovering {len(keys)} folds with {args.workers} workers; "
        f"{threads_per_worker} XGBoost threads per worker; "
        f"checkpoints: {args.output_dir / 'checkpoints'}",
        flush=True,
    )
    print_lock = threading.Lock()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                recover_fold,
                key,
                cfg=cfg,
                X=X,
                y=y,
                groups=groups,
                splits=splits,
                logged_params=logged_params,
                fallback_params=fallback_params,
                root=args.output_dir,
                print_lock=print_lock,
            ): key
            for key in keys
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results.append(future.result())
            except BaseException as exc:
                atomic_write_json(
                    checkpoint_dir(args.output_dir, *key) / "status.json",
                    {
                        "participant_id": key[0],
                        "fold": key[1],
                        "status": "failed",
                        "error": repr(exc),
                    },
                )
                raise

    write_aggregate(args.output_dir, results)
    print(
        f"[done] recovered selected features for {len(results)} folds; "
        f"aggregate files written to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
