"""Generic per-participant train/eval driver.

Replaces the hand-rolled per-participant loops in CNV_XGB_4.3.py /
CNV_LSTM_3.py / CNV_ML_SVM_1.py. Picks a model factory by name, applies a
configurable feature-selection schedule, runs nested hyperparameter search,
and returns fold-level metrics.

Smoke-friendliness: every heavy step (RFECV iterations, hyperparameter search,
SHAP) is driven by ``cfg["modeling"]`` and shrinks automatically when the
smoke config is active.
"""

from __future__ import annotations

import copy
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (
    GridSearchCV,
    HalvingRandomSearchCV,
    KFold,
    ParameterGrid,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from ..config import apply_participant_override
from ..features.assemble import build_for_participant
from ..features.tensor import build_tensor_for_participant
from ..io import write_csv, run_dir, ensure_dir
from ..logging_utils import get_logger, make_run_id, stamp_run
from ..progress import FoldETA, CohortETA
from ..resources import resolve_n_jobs
from . import feature_selection as fs
from . import xgb as xgb_factory
from . import svm as svm_factory
from . import lstm as lstm_factory
from . import logistic as logistic_factory
from . import riemannian as riemannian_factory
from .normalization import maybe_prefix_param_grid, maybe_wrap_estimator, unwrap_classifier
from .evaluate import participant_metrics, cv_rollup


log = get_logger(__name__)


# --- Model registry -------------------------------------------------------
# Each entry describes how to build the model, the hyperparameter grid, and
# which optimization stages apply. ``data_representation`` selects the
# training data path: "tabular" uses the flat feature parquet from
# features.assemble; "tensor" loads (n_epochs, n_channels, n_times) data
# from features.tensor and skips the feature-selection stages entirely
# (correlation drop, k-best, RFECV, gain, SHAP -- none apply to tensor input).
MODEL_FACTORIES: dict[str, dict] = {
    "xgb": {
        "make":         xgb_factory.make_xgb,
        "param_grid":   xgb_factory.param_grid,
        "rfecv_base":   xgb_factory.make_rfecv_base,
        "supports_gain": True,
        "supports_shap": True,
        "data_representation": "tabular",
    },
    "svm": {
        "make":         svm_factory.make_svm,
        "param_grid":   svm_factory.param_grid,
        "rfecv_base":   None,
        "supports_gain": False,
        "supports_shap": False,
        "data_representation": "tabular",
    },
    "lstm": {
        "make":         lstm_factory.make_lstm,
        "param_grid":   lstm_factory.param_grid,
        "rfecv_base":   None,
        "supports_gain": False,
        "supports_shap": False,
        "data_representation": "tabular",  # legacy n_timesteps=1 path
    },
    "logistic": {
        "make":         logistic_factory.make_logistic,
        "param_grid":   logistic_factory.param_grid,
        "rfecv_base":   None,
        "supports_gain": False,
        "supports_shap": False,
        "data_representation": "tabular",
    },
    "riemannian": {
        "make":         riemannian_factory.make_riemannian,
        "param_grid":   riemannian_factory.param_grid,
        "rfecv_base":   None,
        "supports_gain": False,
        "supports_shap": False,
        "data_representation": "tensor",
    },
}


def _participant_metrics_path(rdir, participant_id: str):
    """Return the per-participant checkpoint path within a run directory."""
    return ensure_dir(rdir / "participants") / f"{participant_id}_metrics.csv"


# ---------------------------------------------------------------------------
def train_one_participant(
    participant_id: str,
    cfg: dict,
    model_name: str,
    *,
    channel_mode: str | None = None,
    cv_mode: str | None = None,
) -> list[dict]:
    """Run nested CV for one participant and return one metrics row per fold."""
    cfg = apply_participant_override(cfg, participant_id)
    factory = MODEL_FACTORIES[model_name]
    representation = factory.get("data_representation", "tabular")

    if representation == "tensor":
        return _train_one_participant_tensor(
            participant_id, cfg, model_name, factory,
            channel_mode=channel_mode, cv_mode=cv_mode,
        )

    df = build_for_participant(participant_id, cfg)
    df = _apply_channel_selection(df, cfg, model_name, channel_mode=channel_mode)
    df = df.dropna(axis=1, how="any")

    groups = df["block_id"] if "block_id" in df.columns else None
    X = df.drop(columns=["condition", "participant_id", "block_id"], errors="ignore")
    y = df["condition"].map({"One": 0, "Two": 1}).astype(int)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    if groups is not None:
        groups = groups.reset_index(drop=True)

    splits = _outer_splits(X, y, groups, cfg, cv_mode=cv_mode)
    log.info(
        "[%s/%s] starting nested CV: %d outer fold(s), inner grid≈%s candidate(s)",
        participant_id, model_name, len(splits),
        _param_grid_size(factory["param_grid"](cfg)),
    )
    eta = FoldETA(total_folds=len(splits))
    rows: list[dict] = []
    for fold_idx, split in enumerate(splits, 1):
        log.info(
            "[%s/%s] outer fold %d/%d starting (cv_mode=%s, repeat=%d, fold=%d)",
            participant_id, model_name, fold_idx, len(splits),
            split.get("cv_mode"), split.get("repeat"), split.get("fold"),
        )
        t0 = time.perf_counter()
        row = _fit_score_split(
            participant_id,
            cfg,
            model_name,
            factory,
            X,
            y,
            split["train_idx"],
            split["test_idx"],
        )
        fold_dt = time.perf_counter() - t0
        eta.record(fold_dt)
        log.info(
            "[%s/%s] outer fold %d/%d done in %.1fs · %s",
            participant_id, model_name, fold_idx, len(splits),
            fold_dt, eta.format(),
        )
        row.update({k: v for k, v in split.items() if not k.endswith("_idx")})
        row["channel_mode"] = _effective_channel_mode(cfg, model_name, channel_mode)
        row["prediction_window"] = cfg.get("_prediction_window", "late_cnv")
        row["window_min_time"] = float(cfg["features"]["min_time"])
        row["window_max_time"] = float(cfg["features"]["max_time"])
        rows.append(row)
    return rows


def _train_one_participant_tensor(
    participant_id: str,
    cfg: dict,
    model_name: str,
    factory: dict,
    *,
    channel_mode: str | None,
    cv_mode: str | None,
) -> list[dict]:
    """Tensor-input training path (Riemannian and future CNN).

    Skips correlation drop / k-best / RFECV / gain / SHAP -- none of those
    are well-defined on ``(n_epochs, n_channels, n_times)`` data. The
    per-fold work is simpler: split tensor, fit search, score.
    """
    log.info("[%s/%s] loading epoch tensor...", participant_id, model_name)
    bundle = build_tensor_for_participant(participant_id, cfg)
    X = np.asarray(bundle["data"])
    y = pd.Series(bundle["labels"]).map({"One": 0, "Two": 1}).astype(int).reset_index(drop=True)
    block_ids = bundle.get("block_ids")
    groups = pd.Series(block_ids).reset_index(drop=True) if block_ids is not None else None
    log.info(
        "[%s/%s] tensor ready: %d epochs × %d channels × %d samples (class balance %d/%d)",
        participant_id, model_name,
        X.shape[0], X.shape[1], X.shape[2],
        int((y == 0).sum()), int((y == 1).sum()),
    )

    splits = _outer_splits_tensor(X, y, groups, cfg, cv_mode=cv_mode)
    grid = factory["param_grid"](cfg)
    inner_splits = int(_cv_config(cfg).get("inner_splits", cfg.get("modeling", {}).get("inner_cv_splits", 2)))
    log.info(
        "[%s/%s] starting nested CV: %d outer fold(s) × %d candidate(s) × %d inner fold(s) "
        "= %d inner fit(s) + %d outer refit(s)",
        participant_id, model_name,
        len(splits), _param_grid_size(grid), inner_splits,
        len(splits) * _param_grid_size(grid) * inner_splits, len(splits),
    )
    eta = FoldETA(total_folds=len(splits))

    rows: list[dict] = []
    for fold_idx, split in enumerate(splits, 1):
        log.info(
            "[%s/%s] outer fold %d/%d starting (cv_mode=%s, repeat=%d, fold=%d, "
            "train=%d, test=%d)",
            participant_id, model_name, fold_idx, len(splits),
            split.get("cv_mode"), split.get("repeat"), split.get("fold"),
            len(split["train_idx"]), len(split["test_idx"]),
        )
        t0 = time.perf_counter()
        row = _fit_score_split_tensor(
            participant_id, cfg, model_name, factory,
            X, y,
            split["train_idx"], split["test_idx"],
        )
        fold_dt = time.perf_counter() - t0
        eta.record(fold_dt)
        log.info(
            "[%s/%s] outer fold %d/%d done in %.1fs · %s",
            participant_id, model_name, fold_idx, len(splits),
            fold_dt, eta.format(),
        )
        row.update({k: v for k, v in split.items() if not k.endswith("_idx")})
        row["channel_mode"] = "full"          # tensor models always use every channel
        row["prediction_window"] = cfg.get("_prediction_window", "late_cnv")
        row["window_min_time"] = float(cfg["features"]["min_time"])
        row["window_max_time"] = float(cfg["features"]["max_time"])
        rows.append(row)
    return rows


def _fit_score_split_tensor(
    participant_id: str,
    cfg: dict,
    model_name: str,
    factory: dict,
    X: np.ndarray,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    """Fit and score one outer fold on tensor input. No feature selection."""
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

    search = _fit_search(
        factory, cfg, model_name, X_train, y_train,
        scale_pos_weight=_scale_pos_weight(y_train),
    )
    log.info("[%s/%s] outer fold best CV=%.3f, params=%s",
             participant_id, model_name, search.best_score_, search.best_params_)

    best = search.best_estimator_
    proba = (
        best.predict_proba(X_test)[:, 1]
        if hasattr(best, "predict_proba")
        else best.decision_function(X_test)
    )
    pred = (proba >= 0.5).astype(int)

    metrics = participant_metrics(np.asarray(y_test), pred, np.asarray(proba))
    metrics["participant_id"] = participant_id
    metrics["model"] = model_name
    # Report the *input* dimensionality: channels × times. The actual feature
    # count after xDAWN + tangent space is determined by the pipeline.
    metrics["n_features_final"] = int(X_train.shape[1] * X_train.shape[2])
    metrics["best_params"] = str(search.best_params_)
    metrics["inner_best_score"] = float(search.best_score_)
    metrics["search_method"] = _search_method(cfg, model_name)
    return metrics


def _outer_splits_tensor(
    X: np.ndarray,
    y: pd.Series,
    groups: pd.Series | None,
    cfg: dict,
    *,
    cv_mode: str | None = None,
) -> list[dict]:
    """Outer splits for tensor input.

    ``sklearn`` splitters work on any input where ``len(X)`` is the sample
    count, so a 3-D ``np.ndarray`` is fine. The chronological sanity check is
    not meaningful without a tabular ``epoch`` column, so it is disabled
    here regardless of ``modeling.cv.chronological_check``.
    """
    cv = _cv_config(cfg)
    mode = cv_mode or cv.get("mode", "repeated_stratified")
    n_splits = _bounded_splits(y, int(cv.get("n_splits", 5)))
    random_state = int(cfg["modeling"].get("random_state", 1))
    rows: list[dict] = []

    if mode == "repeated_stratified":
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=int(cv.get("n_repeats", 20)),
            random_state=random_state,
        )
        for i, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
            rows.append({
                "cv_mode": "repeated_stratified",
                "repeat": i // n_splits,
                "fold":   i % n_splits,
                "train_idx": train_idx,
                "test_idx":  test_idx,
            })
        return rows

    if mode == "grouped":
        if groups is None or groups.nunique(dropna=False) < n_splits:
            log.warning("Grouped CV requested without enough block_id groups; falling back to repeated stratified.")
            return _outer_splits_tensor(X, y, groups, cfg, cv_mode="repeated_stratified")
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(np.zeros(len(y)), y, groups)
        ):
            rows.append({
                "cv_mode": "grouped",
                "repeat":  0,
                "fold":    fold,
                "train_idx": train_idx,
                "test_idx":  test_idx,
            })
        return rows

    if mode == "chronological":
        # No tabular ordering key on tensor data; fall back to the natural
        # epoch order with a no-shuffle KFold.
        splitter = KFold(n_splits=n_splits, shuffle=False)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(np.arange(len(y)))):
            rows.append({
                "cv_mode": "chronological",
                "repeat":  0,
                "fold":    fold,
                "train_idx": train_idx,
                "test_idx":  test_idx,
            })
        return rows

    raise ValueError("cv_mode must be one of: repeated_stratified, grouped, chronological")


def _fit_score_split(
    participant_id: str,
    cfg: dict,
    model_name: str,
    factory: dict,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    """Fit feature selection and hyperparameter search inside one outer fold."""
    mcfg = cfg["modeling"]
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

    # 1. Correlation drop (cheap)
    keep = fs.correlation_drop(X_train, threshold=float(mcfg.get("correlation_threshold", 0.9)))
    X_train, X_test = X_train[keep], X_test[keep]

    # 2. Univariate top-K
    keep = fs.select_kbest(X_train, y_train, k=int(mcfg.get("k_best", 500)))
    X_train, X_test = X_train[keep], X_test[keep]

    # 3. Iterated RFECV (XGB only -- others fall through)
    rfecv_cfg = mcfg.get("rfecv", {})
    rfecv_enabled = bool(rfecv_cfg.get("enabled", True))
    if rfecv_enabled and factory["rfecv_base"] is not None:
        scale_pos_weight = _scale_pos_weight(y_train)
        rfecv_base = factory["rfecv_base"](cfg, scale_pos_weight=scale_pos_weight)
        keep, _imp = fs.rfecv_iterated(
            X_train, y_train, rfecv_base,
            n_iterations=int(rfecv_cfg.get("n_iterations", 5)),
            step=float(rfecv_cfg.get("step", 0.05)),
            min_features_to_select=int(rfecv_cfg.get("min_features_to_select", 200)),
            scoring=rfecv_cfg.get("scoring", "roc_auc"),
            n_jobs=1,
        )
        X_train, X_test = X_train[keep], X_test[keep]
    elif not rfecv_enabled:
        log.info("[%s/%s] RFECV disabled by config; skipping.",
                 participant_id, model_name)

    scale_pos_weight = _scale_pos_weight(y_train)
    search = _fit_search(
        factory, cfg, model_name, X_train, y_train,
        scale_pos_weight=scale_pos_weight,
    )
    log.info("[%s/%s] outer fold best CV=%.3f, params=%s",
             participant_id, model_name, search.best_score_, search.best_params_)

    best = search.best_estimator_
    best_classifier = unwrap_classifier(best)

    # 4. (Optional) gain prune + retrain
    gp = mcfg.get("gain_prune", {}) or {}
    gain_prune_enabled = bool(gp.get("enabled", True))
    gain_prune_refit = bool(gp.get("refit", True))
    if (
        gain_prune_enabled
        and gain_prune_refit
        and factory["supports_gain"]
        and hasattr(best_classifier, "feature_importances_")
    ):
        keep_gain = fs.gain_prune(
            best_classifier.feature_importances_, X_train.columns.tolist(),
            mode=gp.get("mode", "zero"),
            percentile=float(gp.get("percentile", 10)),
            absolute=float(gp.get("absolute", 0.001)),
        )
        if keep_gain and len(keep_gain) < X_train.shape[1]:
            X_train, X_test = X_train[keep_gain], X_test[keep_gain]
            search = _fit_search(
                factory, cfg, model_name, X_train, y_train,
                scale_pos_weight=scale_pos_weight,
            )
            best = search.best_estimator_
            best_classifier = unwrap_classifier(best)
    elif not (gain_prune_enabled and gain_prune_refit):
        log.info("[%s/%s] gain-prune refit disabled by config; skipping.",
                 participant_id, model_name)

    # 5. (Optional) SHAP prune + retrain
    shap_cfg = mcfg.get("shap_prune", {}) or {}
    shap_quantile = float(
        shap_cfg.get("quantile", mcfg.get("shap_prune_quantile", 0.2))
    )
    shap_enabled = bool(shap_cfg.get("enabled", shap_quantile > 0))
    shap_refit = bool(shap_cfg.get("refit", True))
    if (
        shap_enabled
        and shap_refit
        and shap_quantile > 0
        and factory["supports_shap"]
    ):
        keep_shap = fs.shap_prune(
            best_classifier,
            X_train,
            quantile=shap_quantile,
        )
        if keep_shap and len(keep_shap) < X_train.shape[1]:
            X_train, X_test = X_train[keep_shap], X_test[keep_shap]
            search = _fit_search(
                factory, cfg, model_name, X_train, y_train,
                scale_pos_weight=scale_pos_weight,
            )
            best = search.best_estimator_
    elif not (shap_enabled and shap_refit and shap_quantile > 0):
        log.info("[%s/%s] SHAP prune disabled by config; skipping.",
                 participant_id, model_name)

    proba = (
        best.predict_proba(X_test)[:, 1]
        if hasattr(best, "predict_proba")
        else best.decision_function(X_test)
    )
    pred = (proba >= 0.5).astype(int)

    metrics = participant_metrics(np.asarray(y_test), pred, np.asarray(proba))
    metrics["participant_id"] = participant_id
    metrics["model"] = model_name
    metrics["n_features_final"] = int(X_train.shape[1])
    metrics["best_params"] = str(search.best_params_)
    metrics["inner_best_score"] = float(search.best_score_)
    metrics["search_method"] = _search_method(cfg, model_name)
    return metrics


def _fit_search(
    factory: dict,
    cfg: dict,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    scale_pos_weight: float,
) -> GridSearchCV | HalvingRandomSearchCV:
    mcfg = cfg["modeling"]
    estimator, param_grid = _make_search_estimator(
        factory, cfg, model_name,
        scale_pos_weight=scale_pos_weight,
        n_features=X_train.shape[1],
    )
    inner_cv = StratifiedKFold(
        n_splits=_bounded_splits(
            y_train,
            int(_cv_config(cfg).get("inner_splits", mcfg.get("inner_cv_splits", 2))),
        ),
        shuffle=True,
        random_state=int(mcfg.get("random_state", 1)),
    )
    search = _make_search_cv(
        estimator=estimator,
        param_grid=param_grid,
        cfg=cfg,
        model_name=model_name,
        cv=inner_cv,
    )
    search.fit(X_train, y_train)
    return search


def _make_search_cv(
    *,
    estimator,
    param_grid: dict,
    cfg: dict,
    model_name: str,
    cv: StratifiedKFold,
) -> GridSearchCV | HalvingRandomSearchCV:
    mcfg = cfg["modeling"]
    method = _search_method(cfg, model_name)
    common = {
        "estimator": estimator,
        "scoring": mcfg.get("scoring", "accuracy"),
        "n_jobs": _search_n_jobs(cfg, model_name),
        "cv": cv,
        "refit": True,
        # modeling.search.verbose -- opt-in per-candidate progress from sklearn.
        # 0=silent (default), 1=one line per candidate completion, 3=one line
        # per candidate × CV-fold pair. Useful when staring at a slow Riemannian
        # or XGB fit and wanting to see something move.
        "verbose": int(mcfg.get("search", {}).get("verbose", 0)),
    }
    if method == "grid":
        return GridSearchCV(param_grid=param_grid, **common)
    if method == "halving_random":
        scfg = mcfg.get("search", {})
        hcfg = scfg.get("halving", {})
        xgb_cfg = mcfg.get("xgb", {})
        max_resources = int(hcfg.get("max_resources", xgb_cfg.get("n_estimators", 1000)))
        min_resources = min(int(hcfg.get("min_resources", 100)), max_resources)
        n_candidates = min(
            int(scfg.get("n_iter", 100)),
            _param_grid_size(param_grid),
        )
        return HalvingRandomSearchCV(
            param_distributions=param_grid,
            n_candidates=max(1, n_candidates),
            resource=hcfg.get("resource", "n_estimators"),
            min_resources=min_resources,
            max_resources=max_resources,
            factor=int(hcfg.get("factor", 3)),
            aggressive_elimination=bool(hcfg.get("aggressive_elimination", False)),
            random_state=int(mcfg.get("random_state", 1)),
            **common,
        )
    raise ValueError("modeling.search.method must be one of: auto, grid, halving_random")


def _search_method(cfg: dict, model_name: str) -> str:
    method = cfg.get("modeling", {}).get("search", {}).get("method", "auto")
    if method == "auto":
        return "halving_random" if model_name == "xgb" else "grid"
    return method


def _search_n_jobs(cfg: dict, model_name: str) -> int:
    value = cfg.get("modeling", {}).get("search", {}).get("n_jobs")
    if value is None and model_name == "xgb":
        return 1
    return resolve_n_jobs(cfg, value, default=-8)


def _param_grid_size(param_grid: dict) -> int:
    return max(1, len(ParameterGrid(param_grid)))


def _outer_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None,
    cfg: dict,
    *,
    cv_mode: str | None = None,
) -> list[dict]:
    cv = _cv_config(cfg)
    mode = cv_mode or cv.get("mode", "repeated_stratified")
    primary = _primary_outer_splits(X, y, groups, cfg, mode)
    if mode != "chronological" and bool(cv.get("chronological_check", True)):
        primary.extend(_chronological_splits(X, y, cfg))
    return primary


def _primary_outer_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None,
    cfg: dict,
    mode: str,
) -> list[dict]:
    cv = _cv_config(cfg)
    n_splits = _bounded_splits(y, int(cv.get("n_splits", 5)))
    random_state = int(cfg["modeling"].get("random_state", 1))
    rows: list[dict] = []

    if mode == "repeated_stratified":
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=int(cv.get("n_repeats", 20)),
            random_state=random_state,
        )
        for i, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            rows.append({
                "cv_mode": "repeated_stratified",
                "repeat": i // n_splits,
                "fold": i % n_splits,
                "train_idx": train_idx,
                "test_idx": test_idx,
            })
        return rows

    if mode == "grouped":
        if groups is None or groups.nunique(dropna=False) < n_splits:
            log.warning("Grouped CV requested without enough block_id groups; using repeated stratified CV.")
            return _primary_outer_splits(X, y, groups, cfg, "repeated_stratified")
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
            rows.append({
                "cv_mode": "grouped",
                "repeat": 0,
                "fold": fold,
                "train_idx": train_idx,
                "test_idx": test_idx,
            })
        return rows

    if mode == "chronological":
        return _chronological_splits(X, y, cfg)

    raise ValueError("cv_mode must be one of: repeated_stratified, grouped, chronological")


def _chronological_splits(X: pd.DataFrame, y: pd.Series, cfg: dict) -> list[dict]:
    cv = _cv_config(cfg)
    X_ordered, y_ordered = _sort_for_chronological(X, y)
    n_splits = _bounded_splits(y_ordered, int(cv.get("n_splits", 5)))
    splitter = KFold(n_splits=n_splits, shuffle=False)
    rows = []
    for fold, (train_pos, test_pos) in enumerate(splitter.split(X_ordered)):
        y_train = y_ordered.iloc[train_pos]
        y_test = y_ordered.iloc[test_pos]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            log.warning("Skipping chronological fold %d because it contains one class.", fold)
            continue
        rows.append({
            "cv_mode": "chronological",
            "repeat": 0,
            "fold": fold,
            "train_idx": X_ordered.index.to_numpy()[train_pos],
            "test_idx": X_ordered.index.to_numpy()[test_pos],
        })
    return rows


def _sort_for_chronological(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    order = pd.DataFrame({"_idx": np.arange(len(X)), "_y": y.to_numpy()}, index=X.index)
    if "epoch" in X.columns:
        order["epoch"] = X["epoch"].to_numpy()
    else:
        order["epoch"] = np.arange(len(X))
    ordered_index = order.sort_values(["epoch", "_y", "_idx"]).index
    return X.loc[ordered_index], y.loc[ordered_index]


def _bounded_splits(y: pd.Series, requested: int) -> int:
    min_class = int(y.value_counts().min())
    return max(2, min(int(requested), min_class))


def _cv_config(cfg: dict) -> dict:
    return cfg.get("modeling", {}).get("cv", {})


def _effective_channel_mode(
    cfg: dict,
    model_name: str,
    channel_mode: str | None = None,
) -> str:
    if model_name in {"riemannian", "cnn"}:
        return "full"
    return channel_mode or cfg.get("channel_selection", {}).get("mode", "full")


def _apply_channel_selection(
    df: pd.DataFrame,
    cfg: dict,
    model_name: str,
    *,
    channel_mode: str | None = None,
) -> pd.DataFrame:
    mode = _effective_channel_mode(cfg, model_name, channel_mode)
    if mode == "full":
        log.info("Channel mode: full (%d columns)", df.shape[1])
        return df
    if mode != "roi":
        raise ValueError("channel mode must be one of: full, roi")

    roi = set(cfg.get("channel_selection", {}).get("roi", {}).get("channels", []))
    if not roi:
        raise ValueError("channel_selection.roi.channels must not be empty")

    keep = [
        col for col in df.columns
        if _keep_column_for_roi(col, roi)
    ]
    log.info(
        "Channel mode: roi (%d channels), columns %d → %d",
        len(roi), df.shape[1], len(keep),
    )
    return df[keep]


def _keep_column_for_roi(column: str, roi: set[str]) -> bool:
    if column in {"epoch", "condition", "participant_id"}:
        return True
    channel = _feature_channel(column)
    return channel is None or channel in roi


def _feature_channel(column: str) -> str | None:
    """Return electrode channel for amplitude/slope/PSD columns, else None."""
    if column.startswith("slope_"):
        m = re.match(r"^slope_(?P<ch>.+?)_bin_\d+$", column)
        return m.group("ch") if m else None
    m = re.match(r"^(?P<ch>.+?)(?:_[A-Za-z]+)?_bin_\d+$", column)
    if not m:
        return None
    ch = m.group("ch")
    # Source-space features also end in _bin_N but are not electrode channels.
    if "-" in ch or ch.endswith(("-lh", "-rh")):
        return None
    return ch


def _scale_pos_weight(y: pd.Series) -> float:
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return (neg / pos) if pos > 0 else 1.0


def _make_model(factory, cfg, model_name, *, scale_pos_weight, n_features):
    if model_name == "lstm":
        return factory["make"](cfg, n_features=n_features)
    try:
        return factory["make"](cfg, scale_pos_weight=scale_pos_weight)
    except TypeError:
        return factory["make"](cfg)


def _make_search_estimator(factory, cfg, model_name, *, scale_pos_weight, n_features):
    base = _make_model(
        factory, cfg, model_name,
        scale_pos_weight=scale_pos_weight,
        n_features=n_features,
    )
    estimator = maybe_wrap_estimator(base, model_name, cfg)
    param_grid = maybe_prefix_param_grid(factory["param_grid"](cfg), estimator)
    return estimator, param_grid


# ---------------------------------------------------------------------------
def _parallel_participants(cfg: dict) -> int:
    """Resolve the number of participants to train concurrently.

    Defaults to 1 (sequential) for backwards compatibility. Negative values
    follow the project convention of "reserve this many logical CPUs".
    """
    raw = (
        cfg.get("modeling", {})
        .get("parallel", {})
        .get("participants")
    )
    if raw is None:
        return 1
    raw = int(raw)
    if raw == 0:
        return 1
    if raw < 0:
        cpus = os.cpu_count() or 1
        return max(1, cpus - abs(raw))
    return raw


def _force_single_threaded_inner(cfg: dict) -> dict:
    """Return a copy of cfg with inner XGB/search workers pinned to one thread.

    Used inside joblib workers so each participant runs single-threaded and
    we don't oversubscribe (joblib workers × XGB threads × sklearn threads).
    """
    out = copy.deepcopy(cfg)
    modeling = out.setdefault("modeling", {})
    modeling.setdefault("xgb", {})["n_jobs"] = 1
    modeling.setdefault("search", {})["n_jobs"] = 1
    # Cap the project-wide n_jobs so anything reading resources.n_jobs also
    # falls back to one thread inside the worker.
    out.setdefault("resources", {})["n_jobs"] = 1
    out["_parallel_worker"] = True
    return out


def _train_one_with_checkpoint(
    pid: str,
    cfg: dict,
    model: str,
    rdir,
    *,
    channel_mode: str | None,
    cv_mode: str | None,
) -> tuple[str, list[dict] | None, Exception | None]:
    """Train (or load) one participant. Returns (pid, rows, exc)."""
    try:
        participant_path = _participant_metrics_path(rdir, pid)
        if participant_path.exists():
            log.info("[%s] checkpoint exists; loading %s", pid, participant_path)
            participant_df = pd.read_csv(participant_path)
            return pid, participant_df.to_dict(orient="records"), None

        participant_rows = train_one_participant(
            pid, cfg, model, channel_mode=channel_mode, cv_mode=cv_mode,
        )
        participant_df = pd.DataFrame(participant_rows)
        write_csv(participant_df, participant_path)
        log.info("[%s] wrote training checkpoint %s", pid, participant_path)
        return pid, participant_rows, None
    except Exception as exc:                              # noqa: BLE001
        return pid, None, exc


def run(
    cfg: dict,
    *,
    model: str = "xgb",
    run_id: str | None = None,
    channel_mode: str | None = None,
    cv_mode: str | None = None,
) -> pd.DataFrame:
    """Stage entry point — train per participant, write report, return DataFrame."""
    if model not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model: {model!r} (choose from {list(MODEL_FACTORIES)})")

    effective_channel_mode = _effective_channel_mode(cfg, model, channel_mode)
    window = cfg.get("_prediction_window", "late_cnv")
    run_id = run_id or make_run_id(prefix=f"{model}_{effective_channel_mode}_{window}")
    rdir = ensure_dir(run_dir(cfg, run_id))
    if cfg.get("logging", {}).get("stamp_runs", True):
        stamp_run(rdir, cfg, model=model)

    n_workers = _parallel_participants(cfg)
    participants = list(cfg["participants"])
    rows: list[dict] = []
    cohort_eta = CohortETA(total_participants=len(participants), n_workers=n_workers)

    if n_workers <= 1:
        for pid in participants:
            _pid, prows, exc = _train_one_with_checkpoint(
                pid, cfg, model, rdir,
                channel_mode=channel_mode, cv_mode=cv_mode,
            )
            cohort_eta.record_completion()
            if exc is not None:
                log.exception("[%s] failed: %s", pid, exc)
            else:
                rows.extend(prows or [])
            log.info("Cohort progress: %s", cohort_eta.format())
    else:
        from joblib import Parallel, delayed

        worker_cfg = _force_single_threaded_inner(cfg)
        log.info(
            "Training %d participants with %d parallel workers (inner threads pinned to 1)",
            len(participants), n_workers,
        )
        delayed_jobs = (
            delayed(_train_one_with_checkpoint)(
                pid, worker_cfg, model, rdir,
                channel_mode=channel_mode, cv_mode=cv_mode,
            )
            for pid in participants
        )
        # Prefer joblib's streaming API (>=1.3) so we can update the cohort
        # ETA as each worker returns rather than waiting for the entire pool
        # to drain. Fall back to a synchronous list on older joblib.
        try:
            results_iter = Parallel(
                n_jobs=n_workers, backend="loky", verbose=10,
                return_as="generator_unordered",
            )(delayed_jobs)
        except TypeError:
            try:
                results_iter = Parallel(
                    n_jobs=n_workers, backend="loky", verbose=10,
                    return_as="generator",
                )(delayed_jobs)
            except TypeError:
                results_iter = Parallel(
                    n_jobs=n_workers, backend="loky", verbose=10,
                )(delayed_jobs)
        for pid, prows, exc in results_iter:
            cohort_eta.record_completion()
            if exc is not None:
                log.exception("[%s] failed: %s", pid, exc)
            else:
                rows.extend(prows or [])
            log.info("Cohort progress: %s", cohort_eta.format())

    df = pd.DataFrame(rows)
    write_csv(df, rdir / "metrics.csv")
    rollup = cv_rollup(df)
    log.info("Cohort rollup: %s", rollup.to_dict(orient="records"))

    rollup.to_csv(rdir / "rollup.csv", index=False)
    return df
