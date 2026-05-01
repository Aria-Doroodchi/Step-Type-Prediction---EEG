"""Generic per-participant train/eval driver.

Replaces the hand-rolled per-participant loops in CNV_XGB_4.3.py /
CNV_LSTM_3.py / CNV_ML_SVM_1.py. Picks a model factory by name, applies a
configurable feature-selection schedule, runs GridSearchCV, evaluates on a
held-out test split, and returns metrics.

Smoke-friendliness: every heavy step (RFECV iterations, GridSearchCV grid
size, SHAP) is driven by ``cfg["modeling"]`` and shrinks automatically when
the smoke config is active.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split

from ..config import apply_participant_override
from ..features.assemble import build_for_participant
from ..io import write_csv, run_dir, ensure_dir
from ..logging_utils import get_logger, make_run_id, stamp_run
from . import feature_selection as fs
from . import xgb as xgb_factory
from . import svm as svm_factory
from . import lstm as lstm_factory
from . import logistic as logistic_factory
from .evaluate import participant_metrics, cohort_rollup


log = get_logger(__name__)


# --- Model registry -------------------------------------------------------
MODEL_FACTORIES: dict[str, dict] = {
    "xgb": {
        "make":         xgb_factory.make_xgb,
        "param_grid":   xgb_factory.param_grid,
        "rfecv_base":   xgb_factory.make_rfecv_base,
        "supports_gain": True,
        "supports_shap": True,
    },
    "svm": {
        "make":         svm_factory.make_svm,
        "param_grid":   svm_factory.param_grid,
        "rfecv_base":   None,
        "supports_gain": False,
        "supports_shap": False,
    },
    "lstm": {
        "make":         lstm_factory.make_lstm,
        "param_grid":   lstm_factory.param_grid,
        "rfecv_base":   None,
        "supports_gain": False,
        "supports_shap": False,
    },
    "logistic": {
        "make":         logistic_factory.make_logistic,
        "param_grid":   logistic_factory.param_grid,
        "rfecv_base":   None,
        "supports_gain": False,
        "supports_shap": False,
    },
}


# ---------------------------------------------------------------------------
def train_one_participant(
    participant_id: str,
    cfg: dict,
    model_name: str,
) -> dict:
    """Run the full feature-selection + model fit for one participant."""
    cfg = apply_participant_override(cfg, participant_id)
    factory = MODEL_FACTORIES[model_name]
    mcfg = cfg["modeling"]

    df = build_for_participant(participant_id, cfg)
    df = df.drop(columns=["participant_id"])
    df = df.dropna(axis=1, how="any")

    X = df.drop(columns=["condition"])
    y = df["condition"].map({"One": 0, "Two": 1}).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=float(mcfg.get("test_size", 0.30)),
        random_state=int(mcfg.get("random_state", 1)),
        stratify=y,
    )

    # 1. Correlation drop (cheap)
    keep = fs.correlation_drop(X_train, threshold=float(mcfg.get("correlation_threshold", 0.9)))
    X_train, X_test = X_train[keep], X_test[keep]

    # 2. Univariate top-K
    keep = fs.select_kbest(X_train, y_train, k=int(mcfg.get("k_best", 500)))
    X_train, X_test = X_train[keep], X_test[keep]

    # 3. Iterated RFECV (XGB only — others fall through)
    if factory["rfecv_base"] is not None:
        scale_pos_weight = _scale_pos_weight(y_train)
        rfecv_base = factory["rfecv_base"](cfg, scale_pos_weight=scale_pos_weight)
        keep, _imp = fs.rfecv_iterated(
            X_train, y_train, rfecv_base,
            n_iterations=int(mcfg["rfecv"]["n_iterations"]),
            step=float(mcfg["rfecv"]["step"]),
            min_features_to_select=int(mcfg["rfecv"]["min_features_to_select"]),
            scoring=mcfg["rfecv"].get("scoring", "roc_auc"),
        )
        X_train, X_test = X_train[keep], X_test[keep]

    # 4. Build model + GridSearchCV
    scale_pos_weight = _scale_pos_weight(y_train)
    base = _make_model(factory, cfg, model_name,
                       scale_pos_weight=scale_pos_weight,
                       n_features=X_train.shape[1])

    inner_cv = StratifiedKFold(
        n_splits=int(mcfg.get("inner_cv_splits", 2)),
        shuffle=True,
        random_state=int(mcfg.get("random_state", 1)),
    )
    grid = GridSearchCV(
        estimator=base,
        param_grid=factory["param_grid"](cfg),
        scoring=mcfg.get("scoring", "accuracy"),
        n_jobs=-1,
        cv=inner_cv,
        refit=True,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    log.info("[%s/%s] best CV=%.3f, params=%s",
             participant_id, model_name, grid.best_score_, grid.best_params_)

    best = grid.best_estimator_

    # 5. (Optional) gain prune + retrain
    if factory["supports_gain"] and hasattr(best, "feature_importances_"):
        gp = mcfg["gain_prune"]
        keep_gain = fs.gain_prune(
            best.feature_importances_, X_train.columns.tolist(),
            mode=gp.get("mode", "zero"),
            percentile=float(gp.get("percentile", 10)),
            absolute=float(gp.get("absolute", 0.001)),
        )
        if keep_gain and len(keep_gain) < X_train.shape[1]:
            X_train, X_test = X_train[keep_gain], X_test[keep_gain]
            grid = GridSearchCV(
                estimator=_make_model(factory, cfg, model_name,
                                      scale_pos_weight=scale_pos_weight,
                                      n_features=X_train.shape[1]),
                param_grid=factory["param_grid"](cfg),
                scoring=mcfg.get("scoring", "accuracy"),
                cv=inner_cv,
                n_jobs=-1,
                refit=True,
            )
            grid.fit(X_train, y_train)
            best = grid.best_estimator_

    # 6. (Optional) SHAP prune + retrain
    if factory["supports_shap"]:
        keep_shap = fs.shap_prune(best, X_train, quantile=float(mcfg.get("shap_prune_quantile", 0.2)))
        if keep_shap and len(keep_shap) < X_train.shape[1]:
            X_train, X_test = X_train[keep_shap], X_test[keep_shap]
            grid = GridSearchCV(
                estimator=_make_model(factory, cfg, model_name,
                                      scale_pos_weight=scale_pos_weight,
                                      n_features=X_train.shape[1]),
                param_grid=factory["param_grid"](cfg),
                scoring=mcfg.get("scoring", "accuracy"),
                cv=inner_cv,
                n_jobs=-1,
                refit=True,
            )
            grid.fit(X_train, y_train)
            best = grid.best_estimator_

    # 7. Evaluate on test set
    proba = best.predict_proba(X_test)[:, 1] if hasattr(best, "predict_proba") else \
            best.decision_function(X_test)
    pred = (proba >= 0.5).astype(int)

    metrics = participant_metrics(np.asarray(y_test), pred, np.asarray(proba))
    metrics["participant_id"] = participant_id
    metrics["model"] = model_name
    metrics["n_features_final"] = int(X_train.shape[1])
    metrics["best_params"] = str(grid.best_params_)
    return metrics


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


# ---------------------------------------------------------------------------
def run(cfg: dict, *, model: str = "xgb", run_id: str | None = None) -> pd.DataFrame:
    """Stage entry point — train per participant, write report, return DataFrame."""
    if model not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model: {model!r} (choose from {list(MODEL_FACTORIES)})")

    run_id = run_id or make_run_id(prefix=model)
    rdir = ensure_dir(run_dir(cfg, run_id))
    if cfg.get("logging", {}).get("stamp_runs", True):
        stamp_run(rdir, cfg, model=model)

    rows = []
    for pid in cfg["participants"]:
        try:
            rows.append(train_one_participant(pid, cfg, model))
        except Exception as exc:                      # noqa: BLE001
            log.exception("[%s] failed: %s", pid, exc)

    df = pd.DataFrame(rows)
    write_csv(df, rdir / "metrics.csv")
    rollup = cohort_rollup(df)
    log.info("Cohort rollup: %s", rollup)

    pd.DataFrame([rollup]).to_csv(rdir / "rollup.csv", index=False)
    return df
