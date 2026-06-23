"""Per-participant 3-class nested-CV driver (standing / straight / diagonal).

Tabular multiclass generalization of the CNV driver. Reuses the CNV CV-split and
inner-search machinery (label-agnostic) and the state funnel/metrics. Funnel:
pre-KBest → correlation drop → ANOVA KBest → multiclass stability selection →
inner hyperparameter search → optional gain prune+refit. SHAP prune is off for
the multiclass baseline. Classes are balanced by subsampling standing to the
stepping count (+ stratified CV), so no per-sample weighting is required.
"""

from __future__ import annotations

import copy
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from eeg_steptype.models.train import (
    _outer_splits,
    _make_search_cv,
    _bounded_splits,
    _param_grid_size,
    _cv_config,
    _apply_channel_selection,
    _effective_channel_mode,
)
from eeg_steptype.models import svm as svm_factory
from . import logistic as logistic_factory

from ..config import apply_participant_override
from ..features.assemble import build_for_participant
from ..io import write_csv, run_dir, ensure_dir
from ..logging_utils import get_logger, make_run_id, stamp_run
from ..progress import FoldETA, CohortETA
from . import feature_selection as fs
from . import xgb as xgb_factory
from .evaluate import participant_metrics, cv_rollup, CLASS_NAMES


log = get_logger(__name__)

LABEL_MAP = {"standing": 0, "straight": 1, "diagonal": 2}

MODEL_FACTORIES = {
    "xgb": {"make": xgb_factory.make_xgb, "param_grid": xgb_factory.param_grid,
            "rfecv_base": xgb_factory.make_rfecv_base, "supports_gain": True},
    "logistic": {"make": logistic_factory.make_logistic, "param_grid": logistic_factory.param_grid,
                 "rfecv_base": None, "supports_gain": False},
    "svm": {"make": svm_factory.make_svm, "param_grid": svm_factory.param_grid,
            "rfecv_base": None, "supports_gain": False},
}


def _balance_standing(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Subsample standing epochs to the stepping count for a balanced problem."""
    scfg = cfg.get("standing", {})
    if str(scfg.get("balance", "per_class_max")) == "none":
        return df
    counts = df["condition"].value_counts()
    n_step = max(int(counts.get("straight", 0)), int(counts.get("diagonal", 0)))
    n_stand = int(counts.get("standing", 0))
    if n_step == 0 or n_stand <= n_step:
        return df
    rs = int(scfg.get("random_state", 42))
    stand = df[df["condition"] == "standing"].sample(n=n_step, random_state=rs)
    other = df[df["condition"] != "standing"]
    out = pd.concat([other, stand], ignore_index=True)
    log.info("Balanced standing %d -> %d (stepping max=%d)", n_stand, n_step, n_step)
    return out


def _apply_ablation(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Restrict feature blocks for the ablation.

    arm:
      'combined'  (all, default)
      'window'    drop sep      -> amplitude+slopes+psd+src   (brief's SEP test)
      'electrode' drop sep+src  -> amplitude+slopes+psd       (tests if src helps)
      'sep'       sep only
    """
    arm = str(cfg.get("modeling", {}).get("ablation", "combined")).lower()
    if arm in ("combined", "all", ""):
        return df
    meta = {"condition", "participant_id", "block_id", "epoch"}
    sep_cols = {c for c in df.columns if c.startswith("sep_")}
    src_cols = {c for c in df.columns if re.search(r"-(?:lh|rh)_bin_-?\d+$", c)}
    if arm == "window":
        keep = [c for c in df.columns if c not in sep_cols]
    elif arm == "electrode":
        keep = [c for c in df.columns if c not in sep_cols and c not in src_cols]
    elif arm == "sep":
        keep = [c for c in df.columns if c in sep_cols or c in meta]
    else:
        raise ValueError(f"unknown ablation arm {arm!r} (combined|window|electrode|sep)")
    log.info("Ablation '%s': %d -> %d columns", arm, df.shape[1], len(keep))
    return df[keep]


def train_one_participant(participant_id, cfg, model_name, *,
                          channel_mode=None, cv_mode=None) -> list[dict]:
    cfg = apply_participant_override(cfg, participant_id)
    factory = MODEL_FACTORIES[model_name]

    df = build_for_participant(participant_id, cfg)
    df = _balance_standing(df, cfg)
    df = _apply_channel_selection(df, cfg, model_name, channel_mode=channel_mode)
    df = _apply_ablation(df, cfg)
    df = df.dropna(axis=1, how="any")

    present = [c for c in CLASS_NAMES if c in set(df["condition"])]
    counts = df["condition"].value_counts().to_dict()
    if len(present) < 3:
        raise ValueError(
            f"[{participant_id}] only {len(present)} of 3 classes present "
            f"({counts}); dropped from the 3-class run.")
    min_count = min(int(counts.get(c, 0)) for c in CLASS_NAMES)
    if min_count < 5:
        raise ValueError(
            f"[{participant_id}] too few epochs in smallest class ({counts}); skipped.")

    groups = df["block_id"] if "block_id" in df.columns else None
    # Drop the epoch INDEX from features: per-condition epoch ranges differ
    # (standing 0..n_stand, stepping 0..~40), so 'epoch' would leak the label.
    X = df.drop(columns=["condition", "participant_id", "block_id", "epoch"],
                errors="ignore")
    y = df["condition"].map(LABEL_MAP).astype(int)
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    if groups is not None:
        groups = groups.reset_index(drop=True)

    splits = _outer_splits(X, y, groups, cfg, cv_mode=cv_mode)
    log.info("[%s/%s] 3-class nested CV: classes=%s, %d epochs, %d outer fold(s)",
             participant_id, model_name, counts, len(y), len(splits))
    eta = FoldETA(total_folds=len(splits))
    rows: list[dict] = []
    for i, split in enumerate(splits, 1):
        t0 = time.perf_counter()
        row = _fit_score_split(participant_id, cfg, model_name, factory, X, y,
                               split["train_idx"], split["test_idx"])
        eta.record(time.perf_counter() - t0)
        log.info("[%s/%s] fold %d/%d done · %s", participant_id, model_name,
                 i, len(splits), eta.format())
        row.update({k: v for k, v in split.items() if not k.endswith("_idx")})
        row["channel_mode"] = _effective_channel_mode(cfg, model_name, channel_mode)
        row["ablation"] = str(cfg.get("modeling", {}).get("ablation", "combined"))
        rows.append(row)
    return rows


def _fit_score_split(participant_id, cfg, model_name, factory, X, y,
                     train_idx, test_idx) -> dict:
    mcfg = cfg["modeling"]
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

    pre_k = mcfg.get("pre_kbest", None)
    if pre_k not in (None, "none", "None", 0) and int(pre_k) < X_train.shape[1]:
        keep = fs.select_kbest(X_train, y_train, k=int(pre_k))
        X_train, X_test = X_train[keep], X_test[keep]

    keep = fs.correlation_drop(X_train, threshold=float(mcfg.get("correlation_threshold", 0.9)))
    X_train, X_test = X_train[keep], X_test[keep]
    keep = fs.select_kbest(X_train, y_train, k=int(mcfg.get("k_best", 500)))
    X_train, X_test = X_train[keep], X_test[keep]

    fscfg = mcfg.get("feature_selection", {}) or {}
    method = str(fscfg.get("method", "stability")).lower()
    if method == "stability":
        scfg = fscfg.get("stability", {}) or {}
        mf = scfg.get("max_features", 150)
        keep, _ = fs.stability_select(
            X_train, y_train,
            n_subsamples=int(scfg.get("n_subsamples", 50)),
            sample_fraction=float(scfg.get("sample_fraction", 0.5)),
            l1_ratio=float(scfg.get("l1_ratio", 0.5)),
            n_lambda=int(scfg.get("n_lambda", 15)),
            threshold=float(scfg.get("threshold", 0.6)),
            max_features=(None if mf in (None, "none", "None") else int(mf)),
            min_features=int(scfg.get("min_features", 10)),
            random_state=int(mcfg.get("random_state", 1)),
        )
        if keep:
            X_train, X_test = X_train[keep], X_test[keep]
    elif method == "none":
        pass
    else:
        log.info("[%s] feature_selection.method=%s not supported in state baseline; "
                 "using current features.", participant_id, method)

    search = _fit_search(factory, cfg, model_name, X_train, y_train)
    log.info("[%s/%s] inner best=%.3f params=%s", participant_id, model_name,
             search.best_score_, search.best_params_)
    best = search.best_estimator_

    gp = mcfg.get("gain_prune", {}) or {}
    if (bool(gp.get("enabled", True)) and bool(gp.get("refit", True))
            and factory["supports_gain"] and hasattr(best, "feature_importances_")):
        keep_gain = fs.gain_prune(best.feature_importances_, X_train.columns.tolist(),
                                  mode=gp.get("mode", "zero"),
                                  percentile=float(gp.get("percentile", 10)),
                                  absolute=float(gp.get("absolute", 0.001)))
        if keep_gain and len(keep_gain) < X_train.shape[1]:
            X_train, X_test = X_train[keep_gain], X_test[keep_gain]
            search = _fit_search(factory, cfg, model_name, X_train, y_train)
            best = search.best_estimator_

    proba = best.predict_proba(X_test)
    pred = np.asarray(proba).argmax(axis=1)
    metrics = participant_metrics(np.asarray(y_test), pred, np.asarray(proba))
    metrics["participant_id"] = participant_id
    metrics["model"] = model_name
    metrics["n_features_final"] = int(X_train.shape[1])
    metrics["best_params"] = str(search.best_params_)
    metrics["inner_best_score"] = float(search.best_score_)
    return metrics


def _fit_search(factory, cfg, model_name, X_train, y_train):
    mcfg = cfg["modeling"]
    estimator = factory["make"](cfg)
    param_grid = factory["param_grid"](cfg)
    n_inner = int(_cv_config(cfg).get("inner_splits", mcfg.get("inner_cv_splits", 3)))
    inner_cv = StratifiedKFold(n_splits=_bounded_splits(y_train, n_inner),
                               shuffle=True, random_state=int(mcfg.get("random_state", 1)))
    search = _make_search_cv(estimator=estimator, param_grid=param_grid, cfg=cfg,
                             model_name=model_name, cv=inner_cv)
    search.fit(X_train, y_train)
    return search


# ---------------------------------------------------------------------------
def _participant_metrics_path(rdir, pid):
    return ensure_dir(rdir / "participants") / f"{pid}_metrics.csv"


def _train_one_with_checkpoint(pid, cfg, model, rdir, *, channel_mode, cv_mode):
    try:
        path = _participant_metrics_path(rdir, pid)
        if path.exists():
            log.info("[%s] checkpoint exists; loading.", pid)
            return pid, pd.read_csv(path).to_dict(orient="records"), None
        rows = train_one_participant(pid, cfg, model, channel_mode=channel_mode, cv_mode=cv_mode)
        write_csv(pd.DataFrame(rows), path)
        return pid, rows, None
    except Exception as exc:  # noqa: BLE001
        return pid, None, exc


def _parallel_participants(cfg) -> int:
    raw = cfg.get("modeling", {}).get("parallel", {}).get("participants")
    if raw is None or int(raw) == 0:
        return 1
    raw = int(raw)
    if raw < 0:
        return max(1, (os.cpu_count() or 1) - abs(raw))
    return raw


def _force_single_threaded_inner(cfg) -> dict:
    out = copy.deepcopy(cfg)
    m = out.setdefault("modeling", {})
    m.setdefault("xgb", {})["n_jobs"] = 1
    m.setdefault("search", {})["n_jobs"] = 1
    out.setdefault("resources", {})["n_jobs"] = 1
    return out


def run(cfg, *, model="xgb", run_id=None, channel_mode=None, cv_mode=None) -> pd.DataFrame:
    if model not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model {model!r}; choose from {list(MODEL_FACTORIES)}")
    eff_cm = _effective_channel_mode(cfg, model, channel_mode)
    run_id = run_id or make_run_id(prefix=f"state_{model}_{eff_cm}")
    rdir = ensure_dir(run_dir(cfg, run_id))
    if cfg.get("logging", {}).get("stamp_runs", True):
        stamp_run(rdir, cfg, model=model)

    participants = list(cfg["participants"])
    n_workers = _parallel_participants(cfg)
    rows: list[dict] = []
    cohort_eta = CohortETA(total_participants=len(participants), n_workers=n_workers)

    if n_workers <= 1:
        for pid in participants:
            _pid, prows, exc = _train_one_with_checkpoint(
                pid, cfg, model, rdir, channel_mode=channel_mode, cv_mode=cv_mode)
            cohort_eta.record_completion()
            if exc is not None:
                log.warning("[%s] skipped: %s", pid, exc)
            else:
                rows.extend(prows or [])
            log.info("Cohort progress: %s", cohort_eta.format())
    else:
        from joblib import Parallel, delayed
        wcfg = _force_single_threaded_inner(cfg)
        log.info("Training %d participants × %d workers", len(participants), n_workers)
        results = Parallel(n_jobs=n_workers, backend="loky", verbose=10)(
            delayed(_train_one_with_checkpoint)(
                pid, wcfg, model, rdir, channel_mode=channel_mode, cv_mode=cv_mode)
            for pid in participants)
        for pid, prows, exc in results:
            cohort_eta.record_completion()
            if exc is not None:
                log.warning("[%s] skipped: %s", pid, exc)
            else:
                rows.extend(prows or [])

    df = pd.DataFrame(rows)
    write_csv(df, rdir / "metrics.csv")
    rollup = cv_rollup(df)
    log.info("Cohort rollup: %s", rollup.to_dict(orient="records"))
    rollup.to_csv(rdir / "rollup.csv", index=False)
    return df
