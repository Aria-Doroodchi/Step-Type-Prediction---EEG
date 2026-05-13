"""Feature-selection layers used by the classical-ML lines.

The original CNV_XGB_4.3.py applied these inline and in this order:

    1. correlation_drop  — drop one of every |corr|>θ pair
    2. select_kbest      — ANOVA F-test top-K
    3. rfecv_iterated    — N-fold RFECV repeated and averaged
    4. gain_prune        — drop features with zero (or low) XGB gain
    5. shap_prune        — drop bottom-quantile features by mean |SHAP|

Each function takes a DataFrame and returns the *list of surviving columns*,
keeping the per-stage logic decoupled from the model code.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.model_selection import StratifiedKFold

from ..logging_utils import get_logger


log = get_logger(__name__)


# ---------------------------------------------------------------------------
def _drop_constant_columns(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (X without constant numeric columns, list of dropped column names).

    A column is "constant" if it has zero variance (or is all-NaN). Such columns
    carry no information, and they trip up downstream ANOVA F-tests with a
    spurious divide-by-zero warning. Filtering them at this seam keeps every
    later feature-selection step well-defined.
    """
    numeric = X.select_dtypes(include=[np.number])
    # nunique(dropna=False) treats NaN as its own value, so all-NaN columns
    # also report 1 unique value -- exactly what we want to drop.
    constant = [c for c in numeric.columns if numeric[c].nunique(dropna=False) <= 1]
    if not constant:
        return X, []
    return X.drop(columns=constant), constant


def correlation_drop(X: pd.DataFrame, threshold: float = 0.9) -> list[str]:
    """Return surviving feature names after dropping highly correlated ones."""
    X, constants = _drop_constant_columns(X)
    if constants:
        log.info("[corr-drop] dropped %d zero-variance column(s) before correlation step",
                 len(constants))
    corr = X.select_dtypes(include=[np.number]).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if any(upper[c] > threshold)]
    keep = [c for c in X.columns if c not in drop]
    log.info("[corr-drop] kept %d / dropped %d at θ=%.2f",
             len(keep), len(drop), threshold)
    return keep


def select_kbest(X: pd.DataFrame, y: pd.Series, k: int) -> list[str]:
    X, constants = _drop_constant_columns(X)
    if constants:
        log.info("[k-best] dropped %d zero-variance column(s) before ANOVA F",
                 len(constants))
    k = min(int(k), X.shape[1])
    sel = SelectKBest(score_func=f_classif, k=k)
    sel.fit(X, y)
    keep = X.columns[sel.get_support()].tolist()
    log.info("[k-best] selected %d / %d (k=%d) via ANOVA F", len(keep), X.shape[1], k)
    return keep


def rfecv_iterated(
    X: pd.DataFrame,
    y: pd.Series,
    estimator,
    *,
    n_iterations: int = 5,
    step: float = 0.05,
    min_features_to_select: int = 200,
    scoring: str = "roc_auc",
    n_splits: int = 2,
    n_jobs: int = 1,
) -> tuple[list[str], np.ndarray]:
    """Run RFECV n times with different folds; return the union of always-kept
    features plus the *mean* feature importances across iterations.

    Returns (kept_columns, mean_importances aligned with X.columns).
    """
    n_iterations = max(1, int(n_iterations))
    importances = np.zeros((n_iterations, X.shape[1]))
    dropped_log: list[list[str]] = []
    selected_log: list[list[str]] = []

    for i in range(n_iterations):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=i + 1)
        sel = RFECV(
            estimator=estimator,
            step=step,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            min_features_to_select=min_features_to_select,
        )
        sel.fit(X, y)

        mask = sel.support_
        idx = np.where(mask)[0]
        importances[i, idx] = sel.estimator_.feature_importances_

        selected_log.append(X.columns[mask].tolist())
        dropped_log.append(X.columns[~mask].tolist())
        log.info("[rfecv it %d/%d] kept=%d, best CV=%.3f",
                 i + 1, n_iterations, mask.sum(),
                 float(sel.cv_results_["mean_test_score"].max()))

    mean_imp = importances.mean(axis=0)
    # Top 80% of features by mean importance is what the original kept.
    rank = pd.Series(mean_imp, index=X.columns).sort_values(ascending=False)
    cutoff = int(len(rank) * 0.80)
    kept = rank.head(cutoff).index.tolist()

    drop_counter = Counter(f for it in dropped_log for f in it)
    always_dropped = sum(1 for c in drop_counter.values() if c == n_iterations)
    log.info("[rfecv] kept top 80%% = %d features; %d dropped in all iters",
             len(kept), always_dropped)
    return kept, mean_imp


def gain_prune(
    feature_importances: np.ndarray,
    feature_names: list[str],
    *,
    mode: str = "zero",
    percentile: float = 10.0,
    absolute: float = 0.001,
) -> list[str]:
    """Drop features by XGB feature_importance_ (gain)."""
    s = pd.Series(feature_importances, index=feature_names)
    if mode == "zero":
        keep = s[s > 0].index.tolist()
    elif mode == "percentile":
        thr = np.percentile(feature_importances, percentile)
        keep = s[s > thr].index.tolist()
    elif mode == "absolute":
        keep = s[s > absolute].index.tolist()
    else:
        raise ValueError(f"Unknown gain_prune mode: {mode}")
    log.info("[gain-prune mode=%s] kept %d / %d", mode, len(keep), len(s))
    return keep


def shap_prune(
    model,
    X: pd.DataFrame,
    *,
    quantile: float = 0.20,
) -> list[str]:
    """Drop bottom `quantile` features by mean |SHAP|."""
    try:
        import shap
    except Exception as exc:                          # noqa: BLE001
        log.warning("shap unavailable (%s); skipping SHAP prune.", exc)
        return list(X.columns)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    s = pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)
    thr = s.quantile(quantile)
    keep = s[s > thr].index.tolist()
    log.info("[shap-prune q=%.2f] kept %d / %d", quantile, len(keep), len(s))
    return keep
