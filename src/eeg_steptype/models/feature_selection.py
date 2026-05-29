"""Feature-selection layers used by the classical-ML lines.

The original CNV_XGB_4.3.py applied these inline and in this order:

    1. correlation_drop  — drop one of every |corr|>θ pair
    2. select_kbest      — ANOVA F-test top-K
    3. rfecv_iterated    — N-fold RFECV repeated and averaged
    4. gain_prune        — drop features with zero (or low) XGB gain
    5. shap_prune        — drop bottom-quantile features by mean |SHAP|

``stability_select`` is the current default in-fold selector (see
``models.train``), replacing the iterated RFECV at step 3: it is more robust at
small per-participant trial counts, model-agnostic, and comes with a
false-discovery bound. ``rfecv_iterated`` is retained for comparison runs.

Each function takes a DataFrame and returns the *list of surviving columns*,
keeping the per-stage logic decoupled from the model code.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

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


def stability_select(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_subsamples: int = 50,
    sample_fraction: float = 0.5,
    l1_ratio: float = 0.5,
    n_lambda: int = 15,
    threshold: float = 0.6,
    max_features: int | None = 150,
    min_features: int = 10,
    random_state: int = 1,
    max_iter: int = 2000,
) -> tuple[list[str], pd.Series]:
    """Complementary-pairs stability selection with an elastic-net base.

    For each of ``n_subsamples`` complementary half-samples we fit an
    elastic-net logistic regression across a regularisation path and record
    which coefficients are non-zero. A feature's *selection probability* is the
    maximum, over the path, of its selection frequency across subsamples
    (Meinshausen & Bühlmann 2010; complementary pairs per Shah & Samworth
    2013). Features whose probability meets ``threshold`` are kept.

    Why this over the old iterated RFECV: it is robust at the small
    per-participant trial counts where a 2-fold RFECV curve is too noisy to pick
    a stable feature count, it is model-agnostic (one selector for logistic /
    svm / xgb), and the selection probabilities carry a false-discovery bound.

    Returns ``(kept_columns, selection_probabilities)`` — the probabilities are
    a Series indexed by the surviving-after-constant-drop feature names, useful
    for logging and for cross-participant stability summaries.
    """
    X, constants = _drop_constant_columns(X)
    if constants:
        log.info("[stability] dropped %d zero-variance column(s) before selection",
                 len(constants))

    cols = list(X.columns)
    p = len(cols)
    if p == 0:
        return [], pd.Series(dtype=float)

    Xz = StandardScaler().fit_transform(X.to_numpy(dtype=float))
    yv = np.asarray(y).astype(int)
    n = Xz.shape[0]

    # Regularisation path. Smaller C = stronger penalty = sparser; spanning a
    # geometric grid lets the "max over path" pick up each feature near its
    # entry point.
    C_grid = np.geomspace(0.01, 10.0, num=max(2, int(n_lambda)))

    n_pairs = max(1, int(n_subsamples) // 2)
    sub_n = max(2, int(round(sample_fraction * n)))
    rng = np.random.default_rng(random_state)

    counts = np.zeros((len(C_grid), p), dtype=float)
    n_fits = 0

    def _fit_mark(idx: np.ndarray) -> None:
        nonlocal n_fits
        if np.unique(yv[idx]).size < 2:
            return
        for li, C in enumerate(C_grid):
            clf = LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=l1_ratio,
                C=float(C), max_iter=max_iter, random_state=random_state,
                tol=1e-3,
            )
            clf.fit(Xz[idx], yv[idx])
            counts[li] += (np.abs(clf.coef_.ravel()) > 1e-8)
        n_fits += 1

    for _ in range(n_pairs):
        perm = rng.permutation(n)
        half = perm[:sub_n]
        comp = perm[sub_n:sub_n * 2] if sub_n * 2 <= n else perm[sub_n:]
        _fit_mark(half)
        _fit_mark(comp)

    if n_fits == 0:
        log.warning("[stability] no valid subsamples (class imbalance); keeping all features")
        return cols, pd.Series(1.0, index=cols)

    freq = counts / n_fits                       # per-lambda selection frequency
    prob = pd.Series(freq.max(axis=0), index=cols).sort_values(ascending=False)

    kept = prob[prob >= threshold].index.tolist()
    if len(kept) < min_features:
        kept = prob.head(min_features).index.tolist()
    if max_features is not None and len(kept) > max_features:
        kept = prob.head(max_features).index.tolist()

    log.info("[stability] %d fit(s) over %d-point path; kept %d / %d "
             "(threshold=%.2f, top prob=%.2f)",
             n_fits, len(C_grid), len(kept), p, threshold,
             float(prob.iloc[0]) if len(prob) else float("nan"))
    return kept, prob


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
