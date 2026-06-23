"""Feature selection for the state module.

Reuses the CNV correlation drop / ANOVA K-best / gain prune (all multiclass-safe
as-is) and provides a **multiclass-safe** ``stability_select`` — the CNV version
does ``clf.coef_.ravel()`` which is wrong when the elastic-net logistic returns
a (n_classes, n_features) coefficient matrix. Here a feature counts as selected
at a given penalty if it is non-zero for *any* class (one-vs-rest union).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Reuse the multiclass-safe stages verbatim.
from eeg_steptype.models.feature_selection import (  # noqa: F401
    correlation_drop,
    select_kbest,
    rfecv_iterated,
    gain_prune,
    shap_prune,
    _drop_constant_columns,
)
from ..logging_utils import get_logger


log = get_logger(__name__)


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
    """Complementary-pairs stability selection, multiclass-safe.

    Identical to the CNV selector except a feature is marked selected at a
    penalty level when |coef| > 0 for ANY class (handles the (n_classes,
    n_features) elastic-net coefficient matrix).
    """
    X, constants = _drop_constant_columns(X)
    if constants:
        log.info("[stability] dropped %d zero-variance column(s)", len(constants))
    cols = list(X.columns)
    p = len(cols)
    if p == 0:
        return [], pd.Series(dtype=float)

    Xz = StandardScaler().fit_transform(X.to_numpy(dtype=float))
    yv = np.asarray(y).astype(int)
    n = Xz.shape[0]
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
                C=float(C), max_iter=max_iter, random_state=random_state, tol=1e-3,
            )
            clf.fit(Xz[idx], yv[idx])
            sel = np.abs(clf.coef_) > 1e-8           # (n_classes, p) or (1, p)
            if sel.ndim > 1:
                sel = sel.any(axis=0)                # union across classes -> (p,)
            counts[li] += sel
        n_fits += 1

    for _ in range(n_pairs):
        perm = rng.permutation(n)
        half = perm[:sub_n]
        comp = perm[sub_n:sub_n * 2] if sub_n * 2 <= n else perm[sub_n:]
        _fit_mark(half)
        _fit_mark(comp)

    if n_fits == 0:
        log.warning("[stability] no valid subsamples; keeping all features")
        return cols, pd.Series(1.0, index=cols)

    freq = counts / n_fits
    prob = pd.Series(freq.max(axis=0), index=cols).sort_values(ascending=False)
    kept = prob[prob >= threshold].index.tolist()
    if len(kept) < min_features:
        kept = prob.head(min_features).index.tolist()
    if max_features is not None and len(kept) > max_features:
        kept = prob.head(max_features).index.tolist()
    log.info("[stability] %d fit(s); kept %d / %d (threshold=%.2f, top prob=%.2f)",
             n_fits, len(kept), p, threshold,
             float(prob.iloc[0]) if len(prob) else float("nan"))
    return kept, prob
