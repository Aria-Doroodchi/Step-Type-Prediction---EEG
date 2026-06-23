"""Multiclass XGBoost factory for the state module.

3-class softprob objective. Class balancing is handled upstream (standing is
balanced to the stepping count + stratified CV), so no scale_pos_weight /
sample_weight is needed; the ``scale_pos_weight`` kwarg is accepted and ignored
for signature compatibility with the training driver. ``num_class`` is inferred
by XGBClassifier from ``y`` and is NOT set explicitly (the sklearn API rejects
a manual num_class).
"""

from __future__ import annotations

from xgboost import XGBClassifier

from ..resources import resolve_n_jobs


def make_xgb(cfg: dict, *, scale_pos_weight: float | None = None, **_kwargs):
    x = cfg["modeling"]["xgb"]
    n_jobs = resolve_n_jobs(cfg, x.get("n_jobs"), default=-8)
    return XGBClassifier(
        n_estimators=int(x.get("n_estimators", 1000)),
        objective=x.get("objective", "multi:softprob"),
        eval_metric=x.get("eval_metric", "mlogloss"),
        tree_method=x.get("tree_method", "hist"),
        random_state=int(cfg["modeling"].get("random_state", 1)),
        n_jobs=n_jobs,
    )


def param_grid(cfg: dict) -> dict:
    return cfg["modeling"]["xgb"]["param_grid"]


def make_rfecv_base(cfg: dict, *, scale_pos_weight: float | None = None, **_kwargs):
    n_jobs = resolve_n_jobs(cfg, cfg["modeling"].get("xgb", {}).get("n_jobs"), default=-8)
    return XGBClassifier(
        n_estimators=800, learning_rate=0.05, max_depth=4, subsample=0.8,
        colsample_bytree=0.7, reg_lambda=1.0, reg_alpha=0.0, gamma=0.0,
        objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
        random_state=int(cfg["modeling"].get("random_state", 1)), n_jobs=n_jobs,
    )
