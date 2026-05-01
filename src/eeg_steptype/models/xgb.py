"""XGBoost model factory + param grid (read from cfg)."""

from __future__ import annotations

from xgboost import XGBClassifier


def make_xgb(cfg: dict, *, scale_pos_weight: float = 1.0):
    x = cfg["modeling"]["xgb"]
    return XGBClassifier(
        n_estimators=int(x.get("n_estimators", 1000)),
        scale_pos_weight=scale_pos_weight,
        objective=x.get("objective", "binary:logistic"),
        eval_metric=x.get("eval_metric", "logloss"),
        tree_method=x.get("tree_method", "hist"),
        random_state=int(cfg["modeling"].get("random_state", 1)),
        n_jobs=int(x.get("n_jobs", -1)),
    )


def param_grid(cfg: dict) -> dict:
    return cfg["modeling"]["xgb"]["param_grid"]


# A small "RFECV-base" XGB used during the iterated RFECV pass; smaller
# n_estimators because RFECV refits many times.
def make_rfecv_base(cfg: dict, *, scale_pos_weight: float = 1.0):
    return XGBClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        reg_alpha=0.0,
        gamma=0.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=int(cfg["modeling"].get("random_state", 1)),
        n_jobs=-1,
    )
