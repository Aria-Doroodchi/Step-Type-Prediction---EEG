"""Multiclass logistic-regression factory for the state module.

The CNV factory uses the ``liblinear`` solver, which cannot do 3-class
classification. Here we use ``lbfgs`` (native multinomial), balanced class
weights, and ``predict_proba`` for macro-OVR AUC. Used for the fast smoke;
xgb is the primary model.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def make_logistic(cfg: dict, *, scale_pos_weight: float | None = None, **_kwargs):
    return LogisticRegression(
        max_iter=2000,
        random_state=int(cfg["modeling"].get("random_state", 1)),
        class_weight="balanced",
        solver="lbfgs",            # supports multinomial (n_classes >= 3)
    )


def param_grid(cfg: dict) -> dict:
    return cfg.get("modeling", {}).get("logistic", {}).get("param_grid", {"C": [0.1, 1.0]})
