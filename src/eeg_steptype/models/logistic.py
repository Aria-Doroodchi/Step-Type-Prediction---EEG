"""Plain logistic-regression factory.

Used by the smoke-test config (`configs/smoke.yaml`) to verify the pipeline
end-to-end in a few seconds rather than a few hours. Tiny grid on purpose.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def make_logistic(cfg: dict, *, scale_pos_weight: float = 1.0):
    rs = int(cfg["modeling"].get("random_state", 1))
    return LogisticRegression(
        max_iter=2000,
        random_state=rs,
        # `scale_pos_weight` for parity with the XGB call; LR uses class_weight.
        class_weight={0: 1.0, 1: float(scale_pos_weight)},
        solver="liblinear",
    )


def param_grid(cfg: dict) -> dict:
    return cfg.get("modeling", {}).get("logistic", {}).get(
        "param_grid",
        {"C": [0.1, 1.0]},  # tiny default for smoke
    )
