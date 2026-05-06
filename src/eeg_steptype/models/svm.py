"""SVM model factory + param grid."""

from __future__ import annotations

from sklearn.svm import SVC


def make_svm(cfg: dict):
    return SVC(
        probability=True,
        random_state=int(cfg["modeling"].get("random_state", 1)),
        class_weight="balanced",
    )


def param_grid(cfg: dict) -> dict:
    return cfg["modeling"]["svm"]["param_grid"]
