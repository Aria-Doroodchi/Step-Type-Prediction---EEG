"""Per-participant metrics + cohort rollup.

Identical metric logic to the original scripts (confusion matrix → per-class
accuracy, AUC, overall) but as functions you can call from anywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


def participant_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total_one = tn + fp
    total_two = fn + tp
    acc_one = tn / total_one if total_one > 0 else 0.0
    acc_two = tp / total_two if total_two > 0 else 0.0
    auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float("nan")
    overall = (tn + tp) / (total_one + total_two) if (total_one + total_two) > 0 else 0.0
    return {
        "total_One": int(total_one),
        "total_Two": int(total_two),
        "correct_One": int(tn),
        "correct_Two": int(tp),
        "accuracy_One": float(acc_one),
        "accuracy_Two": float(acc_two),
        "overall_accuracy": float(overall),
        "auc": float(auc),
    }


def cohort_rollup(per_participant: pd.DataFrame) -> dict:
    """Aggregate per-participant rows into overall accuracy figures."""
    if per_participant.empty:
        return {"overall_accuracy": float("nan")}
    total_one = int(per_participant["total_One"].sum())
    total_two = int(per_participant["total_Two"].sum())
    correct_one = int(per_participant["correct_One"].sum())
    correct_two = int(per_participant["correct_Two"].sum())
    overall = ((correct_one + correct_two) / (total_one + total_two)
               if (total_one + total_two) > 0 else 0.0)
    return {
        "total_One": total_one,
        "total_Two": total_two,
        "correct_One": correct_one,
        "correct_Two": correct_two,
        "accuracy_One": (correct_one / total_one) if total_one else 0.0,
        "accuracy_Two": (correct_two / total_two) if total_two else 0.0,
        "overall_accuracy": float(overall),
    }
