"""3-class metrics + cohort rollup (standing / straight / diagonal).

Generalizes the CNV binary metrics: 3×3 confusion, per-class recall, overall
accuracy, macro one-vs-rest AUC (the primary metric), and macro-F1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


LABELS = [0, 1, 2]
CLASS_NAMES = ["standing", "straight", "diagonal"]


def participant_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """One fold's metrics. ``y_proba`` is (n, 3) class-probability columns."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    totals = cm.sum(axis=1)
    correct = np.diag(cm)
    overall = float(correct.sum() / cm.sum()) if cm.sum() else 0.0

    out: dict[str, float] = {}
    for i, name in enumerate(CLASS_NAMES):
        out[f"total_{name}"] = int(totals[i])
        out[f"correct_{name}"] = int(correct[i])
        out[f"accuracy_{name}"] = float(correct[i] / totals[i]) if totals[i] else 0.0
    out["overall_accuracy"] = overall
    out["macro_f1"] = float(f1_score(y_true, y_pred, labels=LABELS, average="macro",
                                     zero_division=0))
    out["macro_auc"] = _macro_ovr_auc(y_true, y_proba)
    # flattened confusion (row=true, col=pred) for diagnostics / heatmaps
    for i, ti in enumerate(CLASS_NAMES):
        for j, tj in enumerate(CLASS_NAMES):
            out[f"cm_{ti}_{tj}"] = int(cm[i, j])
    return out


def _macro_ovr_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    y_proba = np.asarray(y_proba)
    if y_proba.ndim != 2 or y_proba.shape[1] != len(LABELS):
        return float("nan")
    if len(np.unique(y_true)) < len(LABELS):
        # OVR macro AUC is undefined if a class is absent from this fold's test set.
        try:
            return float(roc_auc_score(y_true, y_proba, multi_class="ovr",
                                       average="macro", labels=LABELS))
        except Exception:
            return float("nan")
    try:
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr",
                                   average="macro", labels=LABELS))
    except Exception:
        return float("nan")


def cohort_rollup(per_participant: pd.DataFrame) -> dict:
    if per_participant.empty:
        return {"overall_accuracy": float("nan")}
    n = len(per_participant)
    row: dict[str, float] = {
        "n_rows": int(n),
        "n_participants": int(per_participant["participant_id"].nunique())
        if "participant_id" in per_participant else 0,
    }
    # pooled per-class recall from summed counts
    for name in CLASS_NAMES:
        tot = int(per_participant.get(f"total_{name}", pd.Series(dtype=int)).sum())
        cor = int(per_participant.get(f"correct_{name}", pd.Series(dtype=int)).sum())
        row[f"total_{name}"] = tot
        row[f"correct_{name}"] = cor
        row[f"accuracy_{name}"] = (cor / tot) if tot else 0.0
    for metric in ("overall_accuracy", "macro_auc", "macro_f1"):
        vals = per_participant[metric].dropna() if metric in per_participant else pd.Series(dtype=float)
        m = int(len(vals))
        row[f"{metric}_mean"] = float(vals.mean()) if m else float("nan")
        row[f"{metric}_sd"] = float(vals.std(ddof=1)) if m > 1 else 0.0
        row[f"{metric}_ci95"] = float(1.96 * vals.std(ddof=1) / np.sqrt(m)) if m > 1 else 0.0
    row["chance_accuracy"] = 1.0 / len(CLASS_NAMES)
    # inner-vs-outer overfit gap (if inner_best_score present)
    if "inner_best_score" in per_participant and "macro_auc" in per_participant:
        gap = (per_participant["inner_best_score"] - per_participant["macro_auc"]).dropna()
        row["overfit_gap_mean"] = float(gap.mean()) if len(gap) else float("nan")
    return row


def cv_rollup(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "cv_mode" not in metrics:
        return pd.DataFrame([cohort_rollup(metrics)])
    rows = []
    for mode, part in metrics.groupby("cv_mode", dropna=False):
        r = cohort_rollup(part)
        r["cv_mode"] = mode
        rows.append(r)
    return pd.DataFrame(rows)
