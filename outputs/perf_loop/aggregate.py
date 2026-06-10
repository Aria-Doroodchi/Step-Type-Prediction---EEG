"""Aggregate a stamped run's metrics.csv into the task-defined cohort AUC + gap.

Objective-function definitions (per the agentic-loop brief):
  * cohort AUC = mean over participants of each participant's mean outer-fold
    held-out AUC, restricted to repeated_stratified folds (chronological-check
    rows excluded).
  * overfit gap = mean over participants of each participant's mean
    (inner_best_score - held-out AUC) over repeated_stratified folds.

Usage:
  python outputs/perf_loop/aggregate.py outputs/runs/<run_id>            # tabular run (metrics.csv)
  python outputs/perf_loop/aggregate.py outputs/runs/<run_id> --pooling  # pooling_comparison.csv
  python outputs/perf_loop/aggregate.py <a.csv> <b.csv> --label A B      # compare arbitrary metrics csvs
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


def _read_metrics(path: str) -> pd.DataFrame:
    if os.path.isdir(path):
        for name in ("metrics.csv", "pooling_comparison.csv"):
            p = os.path.join(path, name)
            if os.path.exists(p):
                path = p
                break
        else:
            raise FileNotFoundError(f"no metrics.csv / pooling_comparison.csv in {path}")
    return pd.read_csv(path)


def summarize(df: pd.DataFrame, *, pooling: bool = False) -> dict:
    """Return cohort AUC + gap per the task definition.

    Restrict to repeated_stratified folds for the per-participant path. The
    pooling path uses pooled_<mode> cv_mode rows and the held_out_participant
    column as the subject key.
    """
    subj_col = "held_out_participant" if "held_out_participant" in df.columns else "participant_id"
    if "cv_mode" in df.columns:
        if pooling or df["cv_mode"].astype(str).str.startswith("pooled_").any():
            prim = df[df["cv_mode"].astype(str).str.startswith("pooled_")].copy()
        else:
            prim = df[df["cv_mode"] == "repeated_stratified"].copy()
    else:
        prim = df.copy()
    prim = prim.dropna(subset=["auc"])
    if prim.empty:
        return {"n_subjects": 0, "n_folds": 0, "cohort_auc": float("nan"), "gap": float("nan")}
    prim["_gap"] = prim["inner_best_score"] - prim["auc"]
    per_subj = prim.groupby(subj_col).agg(
        auc=("auc", "mean"),
        gap=("_gap", "mean"),
        inner=("inner_best_score", "mean"),
        folds=("auc", "size"),
    )
    return {
        "n_subjects": int(per_subj.shape[0]),
        "n_folds": int(prim.shape[0]),
        "cohort_auc": float(per_subj["auc"].mean()),
        "cohort_auc_sd": float(per_subj["auc"].std(ddof=1)) if per_subj.shape[0] > 1 else 0.0,
        "inner_cv": float(per_subj["inner"].mean()),
        "gap": float(per_subj["gap"].mean()),
        "per_subject": per_subj["auc"].round(4).to_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--pooling", action="store_true")
    ap.add_argument("--label", nargs="*", default=None)
    ap.add_argument("--only", nargs="*", default=None,
                    help="Restrict to these participant ids (canonical-cohort filter).")
    args = ap.parse_args()
    for i, path in enumerate(args.paths):
        df = _read_metrics(path)
        if args.only:
            subj_col = "held_out_participant" if "held_out_participant" in df.columns else "participant_id"
            df = df[df[subj_col].isin(args.only)].copy()
        if "cv_mode" in df.columns and df["cv_mode"].astype(str).str.startswith("pooled_").any():
            # pooling run: report each mode separately
            for mode, part in df.groupby("cv_mode"):
                s = summarize(part, pooling=True)
                print(f"[{path}::{mode}] subj={s['n_subjects']} folds={s['n_folds']} "
                      f"cohortAUC={s['cohort_auc']:.4f} (sd {s['cohort_auc_sd']:.3f}) "
                      f"innerCV={s['inner_cv']:.4f} gap={s['gap']:+.4f}")
            continue
        s = summarize(df, pooling=args.pooling)
        label = (args.label[i] if args.label and i < len(args.label) else path)
        print(f"[{label}] subj={s['n_subjects']} folds={s['n_folds']} "
              f"cohortAUC={s['cohort_auc']:.4f} (sd {s['cohort_auc_sd']:.3f}) "
              f"innerCV={s['inner_cv']:.4f} gap={s['gap']:+.4f}")
        print(f"    per-subject AUC: {s['per_subject']}")


if __name__ == "__main__":
    main()
