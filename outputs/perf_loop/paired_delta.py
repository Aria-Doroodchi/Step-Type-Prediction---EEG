"""Paired per-subject delta between two pooling modes on one matched frame.

per_participant and partial share IDENTICAL test folds (same (subject, fold)
keys), so their difference is a clean paired estimate of what pooling buys. This
reports, for a pooling_comparison.csv:

  * per-subject mean AUC for each mode (averaged over that subject's folds),
  * the per-subject paired delta (partial - per_participant),
  * the cohort mean delta, its SE, a paired t-stat (n = #subjects), and the
    fraction of subjects that improved,
  * the gap (inner_best_score - auc) per mode.

This matches the task's GATE definition (prefer paired per-subject deltas) and
the prior loop's reported "paired delta / t / k of n up" numbers.

Usage:
  python outputs/perf_loop/paired_delta.py outputs/runs/<run_id> \
      [--a per_participant] [--b partial] [--only P01 P02 ...]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def _load(path: str) -> pd.DataFrame:
    if os.path.isdir(path):
        path = os.path.join(path, "pooling_comparison.csv")
    return pd.read_csv(path)


def paired(df: pd.DataFrame, a: str, b: str, only: list[str] | None = None) -> dict:
    df = df.copy()
    df["cv_mode"] = df["cv_mode"].astype(str)
    subj = "held_out_participant"
    if only:
        df = df[df[subj].isin(only)]
    A = df[df["cv_mode"] == f"pooled_{a}"]
    B = df[df["cv_mode"] == f"pooled_{b}"]
    if A.empty or B.empty:
        raise SystemExit(f"missing rows for pooled_{a} or pooled_{b}")

    # Per-subject mean AUC (averaged over that subject's folds).
    a_s = A.groupby(subj)["auc"].mean()
    b_s = B.groupby(subj)["auc"].mean()
    common = sorted(set(a_s.index) & set(b_s.index))
    a_s, b_s = a_s.loc[common], b_s.loc[common]
    delta = (b_s - a_s)

    n = len(common)
    mean_d = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    t = mean_d / se if se and not np.isnan(se) and se > 0 else float("nan")
    up = int((delta > 0).sum())

    def _gap(frame: pd.DataFrame) -> float:
        g = frame.assign(_g=frame["inner_best_score"] - frame["auc"])
        return float(g.groupby(subj)["_g"].mean().mean())

    return {
        "n_subjects": n,
        "a": a, "b": b,
        "a_cohort": float(a_s.mean()), "b_cohort": float(b_s.mean()),
        "a_gap": _gap(A[A[subj].isin(common)]), "b_gap": _gap(B[B[subj].isin(common)]),
        "mean_delta": mean_d, "se": se, "t": t, "up": up,
        "per_subject_delta": delta.round(4).to_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--a", default="per_participant")
    ap.add_argument("--b", default="partial")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    r = paired(_load(args.path), args.a, args.b, args.only)
    print(f"PAIRED {r['b']} - {r['a']}  (n={r['n_subjects']} subjects)")
    print(f"  {r['a']:16s} cohort AUC {r['a_cohort']:.4f}  gap {r['a_gap']:+.4f}")
    print(f"  {r['b']:16s} cohort AUC {r['b_cohort']:.4f}  gap {r['b_gap']:+.4f}")
    print(f"  paired delta = {r['mean_delta']:+.4f}  SE {r['se']:.4f}  "
          f"t = {r['t']:.2f}  ({r['up']}/{r['n_subjects']} subjects up)")
    print(f"  per-subject delta: {r['per_subject_delta']}")


if __name__ == "__main__":
    main()
