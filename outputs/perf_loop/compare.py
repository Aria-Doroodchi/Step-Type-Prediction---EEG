"""Paired candidate-vs-baseline comparison on matched folds.

All runs share random_state=1, so a candidate and the baseline see identical
outer folds per subject. The per-subject AUC delta is therefore a *paired*
estimate — far less noisy than the raw 8-subject cohort SD — which is the
right basis for gating a noisy 8-subject screen.

Usage:
  python outputs/perf_loop/compare.py <baseline_run> <candidate_run> [--only P.. ..]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "outputs/perf_loop")
from aggregate import _read_metrics, summarize  # noqa: E402


def _per_subject(df: pd.DataFrame, only=None) -> pd.DataFrame:
    subj = "held_out_participant" if "held_out_participant" in df.columns else "participant_id"
    if "cv_mode" in df.columns:
        if df["cv_mode"].astype(str).str.startswith("pooled_").any():
            df = df[df["cv_mode"].astype(str).str.startswith("pooled_")]
        else:
            df = df[df["cv_mode"] == "repeated_stratified"]
    df = df.dropna(subset=["auc"]).copy()
    if only:
        df = df[df[subj].isin(only)]
    df["_gap"] = df["inner_best_score"] - df["auc"]
    g = df.groupby(subj).agg(auc=("auc", "mean"), gap=("_gap", "mean"))
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    b = _per_subject(_read_metrics(args.baseline), args.only)
    c = _per_subject(_read_metrics(args.candidate), args.only)
    common = b.index.intersection(c.index)
    b, c = b.loc[common], c.loc[common]

    d_auc = c["auc"] - b["auc"]
    d_gap = c["gap"] - b["gap"]
    n = len(common)
    se = d_auc.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    print(f"baseline={args.baseline}")
    print(f"candidate={args.candidate}")
    print(f"n_subjects (paired) = {n}")
    print(f"cohort AUC: base={b['auc'].mean():.4f} -> cand={c['auc'].mean():.4f}  "
          f"delta={d_auc.mean():+.4f}  (paired SE {se:.4f}, "
          f"~{d_auc.mean()/se:+.2f} SE)" if se == se and se > 0 else
          f"cohort AUC: base={b['auc'].mean():.4f} -> cand={c['auc'].mean():.4f}  delta={d_auc.mean():+.4f}")
    print(f"gap:        base={b['gap'].mean():+.4f} -> cand={c['gap'].mean():+.4f}  "
          f"delta={d_gap.mean():+.4f}  (guardrail: must be <= +0.03)")
    wins = int((d_auc > 0).sum())
    print(f"per-subject AUC improved in {wins}/{n} subjects")
    print("per-subject AUC delta:")
    for s in common:
        print(f"   {s}: {b.loc[s,'auc']:.3f} -> {c.loc[s,'auc']:.3f}  ({d_auc[s]:+.3f})")


if __name__ == "__main__":
    main()
