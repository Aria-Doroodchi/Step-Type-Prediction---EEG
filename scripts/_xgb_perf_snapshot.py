"""Ad-hoc XGB current-performance snapshot.

Reproduces the express-tier XGB nested-CV on cached features for two windows:
  - full_cnv  / stats_pyramid_core   (fills the previously un-aggregated run)
  - late_cnv  / stats_pyramid_core   (reproduction check vs the recorded 0.5576)

Writes a tidy per-run rollup to outputs/runs/_snapshot_<tag>/ and prints a
one-line summary per run. Uses the saved express config so the protocol matches
the recorded screening runs (5x2 repeated stratified outer, inner 2-fold,
HalvingRandomSearchCV n_iter=25, stability selection + gain prune).
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

from eeg_steptype.models import train  # noqa: E402


BASE_CFG = "outputs/runs/bin_full_cnv_stats_pyramid_core_xgb/config.yaml"


def run_window(tag: str, window: str, tmin: float, tmax: float) -> dict:
    cfg = yaml.safe_load(open(BASE_CFG))
    cfg["_prediction_window"] = window
    cfg["features"]["min_time"] = tmin
    cfg["features"]["max_time"] = tmax

    participants = list(cfg["participants"])
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, pid in enumerate(participants, 1):
        pt = time.perf_counter()
        try:
            prows = train.train_one_participant(pid, cfg, "xgb")
            rows.extend(prows)
            df = pd.DataFrame(prows)
            prim = df[df["cv_mode"] == "repeated_stratified"]
            print(
                f"[{tag}] {pid} ({i}/{len(participants)}) "
                f"testAUC={prim['auc'].mean():.3f} "
                f"innerCV={prim['inner_best_score'].mean():.3f} "
                f"feats={int(prim['n_features_final'].mean())} "
                f"({time.perf_counter()-pt:.0f}s)",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{tag}] {pid} FAILED: {exc}", flush=True)

    full = pd.DataFrame(rows)
    full.to_csv(f"outputs/runs/_snapshot_{tag}_metrics.csv", index=False)
    prim = full[full["cv_mode"] == "repeated_stratified"]
    summary = {
        "tag": tag,
        "window": window,
        "n_participants": int(prim["participant_id"].nunique()),
        "test_auc_mean": float(prim["auc"].mean()),
        "test_auc_sd": float(prim.groupby("participant_id")["auc"].mean().std()),
        "inner_cv_mean": float(prim["inner_best_score"].mean()),
        "overfit_gap": float(prim["inner_best_score"].mean() - prim["auc"].mean()),
        "acc_mean": float(prim["overall_accuracy"].mean()),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    print("SUMMARY:", summary, flush=True)
    return summary


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    out = []
    if which in ("both", "full"):
        out.append(run_window("full_cnv_xgb", "full_cnv", 0.0, 2.0))
    if which in ("both", "late"):
        out.append(run_window("late_cnv_xgb", "late_cnv", 1.0, 2.0))
    pd.DataFrame(out).to_csv("outputs/runs/_snapshot_summary.csv", index=False)
    print("DONE", flush=True)
