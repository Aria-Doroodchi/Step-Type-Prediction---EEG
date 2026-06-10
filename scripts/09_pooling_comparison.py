"""Compare per-participant, partial-pooling, and full-pooling workflows.

Motivation: the per-participant XGB models overfit the inner CV by ~0.17-0.24
AUC. Pooling data across subjects is the strongest lever on that gap. This
script runs all three workflows on one shared pooled feature frame (so they are
perfectly matched on features and -- for per_participant vs partial -- on test
folds), then reports the inner-vs-outer overfitting gap and held-out AUC per
mode.

Usage
-----
    python scripts/09_pooling_comparison.py --config configs/pooling_compare.yaml
    python scripts/09_pooling_comparison.py --config configs/pooling_compare.yaml \
        --modes per_participant partial full --model xgb

Writes ``outputs/runs/<run_id>/pooling_comparison.csv`` (per-fold rows) and
``pooling_summary.csv`` (one row per mode), and prints a comparison table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eeg_steptype.config import load_config
from eeg_steptype.io import ensure_dir, run_dir, write_csv
from eeg_steptype.logging_utils import get_logger, make_run_id, setup_logging, stamp_run
from eeg_steptype.models import pooling


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", nargs="+", default=["configs/pooling_compare.yaml"],
                   help="YAML overlay(s) on default.yaml.")
    p.add_argument("--model", default="xgb", choices=["xgb", "svm", "logistic"])
    p.add_argument("--participants", nargs="*", default=None,
                   help="Override the participant subset from the config.")
    p.add_argument("--modes", nargs="+", default=list(pooling.POOLING_MODES),
                   choices=list(pooling.POOLING_MODES))
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    if args.participants:
        cfg["participants"] = list(args.participants)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    log = get_logger("scripts.09_pooling_comparison")

    run_id = args.run_id or make_run_id(prefix=f"pooling_compare_{args.model}")
    rdir = ensure_dir(run_dir(cfg, run_id))
    if cfg.get("logging", {}).get("stamp_runs", True):
        stamp_run(rdir, cfg, model=args.model)

    participants = list(cfg["participants"])
    log.info("Pooling comparison: model=%s, %d subjects, modes=%s",
             args.model, len(participants), args.modes)

    # Build the shared pooled feature frame ONCE and reuse across modes so the
    # only thing differing between workflows is how data is shared.
    pooled = pooling.build_pooled_frame(cfg, participants, args.model)

    all_rows: list[dict] = []
    summaries: list[dict] = []
    for mode in args.modes:
        log.info("=== mode: %s ===", mode)
        rows = pooling.train_pooled(cfg, args.model, mode=mode, pooled_frame=pooled)
        for r in rows:
            r["mode"] = mode
        all_rows.extend(rows)
        gap = pooling.overfit_gap(rows)
        gap["mode"] = mode
        summaries.append(gap)
        log.info("[%s] test_auc=%.3f inner_cv=%.3f gap=%+.3f (n=%d folds, %d subj)",
                 mode, gap["test_auc"], gap["inner_cv"], gap["gap"], gap["n"], gap["n_subjects"])

    metrics_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summaries)[
        ["mode", "n", "n_subjects", "test_auc", "test_auc_sd", "inner_cv", "gap"]
    ]
    write_csv(metrics_df, rdir / "pooling_comparison.csv")
    write_csv(summary_df, rdir / "pooling_summary.csv")

    print("\n=== Pooling comparison (inner-vs-outer overfitting gap) ===")
    print(f"model={args.model}  subjects={len(participants)}  "
          f"features={pooled.shape[1] - 4}  window={cfg.get('_prediction_window')}")
    print(summary_df.round(3).to_string(index=False))
    print(f"\nWrote: {rdir / 'pooling_summary.csv'}")
    # Headline: how much each pooling strategy shrinks the gap vs the baseline.
    base = summary_df.loc[summary_df["mode"] == "per_participant", "gap"]
    if not base.empty:
        b = float(base.iloc[0])
        for _, r in summary_df.iterrows():
            if r["mode"] != "per_participant":
                print(f"  {r['mode']:15s} gap {r['gap']:+.3f}  "
                      f"(baseline {b:+.3f}, change {r['gap'] - b:+.3f})")


if __name__ == "__main__":
    main()
