"""Recover an interrupted partial-pooling run from its console log.

This utility is intentionally narrow: it parses completed fold summaries from
the original log, reruns explicitly requested missing folds, and writes a
minimal combined CSV plus a Markdown report.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from eeg_steptype.models import pooling
from eeg_steptype.models import train as train_module


FOLD_RE = re.compile(
    r"(?P<time>\d{2}:\d{2}:\d{2}).*"
    r"\[pooling/partial\] (?P<participant>\S+) fold (?P<fold>\d+): "
    r"testAUC=(?P<auc>[0-9.]+) innerCV=(?P<inner>[0-9.]+) "
    r"\(train=(?P<train>\d+) epochs/(?P<subjects>\d+) subj\)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--participant", default="P39")
    parser.add_argument("--folds", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_saved_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_completed_folds(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    raw_prefix = path.read_bytes()[:2]
    encoding = "utf-16" if raw_prefix in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
    with path.open(encoding=encoding, errors="replace") as handle:
        for line in handle:
            match = FOLD_RE.search(line)
            if not match:
                continue
            rows.append(
                {
                    "participant_id": match["participant"],
                    "held_out_participant": match["participant"],
                    "fold": int(match["fold"]),
                    "auc": float(match["auc"]),
                    "inner_best_score": float(match["inner"]),
                    "n_train": int(match["train"]),
                    "n_train_subjects": int(match["subjects"]),
                    "pooling_mode": "partial",
                    "cv_mode": "pooled_partial",
                    "source": "recovered_from_log",
                }
            )
    return pd.DataFrame(rows)


def run_missing_folds(
    cfg: dict,
    participant: str,
    folds: list[int],
) -> pd.DataFrame:
    participants = list(cfg["participants"])
    pooled_frame = pooling.build_pooled_frame(
        cfg, participants, "xgb", channel_mode="full"
    )
    X, y, groups = pooling._prep_xyg(pooled_frame)
    subject_idx = np.where(groups == participant)[0]
    other_idx = np.where(groups != participant)[0]
    splits = pooling._subject_inner_splits(cfg, subject_idx, y)
    factory = train_module.MODEL_FACTORIES["xgb"]

    rows: list[dict] = []
    for fold in folds:
        if fold < 0 or fold >= len(splits):
            raise ValueError(
                f"Fold {fold} is invalid; {participant} has {len(splits)} folds"
            )
        train_pos, test_pos = splits[fold]
        train_idx = np.concatenate([subject_idx[train_pos], other_idx])
        test_idx = subject_idx[test_pos]
        print(
            f"[recovery] running {participant} fold {fold} "
            f"({len(train_idx)} train / {len(test_idx)} test)",
            flush=True,
        )
        row = train_module._fit_score_split(
            participant,
            cfg,
            "xgb",
            factory,
            X,
            y,
            train_idx,
            test_idx,
            groups=groups,
        )
        row.update(
            {
                "pooling_mode": "partial",
                "held_out_participant": participant,
                "cv_mode": "pooled_partial",
                "repeat": 0,
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_train_subjects": int(pd.unique(groups[train_idx]).size),
                "prediction_window": cfg.get("_prediction_window", "full_cnv"),
                "window_min_time": float(cfg["features"]["min_time"]),
                "window_max_time": float(cfg["features"]["max_time"]),
                "source": "rerun_after_interruption",
            }
        )
        rows.append(row)
        print(
            f"[recovery] completed {participant} fold {fold}: "
            f"AUC={row['auc']:.6f}, innerCV={row['inner_best_score']:.6f}",
            flush=True,
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in frame[columns].iterrows()
    ]
    return "\n".join([header, separator, *rows])


def write_report(
    output_path: Path,
    combined: pd.DataFrame,
    rerun: pd.DataFrame,
    config_path: Path,
    log_path: Path,
) -> None:
    participant_summary = (
        combined.groupby("participant_id", sort=False)
        .agg(
            completed_folds=("fold", "count"),
            mean_auc=("auc", "mean"),
            mean_inner_cv=("inner_best_score", "mean"),
        )
        .reset_index()
    )
    participant_summary["mean_auc"] = participant_summary["mean_auc"].map(
        lambda value: f"{value:.4f}"
    )
    participant_summary["mean_inner_cv"] = participant_summary[
        "mean_inner_cv"
    ].map(lambda value: f"{value:.4f}")

    rerun_summary = rerun[
        [
            "participant_id",
            "fold",
            "auc",
            "inner_best_score",
            "overall_accuracy",
            "n_features_final",
        ]
    ].copy()
    for column in ["auc", "inner_best_score", "overall_accuracy"]:
        rerun_summary[column] = rerun_summary[column].map(
            lambda value: f"{value:.4f}"
        )

    auc = combined["auc"].astype(float)
    inner = combined["inner_best_score"].astype(float)
    auc_ci = 1.96 * auc.std(ddof=1) / np.sqrt(len(auc))
    inner_ci = 1.96 * inner.std(ddof=1) / np.sqrt(len(inner))
    recovered_count = int((combined["source"] == "recovered_from_log").sum())
    rerun_count = int((combined["source"] == "rerun_after_interruption").sum())

    text = f"""# Interrupted XGBoost Partial-Pooling Recovery Report

Generated: {datetime.now().astimezone().isoformat(timespec="seconds")}

## Recovery Status

- Original command: `python scripts\\04_train.py --model xgb --config configs\\pooling.yaml --n-jobs -4`
- Saved configuration: `{config_path}`
- Surviving console log: `{log_path}`
- Completed fold summaries recovered from log: **{recovered_count}**
- Missing folds rerun: **{rerun_count}**
- Combined fold count: **{len(combined)} / 150**
- Participants represented: **{combined["participant_id"].nunique()} / 30**

## Available Aggregate Results

| Metric | Value |
| --- | ---: |
| Mean held-out AUC | {auc.mean():.4f} |
| Held-out AUC SD | {auc.std(ddof=1):.4f} |
| Held-out AUC 95% CI half-width | {auc_ci:.4f} |
| Mean inner-CV AUC | {inner.mean():.4f} |
| Inner-CV AUC SD | {inner.std(ddof=1):.4f} |
| Inner-CV AUC 95% CI half-width | {inner_ci:.4f} |
| Inner minus held-out AUC gap | {(inner.mean() - auc.mean()):.4f} |

## Rerun Folds

{markdown_table(rerun_summary, list(rerun_summary.columns))}

## Per-Participant AUC Summary

{markdown_table(participant_summary, list(participant_summary.columns))}

## Data Availability And Limitations

The original pooled workflow held all fold rows in memory and was designed to
write `metrics.csv` and `rollup.csv` only after every fold completed. The power
loss therefore prevented those normal artifacts from being written.

The surviving console log records participant, fold, held-out AUC, inner-CV
score, training epoch count, and training-subject count for the first 147
folds. Those values were recovered exactly as printed, at three-decimal
precision. The final three folds were rerun from the saved configuration and
retain the full metric row.

Consequently, the 150-fold AUC and inner-CV aggregates are available, but
cohort-wide confusion-matrix totals, class-specific accuracy, overall accuracy,
feature counts, and best parameters cannot be reconstructed for the first 147
folds. Aggregate calculations involving recovered rows inherit the log's
three-decimal rounding.

## Generated Files

- `combined_recovered_metrics.csv`: minimal 150-fold table using all available data.
- `rerun_P39_folds_2_3_4_full_metrics.csv`: full metric rows for the three rerun folds.
- `recovery_report.md`: this report.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_saved_config(args.config)
    recovered = parse_completed_folds(args.log)
    if recovered.empty:
        raise RuntimeError(f"No completed pooled folds found in {args.log}")

    requested = {(args.participant, fold) for fold in args.folds}
    existing = set(zip(recovered["participant_id"], recovered["fold"]))
    overlap = requested & existing
    if overlap:
        raise RuntimeError(f"Requested folds already exist in the log: {sorted(overlap)}")

    rerun = run_missing_folds(cfg, args.participant, args.folds)
    rerun_path = args.output_dir / "rerun_P39_folds_2_3_4_full_metrics.csv"
    rerun.to_csv(rerun_path, index=False)

    combined = pd.concat([recovered, rerun], ignore_index=True, sort=False)
    combined = combined.sort_values(["participant_id", "fold"]).reset_index(drop=True)
    if combined.duplicated(["participant_id", "fold"]).any():
        raise RuntimeError("Combined output contains duplicate participant/fold rows")
    combined_path = args.output_dir / "combined_recovered_metrics.csv"
    combined.to_csv(combined_path, index=False)

    report_path = args.output_dir / "recovery_report.md"
    write_report(report_path, combined, rerun, args.config, args.log)
    print(f"[recovery] wrote {combined_path}", flush=True)
    print(f"[recovery] wrote {rerun_path}", flush=True)
    print(f"[recovery] wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
