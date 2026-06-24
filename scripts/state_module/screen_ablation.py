"""SCREEN ablation: run the 4 feature-block arms on the prepared SCREEN cohort,
build diagnostic figures, and print a comparison table.

Run AFTER `run_state.py --config configs/state/screen.yaml --stages preprocess
src features --force` has built epochs/src/features for the 8 SCREEN participants.

Arms:
  combined   amplitude+slopes+psd+src+sep   (full model)
  window     amplitude+slopes+psd+src        (drop SEP — the brief's SEP test)
  electrode  amplitude+slopes+psd            (drop SEP+src — does src help?)
  sep        sep only                        (SEP standalone signal)
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from eeg_statetype.config import load_config
from eeg_statetype.logging_utils import setup_logging, get_logger
from eeg_statetype.models.train import run as run_train
from eeg_statetype.models.evaluate import cv_rollup, CLASS_NAMES
from eeg_statetype.viz import results as viz_results

ARMS = ["combined", "window", "electrode", "sep"]


def main(model: str = "xgb"):
    setup_logging("INFO")
    log = get_logger("state.screen_ablation")
    cfg = load_config(["configs/state/screen.yaml"])
    pids = list(cfg["participants"])

    rollups: dict[str, pd.DataFrame] = {}
    metrics: dict[str, pd.DataFrame] = {}
    for arm in ARMS:
        log.info("######## ablation arm: %s ########", arm)
        c = copy.deepcopy(cfg)
        c.setdefault("modeling", {})["ablation"] = arm
        df = run_train(c, model=model, run_id=f"state_screen_{model}_{arm}")
        metrics[arm] = df
        rollups[arm] = cv_rollup(df)

    # figures from the combined arm + ablation bars
    viz.results.build_report(cfg, metrics["combined"], pids, tag=f"screen_{model}")
    viz.results.ablation_bars(rollups, viz.results.figs_dir(cfg) / f"ablation_{model}.png")

    # comparison table
    print("\n" + "=" * 78)
    print(f"SCREEN ablation ({model}, {len(pids)} participants, "
          f"chance: acc 0.333 / AUC 0.5)")
    print("=" * 78)
    hdr = f"{'arm':10s} {'macroAUC':>9s} {'acc':>6s} {'mF1':>6s} " + \
          " ".join(f"{c[:4]:>6s}" for c in CLASS_NAMES) + f" {'gap':>6s}"
    print(hdr)
    for arm in ARMS:
        r = rollups[arm].iloc[0]
        print(f"{arm:10s} {r['macro_auc_mean']:9.3f} {r.get('overall_accuracy_mean', float('nan')):6.3f} "
              f"{r.get('macro_f1_mean', float('nan')):6.3f} "
              + " ".join(f"{r[f'accuracy_{c}']:6.3f}" for c in CLASS_NAMES)
              + f" {r.get('overfit_gap_mean', float('nan')):6.3f}")
    print("=" * 78)
    print("Key questions: combined vs window = does SEP add? | window vs electrode = does src add?")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "xgb")
