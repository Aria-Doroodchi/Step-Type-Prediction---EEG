"""State-module pipeline driver (3-class: standing / straight / diagonal).

Sibling of run.py for the CNV pipeline. Stages run in sequence:
    preprocess -> src -> features -> train

Usage:
    python run_state.py --stages preprocess --participants P06 P03
    python run_state.py --stages src features train --model xgb
    python run_state.py --model logistic --participants P06 P03   # fast smoke
    python run_state.py --config configs/state/smoke.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from eeg_statetype.config import load_config
from eeg_statetype.logging_utils import setup_logging, get_logger
from eeg_statetype.preprocessing import pipeline as preprocess
from eeg_statetype.source_localization import pipeline as src_loc
from eeg_statetype.features import assemble as features
from eeg_statetype.models.train import run as run_train


STAGES = ["preprocess", "src", "features", "train"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", default=None,
                   help="Extra YAML overlay(s) merged on top of configs/state/default.yaml.")
    p.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    p.add_argument("--participants", nargs="*")
    p.add_argument("--model", default="xgb")
    p.add_argument("--run-id", default=None)
    p.add_argument("--channel-mode", choices=["full", "roi"], default=None)
    p.add_argument("--cv-mode", choices=["repeated_stratified", "grouped"], default=None)
    p.add_argument("--ablation", choices=["combined", "window", "electrode", "sep"],
                   default=None, help="Feature-block ablation arm for the train stage.")
    p.add_argument("--parallel-participants", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.parallel_participants is not None:
        cfg.setdefault("modeling", {}).setdefault("parallel", {})[
            "participants"] = int(args.parallel_participants)
    if args.n_jobs is not None:
        cfg.setdefault("resources", {})["n_jobs"] = int(args.n_jobs)
    if args.ablation is not None:
        cfg.setdefault("modeling", {})["ablation"] = args.ablation

    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    log = get_logger("run_state")

    pids = args.participants or cfg["participants"]
    if args.participants:
        cfg["participants"] = list(args.participants)
    log.info("State run: stages=%s model=%s participants=%d",
             args.stages, args.model, len(pids))

    for stage in args.stages:
        log.info("\n" + "=" * 70 + f"\nStage: {stage}\n" + "=" * 70)
        if stage == "preprocess":
            for pid in pids:
                preprocess.run(pid, cfg, force=args.force)
        elif stage == "src":
            for pid in pids:
                src_loc.run(pid, cfg, force=args.force)
        elif stage == "features":
            for pid in pids:
                features.run(pid, cfg, force=args.force)
        elif stage == "train":
            run_train(cfg, model=args.model, run_id=args.run_id,
                      channel_mode=args.channel_mode, cv_mode=args.cv_mode)


if __name__ == "__main__":
    main()
