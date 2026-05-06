"""Single-process pipeline driver. Runs stages in sequence.

Usage:
    python run.py --config configs/default.yaml
    python run.py --stages preprocess src features train --model xgb
    python run.py --config configs/smoke.yaml --model logistic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from eeg_steptype.config import apply_prediction_window, load_config
from eeg_steptype.logging_utils import setup_logging, get_logger
from eeg_steptype.preprocessing import pipeline as preprocess
from eeg_steptype.source_localization import pipeline as src_loc
from eeg_steptype.features import assemble as features
from eeg_steptype.models.train import run as run_train


STAGES = ["preprocess", "src", "features", "train"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    p.add_argument("--participants", nargs="*")
    p.add_argument("--model", default="xgb")
    p.add_argument("--channel-mode", choices=["full", "roi"], default=None)
    p.add_argument("--cv-mode", choices=["repeated_stratified", "grouped", "chronological"], default=None)
    p.add_argument("--prediction-window", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--participant-override-mode",
        choices=["raw_assembly_only", "full", "none"],
        default=None,
        help="How much of configs/overrides/Pxx.yaml to apply.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg = apply_prediction_window(cfg, args.prediction_window)
    if args.participant_override_mode:
        cfg.setdefault("participant_overrides", {})["mode"] = args.participant_override_mode
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    log = get_logger("run")

    pids = args.participants or cfg["participants"]

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
            run_train(cfg, model=args.model, channel_mode=args.channel_mode, cv_mode=args.cv_mode)


if __name__ == "__main__":
    main()
