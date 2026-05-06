"""Stage 4 — feature parquets → trained model + metrics CSV.

Usage:
    python scripts/04_train.py --model xgb
    python scripts/04_train.py --model lstm  --config configs/default.yaml
    python scripts/04_train.py --model logistic --config configs/smoke.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_steptype.config import apply_prediction_window, load_config
from eeg_steptype.logging_utils import setup_logging, get_logger
from eeg_steptype.models.train import run as run_train, MODEL_FACTORIES


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--model", choices=sorted(MODEL_FACTORIES.keys()), default="xgb")
    p.add_argument("--run-id", default=None)
    p.add_argument(
        "--prediction-window",
        default=None,
        help="Named prediction window from config, e.g. late_cnv or full_cnv.",
    )
    p.add_argument(
        "--channel-mode",
        choices=["full", "roi"],
        default=None,
        help="Training-time channel feature subset. Use twice for comparison: full and roi.",
    )
    p.add_argument(
        "--cv-mode",
        choices=["repeated_stratified", "grouped", "chronological"],
        default=None,
        help="Outer CV strategy. Defaults to modeling.cv.mode from the config.",
    )
    p.add_argument(
        "--participant-override-mode",
        choices=["raw_assembly_only", "full", "none"],
        default=None,
        help="Accepted for workflow consistency; training uses already-built features.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg = apply_prediction_window(cfg, args.prediction_window)
    if args.participant_override_mode:
        cfg.setdefault("participant_overrides", {})["mode"] = args.participant_override_mode
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    log = get_logger("scripts.04_train")
    log.info(
        "Training model=%s channel_mode=%s on %d participants",
        args.model, args.channel_mode or cfg.get("channel_selection", {}).get("mode", "full"),
        len(cfg["participants"]),
    )

    run_train(
        cfg,
        model=args.model,
        run_id=args.run_id,
        channel_mode=args.channel_mode,
        cv_mode=args.cv_mode,
    )


if __name__ == "__main__":
    main()
