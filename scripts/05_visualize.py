"""Stage 5 — plot the latest run's metrics.

Usage:
    python scripts/05_visualize.py --run outputs/runs/xgb_20260101_120000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_steptype.config import load_config
from eeg_steptype.logging_utils import setup_logging, get_logger
from eeg_steptype.viz.results import plot_per_participant_accuracy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--run", required=True, help="Path to a runs/<id>/ directory")
    p.add_argument(
        "--participant-override-mode",
        choices=["raw_assembly_only", "full", "none"],
        default=None,
        help="Accepted for workflow consistency; visualization uses run outputs.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.participant_override_mode:
        cfg.setdefault("participant_overrides", {})["mode"] = args.participant_override_mode
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    log = get_logger("scripts.05_visualize")

    run = Path(args.run)
    metrics = run / "metrics.csv"
    if not metrics.exists():
        raise SystemExit(f"No metrics.csv at {metrics}")
    plot_per_participant_accuracy(metrics, run / "per_participant_accuracy.png")
    log.info("Done.")


if __name__ == "__main__":
    main()
