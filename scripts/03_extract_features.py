"""Stage 3 — epochs (+ src CSV) → cached per-participant feature parquet.

Usage:
    python scripts/03_extract_features.py
    python scripts/03_extract_features.py --participants P25 --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_steptype.config import apply_prediction_window, load_config
from eeg_steptype.logging_utils import setup_logging, get_logger
from eeg_steptype.features import assemble


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--participants", nargs="*")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--prediction-window",
        default=None,
        help="Named prediction window from config, e.g. late_cnv or full_cnv.",
    )
    p.add_argument(
        "--participant-override-mode",
        choices=["raw_assembly_only", "full", "none"],
        default=None,
        help="Accepted for workflow consistency; feature extraction does not apply participant YAMLs.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg = apply_prediction_window(cfg, args.prediction_window)
    if args.participant_override_mode:
        cfg.setdefault("participant_overrides", {})["mode"] = args.participant_override_mode
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    log = get_logger("scripts.03_extract_features")

    pids = args.participants or cfg["participants"]
    log.info("Extracting features for %d participants", len(pids))
    for pid in pids:
        try:
            assemble.run(pid, cfg, force=args.force)
        except Exception as exc:                      # noqa: BLE001
            log.exception("[%s] failed: %s", pid, exc)


if __name__ == "__main__":
    main()
