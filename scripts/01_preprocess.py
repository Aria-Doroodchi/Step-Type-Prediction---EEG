"""Stage 1 — raw .bdf → cleaned epoch .fif files (one per condition).

Usage:
    python scripts/01_preprocess.py
    python scripts/01_preprocess.py --participants P25
    python scripts/01_preprocess.py --config configs/smoke.yaml --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `eeg_steptype` importable when running this script directly without
# `pip install -e .`. Once installed, this is a no-op.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_steptype.config import load_config
from eeg_steptype.logging_utils import setup_logging, get_logger
from eeg_steptype.preprocessing import pipeline as preprocess


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="Config YAML (default: configs/default.yaml)")
    p.add_argument("--participants", nargs="*", help="Subset of participant IDs")
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    p.add_argument(
        "--participant-override-mode",
        choices=["raw_assembly_only", "full", "none"],
        default=None,
        help="How much of configs/overrides/Pxx.yaml to apply.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.participant_override_mode:
        cfg.setdefault("participant_overrides", {})["mode"] = args.participant_override_mode
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    log = get_logger("scripts.01_preprocess")

    pids = args.participants or cfg["participants"]
    log.info("Preprocessing %d participants", len(pids))
    for pid in pids:
        try:
            preprocess.run(pid, cfg, force=args.force)
        except Exception as exc:                      # noqa: BLE001
            log.exception("[%s] failed: %s", pid, exc)


if __name__ == "__main__":
    main()
