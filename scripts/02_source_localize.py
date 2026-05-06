"""Stage 2 — cleaned epochs → per-participant source-localized CSV.

Usage:
    python scripts/02_source_localize.py
    python scripts/02_source_localize.py --participants P25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_steptype.config import load_config
from eeg_steptype.logging_utils import setup_logging, get_logger
from eeg_steptype.source_localization import pipeline as src_loc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--participants", nargs="*")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--participant-override-mode",
        choices=["raw_assembly_only", "full", "none"],
        default=None,
        help="Accepted for workflow consistency; source localization does not apply participant YAMLs.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.participant_override_mode:
        cfg.setdefault("participant_overrides", {})["mode"] = args.participant_override_mode
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    log = get_logger("scripts.02_source_localize")

    pids = args.participants or cfg["participants"]
    log.info("Source-localizing %d participants", len(pids))
    for pid in pids:
        try:
            src_loc.run(pid, cfg, force=args.force)
        except Exception as exc:                      # noqa: BLE001
            log.exception("[%s] failed: %s", pid, exc)


if __name__ == "__main__":
    main()
