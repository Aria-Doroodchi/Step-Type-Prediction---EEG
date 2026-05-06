"""Preflight checks for a real-data pipeline run.

Usage:
    python scripts/00_preflight.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_steptype.config import load_config
from eeg_steptype.preflight import run_preflight


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    run_preflight(cfg)
    print("Preflight OK")


if __name__ == "__main__":
    main()
