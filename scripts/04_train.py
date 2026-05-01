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

from eeg_steptype.config import load_config
from eeg_steptype.logging_utils import setup_logging, get_logger
from eeg_steptype.models.train import run as run_train, MODEL_FACTORIES


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--model", choices=sorted(MODEL_FACTORIES.keys()), default="xgb")
    p.add_argument("--run-id", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    log = get_logger("scripts.04_train")
    log.info("Training model=%s on %d participants", args.model, len(cfg["participants"]))

    run_train(cfg, model=args.model, run_id=args.run_id)


if __name__ == "__main__":
    main()
