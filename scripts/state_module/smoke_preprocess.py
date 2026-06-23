"""Smoke-run the state preprocessing on a few participants.

Usage: python scripts/state_module/smoke_preprocess.py [P06 P03 ...]
Default smoke set: P06 (1024 Hz Stim + 2048 Hz Standing — exercises per-file
sfreq end-to-end) and P03 (split Stim recording — exercises crop+concat).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from eeg_statetype.config import load_config
from eeg_statetype.logging_utils import setup_logging, get_logger
from eeg_statetype.preprocessing import pipeline as preprocess

SMOKE = ["P06", "P03"]


def main(pids):
    setup_logging("INFO")
    log = get_logger("state.smoke")
    cfg = load_config()
    log.info("State preprocess smoke: participants=%s", pids)
    for pid in pids:
        t0 = time.perf_counter()
        log.info("######## %s START ########", pid)
        try:
            preprocess.run(pid, cfg, force=True)
            log.info("######## %s DONE in %.1f min ########", pid,
                     (time.perf_counter() - t0) / 60.0)
        except Exception:
            log.exception("######## %s FAILED ########", pid)


if __name__ == "__main__":
    main(sys.argv[1:] or SMOKE)
