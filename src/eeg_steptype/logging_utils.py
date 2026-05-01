"""Logger setup + per-run reproducibility stamping."""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(level: str = "INFO", logfile: Path | None = None) -> logging.Logger:
    """Configure the root logger; idempotent."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        root.addHandler(sh)

    if logfile is not None:
        logfile = Path(logfile)
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        root.addHandler(fh)

    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Run stamping
# ---------------------------------------------------------------------------
def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parents[2],
        )
        return out.decode("ascii").strip()
    except Exception:
        return "unknown"


def stamp_run(run_dir: Path, cfg: dict, *, model: str | None = None) -> None:
    """Write config snapshot + git SHA + python/env info to ``run_dir``."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    with open(run_dir / "git_sha.txt", "w", encoding="utf-8") as fh:
        fh.write(git_sha() + "\n")

    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "model": model,
        "argv": sys.argv,
    }
    with open(run_dir / "env.json", "w", encoding="utf-8") as fh:
        json.dump(env, fh, indent=2)
