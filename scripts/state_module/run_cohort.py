"""Robust per-participant cohort runner for the state module.

Runs preprocess -> src -> features for each participant in its OWN subprocess
with a per-stage timeout. This is the lesson from the 22 h stall: one long-lived
process for the whole cohort is fragile — a single Picard hang / crash stalls
everything silently. Here each (participant, stage) is isolated, so:
  * memory is released between participants,
  * a hang is killed by the timeout and the batch continues,
  * the run is resumable (idempotent skip unless --force),
  * a progress heartbeat is written for live monitoring.

Usage:
  python scripts/state_module/run_cohort.py --config configs/state/screen.yaml \
      --stages preprocess src features --force
Monitor:  tail -f outputs/state_module/logs/cohort_progress.txt
Train (after): run separately (joblib-parallel), e.g. screen_ablation.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYEXE = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOGDIR = ROOT / "outputs" / "state_module" / "logs"
PROGRESS = LOGDIR / "cohort_progress.txt"

STAGE_TIMEOUT = {"preprocess": 1200, "src": 3000, "features": 900}  # seconds


def _load_participants(config: str) -> list[str]:
    sys.path.insert(0, str(ROOT / "src"))
    from eeg_statetype.config import load_config
    return list(load_config([config] if config else None)["participants"])


def _hb(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run_one(pid: str, stage: str, config: str, force: bool) -> tuple:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    plog = LOGDIR / f"cohort_{stage}_{pid}.log"
    cmd = [PYEXE, "run_state.py", "--config", config, "--stages", stage,
           "--participants", pid, "--n-jobs", "-4"]
    if force:
        cmd.append("--force")
    env = {"PYTHONUTF8": "1", "PYTHONWARNINGS": "ignore::FutureWarning"}
    import os
    full_env = {**os.environ, **env}
    t0 = time.time()
    try:
        with open(plog, "w", encoding="utf-8") as fh:
            r = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                               timeout=STAGE_TIMEOUT[stage], env=full_env)
        return (str(r.returncode), time.time() - t0)
    except subprocess.TimeoutExpired:
        return ("TIMEOUT", time.time() - t0)
    except Exception as exc:  # noqa: BLE001
        return (f"ERR:{exc!r}", time.time() - t0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/state/screen.yaml")
    p.add_argument("--stages", nargs="+", default=["preprocess", "src", "features"])
    p.add_argument("--participants", nargs="*")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    pids = args.participants or _load_participants(args.config)
    _hb(f"==== COHORT START: {len(pids)} participants, stages={args.stages}, "
        f"config={args.config}, force={args.force} ====")
    summary = []
    for i, pid in enumerate(pids, 1):
        for stage in args.stages:
            rc, dt = run_one(pid, stage, args.config, args.force)
            status = "ok" if rc == "0" else f"FAIL({rc})"
            _hb(f"[{i}/{len(pids)}] {pid} {stage}: {status} in {dt/60:.1f} min")
            summary.append((pid, stage, rc, dt))
            if rc != "0":
                _hb(f"    -> {pid} {stage} not ok; skipping remaining stages for {pid}")
                break
    ok = sum(1 for _, _, rc, _ in summary if rc == "0")
    _hb(f"==== COHORT DONE: {ok}/{len(summary)} (participant,stage) steps ok ====")
    bad = [(pid, st, rc) for pid, st, rc, _ in summary if rc != "0"]
    if bad:
        _hb("FAILURES: " + "; ".join(f"{pid}/{st}={rc}" for pid, st, rc in bad))


if __name__ == "__main__":
    main()
