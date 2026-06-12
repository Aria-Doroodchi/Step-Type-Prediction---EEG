# Project instructions

EEG step-type classification (straight `One` vs diagonal `Two` from CNV signals),
per-participant nested CV. MSc thesis project.

## Running tasks — estimate, time-log, and investigate overruns

Whenever you run a task that takes more than a few seconds (python scripts, training
runs, `cmd`/`powershell`/`bash` jobs, feature builds, test suites):

1. **Estimate first.** Before launching, state a rough expected duration / ETA
   (e.g. "≈45 min, ETA 17:00"), grounded in known per-unit costs (e.g. PSD Morlet
   build ≈1.5–2 min/condition, a pooled fold ≈10–30 s).
2. **Time-log.** Record the actual start time and elapsed/end time when it finishes
   (the run logs already carry `HH:MM:SS` stamps — cite them). Note whether it came
   in on, under, or over estimate.
3. **Investigate overruns.** If a task takes *significantly* longer than estimated
   (rule of thumb: >1.5–2× the estimate, or visibly stalled), STOP waiting passively
   and investigate: check the live log for the last progress line, confirm it is not
   stuck on one item (a deadlock, a swap-thrash, a silent fallback to the wrong/30-subject
   config), and fix the root cause before continuing. Do not just let it run.

This applies throughout the agentic perf/CNN loops. `scripts/09_pooling_comparison.py`
is **not** checkpointed (writes CSV only at the end), so a stall that runs past estimate
risks losing the whole run — catching it early matters.

## Key context

- Work happens on `perf/agentic-improvements` (XGB/CNN perf loop) and `feat/stim-module`.
  Source of truth for perf results: `outputs/perf_loop/LEDGER.md`.
- venvs: `.venv` = Python 3.14 (classical XGB/sklearn); `.venv312` = Python 3.12 + TF
  (neural only). Always run with `PYTHONUTF8=1` (a θ glyph crashes cp1252 console logging).
- Feature caches are keyed `{window}_{bin}_{cache_tag}` and do **not** include the block
  list — change blocks ⇒ bump `features.cache_tag` or you silently reuse the wrong parquet.
- `--config <nonexistent>` silently falls back to `default.yaml` (30 participants). Always
  verify the participant count and feature column count in the first log lines of a run.
