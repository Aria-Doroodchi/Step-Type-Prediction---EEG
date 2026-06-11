#!/usr/bin/env bash
# Round 2 — POOLED XGB candidate screens (8-subject subset from pooling_compare.yaml).
# Each candidate layers on pooling_compare.yaml and runs partial pooling via scripts/09.
# Test folds are deterministic in cfg, so candidate-vs-baseline partial is PAIRED.
# PYTHONUTF8=1 avoids the cp1252 theta logging crash.
set -u
cd "C:/Users/Ali D/Documents/ML"
PY="./.venv/Scripts/python.exe"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
S9="scripts/09_pooling_comparison.py"

run () {  # run <run_id> <modes> [extra_overlay ...]
  local rid="$1"; local modes="$2"; shift 2
  rm -rf "outputs/runs/$rid" 2>/dev/null
  echo ">>> $rid  modes=[$modes]  overlays=[configs/pooling_compare.yaml $*]"
  $PY $S9 --config configs/pooling_compare.yaml "$@" \
      --model xgb --modes $modes --run-id "$rid" \
      > "outputs/perf_loop/$rid.log" 2>&1
  echo "exit=$? ($rid)"
  $PY outputs/perf_loop/aggregate.py "outputs/runs/$rid"
}

echo "######## ROUND 2 POOLED XGB SCREENS ########"
run r2_pool_base8  "per_participant partial"
run r2_pool_loose8 "partial" configs/_cand_pool_loose.yaml
run r2_pool_grid8  "partial" configs/_cand_pool_grid.yaml
echo "######## ROUND 2 SCREENS DONE ########"
