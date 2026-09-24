#!/usr/bin/env bash
# Overnight 2026-09-24: full-data Dreyer 2023 runs (the warm-up proxy), CPU.
#
#   bash ~/codabench/scripts/overnight_2026-09-24.sh
#
# Resumable: each finished step leaves logs/overnight_2026-09-24/<step>.done
# and is skipped on re-run (benchopt's own cache also skips finished solver
# runs). Every step has a timeout so one stall cannot eat the night.
# Progress: logs/overnight_2026-09-24/STATUS.md; results: RESULTS.md.
#
# Estimated total ~2.5 h (see LOG.md for the per-step estimates).

set -u
source "$HOME/codabench/env.sh" >/dev/null
cd "$HOME/codabench/2026-competition"
TAG=overnight_2026-09-24
LOGDIR="$HOME/codabench/logs/$TAG"
mkdir -p "$LOGDIR"
S="$HOME/codabench/solvers/bci_decoding"
STATUS="$LOGDIR/STATUS.md"
[ -f "$STATUS" ] || printf '| time | event |\n|---|---|\n' > "$STATUS"

DATA='BCI[study=dreyer2023]'
OBJ='BCI-decoding[training=True]'

step() {                       # step <name> <timeout> <benchopt args...>
  local name=$1 timeout=$2; shift 2
  if [ -f "$LOGDIR/$name.done" ]; then echo "| $(date +%T) | skip $name (done) |" >> "$STATUS"; return; fi
  echo "| $(date +%T) | START $name |" >> "$STATUS"
  benchopt run tracks/bci_decoding -d "$DATA" -o "$OBJ" --no-plot --no-html \
      --timeout "$timeout" --output "${TAG}_$name" "$@" > "$LOGDIR/$name.log" 2>&1
  local rc=$?
  echo "| $(date +%T) | END $name rc=$rc |" >> "$STATUS"
  [ $rc -eq 0 ] && touch "$LOGDIR/$name.done"
}

# 1. fast upstream floors (minutes)
step baselines 30m -s MeanLogReg -s Torch-Linear

# 2. Riemann-StepType: deterministic, one run per config (4 configs)
step riemann 90m -s "$S/riemann_steptype.py[reference=['none','car'],use_xdawn=[True,False]]"

# 3. EEGNet-StepType: reference x patience x 3 seeds (12 runs)
step eegnet_steptype 90m \
    -s "$S/eegnet_steptype.py[reference=['none','car'],patience=[10,20],seed=[33,34,35]]"

# 4. upstream braindecode EEGNet (20 epochs), 3 benchopt seeds.
#    Its outputs/ folder is shared across runs: clear stale weights first.
for sd in 1 2 3; do
  rm -rf tracks/bci_decoding/outputs/EEGNet
  step "eegnet_upstream_seed$sd" 90m -s EEGNet --seed "$sd"
done

# 5. summary table
python "$HOME/codabench/scripts/summarize_runs.py" \
    tracks/bci_decoding/outputs/"${TAG}"_*.parquet > "$LOGDIR/RESULTS.md" 2>&1
echo "| $(date +%T) | ALL DONE (RESULTS.md written) |" >> "$STATUS"
