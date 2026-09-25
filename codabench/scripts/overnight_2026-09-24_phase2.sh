#!/usr/bin/env bash
# Overnight 2026-09-24, phase 2: follow-ups to phase 1 on full Dreyer 2023.
# Starts after phase 1 (waits for its ALL DONE). Same resumable step() pattern.
#
#   a) Riemann-StepType xDAWN variants: nfilter 2/4/8, and a 1-40 Hz
#      band-pass (drops sub-1 Hz drift, keeps the 1-4 Hz slow potential).
#   b) EEGNet-StepType trained longer (100 epochs, patience 20): validation
#      loss was still falling at 50 epochs in phase 1.
#   c) EEGNet-StepType with per-window z-scoring (standardize=True).
# b and c run for both references x 3 seeds, so no config is picked on the
# test split. Estimated ~3.5 h.

set -u
P1="$HOME/codabench/logs/overnight_2026-09-24/STATUS.md"
until grep -q "ALL DONE" "$P1" 2>/dev/null; do sleep 60; done

source "$HOME/codabench/env.sh" >/dev/null
cd "$HOME/codabench/2026-competition"
TAG=overnight_2026-09-24_p2
LOGDIR="$HOME/codabench/logs/$TAG"
mkdir -p "$LOGDIR"
S="$HOME/codabench/solvers/bci_decoding"
STATUS="$LOGDIR/STATUS.md"
[ -f "$STATUS" ] || printf '| time | event |\n|---|---|\n' > "$STATUS"
DATA='BCI[study=dreyer2023]'
OBJ='BCI-decoding[training=True]'

step() {
  local name=$1 timeout=$2; shift 2
  if [ -f "$LOGDIR/$name.done" ]; then echo "| $(date +%T) | skip $name (done) |" >> "$STATUS"; return; fi
  echo "| $(date +%T) | START $name |" >> "$STATUS"
  benchopt run tracks/bci_decoding -d "$DATA" -o "$OBJ" --no-plot --no-html \
      --timeout "$timeout" --output "${TAG}_$name" "$@" > "$LOGDIR/$name.log" 2>&1
  local rc=$?
  echo "| $(date +%T) | END $name rc=$rc |" >> "$STATUS"
  [ $rc -eq 0 ] && touch "$LOGDIR/$name.done"
}

step riemann_xdawn 60m \
    -s "$S/riemann_steptype.py[nfilter=[2,4,8],reference=['none','car']]" \
    -s "$S/riemann_steptype.py[bandpass=['1to40'],reference=['none','car']]"

step eegnet_long 150m \
    -s "$S/eegnet_steptype.py[n_epochs=100,patience=20,reference=['none','car'],seed=[33,34,35]]"

step eegnet_zscore 90m \
    -s "$S/eegnet_steptype.py[standardize=True,reference=['none','car'],seed=[33,34,35]]"

python "$HOME/codabench/scripts/summarize_runs.py" \
    tracks/bci_decoding/outputs/overnight_2026-09-24_*.parquet > "$LOGDIR/RESULTS_all.md" 2>&1
echo "| $(date +%T) | ALL DONE (RESULTS_all.md written) |" >> "$STATUS"
