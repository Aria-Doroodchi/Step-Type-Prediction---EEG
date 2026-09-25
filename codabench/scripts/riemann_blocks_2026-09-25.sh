#!/usr/bin/env bash
# 2026-09-25: Riemann-StepType + the two new optional feature blocks
# (slow_block = binned < 4 Hz waveform, filterbank = 4-band tangent space),
# full Dreyer 2023 (train 12,392 -> test 5,040 windows), CPU.
#
#   bash ~/codabench/scripts/riemann_blocks_2026-09-25.sh
#
# Grid: reference=none, use_xdawn=True x slow_block x filterbank (2 x 2),
# plus use_xdawn=False with both blocks. One step per config so STATUS.md
# carries a wall time for each. Resumable like overnight_2026-09-24.sh: a
# finished step leaves <step>.done and is skipped on re-run.
# Progress: logs/riemann_blocks_2026-09-25/STATUS.md; results: RESULTS.md.
#
# Estimated total ~7-8 min (a full Riemann fit is ~45 s; filter bank adds
# 4 tangent-space fits).

set -u
source "$HOME/codabench/env.sh" >/dev/null
cd "$HOME/codabench/2026-competition"
TAG=riemann_blocks_2026-09-25
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

R="$S/riemann_steptype.py[reference=none,use_xdawn"

# 1. baseline: must reproduce 0.754 (blocks off)
step base      15m -s "$R=True,slow_block=False,filterbank=False]"
# 2-4. each block alone on top of xDAWN, then both
step slow      15m -s "$R=True,slow_block=True,filterbank=False]"
step fb        15m -s "$R=True,slow_block=False,filterbank=True]"
step slow_fb   15m -s "$R=True,slow_block=True,filterbank=True]"
# 5. both blocks without xDAWN (does the slow block replace it?)
step noxdawn_slow_fb 15m -s "$R=False,slow_block=True,filterbank=True]"

# 6. summary table
python "$HOME/codabench/scripts/summarize_runs.py" \
    tracks/bci_decoding/outputs/"${TAG}"_*.parquet > "$LOGDIR/RESULTS.md" 2>&1
echo "| $(date +%T) | ALL DONE (RESULTS.md written) |" >> "$STATUS"
