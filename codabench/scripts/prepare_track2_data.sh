#!/usr/bin/env bash
# Download + pre-extract the Track 2 (BCI decoding) datasets with benchopt.
#
#   bash ~/codabench/scripts/prepare_track2_data.sh                 # all three
#   bash ~/codabench/scripts/prepare_track2_data.sh dreyer2023      # just one
#
# `benchopt prepare` is idempotent: a finished study is a warm cache and
# returns quickly, so re-running after an interruption resumes where it
# stopped. Each study is retried up to 3 times (network hiccups).
# Data lands in $BENCHOPT_DATA_HOME/neural_compet (set by env.sh, on the WSL
# ext4 disk). Log: ~/codabench/logs/prepare_track2_<date>.log

set -u
source "$HOME/codabench/env.sh" >/dev/null
cd "$HOME/codabench/2026-competition"

# Smallest first: tangermann2012 = quick real-data smoke test;
# dreyer2023 = the Codabench warm-up proxy; stieger2021 = published baseline.
if [ $# -gt 0 ]; then STUDIES=("$@"); else STUDIES=(tangermann2012 dreyer2023 stieger2021); fi

LOG="$HOME/codabench/logs/prepare_track2_$(date +%Y-%m-%d_%H%M).log"
echo "log: $LOG"

for study in "${STUDIES[@]}"; do
  for attempt in 1 2 3; do
    echo "=== $(date +%T) START $study (attempt $attempt)" | tee -a "$LOG"
    benchopt prepare tracks/bci_decoding -d "BCI[study=$study]" >>"$LOG" 2>&1
    rc=$?
    size=$(du -sh "$BENCHOPT_DATA_HOME" 2>/dev/null | cut -f1)
    echo "=== $(date +%T) END $study rc=$rc data=$size" | tee -a "$LOG"
    [ $rc -eq 0 ] && break
    sleep 30
  done
done
echo "=== $(date +%T) ALL DONE" | tee -a "$LOG"
