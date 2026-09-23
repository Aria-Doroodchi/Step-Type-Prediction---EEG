#!/usr/bin/env bash
# Download + pre-extract the Track 2 (BCI decoding) datasets with benchopt.
#
#   bash ~/codabench/scripts/prepare_track2_data.sh                 # tangermann2012 + dreyer2023
#   bash ~/codabench/scripts/prepare_track2_data.sh dreyer2023      # just one
#
# stieger2021 is 399 GB and is NOT in the default list: C: (which holds the
# WSL disk) has ~118 GB free. Pass it explicitly only after moving the data
# somewhere big (see TRACK2_BCI.md); the script refuses if < 450 GB is free.
#
# `benchopt prepare` is idempotent: a finished study is a warm cache and
# returns quickly, so re-running after an interruption resumes where it
# stopped. Each study is retried up to 3 times (network hiccups).
# Data lands in $BENCHOPT_DATA_HOME/neural_compet (set by env.sh, on the WSL
# ext4 disk). Log: ~/codabench/logs/prepare_track2_<date>.log

set -u
source "$HOME/codabench/env.sh" >/dev/null
cd "$HOME/codabench/2026-competition"

# Smallest first: tangermann2012 (1.2 GB) = quick real-data smoke test;
# dreyer2023 (~21 GB) = the Codabench warm-up proxy.
if [ $# -gt 0 ]; then STUDIES=("$@"); else STUDIES=(tangermann2012 dreyer2023); fi

LOG="$HOME/codabench/logs/prepare_track2_$(date +%Y-%m-%d_%H%M).log"
echo "log: $LOG"

for study in "${STUDIES[@]}"; do
  if [ "$study" = stieger2021 ]; then
    # WSL's own df shows the virtual disk's size, not the free space of C:
    # that actually backs it, so take the smaller of the two.
    free_gb=$(df -BG --output=avail "$BENCHOPT_DATA_HOME" /mnt/c | tail -n +2 | tr -dc '0-9\n' | sort -n | head -1)
    if [ "$free_gb" -lt 450 ]; then
      echo "=== $(date +%T) SKIP stieger2021: 399 GB needed, only ${free_gb} GB free under $BENCHOPT_DATA_HOME" | tee -a "$LOG"
      continue
    fi
  fi
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
