#!/usr/bin/env bash
# Phase 0 (sealed-phase prep, 2026-09-25): build the cross-session window
# caches for the proxies. Resumable: a study whose cache exists is skipped.
#   bash ~/codabench/scripts/sealed_p0_caches.sh [study ...]
set -u
source "$HOME/codabench/env.sh" >/dev/null
LOGDIR="$HOME/codabench/logs/sealed_p0"
mkdir -p "$LOGDIR"
for s in "${@:-zhou2016 tangermann2012 scherer2015 zyma2019}"; do
  for study in $s; do
    echo "| $(date +%T) | START cache $study |" >> "$LOGDIR/STATUS.md"
    timeout 30m python "$HOME/codabench/analysis/xsess_cache.py" "$study" \
        > "$LOGDIR/cache_$study.log" 2>&1
    echo "| $(date +%T) | END cache $study rc=$? |" >> "$LOGDIR/STATUS.md"
  done
done
echo "| $(date +%T) | ALL DONE caches |" >> "$LOGDIR/STATUS.md"
