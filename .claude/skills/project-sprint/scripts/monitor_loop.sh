#!/usr/bin/env bash
# Event stream for the Monitor tool. Every INTERVAL seconds it prints:
#   - new END / ALL DONE rows of STATUS.md (on (re)start it replays earlier ones:
#     harmless),
#   - the first crash signature (Traceback|Error|Killed|OOM) found in each step log,
#   - every non-OK watchdog verdict (SLOW / STALLED / DEAD / FAILED).
# It exits after ALL DONE, so a silent Monitor means healthy only because every
# terminal state is covered.
#   bash monitor_loop.sh <logdir> [interval_s=300]
# Monitor caps a watch at 30 min: re-arm it until ALL DONE.

LOGDIR=${1:?usage: monitor_loop.sh <logdir> [interval_s]}
INTERVAL=${2:-300}
WATCH="$(cd "$(dirname "$0")" && pwd)/watch_run.sh"
seen=0
seen_err="|"
while true; do
  n=$(wc -l < "$LOGDIR/STATUS.md" 2>/dev/null || echo 0)
  if [ "$n" -gt "$seen" ]; then
    tail -n +$((seen + 1)) "$LOGDIR/STATUS.md" | grep -E "END|ALL DONE"
    seen=$n
  fi
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    case "$seen_err" in *"|$f|"*) continue ;; esac
    echo "CRASH-SIGNATURE in $(basename "$f"): $(grep -E "Traceback|Error|Killed|OOM" "$f" | tail -1 | cut -c1-200)"
    seen_err="$seen_err$f|"
  done < <(grep -lE "Traceback|Error|Killed|OOM|MemoryError" "$LOGDIR"/*.log 2>/dev/null)
  # live problems repeat every pass; a FAILED step is reported once (until re-run)
  while IFS= read -r line; do
    case "$line" in
      FAIL*) case "$seen_err" in *"|$line|"*) continue ;; esac
             seen_err="$seen_err$line|" ;;
    esac
    echo "$line"
  done < <(bash "$WATCH" "$LOGDIR" | grep -E "verdict=(SLOW|STALLED|DEAD|FAILED)")
  if grep -q "ALL DONE" "$LOGDIR/STATUS.md" 2>/dev/null; then
    echo "ALL DONE in $LOGDIR"
    exit 0
  fi
  sleep "$INTERVAL"
done
