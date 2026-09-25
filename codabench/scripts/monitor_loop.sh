#!/usr/bin/env bash
# Event stream for the Monitor tool: new STATUS rows (END/ALL DONE), new
# crash signatures in step logs, and every non-OK watchdog verdict, checked
# every INTERVAL s. Exits after ALL DONE.
#   bash ~/codabench/scripts/monitor_loop.sh ~/codabench/logs/sealed_p1 [300]
LOGDIR=${1:?logdir}; INTERVAL=${2:-300}
W="$HOME/codabench/scripts/watch_run.sh"
seen=0; seen_err=""
while true; do
  n=$(wc -l < "$LOGDIR/STATUS.md" 2>/dev/null || echo 0)
  if [ "$n" -gt "$seen" ]; then
    tail -n +$((seen + 1)) "$LOGDIR/STATUS.md" | grep -E "END|ALL DONE" ; seen=$n
  fi
  for f in $(grep -lE "Traceback|Error|Killed|OOM" "$LOGDIR"/*.log 2>/dev/null); do
    case " $seen_err " in *" $f "*) ;; *)
      echo "CRASH-SIGNATURE in $(basename "$f"): $(grep -E "Traceback|Error|Killed|OOM" "$f" | tail -1 | cut -c1-200)"
      seen_err="$seen_err $f";;
    esac
  done
  bash "$W" "$LOGDIR" | grep -E "verdict=(SLOW|STALLED|DEAD|FAILED)"
  if grep -q "ALL DONE" "$LOGDIR/STATUS.md" 2>/dev/null; then
    echo "ALL DONE in $LOGDIR"; exit 0
  fi
  sleep "$INTERVAL"
done
