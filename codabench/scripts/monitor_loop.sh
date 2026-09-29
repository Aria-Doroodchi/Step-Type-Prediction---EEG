#!/usr/bin/env bash
# Event stream for the Monitor tool: new STATUS rows (step ENDs, gate / weight /
# zip lines and every row that ends a run), new crash signatures in step logs,
# and every non-OK watchdog verdict, checked every INTERVAL s.
#   bash ~/codabench/scripts/monitor_loop.sh ~/codabench/logs/sealed_p1 [300]
# Exits 0 after ALL DONE (anywhere in STATUS.md, as before) and after a row that
# ends a run without ALL DONE: release_ablations.sh's "lanes .. finished; <step>
# not done" (a STEPS subset) and train_sealed.sh's "stopped after <step>
# (STOP_AFTER, rc=0)"; exits 1 after an ERROR: / STOPPED (a failed step, a gate)
# / "stopped by a signal" row or a STOP_AFTER step with rc != 0.
# Such a row ends the watch when it appears while watching, or when it is still
# the last row one INTERVAL after the start (a run that stopped just before the
# Monitor started; a resumed run appends rows within seconds, so an old stop row
# of the same folder does not end the watch of the new run).
LOGDIR=${1:?logdir}; INTERVAL=${2:-300}
W="$HOME/codabench/scripts/watch_run.sh"
SHOW="END|ALL DONE|ERROR|NO ZIP|STOPPED|stopped|WARNING|gate:|blend weight:|zipped|finished;"
STOP_OK="stopped after [^ ]* \(STOP_AFTER, rc=0\)|lanes [A-Z]* finished;"
STOP_BAD="ERROR:|STOPPED|stopped by a signal|stopped after "   # (after STOP_OK: a failed STOP_AFTER step)
seen=0; seen_err=""; first=1; prev_n=-1
while true; do
  n=$(wc -l < "$LOGDIR/STATUS.md" 2>/dev/null || echo 0)
  new=0
  if [ "$n" -gt "$seen" ]; then
    tail -n +$((seen + 1)) "$LOGDIR/STATUS.md" | grep -E "$SHOW" ; seen=$n; new=1
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
  last=$(tail -n 1 "$LOGDIR/STATUS.md" 2>/dev/null)
  # a stop row that appeared while watching, or an unchanged one after an interval
  if { [ "$new" = 1 ] && [ "$first" = 0 ]; } || [ "$n" = "$prev_n" ]; then
    if echo "$last" | grep -qE "$STOP_OK"; then
      echo "RUN ENDED without ALL DONE in $LOGDIR:$(echo "$last" | tr -d '|')"; exit 0
    elif echo "$last" | grep -qE "$STOP_BAD"; then
      echo "RUN STOPPED in $LOGDIR:$(echo "$last" | tr -d '|')"; exit 1
    fi
  fi
  first=0; prev_n=$n
  sleep "$INTERVAL"
done
