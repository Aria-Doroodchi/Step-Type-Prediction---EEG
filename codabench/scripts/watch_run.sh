#!/usr/bin/env bash
# Watchdog for a sealed_* log dir: one line per step.
#   bash ~/codabench/scripts/watch_run.sh ~/codabench/logs/sealed_p1
# RUN lines: elapsed vs ETA, seconds since the step log last grew, python
# alive + CPU %, RSS, free RAM/swap, verdict:
#   STALLED = log idle > 10 min; SLOW = elapsed > 1.5 x ETA;
#   DEAD = no live process, no .done and no END row.
LOGDIR=${1:?logdir}
STATUS="$LOGDIR/STATUS.md"
now=$(date +%s)
avail=$(free -g | awk '/Mem:/{print $7}'); swap=$(free -g | awk '/Swap:/{print $3}')
for s in "$LOGDIR"/*.start; do
  [ -e "$s" ] || continue
  name=$(basename "$s" .start)
  read -r t0 eta < "$s"
  el=$((now - t0)); log="$LOGDIR/$name.log"
  if [ -f "$LOGDIR/$name.done" ]; then
    echo "DONE $name"; continue
  fi
  if grep -q "END $name rc" "$STATUS" 2>/dev/null; then
    echo "FAIL $name $(grep "END $name rc" "$STATUS" | tail -1 | tr -d '|') verdict=FAILED"; continue
  fi
  pid=$(cat "$LOGDIR/$name.pid" 2>/dev/null)
  idle=$(( now - $(stat -c %Y "$log" 2>/dev/null || echo "$now") ))
  child=$(pgrep -P "$pid" 2>/dev/null | head -1)
  if [ -n "$child" ]; then
    read -r cpu rss <<< "$(ps -o %cpu=,rss= -p "$child")"
    alive="pid=$child cpu=${cpu}% rss=$((rss / 1024))M"
  else
    alive="no-process"
  fi
  v=OK
  [ "$el" -gt $((eta * 3 / 2)) ] && v=SLOW
  [ "$idle" -gt 600 ] && v=STALLED
  [ "$alive" = "no-process" ] && v=DEAD
  echo "RUN $name elapsed=$((el / 60))m/eta=$((eta / 60))m idle=${idle}s $alive avail=${avail}G swap=${swap}G last=\"$(tail -c 300 "$log" | tr '\r' '\n' | grep -v '^$' | tail -1 | cut -c1-110)\" verdict=$v"
done
grep -q "ALL DONE" "$STATUS" 2>/dev/null && echo "ALL DONE ($LOGDIR)"
exit 0
