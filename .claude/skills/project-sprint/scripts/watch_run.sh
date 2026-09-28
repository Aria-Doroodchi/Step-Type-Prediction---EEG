#!/usr/bin/env bash
# Watchdog for a step_lib log dir: one line per step.
#   bash watch_run.sh <logdir>
#
# DONE <step>                      finished (has .done)
# FAIL <step> ... verdict=FAILED   END row with a non-zero rc
# RUN  <step> elapsed=…m/eta=…m idle=…s pid/cpu/rss free-RAM last="…" verdict=V
#   V = OK | SLOW (elapsed > 1.5 x ETA) | STALLED (log idle > 10 min)
#       | DEAD (no live process, no .done, no END row)
# Works in WSL/Linux; in Git Bash (no pgrep/free) it still reports liveness,
# idle time and the verdict, without CPU/RSS/RAM.

LOGDIR=${1:?usage: watch_run.sh <logdir>}
STATUS="$LOGDIR/STATUS.md"
now=$(date +%s)
mem=""
if command -v free >/dev/null 2>&1; then
  mem="avail=$(free -g | awk '/Mem:/{print $7}')G swap=$(free -g | awk '/Swap:/{print $3}')G"
fi

for s in "$LOGDIR"/*.start; do
  [ -e "$s" ] || continue
  name=$(basename "$s" .start)
  read -r t0 eta < "$s"
  el=$((now - t0))
  log="$LOGDIR/$name.log"
  if [ -f "$LOGDIR/$name.done" ]; then
    echo "DONE $name"
    continue
  fi
  if grep -q "END $name rc" "$STATUS" 2>/dev/null; then
    echo "FAIL $name $(grep "END $name rc" "$STATUS" | tail -1 | tr -d '|') verdict=FAILED"
    continue
  fi
  pid=$(cat "$LOGDIR/$name.pid" 2>/dev/null)
  mtime=$(stat -c %Y "$log" 2>/dev/null || echo "$now")
  idle=$((now - mtime))
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    alive="pid=$pid"
    if command -v pgrep >/dev/null 2>&1; then
      child=$(pgrep -P "$pid" 2>/dev/null | head -1)
      if [ -n "$child" ]; then
        alive="pid=$child $(ps -o %cpu=,rss= -p "$child" 2>/dev/null \
                 | awk '{printf "cpu=%s%% rss=%dM", $1, $2 / 1024}')"
      fi
    fi
  else
    alive="no-process"
  fi
  v=OK
  [ "$el" -gt $((eta * 3 / 2)) ] && v=SLOW
  [ "$idle" -gt 600 ] && v=STALLED
  [ "$alive" = "no-process" ] && v=DEAD
  last=$(tail -c 400 "$log" 2>/dev/null | tr '\r' '\n' | grep -v '^$' | tail -1 | cut -c1-110)
  echo "RUN $name elapsed=$((el / 60))m/eta=$((eta / 60))m idle=${idle}s $alive $mem last=\"$last\" verdict=$v"
done
if grep -q "ALL DONE" "$STATUS" 2>/dev/null; then
  echo "ALL DONE ($LOGDIR)"
fi
exit 0
