# Shared step runner for the sealed-phase proxy scripts (sealed_p*.sh).
# Source it after setting TAG:   TAG=p1; source "$HOME/codabench/scripts/sealed_lib.sh"
#
# step <name> <eta_min> <timeout> <cmd...>
#   skips a step whose <name>.done exists (resumable); writes <name>.start
#   ("epoch_start eta_seconds"), <name>.pid, <name>.log, START/END rows in
#   STATUS.md. The watchdog (watch_run.sh) reads those files.
set -u
source "$HOME/codabench/env.sh" >/dev/null
export PYTHONUTF8=1
# Cap every thread pool (OpenBLAS defaults to all 20 cores per process; two
# lanes x 20 spinning BLAS threads thrashed the first p1 launch, 2026-09-25).
export XS_THREADS=${XS_THREADS:-10}
export OMP_NUM_THREADS=$XS_THREADS OPENBLAS_NUM_THREADS=$XS_THREADS MKL_NUM_THREADS=$XS_THREADS
LOGDIR="$HOME/codabench/logs/sealed_$TAG"
mkdir -p "$LOGDIR"
STATUS="$LOGDIR/STATUS.md"
[ -f "$STATUS" ] || printf '| time | event |\n|---|---|\n' > "$STATUS"
RUN="$HOME/codabench/analysis/sealed_run.py"

step() {
  local name=$1 eta=$2 to=$3; shift 3
  if [ -f "$LOGDIR/$name.done" ]; then
    echo "| $(date +%T) | skip $name (done) |" >> "$STATUS"; return 0
  fi
  echo "$(date +%s) $((eta * 60))" > "$LOGDIR/$name.start"
  echo "| $(date +%T) | START $name (ETA $eta min) |" >> "$STATUS"
  timeout "$to" "$@" > "$LOGDIR/$name.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$LOGDIR/$name.pid"
  wait "$pid"; local rc=$?
  echo "| $(date +%T) | END $name rc=$rc |" >> "$STATUS"
  [ "$rc" -eq 0 ] && touch "$LOGDIR/$name.done"
  return 0
}
