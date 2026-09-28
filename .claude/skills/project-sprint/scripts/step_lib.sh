# Resumable step runner for sprint phase scripts (bundled with the project-sprint
# skill; codabench/scripts/sealed_lib.sh is the Track 2 variant it came from).
#
# In a phase script:
#   LOGDIR=/path/to/logs/<tag>                          # required
#   ENV_SETUP='source ~/codabench/env.sh >/dev/null'    # optional, eval'ed once
#   THREADS_PER_JOB=10                                  # optional (default 10)
#   source "<skill>/scripts/step_lib.sh"
#   step <name> <eta_min> <timeout> <cmd...>            # e.g. step fit 40 120m python x.py
#   finish_if_done <step names...>                      # ALL DONE once every step is done
#
# step skips a step whose <name>.done exists. Otherwise it writes <name>.start
# ("epoch_start eta_seconds", read by watch_run.sh), <name>.pid, START/END rows in
# STATUS.md, runs the command under `timeout` with all output in <name>.log, and
# touches <name>.done when the command exits 0. Re-running the phase script
# therefore resumes where it stopped.
#
# Thread pools are capped per job: two uncapped lanes each spawning a full set of
# OpenBLAS threads made a 9 s fit take > 150 s on this machine (2026-09-25).

set -u
: "${LOGDIR:?set LOGDIR before sourcing step_lib.sh}"
if [ -n "${ENV_SETUP:-}" ]; then eval "$ENV_SETUP"; fi
export PYTHONUTF8=1
THREADS_PER_JOB=${THREADS_PER_JOB:-10}
export OMP_NUM_THREADS=$THREADS_PER_JOB OPENBLAS_NUM_THREADS=$THREADS_PER_JOB \
       MKL_NUM_THREADS=$THREADS_PER_JOB NUMEXPR_NUM_THREADS=$THREADS_PER_JOB \
       XS_THREADS=$THREADS_PER_JOB
mkdir -p "$LOGDIR"
STATUS="$LOGDIR/STATUS.md"
[ -f "$STATUS" ] || printf '| time | event |\n|---|---|\n' > "$STATUS"

status_row() { echo "| $(date +%T) | $* |" >> "$STATUS"; }

step() {
  local name=$1 eta=$2 to=$3
  shift 3
  if [ -f "$LOGDIR/$name.done" ]; then
    status_row "skip $name (done)"
    return 0
  fi
  echo "$(date +%s) $((eta * 60))" > "$LOGDIR/$name.start"
  status_row "START $name (ETA $eta min)"
  timeout "$to" "$@" > "$LOGDIR/$name.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$LOGDIR/$name.pid"
  wait "$pid"
  local rc=$?
  status_row "END $name rc=$rc"
  [ "$rc" -eq 0 ] && touch "$LOGDIR/$name.done"
  return 0
}

finish_if_done() {
  local s
  for s in "$@"; do
    if [ ! -f "$LOGDIR/$s.done" ]; then
      status_row "lanes ${LANES:-all} finished; $s not done"
      return 0
    fi
  done
  status_row "ALL DONE"
}
