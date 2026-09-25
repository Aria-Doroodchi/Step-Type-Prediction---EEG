#!/usr/bin/env bash
# Quick "does it train + score on real data" test of our Track 2 solvers,
# capped to a few batches so it runs in minutes on CPU.
#
#   bash ~/codabench/scripts/test_track2_solvers.sh                  # dreyer2023, 40 batches
#   bash ~/codabench/scripts/test_track2_solvers.sh tangermann2012 20
#   bash ~/codabench/scripts/test_track2_solvers.sh dreyer2023 40 \
#        "bandpass=[none,8to30],reference=[none,car,laplacian]"     # grid
#
# The optional 3rd argument is appended to the parameters of each solver
# that defines all of its keys, so a value list in it becomes a grid (one run
# per combination). A solver lacking a key is skipped (benchopt would abort
# the whole run on an unknown parameter), e.g. "slow_block=[True]" runs
# Riemann-StepType only.
#
# Trains EEGNet-StepType (3 epochs) and Riemann-StepType on the first
# MAX_BATCHES training batches (x64 windows), then scores on the FULL test
# split: a pipeline check, not a real result. Prints a score table; log in
# ~/codabench/logs/test_track2_<study>_<time>.log

set -u
STUDY="${1:-dreyer2023}"
MAX_BATCHES="${2:-40}"
GRID="${3:+,$3}"
source "$HOME/codabench/env.sh" >/dev/null
cd "$HOME/codabench/2026-competition"
LOG="$HOME/codabench/logs/test_track2_${STUDY}_$(date +%Y-%m-%d_%H%M).log"
SOLVERS="$HOME/codabench/solvers/bci_decoding"

KEYS=$(printf '%s' "${3:-}" | grep -oE '[A-Za-z_]+=' | tr -d '=')
supports() {                   # supports <solver file>: defines every grid key
  local k; for k in $KEYS; do grep -q "\"$k\"" "$1" || return 1; done
}
RUN=()
if supports "$SOLVERS/eegnet_steptype.py"; then
  RUN+=(-s "$SOLVERS/eegnet_steptype.py[max_batches=$MAX_BATCHES,n_epochs=3$GRID]")
else echo "skip EEGNet-StepType (grid keys not among its parameters)"; fi
if supports "$SOLVERS/riemann_steptype.py"; then
  RUN+=(-s "$SOLVERS/riemann_steptype.py[max_batches=$MAX_BATCHES$GRID]")
else echo "skip Riemann-StepType (grid keys not among its parameters)"; fi

echo "START $(date +%T) study=$STUDY max_batches=$MAX_BATCHES grid=${3:-none} log=$LOG"
benchopt run tracks/bci_decoding -d "BCI[study=$STUDY]" \
    "${RUN[@]}" -s MeanLogReg \
    -o "BCI-decoding[training=True]" --no-plot --no-html >"$LOG" 2>&1
rc=$?
echo "END $(date +%T) rc=$rc"
grep -E "Error|Traceback" "$LOG" | tail -6
python - <<'EOF'
import glob, re, pandas as pd
f = sorted(glob.glob("tracks/bci_decoding/outputs/benchopt_run_*.parquet"))[-1]
df = pd.read_parquet(f)
df["solver"] = df.solver_name.str.split("[").str[0]
for k in ("reference", "bandpass"):
    df[k] = df.solver_name.map(
        lambda s, k=k: (re.search(k + r"=([^,\]]+)", s) or [None, "-"])[1])
print(df[["solver", "reference", "bandpass", "objective_balanced_accuracy",
          "objective_accuracy", "time"]]
      .sort_values(["solver", "reference", "bandpass"]).to_string(index=False))
EOF
