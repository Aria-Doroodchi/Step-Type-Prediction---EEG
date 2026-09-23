#!/usr/bin/env bash
# Quick "does it train + score on real data" test of our Track 2 solvers,
# capped to a few batches so it runs in minutes on CPU.
#
#   bash ~/codabench/scripts/test_track2_solvers.sh                  # dreyer2023, 40 batches
#   bash ~/codabench/scripts/test_track2_solvers.sh tangermann2012 20
#
# Trains EEGNet-StepType (3 epochs) and Riemann-StepType on the first
# MAX_BATCHES training batches (x64 windows), then scores on the FULL test
# split: a pipeline check, not a real result. Prints a score table; log in
# ~/codabench/logs/test_track2_<study>_<time>.log

set -u
STUDY="${1:-dreyer2023}"
MAX_BATCHES="${2:-40}"
source "$HOME/codabench/env.sh" >/dev/null
cd "$HOME/codabench/2026-competition"
LOG="$HOME/codabench/logs/test_track2_${STUDY}_$(date +%Y-%m-%d_%H%M).log"
SOLVERS="$HOME/codabench/solvers/bci_decoding"

echo "START $(date +%T) study=$STUDY max_batches=$MAX_BATCHES log=$LOG"
benchopt run tracks/bci_decoding -d "BCI[study=$STUDY]" \
    -s "$SOLVERS/eegnet_steptype.py[max_batches=$MAX_BATCHES,n_epochs=3]" \
    -s "$SOLVERS/riemann_steptype.py[max_batches=$MAX_BATCHES]" \
    -s MeanLogReg \
    -o "BCI-decoding[training=True]" --no-plot --no-html >"$LOG" 2>&1
rc=$?
echo "END $(date +%T) rc=$rc"
grep -E "fitting on|epoch [0-9]+/|early stop|Error|Traceback" "$LOG" | tail -12
python - <<'EOF'
import glob, pandas as pd
f = sorted(glob.glob("tracks/bci_decoding/outputs/benchopt_run_*.parquet"))[-1]
df = pd.read_parquet(f)
df["solver"] = df.solver_name.str.split("[").str[0]
print(df[["solver", "objective_balanced_accuracy", "objective_accuracy",
          "objective_n_classes", "time"]].to_string(index=False))
EOF
