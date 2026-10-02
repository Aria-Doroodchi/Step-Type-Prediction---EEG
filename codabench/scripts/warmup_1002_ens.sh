#!/usr/bin/env bash
# Warm-up honest try 1 (2026-10-02): 5-member EEGNet softmax ensemble on clean Dreyer
# (train subjects 1-60 + 82-87, test 61-81). Compare to WU1 0.820 (config mean 0.806 +/- 0.016).
# ETA ~1.8-2.5 h (5 x ~21 min). Foreground; run it with run_in_background.
set -u
source "$HOME/codabench/env.sh" >/dev/null
cd "$HOME/codabench/2026-competition"
TAG=warmup_1002_ens
LOGDIR="$HOME/codabench/logs/$TAG"
mkdir -p "$LOGDIR"
S="$HOME/codabench/solvers/bci_decoding"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
echo "$(date +%T) START ens5" >> "$LOGDIR/STATUS.md"
benchopt run tracks/bci_decoding -d 'BCI[study=dreyer2023]' -s "$S/eegnet_steptype_ens.py" \
    -o 'BCI-decoding[training=True]' --no-plot --no-html --no-cache --timeout 240m \
    --output "${TAG}_ens5" > "$LOGDIR/ens5.log" 2>&1
rc=$?
echo "$(date +%T) END ens5 rc=$rc" >> "$LOGDIR/STATUS.md"
python -c "import pandas as pd; print(pd.read_parquet('tracks/bci_decoding/outputs/${TAG}_ens5.parquet').filter(like='objective').T)" >> "$LOGDIR/STATUS.md" 2>&1
