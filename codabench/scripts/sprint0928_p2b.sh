#!/usr/bin/env bash
# Sprint 2026-09-28, Phase 2b: solver sizing at the worst case the sealed data
# could arrive in, 47 channels at 500 Hz (mock_sealed_500: 14,400 windows x 47
# x 2000 samples; the solver keeps its default chans="eeg", i.e. 43 channels).
# Runs ALONE (brief: the 500 Hz sizing run gets the machine to itself).
#   1. train through benchopt with blend_w="auto" (the self-chosen weight, the
#      slowest fit) into a log-dir submission folder, /usr/bin/time -v
#   2. read-only inference replay at 10 threads, then at 2 threads (a stand-in
#      for an unknown scoring CPU): must reproduce step 1's score exactly
# Sizing rule (brief Phase 2): fit <= 45 min, peak RSS <= 20 GB, predict of the
# whole test set <= 10 min at 10 threads and <= 30 min at 2 threads.
#   wsl.exe bash -lc 'bash ~/codabench/scripts/sprint0928_p2b.sh'
TAG=sprint0928_p2b
export XS_THREADS=10
source "$HOME/codabench/scripts/sealed_lib.sh"
export BENCHOPT_DATA_HOME="$HOME/neuralbench/benchopt_data"
cd "$HOME/codabench/2026-competition" || exit 1
DS="../datasets/mock_sealed.py[study=mock_sealed_500]"
SUB="$LOGDIR/submission"
RO=/tmp/sprint0928_p2b_replay
TIME=(/usr/bin/time -v)
BO=(benchopt run tracks/bci_decoding -d "$DS" --no-plot --no-html --no-cache)

step train500 45 120m "${TIME[@]}" env COMPET_SUBMISSION_DIR="$SUB" "${BO[@]}" \
    -s "../solvers/bci_decoding/riemann_sealed.py[blend_w=auto]" \
    -o "BCI-decoding[training=True]" --output sprint0928_p2b_train500
if [ -f "$LOGDIR/train500.done" ]; then
  rm -rf "$RO"; cp -r "$SUB" "$RO"; chmod -R a-w "$RO"
  step replay500_t10 10 60m "${TIME[@]}" env COMPET_SUBMISSION_DIR="$RO" "${BO[@]}" \
      -s "$RO/submission.py" --output sprint0928_p2b_replay500_t10
  step replay500_t2 30 90m "${TIME[@]}" env COMPET_SUBMISSION_DIR="$RO" XS_THREADS=2 \
      OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "${BO[@]}" \
      -s "$RO/submission.py" --output sprint0928_p2b_replay500_t2
  chmod -R u+w "$RO"
fi
python "$HOME/codabench/scripts/summarize_runs.py" \
    tracks/bci_decoding/outputs/sprint0928_p2b_*.parquet > "$LOGDIR/RESULTS.md" 2>&1
for s in train500 replay500_t10 replay500_t2; do
  [ -f "$LOGDIR/$s.done" ] || { echo "| $(date +%T) | $s not done |" >> "$STATUS"; exit 0; }
done
echo "| $(date +%T) | ALL DONE |" >> "$STATUS"
