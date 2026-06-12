#!/usr/bin/env bash
# Run one candidate (screen: 8-subject express, or confirm: 20-subject express)
# on the standard 2.3k feature set, then aggregate to cohort AUC + gap.
#
# Usage:
#   screen.sh <run_id> <model> <scope> [extra_overlay.yaml ...]
#     scope = screen  -> 8-subject subset (_perf_subset8.yaml)
#     scope = confirm -> full 20-subject cohort (no subset overlay)
# Neural models (cnn/eegnet/eegnext) use the Python-3.12 venv automatically.
set -u
RID="$1"; MODEL="$2"; SCOPE="$3"; shift 3

PYBIN="./.venv/Scripts/python.exe"
case "$MODEL" in
  cnn|eegnet|eegnext) PYBIN="./.venv312/Scripts/python.exe" ;;
esac

# Base overlays: express CV + the standard fast feature set.
OVERLAYS="configs/express.yaml configs/_perf_fast.yaml"
WORKERS=8
if [ "$SCOPE" = "screen" ]; then
  OVERLAYS="$OVERLAYS configs/_perf_subset8.yaml"
elif [ "$SCOPE" = "confirm" ]; then
  OVERLAYS="$OVERLAYS configs/_perf_cohort20.yaml"
  WORKERS=12
fi

rm -rf "outputs/runs/$RID" 2>/dev/null
echo ">>> $RID  model=$MODEL scope=$SCOPE overlays=[$OVERLAYS $*]"
$PYBIN scripts/04_train.py --model "$MODEL" \
    --config $OVERLAYS "$@" \
    --run-id "$RID" --parallel-participants $WORKERS \
    > "outputs/perf_loop/$RID.log" 2>&1
echo "exit=$? ($RID)"
./.venv/Scripts/python.exe outputs/perf_loop/aggregate.py "outputs/runs/$RID"
