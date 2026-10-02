#!/usr/bin/env bash
# Sprint 2026-10-01, Phase 8 regression after the harness ctx_min change
# (sealed_run.aligned_data, router-psdctx): the committed replica flow on
# mock_sealed_s (pairs of 36 windows, none small) must reproduce its rows.
#   wsl.exe bash -lc 'bash ~/codabench/scripts/sprint1001_p8.sh'
TAG=s1001
export XS_THREADS=${XS_THREADS:-5}
source "$HOME/codabench/scripts/sealed_lib.sh"
DH=$HOME/neuralbench/benchopt_data
L="$HOME/codabench/logs"
[ -f "$LOGDIR/p8_rep.done" ] || rm -rf "$L/s1001_rep3_all" "$L/sealed_s1001_rep3_all"
step p8_rep 10 60m env SPLIT=calib:3 TEST_SUBJECTS=0,1,2,3,4,5,6,7,8,9 BLEND_W=auto GATE=replica \
  CHANS=eeg RECIPE_SPEC=riemann:xd=1,fb=1 WCV=loso RECIPE_ALIGN=router-psdctx:riemann WREF=all \
  RUN_NAME=s1001_rep3_all DATASET="../datasets/mock_sealed.py[study=mock_sealed_s,split=replica_full]" \
  bash "$HOME/codabench/scripts/train_sealed.sh" "$DH" mock_sealed_s
{
  echo "# Phase 8 regression (harness ctx_min), $(date '+%F %T')"
  grep -h "blend weight:\|fold scores per w\|gate:" "$L/s1001_rep3_all/STATUS.md" | tr -d '|' | sed 's/^/- /'
  python "$HOME/codabench/analysis/compare_results.py" "$L/sealed_replica_mock_sealed_s" \
    "$L/sealed_s1001_rep3_all" mock_sealed_s | tail -2
} > "$LOGDIR/RESULTS_p8.md" 2>&1
echo "| $(date +%T) | p8 finished |" >> "$STATUS"
