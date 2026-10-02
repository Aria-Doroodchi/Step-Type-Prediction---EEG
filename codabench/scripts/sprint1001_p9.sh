#!/usr/bin/env bash
# Sprint 2026-10-01, integration-review fix check: RELEASE_DAY section 7 step 2
# (final run, baked weight, HARNESS_FROM the step-1 replica) on the full-size
# rehearsal (mock_sealed_120, harness rows of s1001_replica120, w 0.75):
#   p9_neg  BLEND_W=0.5 (not step 1's weight): must stop with the new ERROR
#   p9_pos  BLEND_W=0.75: must pass and zip; its wall time is step 2's budget
#   wsl.exe bash -lc 'bash ~/codabench/scripts/sprint1001_p9.sh'
TAG=s1001
export XS_THREADS=${XS_THREADS:-10}
source "$HOME/codabench/scripts/sealed_lib.sh"
DH=$HOME/neuralbench/benchopt_data
S=mock_sealed_120
L="$HOME/codabench/logs"
DEC="RECIPE_SPEC=riemann:xd=1,fb=1,x=bpt4 RECIPE_ALIGN=router-psdctx:riemann CHANS=eeg WCV=loso WREF=strict"
W1=$(grep -o "solver auto w=[0-9.]*\|trained with the baked w=[0-9.]*" "$L/s1001_replica120/STATUS.md" | tail -1 | grep -o "[0-9.]*$")
echo "| $(date +%T) | p9 step-1 weight W1=$W1 |" >> "$STATUS"
for r in s1001_final120_neg s1001_final120; do
  [ -f "$LOGDIR/p9_${r##*_}.done" ] || rm -rf "$L/$r" "$L/sealed_$r"
done
# shellcheck disable=SC2086
step p9_neg 5 30m env $DEC SPLIT=calib:3 TEST_SUBJECTS=0,1,2,3,4,5,6,7,8,9 BLEND_W=0.5 GATE=final \
  RUN_NAME=s1001_final120_neg HARNESS_FROM=s1001_replica120 \
  DATASET="../datasets/mock_sealed.py[study=$S]" bash "$HOME/codabench/scripts/train_sealed.sh" "$DH" "$S"
# shellcheck disable=SC2086
step p9_pos 20 90m env $DEC SPLIT=calib:3 TEST_SUBJECTS=0,1,2,3,4,5,6,7,8,9 BLEND_W=$W1 GATE=final \
  RUN_NAME=s1001_final120 HARNESS_FROM=s1001_replica120 \
  DATASET="../datasets/mock_sealed.py[study=$S]" bash "$HOME/codabench/scripts/train_sealed.sh" "$DH" "$S"
{
  echo "# step-2 guard check, $(date '+%F %T') (W1=$W1)"
  for r in s1001_final120_neg s1001_final120; do
    echo; echo "## $r"
    grep -hE "ERROR|WARNING|HARNESS_FROM|blend weight|gate:|zipped|NO ZIP|START|END" "$L/$r/STATUS.md" | tr -d '|' | sed 's/^/- /'
  done
} > "$LOGDIR/RESULTS_p9.md" 2>&1
echo "| $(date +%T) | p9 finished |" >> "$STATUS"
