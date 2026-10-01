#!/usr/bin/env bash
# Sprint 2026-10-01, Phase 4: re-verify after the strict-reference review fixes
# (lazy global reference, harness folds written into Xa, fold-score line in
# train_sealed.sh). Two replica flows on mock_sealed_s (router-psdctx,
# BLEND_W=auto, WCV=loso), in parallel:
#   p3b_all     WREF=all    -> rows = committed sealed_replica_mock_sealed_s
#   p3b_strict  WREF=strict -> rows = sealed_s1001_rep_strict (before the fixes)
# both: MATCH and the new "fold scores per w ... EQUAL (4 dp)" line.
#   wsl.exe bash -lc 'bash ~/codabench/scripts/sprint1001_p3b.sh'
TAG=s1001
export XS_THREADS=${XS_THREADS:-5}
source "$HOME/codabench/scripts/sealed_lib.sh"
DH=$HOME/neuralbench/benchopt_data
FULL=0,1,2,3,4,5,6,7,8,9
L="$HOME/codabench/logs"
S="$HOME/codabench/scripts"
REP="SPLIT=calib:3 TEST_SUBJECTS=$FULL BLEND_W=auto GATE=replica CHANS=eeg RECIPE_SPEC=riemann:xd=1,fb=1 WCV=loso RECIPE_ALIGN=router-psdctx:riemann"
DSR="../datasets/mock_sealed.py[study=mock_sealed_s,split=replica_full]"
fresh() { [ -f "$LOGDIR/$1.done" ] || rm -rf "$L/$2" "$L/sealed_$2"; }
for w in all strict; do
  fresh p3b_$w s1001_rep2_$w
done
step p3b_all 12 60m env $REP WREF=all RUN_NAME=s1001_rep2_all DATASET="$DSR" \
  bash "$S/train_sealed.sh" "$DH" mock_sealed_s &
step p3b_strict 12 60m env $REP WREF=strict RUN_NAME=s1001_rep2_strict DATASET="$DSR" \
  bash "$S/train_sealed.sh" "$DH" mock_sealed_s &
wait
{
  echo "# Sprint 2026-10-01 Phase 4 re-verification after the strict review fixes, $(date '+%F %T')"
  for r in s1001_rep2_all s1001_rep2_strict; do
    echo; echo "## $r"
    grep -h "blend weight:\|fold scores per w\|gate:" "$L/$r/STATUS.md" | tr -d '|' | sed 's/^/- /'
  done
  echo; echo "## rows: s1001_rep2_all vs committed replica_mock_sealed_s"
  python "$HOME/codabench/analysis/compare_results.py" "$L/sealed_replica_mock_sealed_s" \
    "$L/sealed_s1001_rep2_all" mock_sealed_s | tail -2
  echo; echo "## rows: s1001_rep2_strict vs s1001_rep_strict (before the fixes)"
  python "$HOME/codabench/analysis/compare_results.py" "$L/sealed_s1001_rep_strict" \
    "$L/sealed_s1001_rep2_strict" mock_sealed_s | tail -2
} > "$LOGDIR/RESULTS_p3b.md" 2>&1
echo "| $(date +%T) | p3b finished |" >> "$STATUS"
