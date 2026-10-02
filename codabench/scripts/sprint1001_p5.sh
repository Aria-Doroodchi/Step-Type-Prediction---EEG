#!/usr/bin/env bash
# Sprint 2026-10-01, Phase 5 (brief addendum): full-size 120 Hz rehearsal of
# the release-day commands as they stand after this sprint (dual LDA, x=bpt4,
# XB=icoh, WREF=strict) on mock_sealed_120 (20 x 6 sessions, 47-ch cache):
#   p5_abl    RELEASE_DAY section 5 (a) without STEPS (all 12 steps, 2 lanes x
#             6 threads); release_ablations.sh writes RESULTS.md (the
#             summarizer's DECISIONS) when every step is done
#   p5_rep    section 7 step 1 (replica, BLEND_W=auto) with the settings of the
#             summarizer's "train_sealed.sh:" line
# The mock's accuracies mean nothing for the recipe: this times the day and
# checks the machinery (train = replay, MATCH, fold scores EQUAL).
#   wsl.exe bash -lc 'bash ~/codabench/scripts/sprint1001_p5.sh'
TAG=s1001
export XS_THREADS=${XS_THREADS:-6}
source "$HOME/codabench/scripts/sealed_lib.sh"
DH=$HOME/neuralbench/benchopt_data
S=mock_sealed_120
FULL=0,1,2,3,4,5,6,7,8,9
L="$HOME/codabench/logs"
step p5_abl 90 300m env TAG=s1001_rel120 STUDY=$S SPLIT=calib:3 TEST_SUBJECTS=$FULL \
  XS_THREADS=6 SPEC=riemann:xd=1,fb=1,x=bpt4 XB=icoh WREF=strict \
  bash "$HOME/codabench/scripts/release_ablations.sh"
R="$L/sealed_s1001_rel120/RESULTS.md"
DECL=$(sed -n 's/^train_sealed.sh: \(.*\) bash .*/\1/p' "$R" 2>/dev/null | tail -1)
if [ -z "$DECL" ]; then
  echo "| $(date +%T) | ERROR: no train_sealed.sh line in $R |" >> "$STATUS"; exit 1
fi
echo "| $(date +%T) | p5 summarizer line: $DECL |" >> "$STATUS"
[ -f "$LOGDIR/p5_rep.done" ] || rm -rf "$L/s1001_replica120" "$L/sealed_s1001_replica120"
# shellcheck disable=SC2086
step p5_rep 45 180m env $DECL GATE=replica RUN_NAME=s1001_replica120 XS_THREADS=10 \
  DATASET="../datasets/mock_sealed.py[study=$S,split=replica_full]" \
  bash "$HOME/codabench/scripts/train_sealed.sh" "$DH" "$S"
grep -hE "gate:|blend weight:|fold scores per w|candidate|zipped|NO ZIP|ERROR|WARNING" \
  "$L/s1001_replica120/STATUS.md" | tr -d '|' > "$LOGDIR/RESULTS_p5_replica.md"
echo "| $(date +%T) | p5 finished |" >> "$STATUS"
