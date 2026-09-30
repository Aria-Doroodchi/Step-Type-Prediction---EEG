#!/usr/bin/env bash
# Sprint 2026-09-29, Phase 4 verification of the adopted block (bpt4) and the
# release-day ablation wiring, on the regression study and the small mock:
#   v_zhou       the release-day regression gate with DEFAULTS (train_sealed.sh
#                on zhou2016_xsess): must reproduce train = replay = 0.770000
#   v_mock_bpt4  train_sealed.sh on mock_sealed_s with
#                RECIPE_SPEC=riemann:xd=1,fb=1,x=bpt4: the candidate bakes
#                xblocks="bpt4"; the gates (train = replay, harness gap) pass
#   v_abl        release_ablations.sh on mock_sealed_s, the release replica
#                split, SPEC with x=bpt4, XB=icoh: steps base / xb / xa
#   v_abl_sum    release_summarize.py on those rows (the blocks rules)
# (The mock's accuracies mean nothing for the recipe: this checks the machinery.)
#   wsl.exe bash -lc 'bash ~/codabench/scripts/sprint0929_f5.sh'
TAG=f0929
export XS_THREADS=${XS_THREADS:-10}
source "$HOME/codabench/scripts/sealed_lib.sh"
DH=$HOME/neuralbench/benchopt_data
FULL=0,1,2,3,4,5,6,7,8,9
R=regress_f0929
[ -f "$LOGDIR/v_zhou.done" ] || rm -rf "$HOME/codabench/logs/$R" "$HOME/codabench/logs/sealed_$R"
step v_zhou 5 30m env SUBMISSION_DIR="$HOME/codabench/logs/$R/submission" RUN_NAME=$R \
  bash "$HOME/codabench/scripts/train_sealed.sh" "$DH" zhou2016_xsess
R2=f0929_mock_bpt4
[ -f "$LOGDIR/v_mock_bpt4.done" ] || rm -rf "$HOME/codabench/logs/$R2" "$HOME/codabench/logs/sealed_$R2"
step v_mock_bpt4 15 60m env DATASET="MockSealed[study=mock_sealed_s]" \
  RECIPE_SPEC=riemann:xd=1,fb=1,x=bpt4 SUBMISSION_DIR="$HOME/codabench/logs/$R2/submission" \
  RUN_NAME=$R2 bash "$HOME/codabench/scripts/train_sealed.sh" "$DH" mock_sealed_s
step v_abl 20 60m env TAG=f0929_abl_s STUDY=mock_sealed_s SPLIT=calib:3 TEST_SUBJECTS=$FULL \
  SPEC=riemann:xd=1,fb=1,x=bpt4 XB=icoh XS_THREADS=5 STEPS="base xb xa" \
  bash "$HOME/codabench/scripts/release_ablations.sh"
step v_abl_sum 3 15m python "$HOME/codabench/analysis/release_summarize.py" --tag f0929_abl_s \
  --study mock_sealed_s --split calib:3 --test_subjects $FULL \
  --spec riemann:xd=1,fb=1,x=bpt4 --xb icoh
for R_ in $R $R2; do
  echo "| $(date +%T) | $R_: $(grep -h 'gate:' "$HOME/codabench/logs/$R_/STATUS.md" 2>/dev/null | tail -1 | tr -d '|')" \
       "$(grep -ch 'ERROR' "$HOME/codabench/logs/$R_/STATUS.md" 2>/dev/null) ERROR rows |" >> "$STATUS"
done
echo "| $(date +%T) | f5 verification finished |" >> "$STATUS"
