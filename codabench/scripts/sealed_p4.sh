#!/usr/bin/env bash
# Phase 4 (sealed-phase prep): the sealed-like 3-class mental-task subset of
# Scherer 2015 (WORD word association, SUB mental subtraction, HAND right-hand
# MI = cache labels 0,1,3), cross-session, plus a Zyma 2019 calculation-vs-rest
# cross-subject sanity check.
#   ALIGN="router-psd:riemann" MODES="pooled persubject" bash ~/codabench/scripts/sealed_p4.sh
# ALIGN / MODES = the Phase 2-3 winners (defaults below). Block ablation of
# Riemann-StepType: all blocks, filter bank only, xDAWN only, broadband+logvar,
# logvar only, no xDAWN; EEGNet-StepType 3 seeds; Phase 3 personalisation.
TAG=p4
export XS_THREADS=10
source "$HOME/codabench/scripts/sealed_lib.sh"
ALIGN=${ALIGN:-router-psd:riemann}
MODES=${MODES:-pooled persubject}
C3="--study scherer2015 --classes 0,1,3"
ABL="riemann:xd=1,fb=1 riemann:xd=1,fb=1,blocks=fb riemann:xd=1,fb=1,blocks=xdawn riemann:xd=1,fb=1,blocks=broad+logvar riemann:xd=1,fb=1,blocks=logvar riemann:xd=0,fb=1 riemann:xd=1,fb=0 meanlr"
echo "| $(date +%T) | config ALIGN=$ALIGN MODES=$MODES |" >> "$STATUS"

laneA() {
  step s3_ablate 25 90m python "$RUN" --tag $TAG $C3 --models $ABL --modes $MODES --aligns none $ALIGN
  step s3_personal 15 60m python "$HOME/codabench/analysis/sealed_personal.py" --tag $TAG $C3 --family riemann:xd=1,fb=1 --align $ALIGN
  step s3_personal_nx 15 60m python "$HOME/codabench/analysis/sealed_personal.py" --tag $TAG $C3 --family riemann:xd=0,fb=1 --align $ALIGN
}
laneB() {
  step zyma 20 60m python "$HOME/codabench/analysis/sealed_zyma.py"
  step s3_eeg 40 150m python "$RUN" --tag $TAG $C3 --models eegnet_st --modes pooled --aligns none $ALIGN --seeds 33 34 35
}
laneA & laneB &
wait
for s in s3_ablate s3_personal s3_personal_nx zyma s3_eeg; do
  [ -f "$LOGDIR/$s.done" ] || { echo "| $(date +%T) | lanes finished; $s not done |" >> "$STATUS"; exit 0; }
done
python "$HOME/codabench/analysis/sealed_summarize.py" --tag $TAG > "$LOGDIR/RESULTS.md" 2>&1
echo "| $(date +%T) | ALL DONE (RESULTS.md written) |" >> "$STATUS"
