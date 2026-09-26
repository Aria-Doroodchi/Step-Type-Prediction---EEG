#!/usr/bin/env bash
# Phase 4 (sealed-phase prep): the sealed-like 3-class mental-task subset of
# Scherer 2015 (WORD word association, SUB mental subtraction, HAND right-hand
# MI = cache labels 0,1,3), cross-session, plus a Zyma 2019 calculation-vs-rest
# cross-subject sanity check.
#   ALIGN_C="router-psd:riemann" ALIGN_RD="online-64:riemann" bash ~/codabench/scripts/sealed_p4.sh
# ALIGN_C = clean alignment (Phase 2 winner for pooled models), ALIGN_RD = the
# rule-dependent online re-centring, reported separately.
# Block ablation of Riemann-StepType: all blocks, filter bank only, xDAWN only,
# broadband + log-var, log-var only, no xDAWN, no filter bank; pooled and
# per-subject; Phase 3 personalisation on the 3-class subset; EEGNet-StepType
# pooled, 3 seeds.
TAG=p4
export XS_THREADS=10
source "$HOME/codabench/scripts/sealed_lib.sh"
ALIGN_C=${ALIGN_C:-router-psd:riemann}
ALIGN_RD=${ALIGN_RD:-online-64:riemann}
C3="--study scherer2015 --classes 0,1,3"
ABL="riemann:xd=1,fb=1 riemann:xd=1,fb=1,blocks=fb riemann:xd=1,fb=1,blocks=xdawn riemann:xd=1,fb=1,blocks=broad+logvar riemann:xd=1,fb=1,blocks=logvar riemann:xd=0,fb=1 riemann:xd=1,fb=0 meanlr"
P="$HOME/codabench/analysis/sealed_personal.py"
echo "| $(date +%T) | config ALIGN_C=$ALIGN_C ALIGN_RD=$ALIGN_RD |" >> "$STATUS"

laneA() {
  step s3_ablate 30 120m python "$RUN" --tag $TAG $C3 --models $ABL --modes pooled persubject --aligns none $ALIGN_C $ALIGN_RD
  step s3_personal 15 60m python "$P" --tag $TAG $C3 --family riemann:xd=1,fb=1 --align $ALIGN_C
  step s3_personal_rd 15 60m python "$P" --tag $TAG $C3 --family riemann:xd=1,fb=1 --align $ALIGN_RD
  step s3_personal_nx 15 60m python "$P" --tag $TAG $C3 --family riemann:xd=0,fb=1 --align $ALIGN_C
}
laneB() {
  step zyma 20 60m python "$HOME/codabench/analysis/sealed_zyma.py"
  step s3_eeg 30 150m python "$RUN" --tag $TAG $C3 --models eegnet_st --modes pooled persubject --aligns none --seeds 33 34 35
  step s3_eeg_al 20 150m python "$RUN" --tag $TAG $C3 --models eegnet_st --modes pooled --aligns oracle:euclid online-64:euclid --seeds 33 34 35
}
LANES=${LANES:-AB}
[[ $LANES == *A* ]] && { laneA & }
[[ $LANES == *B* ]] && { laneB & }
wait
for s in s3_ablate s3_personal s3_personal_rd s3_personal_nx zyma s3_eeg s3_eeg_al; do
  [ -f "$LOGDIR/$s.done" ] || { echo "| $(date +%T) | lanes $LANES finished; $s not done |" >> "$STATUS"; exit 0; }
done
python "$HOME/codabench/analysis/sealed_summarize.py" --tag $TAG > "$LOGDIR/RESULTS.md" 2>&1
echo "| $(date +%T) | ALL DONE (RESULTS.md written) |" >> "$STATUS"
