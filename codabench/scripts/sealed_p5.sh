#!/usr/bin/env bash
# Phase 5 (sealed-phase prep): cross-dataset pre-training of EEGNet-StepType on
# the MI datasets (Dreyer 2023 + Tangermann 2012 + Zhou 2016, all sessions,
# 11 shared channels, one head per dataset), fine-tuned per Scherer subject on
# the sealed-like 3 classes (WORD, SUB, HAND); from scratch vs fine-tuned.
#   P5_ALIGN="router-psd:euclid" bash ~/codabench/scripts/sealed_p5.sh
# Lane A: raw pre-training (3 seeds) -> fine-tune without alignment.
# Lane B: Euclidean-aligned pre-training (3 seeds) -> fine-tune with P5_ALIGN.
# Then the Riemann analogue (tangent-space reference from MI+Scherer vs Scherer).
TAG=p5
export XS_THREADS=10
source "$HOME/codabench/scripts/sealed_lib.sh"
P5_ALIGN=${P5_ALIGN:-router-psd:euclid}
PT="$HOME/codabench/analysis/sealed_pretrain.py"
echo "| $(date +%T) | config P5_ALIGN=$P5_ALIGN |" >> "$STATUS"

laneA() {
  for s in 33 34 35; do step pre_none_$s 25 90m python "$PT" pretrain --align none --seed $s; done
  step ft_none 25 120m python "$PT" finetune --align none --seeds 33 34 35
  step riem_ref 5 30m python "$PT" riemann
}
laneB() {
  for s in 33 34 35; do step pre_euclid_$s 25 90m python "$PT" pretrain --align euclid --seed $s; done
  step ft_align 25 120m python "$PT" finetune --align "$P5_ALIGN" --seeds 33 34 35
}
laneA & laneB &
wait
for s in pre_none_33 pre_none_34 pre_none_35 pre_euclid_33 pre_euclid_34 pre_euclid_35 ft_none ft_align riem_ref; do
  [ -f "$LOGDIR/$s.done" ] || { echo "| $(date +%T) | lanes finished; $s not done |" >> "$STATUS"; exit 0; }
done
python "$HOME/codabench/analysis/sealed_summarize.py" --tag $TAG > "$LOGDIR/RESULTS.md" 2>&1
echo "| $(date +%T) | ALL DONE (RESULTS.md written) |" >> "$STATUS"
