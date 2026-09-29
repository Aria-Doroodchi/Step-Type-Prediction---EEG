#!/usr/bin/env bash
# Sprint 2026-09-29, Phase 1 (alongside the build): baseline reproduction and
# the no-new-code candidates (slow block, re-referencing controls).
# Brief: codabench/prompts/2026-09-29_temporal_spatial_features.md
#   wsl.exe bash -lc 'bash ~/codabench/scripts/sprint0929_f1.sh'   (LANES=A|B|AB)
# Lane A = every Scherer run (3-class, then 5-class: one results file per study,
# so Scherer stays in one lane); lane B = Tangermann + Zhou. Resumable: steps
# with .done are skipped, and sealed_run.py skips configs already in the file.
TAG=f0929
export XS_THREADS=${XS_THREADS:-8}
source "$HOME/codabench/scripts/sealed_lib.sh"
AL=router-psd:riemann
B=riemann:xd=1,fb=1
SPECS="$B $B,sl=1 $B,ref=laplacian $B,ref=car"

laneA() {
  step s3_base 12 60m python "$RUN" --tag $TAG --study scherer2015 --classes 0,1,3 \
    --models $SPECS --modes pooled persubject --aligns none $AL
  step s5_base 10 60m python "$RUN" --tag $TAG --study scherer2015 \
    --models $SPECS --modes pooled persubject --aligns $AL
}
laneB() {
  step tg_base 14 60m python "$RUN" --tag $TAG --study tangermann2012 \
    --models $SPECS --modes pooled persubject --aligns $AL
  step zh_base 4 30m python "$RUN" --tag $TAG --study zhou2016 \
    --models $SPECS --modes pooled persubject --aligns $AL
}
LANES=${LANES:-AB}
[[ $LANES == *A* ]] && { laneA & }
[[ $LANES == *B* ]] && { laneB & }
wait
for s in s3_base s5_base tg_base zh_base; do
  [ -f "$LOGDIR/$s.done" ] || { echo "| $(date +%T) | f1 lanes $LANES finished; $s not done |" >> "$STATUS"; exit 0; }
done
echo "| $(date +%T) | f1 ALL DONE |" >> "$STATUS"
