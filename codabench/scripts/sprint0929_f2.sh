#!/usr/bin/env bash
# Sprint 2026-09-29, Phase 2: screen the temporal (T) and spatial (S) feature
# blocks of analysis/xfeat_*.py against the recipe union riemann:xd=1,fb=1.
# Brief: codabench/prompts/2026-09-29_temporal_spatial_features.md § 5 Phase 2.
#   FAM=TS LANES=AB wsl.exe bash -lc 'bash ~/codabench/scripts/sprint0929_f2.sh'
# FAM picks the families to run now (T, S or TS); re-running with more
# families adds the rest (done steps and done configs are skipped).
# Lane A = Scherer 3-class (tag f0929) then Zhou; lane B = Tangermann then
# Scherer 5-class (tag f0929s5, its own results file: two lanes never append
# to one file). Baselines and the no-new-code controls ran in f1.
TAG=f0929
export XS_THREADS=${XS_THREADS:-10}
source "$HOME/codabench/scripts/sealed_lib.sh"
FAM=${FAM:-TS}
AL=router-psd:riemann
B=riemann:xd=1,fb=1
SPECS_T="$B,x=tseg2 $B,x=tseg3 $B,x=acm3x2 $B,x=acm2x4 $B,blocks=xdawn+broad+logvar,x=fb8 $B,x=fbd $B,x=bpt4"
SPECS_S="$B,x=fblv $B,x=fbrlv $B,x=reg $B,x=csp8 $B,x=icoh"
C3="--study scherer2015 --classes 0,1,3"

fam() {   # fam <T|S> <stepname> <eta> <timeout> <args...>: run only if FAM has it
  local f=$1; shift
  [[ $FAM == *$f* ]] || return 0
  local specs; [ "$f" = T ] && specs=$SPECS_T || specs=$SPECS_S
  local name=$1 eta=$2 to=$3; shift 3
  step "$name" "$eta" "$to" python "$RUN" "$@" --models $specs
}

laneA() {
  for f in T S; do
    fam $f s3_$f 45 150m --tag $TAG $C3 --modes pooled persubject --aligns none $AL
  done
  for f in T S; do
    fam $f zh_$f 10 60m --tag $TAG --study zhou2016 --modes pooled persubject --aligns $AL
  done
}
laneB() {
  for f in T S; do
    fam $f tg_$f 30 120m --tag $TAG --study tangermann2012 --modes pooled persubject --aligns $AL
  done
  for f in T S; do
    fam $f s5_$f 35 120m --tag ${TAG}s5 --study scherer2015 --modes pooled persubject --aligns $AL
  done
}
LANES=${LANES:-AB}
[[ $LANES == *A* ]] && { laneA & }
[[ $LANES == *B* ]] && { laneB & }
wait
for s in s3_T s3_S zh_T zh_S tg_T tg_S s5_T s5_S; do
  [ -f "$LOGDIR/$s.done" ] || { echo "| $(date +%T) | f2 FAM=$FAM LANES=$LANES finished; $s not done |" >> "$STATUS"; exit 0; }
done
python "$HOME/codabench/analysis/f0929_summarize.py" > "$LOGDIR/RESULTS_screen.md" 2>&1
echo "| $(date +%T) | f2 ALL DONE (RESULTS_screen.md written) |" >> "$STATUS"
