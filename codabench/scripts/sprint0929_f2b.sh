#!/usr/bin/env bash
# Sprint 2026-09-29, Phase 2 (second half): screen the spatial blocks (S) and
# the two EDA-motivated candidates (E: tcut1000, fbfrom1000; LOG 18:45) with the
# same configs as sprint0929_f2.sh (which ran the temporal family). Run it only
# after f2's lanes have ended: the lanes append to the same per-study files.
#   FAM=SE LANES=AB wsl.exe bash -lc 'bash ~/codabench/scripts/sprint0929_f2b.sh'
TAG=f0929
export XS_THREADS=${XS_THREADS:-10}
source "$HOME/codabench/scripts/sealed_lib.sh"
FAM=${FAM:-SE}
AL=router-psd:riemann
B=riemann:xd=1,fb=1
SPECS_S="$B,x=fblv $B,x=fbrlv $B,x=reg $B,x=csp8 $B,x=icoh"
SPECS_E="$B,x=tcut1000 $B,blocks=xdawn+broad+logvar,x=fbfrom1000"
C3="--study scherer2015 --classes 0,1,3"
for s in s3_T zh_T tg_T s5_T; do
  pid=$(cat "$LOGDIR/$s.pid" 2>/dev/null)
  if [ -n "$pid" ] && [ ! -f "$LOGDIR/$s.done" ] && kill -0 "$pid" 2>/dev/null; then
    echo "| $(date +%T) | f2b refused: $s (pid $pid) still running |" >> "$STATUS"; exit 1
  fi
done

fam() {   # fam <S|E> <stepname> <eta> <timeout> <args...>: run only if FAM has it
  local f=$1; shift
  [[ $FAM == *$f* ]] || return 0
  local specs; [ "$f" = S ] && specs=$SPECS_S || specs=$SPECS_E
  local name=$1 eta=$2 to=$3; shift 3
  step "$name" "$eta" "$to" python "$RUN" "$@" --models $specs
}

laneA() {
  for f in S E; do fam $f s3_$f 40 150m --tag $TAG $C3 --modes pooled persubject --aligns none $AL; done
  for f in S E; do fam $f zh_$f 8 60m --tag $TAG --study zhou2016 --modes pooled persubject --aligns $AL; done
}
laneB() {
  for f in S E; do fam $f tg_$f 25 120m --tag $TAG --study tangermann2012 --modes pooled persubject --aligns $AL; done
  for f in S E; do fam $f s5_$f 30 120m --tag ${TAG}s5 --study scherer2015 --modes pooled persubject --aligns $AL; done
}
LANES=${LANES:-AB}
[[ $LANES == *A* ]] && { laneA & }
[[ $LANES == *B* ]] && { laneB & }
wait
for s in s3_S s3_E zh_S zh_E tg_S tg_E s5_S s5_E; do
  [ -f "$LOGDIR/$s.done" ] || { echo "| $(date +%T) | f2b FAM=$FAM LANES=$LANES finished; $s not done |" >> "$STATUS"; exit 0; }
done
python "$HOME/codabench/analysis/f0929_summarize.py" > "$LOGDIR/RESULTS_screen.md" 2>&1
echo "| $(date +%T) | f2b ALL DONE (RESULTS_screen.md written) |" >> "$STATUS"
