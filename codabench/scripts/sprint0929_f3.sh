#!/usr/bin/env bash
# Sprint 2026-09-29, Phase 3: confirm the advancing feature blocks under the
# deployed recipe (blend_calib, router ids, clean router alignment).
# Brief: codabench/prompts/2026-09-29_temporal_spatial_features.md § 5 Phase 3.
#   CANDS="riemann:xd=1,fb=1,x=<id> ..." wsl.exe bash -lc 'bash ~/codabench/scripts/sprint0929_f3.sh'
# The baseline riemann:xd=1,fb=1 always runs first. Lane A = Scherer 3-class
# (tag f0929p) then Zhou, then online-64 on Scherer 3-class (rule-dependent,
# information only); lane B = Tangermann (tag f0929p) then Scherer 5-class (tag
# f0929p5, its own results file). One step per (study, spec): resumable.
TAG=f0929
export XS_THREADS=${XS_THREADS:-10}
source "$HOME/codabench/scripts/sealed_lib.sh"
: "${CANDS:?set CANDS to the advancing specs}"
P="$HOME/codabench/analysis/sealed_personal.py"
AL=router-psd:riemann
ALL="riemann:xd=1,fb=1 $CANDS"
C3="--study scherer2015 --classes 0,1,3"
sid() { echo "$1" | md5sum | cut -c1-8; }   # step-name suffix per spec
for s in $ALL; do echo "| $(date +%T) | f3 spec $(sid "$s") = $s |" >> "$STATUS"; done

laneA() {
  for s in $ALL; do step p3_s3_$(sid "$s") 12 60m python "$P" --tag f0929p $C3 --family "$s" --align $AL; done
  for s in $ALL; do step p3_zh_$(sid "$s") 4 30m python "$P" --tag f0929p --study zhou2016 --family "$s" --align $AL; done
  for s in $ALL; do step p3_s3on_$(sid "$s") 10 60m python "$P" --tag f0929p $C3 --family "$s" --align online-64:riemann; done
}
laneB() {
  for s in $ALL; do step p3_tg_$(sid "$s") 14 60m python "$P" --tag f0929p --study tangermann2012 --family "$s" --align $AL; done
  for s in $ALL; do step p3_s5_$(sid "$s") 14 60m python "$P" --tag f0929p5 --study scherer2015 --family "$s" --align $AL; done
}
LANES=${LANES:-AB}
[[ $LANES == *A* ]] && { laneA & }
[[ $LANES == *B* ]] && { laneB & }
wait
for s in $ALL; do for st in s3 zh s3on tg s5; do
  [ -f "$LOGDIR/p3_${st}_$(sid "$s").done" ] || { echo "| $(date +%T) | f3 LANES=$LANES finished; p3_${st}_$(sid "$s") not done |" >> "$STATUS"; exit 0; }
done; done
python "$HOME/codabench/analysis/f0929_summarize.py" --tags f0929 f0929s5 f0929p f0929p5 > "$LOGDIR/RESULTS_confirm.md" 2>&1
echo "| $(date +%T) | f3 ALL DONE (RESULTS_confirm.md written) |" >> "$STATUS"
