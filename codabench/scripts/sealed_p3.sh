#!/usr/bin/env bash
# Phase 3 (sealed-phase prep): pooling vs personalisation, with the Phase 2
# alignment winner applied. Variants per family (analysis/sealed_personal.py):
# pooled | calib (pooled + per-subject LDA / last-layer fine-tune) | blend
# (weight chosen on training data only) | persubject; personal variants with
# the true id (oracle-id) and the log-PSD router's id (router-id).
#   ALIGN_R="router-psd:riemann" ALIGN_E="router-psd:euclid" bash ~/codabench/scripts/sealed_p3.sh
# Lane A: Tangermann (+ Zhou); lane B: Scherer 5-class. 10 threads each.
TAG=p3
export XS_THREADS=10
source "$HOME/codabench/scripts/sealed_lib.sh"
ALIGN_R=${ALIGN_R:-router-psd:riemann}
ALIGN_E=${ALIGN_E:-router-psd:euclid}
P="$HOME/codabench/analysis/sealed_personal.py"
echo "| $(date +%T) | config ALIGN_R=$ALIGN_R ALIGN_E=$ALIGN_E |" >> "$STATUS"

laneA() {
  step t_rx  8  45m  python "$P" --tag $TAG --study tangermann2012 --family riemann:xd=1,fb=1 --align $ALIGN_R
  step t_rn  6  45m  python "$P" --tag $TAG --study tangermann2012 --family riemann:xd=0,fb=1 --align $ALIGN_R
  step z_rx  3  30m  python "$P" --tag $TAG --study zhou2016 --family riemann:xd=1,fb=1 --align $ALIGN_R
  step z_rn  3  30m  python "$P" --tag $TAG --study zhou2016 --family riemann:xd=0,fb=1 --align $ALIGN_R
  step z_e   8  45m  python "$P" --tag $TAG --study zhou2016 --family eegnet_st --align $ALIGN_E --seeds 33 34 35
  step t_e   35 120m python "$P" --tag $TAG --study tangermann2012 --family eegnet_st --align $ALIGN_E --seeds 33 34 35
}
laneB() {
  step s_rx  12 60m  python "$P" --tag $TAG --study scherer2015 --family riemann:xd=1,fb=1 --align $ALIGN_R
  step s_rn  10 60m  python "$P" --tag $TAG --study scherer2015 --family riemann:xd=0,fb=1 --align $ALIGN_R
  step s_e   15 90m  python "$P" --tag $TAG --study scherer2015 --family eegnet_st --align $ALIGN_E --seeds 33 34 35
}
LANES=${LANES:-AB}
[[ $LANES == *A* ]] && { laneA & }
[[ $LANES == *B* ]] && { laneB & }
wait
for s in t_rx t_rn z_rx z_rn z_e t_e s_rx s_rn s_e; do
  [ -f "$LOGDIR/$s.done" ] || { echo "| $(date +%T) | lanes $LANES finished; $s not done yet |" >> "$STATUS"; exit 0; }
done
python "$HOME/codabench/analysis/sealed_summarize.py" --tag $TAG > "$LOGDIR/RESULTS.md" 2>&1
echo "| $(date +%T) | ALL DONE (RESULTS.md written) |" >> "$STATUS"
