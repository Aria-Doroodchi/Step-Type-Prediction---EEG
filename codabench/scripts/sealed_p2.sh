#!/usr/bin/env bash
# Phase 2 (sealed-phase prep): alignment without ids at predict time.
#   bash ~/codabench/scripts/sealed_p2.sh            # both lanes
#   LANES=B bash ~/codabench/scripts/sealed_p2.sh    # one lane (A or B) only
# Signal-level whitening X <- R^-1/2 X, R = Euclidean (EA) or Riemannian mean
# of the group's window covariances. Conditions (see analysis/sealed_run.py):
#   none | oracle (test session's own stats) | trainonly (global train ref) |
#   router-psd (per-window subject id from a log-PSD LDA router; clean) |
#   routerb-psd (64-window batch vote; rule-dependent) | batch (64-window
#   batch stats; rule-dependent)
# Riemann-StepType xDAWN on/off with the filter bank (the two strongest p1
# variants), pooled + persubject, on Tangermann, Scherer (5-class), Zhou.
# EEGNet-StepType (pooled, 3 seeds) on the key conditions: steps *_eeg.
# Lane A = Tangermann + Zhou, lane B = Scherer; 10 threads each. Resumable.
TAG=p2
export XS_THREADS=10
source "$HOME/codabench/scripts/sealed_lib.sh"
RIEM="riemann:xd=1,fb=1 riemann:xd=0,fb=1"
AL="none oracle:euclid oracle:riemann trainonly:euclid trainonly:riemann router-psd:euclid router-psd:riemann routerb-psd:riemann batch:riemann"
M="--modes pooled persubject --aligns $AL"
EAL="--modes pooled --aligns oracle:euclid router-psd:euclid trainonly:euclid --seeds 33 34 35"
# Tangermann EEGNet runs take ~11 min each (ES up to 100 epochs + refit): 2 conditions only
EAL_T="--modes pooled --aligns oracle:euclid router-psd:euclid --seeds 33 34 35"
STEPS="t_riem z_riem z_eeg t_eeg s_riem s_eeg"

laneA() {
  step t_riem 40 120m python "$RUN" --tag $TAG --study tangermann2012 --models $RIEM $M
  step z_riem 8  45m  python "$RUN" --tag $TAG --study zhou2016 --models $RIEM $M
  step z_eeg  15 60m  python "$RUN" --tag $TAG --study zhou2016 --models eegnet_st $EAL
  step t_eeg  70 180m python "$RUN" --tag $TAG --study tangermann2012 --models eegnet_st $EAL_T
}
laneB() {
  step s_riem 55 150m python "$RUN" --tag $TAG --study scherer2015 --models $RIEM $M
  step s_eeg  40 180m python "$RUN" --tag $TAG --study scherer2015 --models eegnet_st $EAL
}
LANES=${LANES:-AB}
[[ $LANES == *A* ]] && { laneA & }
[[ $LANES == *B* ]] && { laneB & }
wait
for s in $STEPS; do [ -f "$LOGDIR/$s.done" ] || { echo "| $(date +%T) | lanes $LANES finished; $s not done yet |" >> "$STATUS"; exit 0; }; done
python "$HOME/codabench/analysis/sealed_summarize.py" --tag p1 --tag $TAG > "$LOGDIR/RESULTS.md" 2>&1
echo "| $(date +%T) | ALL DONE (RESULTS.md written) |" >> "$STATUS"
