#!/usr/bin/env bash
# Phase 1 (sealed-phase prep): cross-session baselines on the proxies.
#   bash ~/codabench/scripts/sealed_p1.sh
# Tangermann 2012 (4-class, 9 x 2 sessions), Scherer 2015 (5-class, 9 x 2),
# Zhou 2016 (3-class, 4 x 3). Test = each subject's last session.
# Models: MeanLogReg; Riemann-StepType xDAWN on/off x filter bank on/off;
# EEGNet-StepType (100 ep, patience 20, ES then refit; 3 seeds); upstream
# braindecode EEGNet (20 ep; 3 seeds). Modes: pooled, persubject.
# Two lanes in parallel, 10 threads each. Resumable (.done + results.jsonl).
TAG=p1
export XS_THREADS=10
source "$HOME/codabench/scripts/sealed_lib.sh"

RIEM="riemann:xd=1,fb=0 riemann:xd=1,fb=1 riemann:xd=0,fb=0 riemann:xd=0,fb=1"
M="--modes pooled persubject --aligns none --seeds 33 34 35"

laneA() {
  step t_fast 4  30m python "$RUN" --tag $TAG --study tangermann2012 --models meanlr $RIEM $M
  step t_bd   8  45m python "$RUN" --tag $TAG --study tangermann2012 --models eegnet_bd $M
  step t_st   40 120m python "$RUN" --tag $TAG --study tangermann2012 --models eegnet_st $M
  step z_fast 2  20m python "$RUN" --tag $TAG --study zhou2016 --models meanlr $RIEM $M
  step z_bd   3  30m python "$RUN" --tag $TAG --study zhou2016 --models eegnet_bd $M
  step z_st   12 60m python "$RUN" --tag $TAG --study zhou2016 --models eegnet_st $M
}
laneB() {
  step s_fast 5  30m python "$RUN" --tag $TAG --study scherer2015 --models meanlr $RIEM $M
  step s_bd   8  45m python "$RUN" --tag $TAG --study scherer2015 --models eegnet_bd $M
  step s_st   40 120m python "$RUN" --tag $TAG --study scherer2015 --models eegnet_st $M
}
laneA & A=$!
laneB & B=$!
wait $A $B
python "$HOME/codabench/analysis/sealed_summarize.py" --tag $TAG > "$LOGDIR/RESULTS.md" 2>&1
echo "| $(date +%T) | ALL DONE (RESULTS.md written) |" >> "$STATUS"
