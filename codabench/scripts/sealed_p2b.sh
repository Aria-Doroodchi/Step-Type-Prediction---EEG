#!/usr/bin/env bash
# Phase 2b (sealed-phase prep): online per-subject re-centring (rule-dependent,
# transductive): each test window is routed (log-PSD router) and whitened with
# the mean of the last N test windows routed to the same subject (training
# reference until N/4 are buffered); windows arrive in recording order, 64 per
# predict() call. N = 64, 128. Riemann xDAWN on/off + FB, pooled + persubject,
# all three proxies; EEGNet-StepType pooled (Euclidean) on Scherer and Zhou.
#   bash ~/codabench/scripts/sealed_p2b.sh
TAG=p2
export XS_THREADS=10
source "$HOME/codabench/scripts/sealed_lib.sh"
RIEM="riemann:xd=1,fb=1 riemann:xd=0,fb=1"
M="--modes pooled persubject --aligns online-64:riemann online-128:riemann"
step z_online 4  30m python "$RUN" --tag $TAG --study zhou2016 --models $RIEM $M
step s_online 12 60m python "$RUN" --tag $TAG --study scherer2015 --models $RIEM $M
step t_online 12 60m python "$RUN" --tag $TAG --study tangermann2012 --models $RIEM $M
step z_online_eeg 6 45m python "$RUN" --tag $TAG --study zhou2016 --models eegnet_st --modes pooled --aligns online-128:euclid --seeds 33 34 35
step s_online_eeg 15 60m python "$RUN" --tag $TAG --study scherer2015 --models eegnet_st --modes pooled --aligns online-128:euclid --seeds 33 34 35
echo "| $(date +%T) | p2b (online) steps finished |" >> "$STATUS"
