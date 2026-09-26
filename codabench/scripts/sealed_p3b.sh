#!/usr/bin/env bash
# Phase 3b (sealed-phase prep): Phase 3 Riemann personalisation rerun with the
# blend_calib variant (pooled blended with pooled-extractor + per-subject LDA,
# weight from training CV), clean router alignment and rule-dependent online.
#   bash ~/codabench/scripts/sealed_p3b.sh
TAG=p3b
export XS_THREADS=10
source "$HOME/codabench/scripts/sealed_lib.sh"
P="$HOME/codabench/analysis/sealed_personal.py"
F=riemann:xd=1,fb=1
for st in zhou2016 tangermann2012 scherer2015; do
  step ${st}_router 12 60m python "$P" --tag $TAG --study $st --family $F --align router-psd:riemann
  step ${st}_online 12 60m python "$P" --tag $TAG --study $st --family $F --align online-64:riemann
done
python "$HOME/codabench/analysis/sealed_summarize.py" --tag $TAG > "$LOGDIR/RESULTS.md" 2>&1
echo "| $(date +%T) | ALL DONE (RESULTS.md written) |" >> "$STATUS"
