#!/usr/bin/env bash
# Sprint 2026-10-01, Phase 7 (brief addendum, pre-registered 20:05): the
# shared-covariance personal LDA screen (analysis/pc_screen.py) on the four
# confirmation proxies, recipe x=bpt4, router-psd, --wcv last; then the rule.
#   wsl.exe bash -lc 'bash ~/codabench/scripts/sprint1001_p7.sh'
TAG=s1001
export XS_THREADS=${XS_THREADS:-4}
source "$HOME/codabench/scripts/sealed_lib.sh"
PC="$HOME/codabench/analysis/pc_screen.py"
step p7_s3 15 60m python "$PC" --tag s1001pc --study scherer2015 --classes 0,1,3
step p7_zh 5 30m python "$PC" --tag s1001pc --study zhou2016
step p7_tg 20 60m python "$PC" --tag s1001pc --study tangermann2012
step p7_s5 20 60m python "$PC" --tag s1001pc --study scherer2015
python "$PC" --summary --tag s1001pc > "$LOGDIR/RESULTS_p7.md" 2>&1
echo "| $(date +%T) | p7 finished |" >> "$STATUS"
