#!/usr/bin/env bash
# Sprint 2026-10-01, Phase 2 G4: the 500 Hz sizing re-measured with the dual
# (n < p) LDA, ALONE (nothing else heavy running), one series: the recipe,
# bpt4 (the default) and bpt4+icoh (release-day step xa). Decision D2 of the
# brief: icoh passes rule 3b's 500 Hz gate iff fit(bpt4+icoh) <= 1.5 x
# fit(recipe) in this series (plus bpt4 with wcv_ref=strict, information). Same harness as sprint0929_f4.sh
# (analysis/xblocks_sizing.py: mock_sealed_500, blend_w="auto", 43 EEG ch).
#   wsl.exe bash -lc 'bash ~/codabench/scripts/sprint1001_sz.sh'
TAG=s1001
export XS_THREADS=${XS_THREADS:-10}
source "$HOME/codabench/scripts/sealed_lib.sh"
P=${PREFIX:-sz3}
SZ="$HOME/codabench/analysis/xblocks_sizing.py"
step "${P}_none" 20 120m python "$SZ" --xblocks ""
step "${P}_bpt4" 22 120m python "$SZ" --xblocks bpt4
step "${P}_bpt4_icoh" 25 150m python "$SZ" --xblocks bpt4_icoh
# the strict blend-weight references (Phase 3) recompute the blocks in each of
# the 6 folds: their cost at the sealed size, with the default blocks
step "${P}_bpt4_strict" 40 150m python "$SZ" --xblocks bpt4 --wcv_ref strict
grep -H "^SIZING\|fit seconds" "$LOGDIR"/${P}_*.log | sed "s|^$LOGDIR/||" > "$LOGDIR/RESULTS_sizing.md"
echo "| $(date +%T) | sz finished ($(grep -c '^[^:]*:SIZING' "$LOGDIR/RESULTS_sizing.md") SIZING lines) |" >> "$STATUS"
