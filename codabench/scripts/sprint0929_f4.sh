#!/usr/bin/env bash
# Sprint 2026-09-29, Phase 4 sizing gate: Riemann-Sealed on the full-size
# 500 Hz mock (mock_sealed_500, blend_w="auto", 43 EEG ch) with and without
# the xblocks under test (brief § 5 Phase 4: fit <= 1.5 x the recipe's, peak
# RSS <= 16 GiB). One heavy lane, each variant in its own process.
#   XBS="bpt4" wsl.exe bash -lc 'bash ~/codabench/scripts/sprint0929_f4.sh'
TAG=f0929
export XS_THREADS=${XS_THREADS:-10}
source "$HOME/codabench/scripts/sealed_lib.sh"
XBS=${XBS:-bpt4}
P=${PREFIX:-sz}     # PREFIX=sz2: a second series (e.g. re-measured alone), its own .done markers
SZ="$HOME/codabench/analysis/xblocks_sizing.py"
step "${P}_none" 35 120m python "$SZ" --xblocks ""
for xb in $XBS; do
  step "${P}_$xb" 55 150m python "$SZ" --xblocks "$xb"
done
grep -H "^SIZING" "$LOGDIR"/sz*_*.log | sed "s|^$LOGDIR/||" > "$LOGDIR/RESULTS_sizing.md"
echo "| $(date +%T) | f4 sizing finished ($(grep -c . "$LOGDIR/RESULTS_sizing.md") SIZING lines) |" >> "$STATUS"
