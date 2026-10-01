#!/usr/bin/env bash
# Sprint 2026-10-01, Phase 2 gates for the dual (n < p) shrinkage LDA (brief
# prompts/2026-10-01_dual_lda_strict_wcv.md § 5 Phase 2). Each run must
# reproduce a committed run (the dual solve is a numerical re-route):
#   lane A  g_zhou      regression flow (train_sealed.sh, zhou2016_xsess): 0.770000
#           g_mock_def  default flow on mock_sealed_s (p = 5073: dual per-subject
#                       and pooled LDAs): 0.483333, rows = sealed_f0929_mock_default
#           g_mock_bpt4 x=bpt4 flow on mock_sealed_s: 0.5125, rows = sealed_f0929_mock_bpt4
#   lane B  g_xbgate    xblocks_gate.py (defaults bit-identity vs HEAD~ solver,
#                       harness parity with icoh: p = 4395 > 4000, dual in both)
#           g_s3icoh    sealed_personal Scherer 3-class bpt4+icoh (p > 4000):
#                       rows = sealed_f0929p
#           g_abl       release_ablations base/xb/xa on mock_sealed_s:
#                       rows = sealed_f0929_abl_s_{A,B}
# then compare_results.py per pair -> logs/sealed_s1001/RESULTS_gates.md
#   REF=<pre-sprint riemann_sealed.py copy> wsl.exe bash -lc 'bash ~/codabench/scripts/sprint1001_g.sh'
TAG=s1001
export XS_THREADS=${XS_THREADS:-5}
source "$HOME/codabench/scripts/sealed_lib.sh"
: "${REF:?set REF to the pre-sprint riemann_sealed.py copy}"
DH=$HOME/neuralbench/benchopt_data
FULL=0,1,2,3,4,5,6,7,8,9
A="$HOME/codabench/analysis"
S="$HOME/codabench/scripts"
L="$HOME/codabench/logs"
fresh() { [ -f "$LOGDIR/$1.done" ] || rm -rf "$L/$2" "$L/sealed_$2"; }

laneA() {
  fresh g_zhou s1001_regress
  step g_zhou 5 30m env SUBMISSION_DIR="$L/s1001_regress/submission" RUN_NAME=s1001_regress \
    bash "$S/train_sealed.sh" "$DH" zhou2016_xsess
  fresh g_mock_def s1001_mock_default
  step g_mock_def 12 60m env DATASET="MockSealed[study=mock_sealed_s]" \
    SUBMISSION_DIR="$L/s1001_mock_default/submission" RUN_NAME=s1001_mock_default \
    bash "$S/train_sealed.sh" "$DH" mock_sealed_s
  fresh g_mock_bpt4 s1001_mock_bpt4
  step g_mock_bpt4 15 60m env DATASET="MockSealed[study=mock_sealed_s]" \
    RECIPE_SPEC=riemann:xd=1,fb=1,x=bpt4 SUBMISSION_DIR="$L/s1001_mock_bpt4/submission" \
    RUN_NAME=s1001_mock_bpt4 bash "$S/train_sealed.sh" "$DH" mock_sealed_s
}
laneB() {
  step g_xbgate 6 30m python "$A/xblocks_gate.py" --ref "$REF" --xblocks icoh
  [ -f "$LOGDIR/g_s3icoh.done" ] || rm -rf "$L/sealed_s1001p"
  step g_s3icoh 20 90m python "$A/sealed_personal.py" --tag s1001p --study scherer2015 \
    --classes 0,1,3 --family riemann:xd=1,fb=1,x=bpt4+icoh --align router-psd:riemann
  [ -f "$LOGDIR/g_abl.done" ] || rm -rf "$L"/sealed_s1001_abl_s*
  step g_abl 25 90m env TAG=s1001_abl_s STUDY=mock_sealed_s SPLIT=calib:3 TEST_SUBJECTS=$FULL \
    SPEC=riemann:xd=1,fb=1,x=bpt4 XB=icoh XS_THREADS=5 STEPS="base xb xa" \
    bash "$S/release_ablations.sh"
}
laneA &
laneB &
wait
{
  echo "# Sprint 2026-10-01 Phase 2 gates (dual LDA): $(date '+%F %T')"
  echo
  for r in s1001_regress s1001_mock_default s1001_mock_bpt4; do
    echo "- $r: $(grep -h 'gate:' "$L/$r/STATUS.md" 2>/dev/null | tail -1 | tr -d '|')"
  done
  echo "- xblocks_gate: $(grep -h '^GATE\|^ALL' "$LOGDIR/g_xbgate.log" | tr '\n' ';')"
  for p in "sealed_regress_f0929 sealed_s1001_regress zhou2016_xsess" \
           "sealed_f0929_mock_default sealed_s1001_mock_default mock_sealed_s" \
           "sealed_f0929_mock_bpt4 sealed_s1001_mock_bpt4 mock_sealed_s" \
           "sealed_f0929p sealed_s1001p scherer2015" \
           "sealed_f0929_abl_s_A,sealed_f0929_abl_s_B sealed_s1001_abl_s_A,sealed_s1001_abl_s_B mock_sealed_s"; do
    set -- $p
    echo; echo "## $2 vs $1 ($3)"; echo
    python "$A/compare_results.py" "$(echo "$1" | sed "s|\([^,]*\)|$L/\1|g")" \
      "$(echo "$2" | sed "s|\([^,]*\)|$L/\1|g")" "$3"
  done
} > "$LOGDIR/RESULTS_gates.md" 2>&1
echo "| $(date +%T) | g gates finished: $(grep -c 'PASS' "$LOGDIR/RESULTS_gates.md") PASS, $(grep -c 'FAIL' "$LOGDIR/RESULTS_gates.md") FAIL lines |" >> "$STATUS"
