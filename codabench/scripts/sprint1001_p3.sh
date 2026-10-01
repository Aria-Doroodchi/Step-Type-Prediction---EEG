#!/usr/bin/env bash
# Sprint 2026-10-01, Phase 3: strict per-fold whitening references in the
# blend-weight search (solver wcv_ref / harness --wref / train_sealed.sh WREF).
# Brief prompts/2026-10-01_dual_lda_strict_wcv.md § 5 Phase 3 (decision D3).
#   lane A (gates, the RELEASE_DAY § 7 step 1 replica flow on mock_sealed_s,
#           BLEND_W=auto, WCV=loso):
#     p3_rep_all     WREF=all, router-psdctx: must reproduce the committed
#                    replica_mock_sealed_s (train 0.602778, w 0.75, MATCH; rows =
#                    sealed_replica_mock_sealed_s)
#     p3_rep_strict  WREF=strict, router-psdctx: solver vs harness MATCH
#     p3_rep_spsd    WREF=strict, router-psd: MATCH (subject references)
#   lane B (zhou2016_xsess, 3 sessions / subject, then information rows):
#     p3_zh_all / p3_zh_strict   train_sealed BLEND_W=auto (WCV=loso): MATCH each
#     p3_i_*         sealed_personal all vs strict on zhou2016 (loso),
#                    scherer2015 3-class and tangermann2012 (last = halves),
#                    recipe x=bpt4, release settings: information only
#   wsl.exe bash -lc 'bash ~/codabench/scripts/sprint1001_p3.sh'
TAG=s1001
export XS_THREADS=${XS_THREADS:-5}
source "$HOME/codabench/scripts/sealed_lib.sh"
DH=$HOME/neuralbench/benchopt_data
FULL=0,1,2,3,4,5,6,7,8,9
L="$HOME/codabench/logs"
S="$HOME/codabench/scripts"
P="$HOME/codabench/analysis/sealed_personal.py"
REP="SPLIT=calib:3 TEST_SUBJECTS=$FULL BLEND_W=auto GATE=replica CHANS=eeg RECIPE_SPEC=riemann:xd=1,fb=1 WCV=loso"
DSR="../datasets/mock_sealed.py[study=mock_sealed_s,split=replica_full]"
fresh() { [ -f "$LOGDIR/$1.done" ] || rm -rf "$L/$2" "$L/sealed_$2"; }
SPEC=riemann:xd=1,fb=1,x=bpt4
REL=(--router_cap 0.5 --wvariant calib --chans eeg)

laneA() {
  fresh p3_rep_all s1001_rep_all
  step p3_rep_all 15 90m env $REP RECIPE_ALIGN=router-psdctx:riemann WREF=all \
    RUN_NAME=s1001_rep_all DATASET="$DSR" bash "$S/train_sealed.sh" "$DH" mock_sealed_s
  fresh p3_rep_strict s1001_rep_strict
  step p3_rep_strict 20 90m env $REP RECIPE_ALIGN=router-psdctx:riemann WREF=strict \
    RUN_NAME=s1001_rep_strict DATASET="$DSR" bash "$S/train_sealed.sh" "$DH" mock_sealed_s
  fresh p3_rep_spsd s1001_rep_spsd
  step p3_rep_spsd 20 90m env $REP RECIPE_ALIGN=router-psd:riemann WREF=strict \
    RUN_NAME=s1001_rep_spsd DATASET="$DSR" bash "$S/train_sealed.sh" "$DH" mock_sealed_s
}
laneB() {
  for w in all strict; do
    fresh p3_zh_$w s1001_zh_$w
    step p3_zh_$w 6 30m env BLEND_W=auto WREF=$w RUN_NAME=s1001_zh_$w \
      SUBMISSION_DIR="$L/s1001_zh_$w/submission" bash "$S/train_sealed.sh" "$DH" zhou2016_xsess
  done
  for w in all strict; do
    step p3_i_zh_$w 5 30m python "$P" --tag s1001i --study zhou2016 --family "$SPEC" \
      --align router-psd:riemann --wcv loso --wref $w "${REL[@]}"
  done
  for w in all strict; do
    step p3_i_s3_$w 12 60m python "$P" --tag s1001i --study scherer2015 --classes 0,1,3 \
      --family "$SPEC" --align router-psd:riemann --wcv last --wref $w "${REL[@]}"
  done
  for w in all strict; do
    step p3_i_tg_$w 15 60m python "$P" --tag s1001i --study tangermann2012 --family "$SPEC" \
      --align router-psd:riemann --wcv last --wref $w "${REL[@]}"
  done
}
laneA &
laneB &
wait
{
  echo "# Sprint 2026-10-01 Phase 3: strict LOSO references, $(date '+%F %T')"
  echo
  for r in replica_mock_sealed_s s1001_rep_all s1001_rep_strict s1001_rep_spsd s1001_zh_all s1001_zh_strict; do
    echo "## $r"
    grep -h "blend weight:\|gate:" "$L/$r/STATUS.md" 2>/dev/null | tr -d '|' | sed 's/^/- /'
    grep -h "blend_w=auto ->" "$L/$r/train.log" 2>/dev/null | sed 's/^/- solver: /'
    grep -h "blend_calib: w_pooled" "$L/$r"/personal_*.log 2>/dev/null | sed 's/^/- harness: /'
    grep -h "wcv=loso.*fold" "$L/$r"/personal_*.log 2>/dev/null | sed 's/^/- harness fold: /'
    echo
  done
  echo "## harness rows: s1001_rep_all vs replica_mock_sealed_s"
  python "$HOME/codabench/analysis/compare_results.py" "$L/sealed_replica_mock_sealed_s" \
    "$L/sealed_s1001_rep_all" mock_sealed_s
  echo
  echo "## information: all vs strict (sealed_personal, $SPEC, router-psd, release settings)"
  echo
  python - "$L/sealed_s1001i" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
print("| study | wcv | wref | blend_calib w | CV per w | test cell (router-id) |\n|---|---|---|---|---|---|")
for f in sorted(d.glob("results_*.jsonl")):
    for line in f.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["spec"].endswith("/blend_calib") and r["mode"] == "router-id":
            print(f"| {r['study']} | {r.get('wcv')} | {r.get('wref', 'all')} | {r.get('blend_calib_w')} "
                  f"| {r.get('blend_calib_cv')} | {r['cell']:.4f} |")
PY
} > "$LOGDIR/RESULTS_p3.md" 2>&1
echo "| $(date +%T) | p3 lane finished |" >> "$STATUS"
