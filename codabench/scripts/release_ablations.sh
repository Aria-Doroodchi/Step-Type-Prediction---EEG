#!/usr/bin/env bash
# Release-day ablations (SEALED_RECIPE.md section 3) on the internal replica of
# the sealed split, as two resumable lanes (sprint 2026-09-28).
#   TAG=rel STUDY=graz2026 bash ~/codabench/scripts/release_ablations.sh
#   TAG=sprint0928_abl_s STUDY=mock_sealed_s XS_THREADS=4 STEPS="base ch_all" LANES=A \
#       bash ~/codabench/scripts/release_ablations.sh      # one <= 10-min chunk
# Split SPLIT (default on the mock only: replica:3, the fully labelled
# participants train on all their sessions, the evaluation participants
# calibrate on sessions 0..2 and are scored on 3..5; any other cache, i.e. meta
# without a "mock" key, needs SPLIT set, and replica:K is refused there, as in
# train_sealed.sh and the harness, 2026-09-29); cell metric over subject x
# session x context cells; the harness router's fallback capped like the
# solver's (CAP=0.5).
# Which split judges what (review 2026-09-28): the release-day DECISIONS are
# taken on SPLIT=calib:3 TEST_SUBJECTS=<the fully labelled participants>, i.e.
# their sessions 3..5 scored after training on everyone's sessions 0..2 (the
# evaluation participants' hidden sessions never train: xsess_lib drops them,
# and the preflight below stops if one would). On the released data that is the
# only valid replica (replica:3 would score the hidden sessions' placeholder
# labels and is refused). On the mock, rehearse the decision rules (machinery
# rules b, c) on that same calib:3 split: under replica:3 the training set holds
# the full participants' decoupled sessions 3..5, which masks the drifting-EMG
# trap. replica:3 on the mock is the organisers' own split, kept for the solver
# gates (iii) / (iv) in train_sealed.sh.
# A TAG's settings (+ the cache build) are kept in logs/sealed_<TAG>/config.txt;
# a rerun of that TAG with other settings, or after the cache was rebuilt, stops.
# The recipe row "base" = sealed_personal blend_calib (router ids at test time):
# family SPEC, alignment ALIGN, chans eeg, pool all, blend-weight CV WCV. Every
# other step changes one thing:
#   lane A (sealed_personal, --wvariant calib):
#     base                           the recipe
#     ch_eeg_eog / ch_eeg_emg / ch_all   --chans eeg+eog | eeg+emg | all (the
#                                    EMG/EOG trap: keep only if it holds on the
#                                    later sessions)
#     al_ctx                         --align router-psdctx:<kind>: whitening and
#                                    routing per (subject, context); only for a
#                                    cache with a context column
#   lane B:
#     xb / xa                        extra feature blocks (sprint 2026-09-29):
#                                    xb = the recipe without its x=<blocks> (when
#                                    SPEC has them), xa = with the blocks of env
#                                    XB added (when XB is set, e.g. XB=icoh)
#     xd0                            the family without xDAWN (xd=1 -> xd=0)
#     wcv_loso (wcv_last)            the other blend-weight CV (6 folds on the
#                                    sealed structure: the slowest step)
#     pool_test                      --pool test: train only on the evaluation
#                                    participants' calibration data
#     online                         --align online-64:<kind>, RULE-DEPENDENT
#                                    (unlabelled test windows): reported apart
#     run                            sealed_run: meanlr + SPEC, pooled and
#                                    per-subject, aligns none / ALIGN (/ psdctx)
# Each lane writes its own harness results (logs/sealed_<TAG>_A/, _B/: never two
# processes on one file); step logs, .done markers and STATUS.md are in
# logs/sealed_<TAG>/. The lanes run side by side, except on a cache over 2 GiB
# (500 Hz: one after the other, X memory-mapped). When every step is done,
# RESULTS.md there holds
#   python ~/codabench/analysis/release_summarize.py --tag <TAG> --study <STUDY>
# (the RESULTS table and the pre-registered release-day DECISIONS).
# Env: TAG, STUDY (required); SPLIT, TEST_SUBJECTS (comma list, harness
# --test_subjects; on the released data the replica is SPLIT=calib:3
# TEST_SUBJECTS=<the fully labelled participants>), SPEC (riemann:xd=1,fb=1), ALIGN
# (router-psd:riemann), WCV (last), CAP (0.5; "none" = uncapped), XB (extra blocks
# that step xa adds to SPEC, e.g. icoh; sprint 2026-09-29), LANES (AB),
# STEPS (space-separated step names: run only those, e.g. to keep each call
# under a time limit; a name that is no step here, or a requested step, i.e. of
# STEPS or else of LANES, without its .done marker at the end, is an ERROR row
# and rc 1, 2026-09-29), XS_THREADS (10 per lane). ETAs below are minutes for the
# full-size sealed data at 10 threads (mock_sealed_s at 4 threads: ~1/3).
: "${TAG:?TAG}" "${STUDY:?STUDY}"
export XS_THREADS=${XS_THREADS:-10}
source "$HOME/codabench/scripts/sealed_lib.sh"
# XSESS_CACHE_ROOT: another cache root (opt-in, as analysis/xsess_cache.py)
CACHE="${XSESS_CACHE_ROOT:-$HOME/neuralbench/xsess_cache}/$STUDY"
# replica:K is the mock's split only (review 2026-09-28, V2): a cache whose meta
# has no "mock" key needs an explicit SPLIT, and replica:K is refused there (the
# harness refuses it too)
read -r IS_MOCK FULL < <(python - "$CACHE" <<'PY'
import json, os, sys
m = json.load(open(os.path.join(sys.argv[1], "meta.json")))
print("mock" if "mock" in m else "real", ",".join(map(str, m.get("full_subjects") or [])) or "-")
PY
) || { echo "| $(date +%T) | ERROR: no readable cache at $CACHE |" >> "$STATUS"; exit 1; }
if [ -z "${SPLIT:-}" ]; then
  if [ "$IS_MOCK" = mock ]; then
    SPLIT=replica:3
  else
    echo "| $(date +%T) | ERROR: $STUDY is not a mock cache (meta has no \"mock\" key): set SPLIT" \
         "explicitly, on the released data SPLIT=calib:3 TEST_SUBJECTS=<the fully labelled" \
         "participants> (meta full_subjects: $FULL) |" >> "$STATUS"
    exit 1
  fi
fi
if [[ $SPLIT == replica:* ]] && [ "$IS_MOCK" != mock ]; then
  echo "| $(date +%T) | ERROR: SPLIT=$SPLIT runs only on a mock cache ($STUDY's meta has no" \
       "\"mock\" key): on the released data use SPLIT=calib:${SPLIT#replica:}" \
       "TEST_SUBJECTS=<the fully labelled participants> (meta full_subjects: $FULL) |" >> "$STATUS"
  exit 1
fi
TEST_SUBJECTS=${TEST_SUBJECTS:-}
SPEC=${SPEC:-riemann:xd=1,fb=1}
ALIGN=${ALIGN:-router-psd:riemann}
WCV=${WCV:-last}
CAP=${CAP:-0.5}
KIND=${ALIGN##*:}
if [ "$WCV" = last ]; then WCV_ALT=loso; else WCV_ALT=last; fi
case $SPEC in *xd=1*) SPEC_NX=${SPEC/xd=1/xd=0};; *) SPEC_NX=;; esac
# extra feature blocks (sprint 2026-09-29; release_summarize.py applies the
# rules): step xb = the recipe WITHOUT its x=<blocks> (only when SPEC has
# them); step xa = the recipe WITH the blocks $XB added (only when XB is set,
# e.g. XB=icoh: x=bpt4 -> x=bpt4+icoh, or x=icoh on a SPEC without blocks)
case $SPEC in
  *,x=*) SPEC_XB=$(echo "$SPEC" | sed 's/,x=[^,]*//')
         SPEC_XA=${XB:+$(echo "$SPEC" | sed "s/\(,x=[^,]*\)/\1+$XB/")};;
  *) SPEC_XB=; SPEC_XA=${XB:+$SPEC,x=$XB};;
esac
HAS_CTX=$([ -f "$CACHE/context.npy" ] && echo 1)
# the steps of each lane on this cache / SPEC / WCV; LANES and STEPS must name
# them (a typo used to end like a finished chunk, and the summary then silently
# applied the recipe default for the step that never ran)
STEPS_A="base ch_eeg_eog ch_eeg_emg ch_all${HAS_CTX:+ al_ctx}"
STEPS_B="${SPEC_NX:+xd0 }${SPEC_XB:+xb }${SPEC_XA:+xa }wcv_$WCV_ALT pool_test online run"
LANES=${LANES:-AB}
if ! [[ $LANES =~ ^[AB]{1,2}$ ]]; then
  echo "| $(date +%T) | ERROR: LANES=$LANES: expected A, B or AB |" >> "$STATUS"; exit 1
fi
for s in ${STEPS:-}; do
  case " $STEPS_A $STEPS_B " in *" $s "*) ;; *)
    echo "| $(date +%T) | ERROR: STEPS names $s, not a step here (lane A: $STEPS_A;" \
         "lane B: $STEPS_B) |" >> "$STATUS"; exit 1;;
  esac
done
COMMON=(--study "$STUDY" --split "$SPLIT")
[ -n "$TEST_SUBJECTS" ] && COMMON+=(--test_subjects "$TEST_SUBJECTS")
[ "$CAP" = none ] || COMMON+=(--router_cap "$CAP")
# 500 Hz caches (> 2 GiB): memory-map X (same values, not in the config key),
# and the two lanes run one after the other (one 500 Hz sealed_run config alone
# peaked at 16.4 GiB of the 31 GB)
BIG=
if [ "$(stat -c %s "$CACHE/X.npy" 2>/dev/null || echo 0)" -gt $((2 * 1024 ** 3)) ]; then
  BIG=1; COMMON+=(--mmap)
fi
P="$HOME/codabench/analysis/sealed_personal.py"
echo "| $(date +%T) | config STUDY=$STUDY SPLIT=$SPLIT TEST_SUBJECTS=${TEST_SUBJECTS:-default}" \
     "SPEC=$SPEC ALIGN=$ALIGN WCV=$WCV" \
     "CAP=$CAP context=${HAS_CTX:-no} LANES=${LANES:-AB} STEPS=${STEPS:-all}" \
     "threads=$XS_THREADS |" >> "$STATUS"

# settings guard: the .done markers of logs/sealed_<TAG>/ belong to one set of
# settings and one build of the cache (meta "built" + X.npy size); a folder from
# before 2026-09-28 has no config.txt and adopts the current settings
FP=$(python - "$CACHE" <<'PY'
import json, os, sys
m = json.load(open(os.path.join(sys.argv[1], "meta.json")))
print(f"built={m.get('built', '?')} X.npy={os.path.getsize(os.path.join(sys.argv[1], 'X.npy'))}")
PY
) || { echo "| $(date +%T) | ERROR: no readable cache at $CACHE |" >> "$STATUS"; exit 1; }
CONF="study=$STUDY split=$SPLIT test_subjects=${TEST_SUBJECTS:-default} spec=$SPEC"
CONF="$CONF align=$ALIGN wcv=$WCV cap=$CAP cache=$FP"
[ -n "${XB:-}" ] && CONF="$CONF xb=$XB"     # (only when set: older config.txt lines stay equal)
if [ -f "$LOGDIR/config.txt" ] && [ "$(cat "$LOGDIR/config.txt")" != "$CONF" ]; then
  echo "| $(date +%T) | ERROR: settings or cache build differ from the ones logs/sealed_$TAG" \
       "ran with ($(cat "$LOGDIR/config.txt")); use another TAG, or move logs/sealed_$TAG" \
       "and its lane folders logs/sealed_${TAG}_A, _B away |" >> "$STATUS"
  exit 1
fi
echo "$CONF" > "$LOGDIR/config.txt"
# preflight: the [data] line of the split; stops if a hidden-test row would train
if ! timeout 30m python "$RUN" --tag "${TAG}_A" "${COMMON[@]}" --chans eeg --models meanlr \
     --check > "$LOGDIR/preflight.log" 2>&1; then
  echo "| $(date +%T) | ERROR: preflight failed (hidden-test rows in train, or a bad split):" \
       "see $LOGDIR/preflight.log |" >> "$STATUS"
  exit 1
fi
echo "| $(date +%T) | preflight: $(grep -m1 -o 'X=.*' "$LOGDIR/preflight.log") |" >> "$STATUS"

rstep() {   # sealed_lib step, unless STEPS names other steps
  if [ -n "${STEPS:-}" ] && [[ " $STEPS " != *" $1 "* ]]; then return 0; fi
  step "$@"
}
pers() {    # pers <lane> <step> <eta_min> <sealed_personal args...>
  local lane=$1 name=$2 eta=$3; shift 3
  rstep "$name" "$eta" 240m python "$P" --tag "${TAG}_$lane" "${COMMON[@]}" --wvariant calib "$@"
}

laneA() {
  pers A base 20 --family "$SPEC" --align "$ALIGN" --wcv "$WCV" --chans eeg
  local c
  for c in eeg+eog eeg+emg all; do
    pers A "ch_${c//+/_}" 20 --family "$SPEC" --align "$ALIGN" --wcv "$WCV" --chans "$c"
  done
  if [ -n "$HAS_CTX" ]; then
    pers A al_ctx 20 --family "$SPEC" --align "router-psdctx:$KIND" --wcv "$WCV" --chans eeg
  fi
}
laneB() {
  if [ -n "$SPEC_XB" ]; then
    pers B xb 20 --family "$SPEC_XB" --align "$ALIGN" --wcv "$WCV" --chans eeg
  fi
  if [ -n "$SPEC_XA" ]; then
    pers B xa 25 --family "$SPEC_XA" --align "$ALIGN" --wcv "$WCV" --chans eeg
  fi
  if [ -n "$SPEC_NX" ]; then
    pers B xd0 15 --family "$SPEC_NX" --align "$ALIGN" --wcv "$WCV" --chans eeg
  fi
  pers B "wcv_$WCV_ALT" 60 --family "$SPEC" --align "$ALIGN" --wcv "$WCV_ALT" --chans eeg
  pers B pool_test 10 --family "$SPEC" --align "$ALIGN" --wcv "$WCV" --chans eeg --pool test
  pers B online 20 --family "$SPEC" --align "online-64:$KIND" --wcv "$WCV" --chans eeg
  rstep run 30 240m python "$RUN" --tag "${TAG}_B" "${COMMON[@]}" --chans eeg \
      --models meanlr "$SPEC" --modes pooled persubject \
      --aligns none "$ALIGN" ${HAS_CTX:+"router-psdctx:$KIND"}
}
[ -n "$HAS_CTX" ] || echo "| $(date +%T) | al_ctx skipped: $STUDY has no context column |" >> "$STATUS"
[ -n "$SPEC_NX" ] || echo "| $(date +%T) | xd0 skipped: SPEC=$SPEC has no xd=1 |" >> "$STATUS"
if [ -n "$BIG" ]; then
  echo "| $(date +%T) | large cache: lanes $LANES run one after the other |" >> "$STATUS"
  [[ $LANES == *A* ]] && laneA
  [[ $LANES == *B* ]] && laneB
else
  [[ $LANES == *A* ]] && { laneA & }
  [[ $LANES == *B* ]] && { laneB & }
fi
wait
# every requested step (STEPS, else every step of LANES) must be done: a step
# that crashed (sealed_lib's step returns 0) or never ran is an ERROR, rc 1
REQ=${STEPS:-$([[ $LANES == *A* ]] && echo "$STEPS_A") $([[ $LANES == *B* ]] && echo "$STEPS_B")}
for s in $REQ; do
  [ -f "$LOGDIR/$s.done" ] || {
    echo "| $(date +%T) | ERROR: step $s requested but not done (see $LOGDIR/$s.log) |" >> "$STATUS"
    exit 1; }
done
for s in $STEPS_A $STEPS_B; do
  [ -f "$LOGDIR/$s.done" ] || { echo "| $(date +%T) | lanes $LANES finished; $s not done |" >> "$STATUS"; exit 0; }
done
python "$HOME/codabench/analysis/release_summarize.py" --tag "$TAG" --study "$STUDY" \
    --split "$SPLIT" --test_subjects "$TEST_SUBJECTS" --spec "$SPEC" --align "$ALIGN" \
    --wcv "$WCV" --cap "$CAP" --xb "${XB:-}" > "$LOGDIR/RESULTS.md" 2>&1
echo "| $(date +%T) | ALL DONE (RESULTS.md written) |" >> "$STATUS"
