#!/usr/bin/env bash
# Train the sealed-phase recipe end to end on a released study (SKELETON,
# 2026-09-26; replica split, self-chosen weight and mock support 2026-09-28;
# the recipe itself is in codabench/SEALED_RECIPE.md).
#
#   bash ~/codabench/scripts/train_sealed.sh <data_home> <study> [<modality/task> [overlay]]
#   e.g. bash ~/codabench/scripts/train_sealed.sh ~/neuralbench/benchopt_data \
#            graz2026 eeg/motor_imagery graz2026
#   dress rehearsal on the mock (the cache must exist: analysis/mock_sealed.py):
#        DATASET="MockSealed[study=mock_sealed_s]" bash ~/codabench/scripts/train_sealed.sh \
#            ~/neuralbench/benchopt_data mock_sealed_s
#
# <data_home>  benchopt data home holding neural_compet/<Study>/ (the study must
#              already be downloaded: `benchopt prepare` or neuralfetch)
# <study>      cache key (also the bci_studies _OVERLAYS key for benchopt)
# <task>       NeuralBench task whose windows/targets define the data (needed
#              only while ~/neuralbench/xsess_cache/<study>/ does not exist)
# [overlay]    dataset overlay in that task's datasets/ folder
#
# Environment. The defaults reproduce the 2026-09-25 proxy runs (SPLIT=last).
# Any other split is a release run and gets the defaults marked (release):
#   ADAPT      none | online (rule-dependent test-time re-centring)
#   BLEND_W    harness (default: the weight sealed_personal chose) | auto (bake
#              blend_w="auto": the solver chooses w in fit by leave-one-
#              calibration-session-out) | <number> (bake that weight)
#   SPLIT      harness split: last (default on the proxies) | replica:K (the
#              default on the mock: its meta names eval_subjects; K =
#              calib_sessions) | calib:K (see xsess_lib.xsess_split). On the
#              released data the internal replica is SPLIT=calib:3
#              TEST_SUBJECTS=<the fully labelled participants' indices> (their
#              sessions 3..5 are scored; the hidden-test rows never train). A
#              cache that looks sealed (context column, eval_subjects, > 3
#              sessions) with SPLIT unset stops here and says so. replica:K on
#              a cache whose meta has no "mock" key stops too (the harness
#              refuses it as well)
#   TEST_SUBJECTS  comma list of subject indices tested under SPLIT (harness
#              --test_subjects; default: the split's own)
#   WCV        harness blend-weight CV: last | loso (default when BLEND_W=auto,
#              the rule of the solver's "auto", so the two weights are comparable)
#   CHANS      eeg (default) | eeg+eog | eeg+emg | all: harness and candidate
#   RECIPE_SPEC   harness family (riemann:xd=1,fb=1); xd / fb are baked into the
#              candidate as use_xdawn / filterbank
#   RECIPE_ALIGN  harness alignment (router-psd:riemann; online-64:riemann under
#              ADAPT=online); its kind (and online buffer) is baked in.
#              router-psdctx:<kind> (the release-day context rule) bakes
#              align="subject_context" (router + whitening per (subject, context)
#              pair); not with ADAPT=online (the solver has no per-pair online mode)
#   ROUTER_CAP harness router fallback cap: none | 0.5 (release, the solver's
#              min(thr, 0.5))
#   WVARIANT   sealed_personal --wvariant: both | calib (release; same weight,
#              without the per-subject full models)
#   DATASET    benchopt -d selector: BCI[study=<study>]; a mock cache (meta has
#              "mock") -> ../datasets/mock_sealed.py[study=<study>]. The name form
#              MockSealed[...] is rewritten to that path (benchopt 1.10 finds the
#              mock dataset only by path). Extra parameters pass through as
#              given, e.g. the release-day replica on the mock (RELEASE_DAY
#              section 7 step 1; test = the full participants' sessions 3..5):
#              DATASET="../datasets/mock_sealed.py[study=<study>,split=replica_full]"
#              with SPLIT=calib:3 TEST_SUBJECTS=<meta full_subjects>
#   GATE       which step-5 gates block the zip (none is in config.txt):
#              replica (default when SPLIT != last: benchopt's test set is the
#              harness's) -> a failed replay, train != replay, a harness gap
#              > 2 points (FLAG) and, with BLEND_W=auto and WCV=loso, a solver
#              vs harness weight MISMATCH all block it | final (default when
#              SPLIT=last, i.e. as before; for the release-day all-data run,
#              whose test set differs from the harness's) -> only a failed
#              replay or train != replay block it; the gap / weight lines are
#              recorded, informational
#   FORCE_ZIP  1: zip despite a weight MISMATCH under GATE=replica (a loud
#              WARNING goes into STATUS.md and RESULTS.md); never overrides a
#              failed replay, train != replay or a gap FLAG
#   HARNESS_FROM <tag>: seed this run's harness folder logs/sealed_<HTAG> from
#              logs/sealed_<tag> (results_*.jsonl + probs/) before the harness
#              steps, so every setting already computed there is skipped (the
#              steps still run: sealed_personal refits its router, then skips).
#              E.g. the release-day final run: RUN_NAME=final_<S>
#              HARNESS_FROM=train_sealed_<S>. Only into a harness folder that has
#              no results yet, and only from a folder with this cache's
#              fingerprint (stale rows are refused), and only from a run whose
#              gate passed: logs/<tag>/STATUS.md, when it exists, must hold ALL
#              DONE and no NO ZIP / STOPPED row (HARNESS_FROM_ANY=1: seed anyway)
#   SUBMISSION_DIR  absolute folder training writes the submission to: the
#              track's outputs/Riemann-Sealed-Cand<suffix>/ (as before: SPLIT=last
#              without RUN_NAME), or <logdir>/submission (release splits, and any
#              run with RUN_NAME, since 2026-09-29), so no outputs/ folder is
#              overwritten. The resolved folder is part of config.txt, and
#              train.hash (its path + the sha256 of its files, written when
#              train succeeds) is checked before the replay and the zip: a folder
#              changed since training stops with an ERROR and no zip
#   STOP_AFTER <step name>: stop after that step (e.g. personal_none, to read
#              the harness numbers before training); rerun without it to go on.
#              A name that is no step of this run is an ERROR
#   RUN_NAME   log folder / harness tag / benchopt output name instead of
#              train_sealed_<study><suffix> (e.g. a regression run that must not
#              touch a committed log folder); its submission goes to
#              <logdir>/submission unless SUBMISSION_DIR says otherwise
#   XS_THREADS threads per job (default 10)
#
# Steps (each resumable via <step>.done in logs/train_sealed_<study><suffix>/,
# suffix _online and/or _auto; the settings are kept in config.txt there, and a
# rerun with other settings stops instead of reusing the old steps). Harness
# rows go to logs/sealed_train_sealed_<study>[_online]/, shared by the BLEND_W
# modes (every row is keyed by its settings); do not run two modes of one study
# at the same time. Both folders hold cache_fingerprint.txt: a rerun after the
# cache was rebuilt stops (stale rows). A failed step stops the script; a
# TERM / INT to it also kills the running step (<step>.pid).
#   1. cache: every window + subject/session/run(/context) metadata (skipped
#      when the cache exists, e.g. the mock)
#   2. EDA   : TODO (parametrise analysis/dreyer_eda.py by study)
#   3. preflight (sealed_run --check: [data] line; stops if a hidden-test row
#      would train), then validate the recipe in the harness (cell metric on
#      SPLIT) and choose the blend weight on training data (sealed_personal, WCV)
#   4. train Riemann-Sealed with benchopt (the settings baked into a candidate
#      copy as defaults); the solver's own weight is logged next to the harness's
#   5. platform replay from a read-only copy (score must match step 4), gate
#      check vs the harness in RESULTS.md, zip into the log dir only if the
#      GATE's gates pass (a failed replay stops the script; the parquets are
#      the ones benchopt reported saving in this run's train.log / replay.log;
#      a blocked run writes no zip, renames older zips of the folder to
#      *.zip.blocked and exits 1)
# Never uploads (the user's action) and never writes codabench/submissions/.
set -u
DATA_HOME=${1:?data_home}; STUDY=${2:?study}; TASK=${3:-}; OVERLAY=${4:-}
ADAPT=${ADAPT:-none}   # "online" only once the organisers allow test-time statistics
BLEND_W=${BLEND_W:-harness}
SUFFIX=$([ "$ADAPT" = online ] && echo _online)
HTAG=${RUN_NAME:-train_sealed_$STUDY$SUFFIX}   # harness tag, shared by the BLEND_W modes
[ "$BLEND_W" = auto ] && SUFFIX=${SUFFIX}_auto
TAG=${RUN_NAME:-train_sealed_$STUDY$SUFFIX}
source "$HOME/codabench/env.sh" >/dev/null
export BENCHOPT_DATA_HOME="$DATA_HOME" PYTHONUTF8=1
export XS_THREADS=${XS_THREADS:-10}
export OMP_NUM_THREADS=$XS_THREADS OPENBLAS_NUM_THREADS=$XS_THREADS MKL_NUM_THREADS=$XS_THREADS
LOGDIR="$HOME/codabench/logs/$TAG"; mkdir -p "$LOGDIR"
STATUS="$LOGDIR/STATUS.md"; [ -f "$STATUS" ] || printf '| time | event |\n|---|---|\n' > "$STATUS"
A="$HOME/codabench/analysis"
# XSESS_CACHE_ROOT: another cache root (opt-in, as analysis/xsess_cache.py)
CACHE="${XSESS_CACHE_ROOT:-$HOME/neuralbench/xsess_cache}/$STUDY"

note() { echo "| $(date +%T) | $* |" >> "$STATUS"; }

# STOP_AFTER must name a step of this run: a typo used to run the whole pipeline
if [ -n "${STOP_AFTER:-}" ]; then
  STEP_NAMES="cache preflight validate personal_$ADAPT train replay"; ok=
  for s in $STEP_NAMES; do [ "$STOP_AFTER" = "$s" ] && ok=1; done
  [ -n "$ok" ] || { note "ERROR: STOP_AFTER=$STOP_AFTER names no step of this run ($STEP_NAMES)"; exit 1; }
fi

# Each step runs in the background with its pid in <step>.pid (as sealed_lib.sh),
# so a TERM / INT to this script (watchdog, outer timeout) also stops the running
# step instead of leaving it appending harness rows with no .done marker.
CHILD=
trap 'note "stopped by a signal (step pid ${CHILD:-none} killed)"
      [ -n "${CHILD:-}" ] && kill "$CHILD" 2>/dev/null; exit 143' TERM INT

step() {   # step <name> <timeout> <cmd...>
  local name=$1 to=$2 rc=0; shift 2
  if [ -f "$LOGDIR/$name.done" ]; then
    echo "| $(date +%T) | skip $name |" >> "$STATUS"
  else
    echo "| $(date +%T) | START $name |" >> "$STATUS"
    timeout "$to" "$@" > "$LOGDIR/$name.log" 2>&1 &
    CHILD=$!; echo "$CHILD" > "$LOGDIR/$name.pid"
    wait "$CHILD"; rc=$?; CHILD=
    echo "| $(date +%T) | END $name rc=$rc |" >> "$STATUS"
    if [ $rc -eq 0 ]; then
      # the submission this training wrote (checked before the replay and the zip)
      [ "$name" = train ] && out_hash > "$LOGDIR/train.hash"
      touch "$LOGDIR/$name.done"
    fi
  fi
  # STOP_AFTER stops here whatever the step's rc (a failed step never goes on)
  if [ "${STOP_AFTER:-}" = "$name" ]; then
    note "stopped after $name (STOP_AFTER, rc=$rc)"; exit $rc
  fi
  # every caller stops on a failed step: say so (monitor_loop.sh's stop cue)
  [ $rc -eq 0 ] || note "STOPPED: step $name failed (rc=$rc), see $LOGDIR/$name.log"
  return $rc
}

# 1. cache (check the [data]/shape line in cache.log before believing anything).
#    An existing cache (the mock, or a rerun) is used as it is.
if [ ! -f "$LOGDIR/cache.done" ] && [ -f "$CACHE/meta.json" ]; then
  note "cache exists: $CACHE (not rebuilt)"; touch "$LOGDIR/cache.done"
fi
if [ ! -f "$LOGDIR/cache.done" ] && [ -z "$TASK" ]; then
  note "ERROR: no cache at $CACHE and no <modality/task> to build it"; exit 1
fi
step cache 60m python "$A/xsess_cache.py" "$STUDY" --task "$TASK" ${OVERLAY:+--overlay "$OVERLAY"} || exit 1

# release structure? (meta names the evaluation participants: sealed data / mock)
# "sealed" = the cache looks like the sealed release: meta hidden_split /
# eval_subjects / context_column, a context.npy, or a subject with > 3 sessions
read -r CALIB_K IS_MOCK SEALED FULL < <(python - "$CACHE" <<'PY'
import json, os, sys
import numpy as np
d = sys.argv[1]
m = json.load(open(os.path.join(d, "meta.json")))
subj, sess = np.load(os.path.join(d, "subj.npy")), np.load(os.path.join(d, "session.npy"))
more3 = max(len(np.unique(sess[subj == s])) for s in np.unique(subj)) > 3
sealed = bool(m.get("hidden_split") or m.get("eval_subjects") or m.get("context_column")
              or os.path.exists(os.path.join(d, "context.npy")) or more3)
print(m.get("calib_sessions", 3) if m.get("eval_subjects") else "-",
      "mock" if "mock" in m else "real", "sealed" if sealed else "proxy",
      ",".join(map(str, m.get("full_subjects") or [])) or "-")
PY
) || { note "ERROR: cannot read $CACHE/meta.json"; exit 1; }
if [ -z "${SPLIT:-}" ]; then
  if [ "$IS_MOCK" = mock ] && [ "$CALIB_K" != - ]; then
    SPLIT=replica:$CALIB_K     # the mock: the organisers' own split (gates iii / iv)
  elif [ "$SEALED" = sealed ]; then
    # the proxies' "last" would validate on the wrong split, and replica:K on the
    # released data would score the hidden sessions' placeholder labels
    note "ERROR: $STUDY looks like the sealed release; set SPLIT=calib:${CALIB_K/-/3}" \
         "TEST_SUBJECTS=<the fully labelled participants' indices> (meta full_subjects:" \
         "$FULL), or SPLIT=last explicitly"
    exit 1
  else
    SPLIT=last
  fi
fi
# replica:K is the organisers' split: only the mock has true labels there
if [[ $SPLIT == replica:* ]] && [ "$IS_MOCK" != mock ]; then
  note "ERROR: SPLIT=$SPLIT runs only on a mock cache ($STUDY's meta has no \"mock\" key):" \
       "on the released data use SPLIT=calib:${SPLIT#replica:} TEST_SUBJECTS=<the fully" \
       "labelled participants' indices> (meta full_subjects: $FULL)"
  exit 1
fi
# which gates block the zip (step 5); not a setting of the computation
if [ -z "${GATE:-}" ]; then
  if [ "$SPLIT" = last ]; then GATE=final; else GATE=replica; fi
fi
case $GATE in
  replica|final) ;;
  *) note "ERROR: GATE=$GATE: expected replica | final"; exit 1;;
esac
FORCE_ZIP=${FORCE_ZIP:-0}
TEST_SUBJECTS=${TEST_SUBJECTS:-}
if [ "$BLEND_W" = auto ]; then WCV=${WCV:-loso}; else WCV=${WCV:-last}; fi
CHANS=${CHANS:-eeg}
# release defaults on any split other than the proxies' "last"
if [ "$SPLIT" = last ]; then DEF_CAP=none DEF_WV=both; else DEF_CAP=0.5 DEF_WV=calib; fi
ROUTER_CAP=${ROUTER_CAP:-$DEF_CAP}; WVARIANT=${WVARIANT:-$DEF_WV}
if [ "$IS_MOCK" = mock ]; then DEF_DATASET="../datasets/mock_sealed.py[study=$STUDY]"
else DEF_DATASET="BCI[study=$STUDY]"; fi
DATASET=${DATASET:-$DEF_DATASET}
case $DATASET in MockSealed\[*) DATASET="../datasets/mock_sealed.py[${DATASET#MockSealed\[}";; esac

# 3. recipe validation, cross-session (SPLIT: last session per subject = test on
#    the proxies; the sealed replica on release caches).
#    RECIPE_* defaults = SEALED_RECIPE.md section 1; override from the env.
RECIPE_SPEC=${RECIPE_SPEC:-riemann:xd=1,fb=1}
# the blend weight must be chosen under the alignment the solver will use
# (Zhou: w=0.75 under the router, 0.5 under online; the wrong one cost 3 points)
if [ "$ADAPT" = online ]; then DEF_ALIGN=online-64:riemann; else DEF_ALIGN=router-psd:riemann; fi
RECIPE_ALIGN=${RECIPE_ALIGN:-$DEF_ALIGN}
# what the candidate can express: xd / fb of the family (harness defaults xd=1,
# fb=0), the reference kind and online buffer of the alignment, and its unit
# (subject, or (subject, context) pair: router-psdctx -> align="subject_context")
XD=1; FB=0; KIND=${RECIPE_ALIGN##*:}; HOW=${RECIPE_ALIGN%%:*}; BUF=64; ALIGN_S=subject
case $RECIPE_SPEC in
  riemann|riemann:*) ;;
  *) note "ERROR: RECIPE_SPEC=$RECIPE_SPEC: the candidate is Riemann-Sealed (riemann:xd=.,fb=.)"; exit 1;;
esac
for kv in $(echo "${RECIPE_SPEC#riemann}" | tr ':,' '  '); do
  case $kv in
    xd=*) XD=${kv#xd=};; fb=*) FB=${kv#fb=};;
    *) note "ERROR: RECIPE_SPEC option $kv has no Riemann-Sealed parameter"; exit 1;;
  esac
done
case $HOW in
  router-psd) ;;
  online-*) BUF=${HOW#online-};;
  router-psdctx)
    if [ "$ADAPT" = online ]; then
      note "ERROR: RECIPE_ALIGN=$RECIPE_ALIGN with ADAPT=online: Riemann-Sealed has no" \
           "per-(subject, context) online re-centring (align=subject_context raises)"; exit 1
    fi
    ALIGN_S=subject_context;;
  *-psdctx) note "ERROR: RECIPE_ALIGN=$RECIPE_ALIGN: of the context alignments the candidate" \
                 "expresses only router-psdctx:<kind> (align=subject_context)"; exit 1;;
  *) note "WARNING: RECIPE_ALIGN=$RECIPE_ALIGN is not what the solver does (router-psd / online-N)";;
esac
XDB=$([ "$XD" = 0 ] && echo False || echo True); FBB=$([ "$FB" = 1 ] && echo True || echo False)

# the submission folder training writes (OUT), resolved here: it is part of the
# settings, and train.hash ties its files to this run's training.
# RUN_NAME (a regression or rehearsal run) never writes the track's outputs/
# folder either (RELEASE_DAY section 9, V1: the section-0 regression would have
# overwritten outputs/Riemann-Sealed-Cand/); the default flow is unchanged
if [ -n "${SUBMISSION_DIR:-}" ] || [ "$SPLIT" != last ] || [ -n "${RUN_NAME:-}" ]; then
  OUT=${SUBMISSION_DIR:-$LOGDIR/submission}; SET_OUT=1
else
  OUT=tracks/bci_decoding/outputs/Riemann-Sealed-Cand$SUFFIX; SET_OUT=
fi
# (relative paths were, and are, relative to 2026-competition/, where benchopt runs)
case $OUT in /*) ;; *) OUT="$HOME/codabench/2026-competition/$OUT";; esac
out_hash() {   # "out=<OUT>" and the sha256 of every file the zip would hold
  echo "out=$OUT"
  if [ -d "$OUT" ]; then
    (cd "$OUT" && find . -type f ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z \
       | xargs -0r sha256sum)
  else
    echo "(no folder)"
  fi
}
check_out() {  # check_out <next action>: OUT still holds what this run's training wrote
  if [ "$(out_hash)" != "$(cat "$LOGDIR/train.hash" 2>/dev/null)" ]; then
    note "ERROR: submission folder changed since training ($OUT differs from" \
         "$LOGDIR/train.hash; checked before the $1): NO ZIP. Restore it, or retrain" \
         "(remove $LOGDIR/train.done)"
    exit 1
  fi
}

CONF="split=$SPLIT test_subjects=${TEST_SUBJECTS:-default} wcv=$WCV chans=$CHANS"
CONF="$CONF spec=$RECIPE_SPEC align=$RECIPE_ALIGN adapt=$ADAPT"
CONF="$CONF blend_w=$BLEND_W router_cap=$ROUTER_CAP wvariant=$WVARIANT dataset=$DATASET"
CONF="$CONF out=$OUT"
HDIR="$HOME/codabench/logs/sealed_$HTAG"
if [ -f "$LOGDIR/config.txt" ] && [ "$(cat "$LOGDIR/config.txt")" != "$CONF" ]; then
  if [ "$(cat "$LOGDIR/config.txt")" = "${CONF% out=*}" ]; then
    # a folder from before 2026-09-29 (no out= in its settings) adopts this OUT
    note "config.txt from before 2026-09-29 (no out=): adopts out=$OUT"
  else
    note "ERROR: settings differ from the ones $LOGDIR ran with" \
         "($(cat "$LOGDIR/config.txt")); move logs/$TAG and its harness rows logs/sealed_$HTAG" \
         "away, use another RUN_NAME, or rerun with those settings"
    exit 1
  fi
fi
# cache fingerprint (meta "built" + X.npy size): a cache rebuilt under the same
# study name must not reuse this run's steps or harness rows (sealed_run /
# sealed_personal skip every config key they already hold). Folders from before
# 2026-09-28 have no fingerprint yet and adopt the current one.
FP=$(python - "$CACHE" <<'PY'
import json, os, sys
m = json.load(open(os.path.join(sys.argv[1], "meta.json")))
print(f"built={m.get('built', '?')} X.npy={os.path.getsize(os.path.join(sys.argv[1], 'X.npy'))}")
PY
) || { note "ERROR: cannot fingerprint $CACHE"; exit 1; }
for f in "$LOGDIR/cache_fingerprint.txt" "$HDIR/cache_fingerprint.txt"; do
  if [ -f "$f" ] && [ "$(cat "$f")" != "$FP" ]; then
    note "ERROR: the $STUDY cache ($FP) was rebuilt since $(dirname "$f") was computed" \
         "($(cat "$f")): its steps / harness rows are stale; move logs/$TAG and" \
         "logs/sealed_$HTAG away or use another RUN_NAME"
    exit 1
  fi
done
# HARNESS_FROM: seed this run's harness rows from another harness tag (same
# cache build), so the harness steps skip every setting already computed there
if [ -n "${HARNESS_FROM:-}" ]; then
  SRC="$HOME/codabench/logs/sealed_$HARNESS_FROM"
  if [ "$SRC" = "$HDIR" ]; then
    note "HARNESS_FROM=$HARNESS_FROM is this run's own harness tag: nothing to seed"
  elif [ ! -d "$SRC" ] || ! ls "$SRC"/results*.jsonl >/dev/null 2>&1; then
    note "ERROR: HARNESS_FROM=$HARNESS_FROM: no harness rows in $SRC"; exit 1
  elif [ ! -f "$SRC/cache_fingerprint.txt" ] || [ "$(cat "$SRC/cache_fingerprint.txt")" != "$FP" ]; then
    note "ERROR: HARNESS_FROM=$HARNESS_FROM: $SRC was computed on another build of the" \
         "$STUDY cache ($(cat "$SRC/cache_fingerprint.txt" 2>/dev/null || echo 'no fingerprint'))," \
         "not this one ($FP): its rows would be stale"
    exit 1
  elif [ "${HARNESS_FROM_ANY:-0}" != 1 ] && [ -f "$HOME/codabench/logs/$HARNESS_FROM/STATUS.md" ] \
       && { ! grep -q "ALL DONE" "$HOME/codabench/logs/$HARNESS_FROM/STATUS.md" \
            || grep -qE "NO ZIP|STOPPED" "$HOME/codabench/logs/$HARNESS_FROM/STATUS.md"; }; then
    note "ERROR: HARNESS_FROM=$HARNESS_FROM: its run did not pass its gate (logs/$HARNESS_FROM/" \
         "STATUS.md needs ALL DONE and no NO ZIP / STOPPED row); check it, then" \
         "HARNESS_FROM_ANY=1 seeds anyway"
    exit 1
  elif ls "$HDIR"/results*.jsonl >/dev/null 2>&1; then
    note "HARNESS_FROM=$HARNESS_FROM: logs/sealed_$HTAG already holds rows" \
         "($(cat "$HDIR/seeded_from.txt" 2>/dev/null || echo 'not seeded')): not seeded again"
  else
    mkdir -p "$HDIR/probs"
    cp -p "$SRC"/results*.jsonl "$HDIR/" \
      && { [ ! -d "$SRC/probs" ] || cp -rp "$SRC/probs/." "$HDIR/probs/"; } \
      || { note "ERROR: HARNESS_FROM=$HARNESS_FROM: copying $SRC into $HDIR failed"; exit 1; }
    echo "$HARNESS_FROM $(date '+%F %T')" > "$HDIR/seeded_from.txt"
    note "HARNESS_FROM: seeded logs/sealed_$HTAG from logs/sealed_$HARNESS_FROM" \
         "($(cat "$HDIR"/results*.jsonl | wc -l) rows, $(ls "$HDIR/probs" | wc -l) probs files)"
  fi
fi
mkdir -p "$HDIR"
echo "$FP" > "$LOGDIR/cache_fingerprint.txt"; echo "$FP" > "$HDIR/cache_fingerprint.txt"
echo "$CONF" > "$LOGDIR/config.txt"
note "config $CONF (harness rows: logs/sealed_$HTAG; cache $FP)"
note "gates blocking the zip: GATE=$GATE$([ "$FORCE_ZIP" = 1 ] && echo ' FORCE_ZIP=1')"

H_ARGS=(--split "$SPLIT" --chans "$CHANS")
[ -n "$TEST_SUBJECTS" ] && H_ARGS+=(--test_subjects "$TEST_SUBJECTS")
[ "$ROUTER_CAP" = none ] || H_ARGS+=(--router_cap "$ROUTER_CAP")
# 500 Hz caches (> 2 GiB): memory-map X (same values, not in the config key)
[ "$(stat -c %s "$CACHE/X.npy" 2>/dev/null || echo 0)" -gt $((2 * 1024 ** 3)) ] && H_ARGS+=(--mmap)
# preflight: the split / channel [data] line, and no hidden-test row may train
step preflight 30m python "$A/sealed_run.py" --tag "$HTAG" --study "$STUDY" --models meanlr \
    --check "${H_ARGS[@]}" \
    || { note "ERROR: preflight failed (hidden-test rows in train, or a bad split): see" \
              "$LOGDIR/preflight.log"; exit 1; }
note "preflight: $(grep -m1 -o 'X=.*' "$LOGDIR/preflight.log")"
step validate 120m python "$A/sealed_run.py" --tag "$HTAG" --study "$STUDY" \
    --models meanlr "$RECIPE_SPEC" --modes pooled persubject --aligns none "$RECIPE_ALIGN" \
    "${H_ARGS[@]}" || exit 1
step personal_$ADAPT 120m python "$A/sealed_personal.py" --tag "$HTAG" --study "$STUDY" \
    --family "$RECIPE_SPEC" --align "$RECIPE_ALIGN" --wcv "$WCV" --wvariant "$WVARIANT" \
    "${H_ARGS[@]}" || exit 1

# 4. benchopt training of the solver (needs the overlay registered in
#    bci_studies _OVERLAYS: see scripts/install_xsess_overlays.sh).
#    blend_calib weight: the one sealed_personal chose on training data under
#    these settings (held-out calibration session / halves, or loso); with
#    BLEND_W=auto the solver chooses its own by loso from the loader's session
#    ids and the harness weight is only the cross-check.
#    (the harness writes to logs/sealed_<harness tag>/, not to this script's LOGDIR)
HW=$(python - "$HOME/codabench/logs/sealed_$HTAG" "$STUDY" "$RECIPE_SPEC" "$RECIPE_ALIGN" \
     "$SPLIT" "$WCV" "$CHANS" "$ROUTER_CAP" "$TEST_SUBJECTS" <<'PY'
import json, pathlib, sys
d, study, spec, align, split, wcv, chans, cap, ts = sys.argv[1:]
cap = None if cap == "none" else float(cap)
# as sealed_run.prepare_data stores it: sorted subject indices, comma-joined
ts = ",".join(map(str, sorted({int(s) for s in ts.split(",") if s}))) or None
p = pathlib.Path(d) / f"results_{study}.jsonl"
rows = []
for line in (p.read_text().splitlines() if p.exists() else []):
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError:
        continue
# this configuration's rows (rows written before 2026-09-28 carry no split
# fields: they are the defaults)
rows = [r for r in rows if "blend_calib_w" in r and r["align"] == align
        and r["spec"].split("/")[0] == spec and r.get("split", "last") == split
        and r.get("wcv", "last") == wcv and r.get("chans", "eeg") == chans
        and r.get("pool", "all") == "all" and r.get("router_cap") == cap
        and r.get("test_subjects") == ts]
if not rows:
    sys.exit(f"no blend_calib_w for spec={spec} align={align} split={split} wcv={wcv} "
             f"chans={chans} router_cap={cap} test_subjects={ts} in {p}")
# the deployed variant (blend_calib, router ids) for the gate check after replay
bc = [r for r in rows if r["spec"].endswith("/blend_calib") and r["mode"] == "router-id"]
bc = bc[-1] if bc else {"pooled": "nan", "cell": "nan"}
print(rows[-1]["blend_calib_w"], bc["pooled"], bc["cell"])
PY
) || { echo "| $(date +%T) | ERROR: blend weight not found |" >> "$STATUS"; exit 1; }
read -r W H_POOLED H_CELL <<< "$HW"
case $BLEND_W in harness) CW=$W;; auto) CW='"auto"';; *) CW=$BLEND_W;; esac
# Codabench instantiates the solver with its DEFAULT parameters: bake the chosen
# settings into a candidate copy as defaults (the frozen-WU1 practice) and train
# that copy with no overrides.
CAND="$LOGDIR/riemann_sealed_cand.py"
sed -e 's/"personal": \["pooled"\]/"personal": ["blend"]/' \
    -e "s/\"blend_w\": \[0.5\]/\"blend_w\": [$CW]/" \
    -e "s/\"adapt\": \[\"none\"\]/\"adapt\": [\"$ADAPT\"]/" \
    -e "s/\"use_xdawn\": \[True\]/\"use_xdawn\": [$XDB]/" \
    -e "s/\"filterbank\": \[True\]/\"filterbank\": [$FBB]/" \
    -e "s/\"kind\": \[\"riemann\"\]/\"kind\": [\"$KIND\"]/" \
    -e "s/\"buffer\": \[64\]/\"buffer\": [$BUF]/" \
    -e "s/\"chans\": \[\"eeg\"\]/\"chans\": [\"$CHANS\"]/" \
    -e "s/\"align\": \[\"subject\"\]/\"align\": [\"$ALIGN_S\"]/" \
    -e "s/name = \"Riemann-Sealed\"/name = \"Riemann-Sealed-Cand$SUFFIX\"/" \
    "$HOME/codabench/solvers/bci_decoding/riemann_sealed.py" > "$CAND"
for want in "\"blend_w\": [$CW]" '"personal": ["blend"]' "\"adapt\": [\"$ADAPT\"]" \
            "\"use_xdawn\": [$XDB]" "\"filterbank\": [$FBB]" "\"kind\": [\"$KIND\"]" \
            "\"buffer\": [$BUF]" "\"chans\": [\"$CHANS\"]" "\"align\": [\"$ALIGN_S\"]"; do
  grep -qF "$want" "$CAND" \
      || { echo "| $(date +%T) | ERROR: candidate defaults not set ($want) |" >> "$STATUS"; exit 1; }
done
# a solver without the subject_context branch would silently fall back to no alignment
if [ "$ALIGN_S" = subject_context ] && ! grep -q 'align == "subject_context"' "$CAND"; then
  note "ERROR: $CAND has no align=\"subject_context\" (solver older than 2026-09-28 agent G)"; exit 1
fi
note "candidate $CAND: personal=blend blend_w=$CW adapt=$ADAPT use_xdawn=$XDB" \
     "filterbank=$FBB kind=$KIND buffer=$BUF chans=$CHANS align=$ALIGN_S"
cd "$HOME/codabench/2026-competition"
# OUT: resolved with the settings above
[ -n "$SET_OUT" ] && export COMPET_SUBMISSION_DIR="$OUT"
# --no-cache: benchopt caches on (dataset, solver) parameters, not on the data or
# the candidate file, so a rerun would silently return the previous score.
# A (re)training voids an earlier replay and hash: the replay must be of this training.
[ -f "$LOGDIR/train.done" ] || rm -f "$LOGDIR/replay.done" "$LOGDIR/train.hash"
step train 180m benchopt run tracks/bci_decoding -d "$DATASET" -s "$CAND" \
    -o "BCI-decoding[training=True]" --no-plot --no-html --no-cache --output "${TAG}_train" || exit 1
unset COMPET_SUBMISSION_DIR
if [ ! -f "$LOGDIR/train.hash" ]; then
  # trained before 2026-09-29 (no hash was recorded): nothing to compare with
  out_hash > "$LOGDIR/train.hash"
  note "WARNING: no train.hash (trained before 2026-09-29): recorded the current $OUT" \
       "as this training's submission"
fi
note "solver: $(grep -m1 -o 'fitting on X=.*' "$LOGDIR/train.log")"
# solver vs harness weight, WV: MATCH | MISMATCH (both chose by loso: a real
# disagreement) | INCOMPARABLE (harness wcv is not loso) | n/a (weight baked)
SW=
if [ "$BLEND_W" = auto ]; then
  SW=$(sed -n 's/.*\[Riemann-Sealed\] blend_w=auto -> \([0-9.]*\) .*/\1/p' "$LOGDIR/train.log" | tail -1)
  grep -q "blend_w='auto' needs session ids" "$LOGDIR/train.log" && SW="0.5 (no session ids)"
  if [ "$SW" = "$W" ]; then V=MATCH; else V="MISMATCH: check the fold scores"; fi
  if [ "$WCV" = loso ]; then WV=${V%%:*}
  else WV=INCOMPARABLE; V="$V (harness wcv=$WCV is not the solver's loso rule: informational)"; fi
  note "blend weight: harness w=$W (split=$SPLIT wcv=$WCV) vs solver auto w=${SW:-not found}: $V"
else
  WV=n/a
  note "blend weight: harness w=$W (split=$SPLIT wcv=$WCV); solver trained with the baked w=$CW"
fi

# 5. replay read-only, inference only; a failed replay stops here (no zip)
check_out replay
R=/tmp/${TAG}_replay; [ -d "$R" ] && chmod -R u+w "$R"; rm -rf "$R"; cp -r "$OUT" "$R"; chmod -R a-w "$R"
COMPET_SUBMISSION_DIR="$R" step replay 120m benchopt run tracks/bci_decoding \
    -d "$DATASET" -s "$R/submission.py" --no-plot --no-html --no-cache --output "${TAG}_replay" \
    || { note "ERROR: replay failed: NO ZIP (see $LOGDIR/replay.log)"; exit 1; }
# this run's result files: the paths benchopt reported saving in the step logs
# (it appends _1, _2, ... to a name that exists, so a name glob or the newest
# mtime can pick another run's file)
saved() {
  grep -ao 'Saving result in: [^ ]*\.parquet' "$LOGDIR/$1.log" 2>/dev/null | tail -1 \
    | sed 's/^Saving result in: //'
}
PT=$(saved train); PR=$(saved replay)
if [ -z "$PT" ] || [ ! -f "$PT" ] || [ -z "$PR" ] || [ ! -f "$PR" ]; then
  note "ERROR: this run's result files are missing (train: ${PT:-not in train.log}," \
       "replay: ${PR:-not in replay.log}): NO ZIP"
  exit 1
fi
python "$HOME/codabench/scripts/summarize_runs.py" "$PT" "$PR" > "$LOGDIR/RESULTS.md" 2>&1
# gate: train score == replay score (exact) and within 2 points of the harness
# blend_calib (router ids) pooled balanced accuracy on the same split
GOUT=$(python - "$PT" "$PR" "$H_POOLED" "$H_CELL" "$W" "$CW" "$LOGDIR/RESULTS.md" <<'PY'
import pathlib, sys
import pandas as pd
pt, pr, hp, hc, hw, cw, res = sys.argv[1:]
hp, hc = float(hp), float(hc)


def score(p):
    return pathlib.Path(p).name, float(pd.read_parquet(p)["objective_balanced_accuracy"].iloc[-1])


(ft, tr), (fr, rp) = score(pt), score(pr)
same = "EQUAL" if tr == rp else "DIFFERENT"
gap = abs(tr - hp)
ok = "OK" if gap <= 0.02 else "FLAG"
with open(res, "a") as f:
    f.write(f"\n## Gate check\n\n| check | value | verdict |\n|---|---|---|\n"
            f"| benchopt train ({ft}) | {tr:.6f} | |\n"
            f"| read-only replay ({fr}) | {rp:.6f} | {same} (exact) |\n"
            f"| harness blend_calib router-id pooled BA (cell {hc:.4f}, w={hw}) | {hp:.6f} | "
            f"abs(train - harness) = {gap:.4f}: {ok} (<= 0.02) |\n"
            f"\nCandidate blend_w = {cw}.\n")
print(f"gate: train {tr:.6f} replay {rp:.6f} {same}; harness blend_calib pooled "
      f"{hp:.6f} (w={hw}), gap {gap:.4f} {ok}")
print(same, ok)
PY
) || { note "ERROR: the gate check failed to run: NO ZIP"; exit 1; }
{ read -r GLINE; read -r SAME GAPV; } <<< "$GOUT"
note "$GLINE"
# which gates block the zip (GATE, see the header): train = replay always; on a
# replica run (benchopt's test set = the harness's) also the gap and a weight
# MISMATCH (FORCE_ZIP=1 overrides the MISMATCH only)
BLOCK=; FORCED=
[ "$SAME" = EQUAL ] || BLOCK="train != replay (exact)"
if [ "$GATE" = replica ]; then
  [ "$GAPV" = OK ] || BLOCK="${BLOCK:+$BLOCK; }harness gap > 0.02 (FLAG)"
  if [ "$WV" = MISMATCH ]; then
    if [ "$FORCE_ZIP" = 1 ]; then
      FORCED="solver vs harness weight MISMATCH (harness w=$W, solver w=${SW:-not found})"
    else
      BLOCK="${BLOCK:+$BLOCK; }solver vs harness weight MISMATCH (FORCE_ZIP=1 zips anyway)"
    fi
  fi
fi
if [ "$GATE" = replica ]; then
  GAPROLE="a FLAG blocks the zip"
  case $WV in
    MATCH|MISMATCH) WROLE="a MISMATCH blocks the zip unless FORCE_ZIP=1";;
    INCOMPARABLE) WROLE="informational (harness wcv=$WCV is not the solver's loso rule)";;
    *) WROLE="nothing to compare (the weight was baked)";;
  esac
else
  GAPROLE="informational under GATE=final"; WROLE="informational under GATE=final"
fi
{ printf '\n## Zip decision (GATE=%s)\n\n' "$GATE"
  echo "- train = read-only replay (exact): $SAME (anything but EQUAL blocks the zip under every GATE)"
  echo "- harness gap: $GAPV ($GAPROLE)"
  echo "- blend weight, solver vs harness: $WV ($WROLE)"
  if [ -n "$BLOCK" ]; then echo "- **NO ZIP**: $BLOCK"
  elif [ -n "$FORCED" ]; then echo "- **ZIP FORCED (FORCE_ZIP=1) despite: $FORCED. Do not upload it before the fold scores explain the difference.**"
  else echo "- zip written"; fi
} >> "$LOGDIR/RESULTS.md"
if [ -n "$BLOCK" ]; then
  note "NO ZIP (GATE=$GATE): $BLOCK"
  for z in "$LOGDIR"/*.zip; do
    [ -e "$z" ] || continue
    mv "$z" "$z.blocked" \
      && note "renamed $(basename "$z") (an older zip of this folder) to $(basename "$z").blocked"
  done
  echo "| $(date +%T) | STOPPED: gate failed, no zip |" >> "$STATUS"
  exit 1
fi
if [ -n "$FORCED" ]; then
  note "WARNING !!! FORCE_ZIP=1: ZIPPING DESPITE A FAILED REPLICA GATE: $FORCED." \
       "Do not upload this zip before the fold scores in train.log and personal_$ADAPT.log" \
       "explain the difference !!!"
fi
# zip into the run's log dir (codabench/submissions/ is the user's folder), only
# the model this run trained and replayed
check_out zip
Z="$LOGDIR/riemann_sealed_${STUDY}_$(date +%F).zip"
python - "$OUT" "$Z" <<'PY' || { note "ERROR: zipping $OUT failed"; exit 1; }
import pathlib, sys, zipfile
src, dst = pathlib.Path(sys.argv[1]), sys.argv[2]
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(src.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            z.write(p, p.relative_to(src))
PY
echo "| $(date +%T) | zipped $Z (NOT uploaded$([ -n "$FORCED" ] && echo '; FORCED, see the WARNING')) |" >> "$STATUS"
echo "| $(date +%T) | ALL DONE |" >> "$STATUS"
