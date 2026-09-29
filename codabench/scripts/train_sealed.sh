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
#              sessions) with SPLIT unset stops here and says so
#   TEST_SUBJECTS  comma list of subject indices tested under SPLIT (harness
#              --test_subjects; default: the split's own)
#   WCV        harness blend-weight CV: last | loso (default when BLEND_W=auto,
#              the rule of the solver's "auto", so the two weights are comparable)
#   CHANS      eeg (default) | eeg+eog | eeg+emg | all: harness and candidate
#   RECIPE_SPEC   harness family (riemann:xd=1,fb=1); xd / fb are baked into the
#              candidate as use_xdawn / filterbank
#   RECIPE_ALIGN  harness alignment (router-psd:riemann; online-64:riemann under
#              ADAPT=online); its kind (and online buffer) is baked in
#   ROUTER_CAP harness router fallback cap: none | 0.5 (release, the solver's
#              min(thr, 0.5))
#   WVARIANT   sealed_personal --wvariant: both | calib (release; same weight,
#              without the per-subject full models)
#   DATASET    benchopt -d selector: BCI[study=<study>]; a mock cache (meta has
#              "mock") -> ../datasets/mock_sealed.py[study=<study>]. The name form
#              MockSealed[...] is rewritten to that path (benchopt 1.10 finds the
#              mock dataset only by path)
#   SUBMISSION_DIR  absolute folder training writes the submission to: the
#              track's outputs/Riemann-Sealed-Cand<suffix>/ (as before), or
#              (release) <logdir>/submission, so no outputs/ folder is overwritten
#   STOP_AFTER <step name>: stop after that step (e.g. personal_none, to read
#              the harness numbers before training); rerun without it to go on
#   RUN_NAME   log folder / harness tag / benchopt output name instead of
#              train_sealed_<study><suffix> (e.g. a regression run that must not
#              touch a committed log folder)
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
#      check vs the harness in RESULTS.md, zip into the log dir
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
    [ $rc -eq 0 ] && touch "$LOGDIR/$name.done"
  fi
  # STOP_AFTER stops here whatever the step's rc (a failed step never goes on)
  if [ "${STOP_AFTER:-}" = "$name" ]; then
    note "stopped after $name (STOP_AFTER, rc=$rc)"; exit $rc
  fi
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
# fb=0), the reference kind and online buffer of the alignment
XD=1; FB=0; KIND=${RECIPE_ALIGN##*:}; HOW=${RECIPE_ALIGN%%:*}; BUF=64
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
  *-psdctx) note "ERROR: RECIPE_ALIGN=$RECIPE_ALIGN: Riemann-Sealed has no (subject, context) router"; exit 1;;
  *) note "WARNING: RECIPE_ALIGN=$RECIPE_ALIGN is not what the solver does (router-psd / online-N)";;
esac
XDB=$([ "$XD" = 0 ] && echo False || echo True); FBB=$([ "$FB" = 1 ] && echo True || echo False)

CONF="split=$SPLIT test_subjects=${TEST_SUBJECTS:-default} wcv=$WCV chans=$CHANS"
CONF="$CONF spec=$RECIPE_SPEC align=$RECIPE_ALIGN adapt=$ADAPT"
CONF="$CONF blend_w=$BLEND_W router_cap=$ROUTER_CAP wvariant=$WVARIANT dataset=$DATASET"
HDIR="$HOME/codabench/logs/sealed_$HTAG"
if [ -f "$LOGDIR/config.txt" ] && [ "$(cat "$LOGDIR/config.txt")" != "$CONF" ]; then
  note "ERROR: settings differ from the ones $LOGDIR ran with" \
       "($(cat "$LOGDIR/config.txt")); move logs/$TAG and its harness rows logs/sealed_$HTAG" \
       "away, use another RUN_NAME, or rerun with those settings"
  exit 1
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
mkdir -p "$HDIR"
echo "$FP" > "$LOGDIR/cache_fingerprint.txt"; echo "$FP" > "$HDIR/cache_fingerprint.txt"
echo "$CONF" > "$LOGDIR/config.txt"
note "config $CONF (harness rows: logs/sealed_$HTAG; cache $FP)"

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
    -e "s/name = \"Riemann-Sealed\"/name = \"Riemann-Sealed-Cand$SUFFIX\"/" \
    "$HOME/codabench/solvers/bci_decoding/riemann_sealed.py" > "$CAND"
for want in "\"blend_w\": [$CW]" '"personal": ["blend"]' "\"adapt\": [\"$ADAPT\"]" \
            "\"use_xdawn\": [$XDB]" "\"filterbank\": [$FBB]" "\"kind\": [\"$KIND\"]" \
            "\"buffer\": [$BUF]" "\"chans\": [\"$CHANS\"]"; do
  grep -qF "$want" "$CAND" \
      || { echo "| $(date +%T) | ERROR: candidate defaults not set ($want) |" >> "$STATUS"; exit 1; }
done
note "candidate $CAND: personal=blend blend_w=$CW adapt=$ADAPT use_xdawn=$XDB" \
     "filterbank=$FBB kind=$KIND buffer=$BUF chans=$CHANS"
cd "$HOME/codabench/2026-competition"
if [ -n "${SUBMISSION_DIR:-}" ] || [ "$SPLIT" != last ]; then
  OUT=${SUBMISSION_DIR:-$LOGDIR/submission}; export COMPET_SUBMISSION_DIR="$OUT"
else
  OUT=tracks/bci_decoding/outputs/Riemann-Sealed-Cand$SUFFIX
fi
# --no-cache: benchopt caches on (dataset, solver) parameters, not on the data or
# the candidate file, so a rerun would silently return the previous score
step train 180m benchopt run tracks/bci_decoding -d "$DATASET" -s "$CAND" \
    -o "BCI-decoding[training=True]" --no-plot --no-html --no-cache --output "${TAG}_train" || exit 1
unset COMPET_SUBMISSION_DIR
note "solver: $(grep -m1 -o 'fitting on X=.*' "$LOGDIR/train.log")"
if [ "$BLEND_W" = auto ]; then
  SW=$(sed -n 's/.*\[Riemann-Sealed\] blend_w=auto -> \([0-9.]*\) .*/\1/p' "$LOGDIR/train.log" | tail -1)
  grep -q "blend_w='auto' needs session ids" "$LOGDIR/train.log" && SW="0.5 (no session ids)"
  if [ "$SW" = "$W" ]; then V=MATCH; else V="MISMATCH: check the fold scores"; fi
  [ "$WCV" = loso ] || V="$V (harness wcv=$WCV is not the solver's loso rule)"
  note "blend weight: harness w=$W (split=$SPLIT wcv=$WCV) vs solver auto w=${SW:-not found}: $V"
else
  note "blend weight: harness w=$W (split=$SPLIT wcv=$WCV); solver trained with the baked w=$CW"
fi

# 5. replay read-only, inference only; then zip (files at the zip root)
R=/tmp/${TAG}_replay; [ -d "$R" ] && chmod -R u+w "$R"; rm -rf "$R"; cp -r "$OUT" "$R"; chmod -R a-w "$R"
COMPET_SUBMISSION_DIR="$R" step replay 120m benchopt run tracks/bci_decoding \
    -d "$DATASET" -s "$R/submission.py" --no-plot --no-html --no-cache --output "${TAG}_replay"
python "$HOME/codabench/scripts/summarize_runs.py" \
    tracks/bci_decoding/outputs/"${TAG}"_train*.parquet \
    tracks/bci_decoding/outputs/"${TAG}"_replay*.parquet > "$LOGDIR/RESULTS.md" 2>&1
# gate: train score == replay score (exact) and within 2 points of the harness
# blend_calib (router ids) pooled balanced accuracy on the same split
GATE=$(python - tracks/bci_decoding/outputs "$TAG" "$H_POOLED" "$H_CELL" "$W" "$CW" \
       "$LOGDIR/RESULTS.md" <<'PY'
import pathlib, sys
import pandas as pd
out, tag, hp, hc, hw, cw, res = sys.argv[1:]
hp, hc = float(hp), float(hc)

def newest(kind):
    ps = sorted(pathlib.Path(out).glob(f"{tag}_{kind}*.parquet"), key=lambda p: p.stat().st_mtime)
    if not ps:
        return "missing", float("nan")
    return ps[-1].name, float(pd.read_parquet(ps[-1])["objective_balanced_accuracy"].iloc[-1])

(ft, tr), (fr, rp) = newest("train"), newest("replay")
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
PY
)
note "$GATE"
# zip into the run's log dir (codabench/submissions/ is the user's folder)
Z="$LOGDIR/riemann_sealed_${STUDY}_$(date +%F).zip"
python - "$OUT" "$Z" <<'PY' && echo "| $(date +%T) | zipped $Z (NOT uploaded) |" >> "$STATUS"
import pathlib, sys, zipfile
src, dst = pathlib.Path(sys.argv[1]), sys.argv[2]
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(src.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            z.write(p, p.relative_to(src))
PY
echo "| $(date +%T) | ALL DONE |" >> "$STATUS"
