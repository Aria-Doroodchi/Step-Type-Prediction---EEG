#!/usr/bin/env bash
# Train the sealed-phase recipe end to end on a released study (SKELETON,
# 2026-09-26; the recipe itself is in codabench/SEALED_RECIPE.md).
#
#   bash ~/codabench/scripts/train_sealed.sh <data_home> <study> <modality/task> [overlay]
#   e.g. bash ~/codabench/scripts/train_sealed.sh ~/neuralbench/benchopt_data \
#            graz2026 eeg/motor_imagery graz2026
#
# <data_home>  benchopt data home holding neural_compet/<Study>/ (the study must
#              already be downloaded: `benchopt prepare` or neuralfetch)
# <study>      cache key (also the bci_studies _OVERLAYS key for benchopt)
# <task>       NeuralBench task whose windows/targets define the data
# [overlay]    dataset overlay in that task's datasets/ folder
#
# Steps (each resumable via <step>.done in logs/train_sealed_<study>/):
#   1. cache: every window + subject/session/run(/context) metadata
#   2. EDA   : TODO (parametrise analysis/dreyer_eda.py by study)
#   3. validate the recipe in the harness (cross-session, cell metric)
#   4. train Riemann-Sealed with benchopt on the study overlay (recipe defaults)
#   5. platform replay from a read-only copy (score must match step 4), zip
#      into the log dir
# Never uploads (the user's action) and never writes codabench/submissions/.
set -u
DATA_HOME=${1:?data_home}; STUDY=${2:?study}; TASK=${3:?modality/task}; OVERLAY=${4:-}
ADAPT=${ADAPT:-none}   # "online" only once the organisers allow test-time statistics
SUFFIX=$([ "$ADAPT" = online ] && echo _online)
TAG=train_sealed_$STUDY$SUFFIX
source "$HOME/codabench/env.sh" >/dev/null
export BENCHOPT_DATA_HOME="$DATA_HOME" PYTHONUTF8=1
export XS_THREADS=${XS_THREADS:-10}
export OMP_NUM_THREADS=$XS_THREADS OPENBLAS_NUM_THREADS=$XS_THREADS MKL_NUM_THREADS=$XS_THREADS
LOGDIR="$HOME/codabench/logs/$TAG"; mkdir -p "$LOGDIR"
STATUS="$LOGDIR/STATUS.md"; [ -f "$STATUS" ] || printf '| time | event |\n|---|---|\n' > "$STATUS"
A="$HOME/codabench/analysis"

step() {   # step <name> <timeout> <cmd...>
  local name=$1 to=$2; shift 2
  [ -f "$LOGDIR/$name.done" ] && { echo "| $(date +%T) | skip $name |" >> "$STATUS"; return 0; }
  echo "| $(date +%T) | START $name |" >> "$STATUS"
  timeout "$to" "$@" > "$LOGDIR/$name.log" 2>&1; local rc=$?
  echo "| $(date +%T) | END $name rc=$rc |" >> "$STATUS"
  [ $rc -eq 0 ] && touch "$LOGDIR/$name.done"
  return $rc
}

# 1. cache (check the [data]/shape line in cache.log before believing anything)
step cache 60m python "$A/xsess_cache.py" "$STUDY" --task "$TASK" ${OVERLAY:+--overlay "$OVERLAY"} || exit 1

# 3. recipe validation, cross-session (last session per subject = test).
#    RECIPE_* defaults = SEALED_RECIPE.md section 1; override from the env.
RECIPE_SPEC=${RECIPE_SPEC:-riemann:xd=1,fb=1}
# the blend weight must be chosen under the alignment the solver will use
# (Zhou: w=0.75 under the router, 0.5 under online; the wrong one cost 3 points)
if [ "$ADAPT" = online ]; then DEF_ALIGN=online-64:riemann; else DEF_ALIGN=router-psd:riemann; fi
RECIPE_ALIGN=${RECIPE_ALIGN:-$DEF_ALIGN}
step validate 120m python "$A/sealed_run.py" --tag "$TAG" --study "$STUDY" \
    --models meanlr "$RECIPE_SPEC" --modes pooled persubject --aligns none "$RECIPE_ALIGN"
step personal_$ADAPT 120m python "$A/sealed_personal.py" --tag "$TAG" --study "$STUDY" \
    --family "$RECIPE_SPEC" --align "$RECIPE_ALIGN"

# 4. benchopt training of the solver (needs the overlay registered in
#    bci_studies _OVERLAYS: see scripts/install_xsess_overlays.sh).
#    blend_calib weight: the benchopt train loader is shuffled and has no
#    session ids, so the solver cannot choose it; take the one sealed_personal
#    chose on training data (held-out calibration session / halves).
#    (the harness writes to logs/sealed_<tag>/, not to this script's LOGDIR)
W=$(python - "$HOME/codabench/logs/sealed_$TAG" "$STUDY" "$RECIPE_ALIGN" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / f"results_{sys.argv[2]}.jsonl"
rows = [r for r in (json.loads(l) for l in p.read_text().splitlines()) if "blend_calib_w" in r
        and r["align"] == sys.argv[3]] if p.exists() else []
if not rows:
    sys.exit(f"no blend_calib_w for align={sys.argv[3]} in {p}")
print(rows[-1]["blend_calib_w"])
PY
) || { echo "| $(date +%T) | ERROR: blend weight not found |" >> "$STATUS"; exit 1; }
# Codabench instantiates the solver with its DEFAULT parameters: bake the chosen
# settings into a candidate copy as defaults (the frozen-WU1 practice) and train
# that copy with no overrides.
CAND="$LOGDIR/riemann_sealed_cand.py"
sed -e 's/"personal": \["pooled"\]/"personal": ["blend"]/' \
    -e "s/\"blend_w\": \[0.5\]/\"blend_w\": [$W]/" \
    -e "s/\"adapt\": \[\"none\"\]/\"adapt\": [\"$ADAPT\"]/" \
    -e "s/name = \"Riemann-Sealed\"/name = \"Riemann-Sealed-Cand$SUFFIX\"/" \
    "$HOME/codabench/solvers/bci_decoding/riemann_sealed.py" > "$CAND"
grep -q "\"blend_w\": \[$W\]" "$CAND" && grep -q '"personal": \["blend"\]' "$CAND" \
    && grep -q "\"adapt\": \[\"$ADAPT\"\]" "$CAND" \
    || { echo "| $(date +%T) | ERROR: candidate defaults not set |" >> "$STATUS"; exit 1; }
echo "| $(date +%T) | candidate $CAND: personal=blend blend_w=$W adapt=$ADAPT |" >> "$STATUS"
cd "$HOME/codabench/2026-competition"
OUT=tracks/bci_decoding/outputs/Riemann-Sealed-Cand$SUFFIX
step train 180m benchopt run tracks/bci_decoding -d "BCI[study=$STUDY]" -s "$CAND" \
    -o "BCI-decoding[training=True]" --no-plot --no-html --output "${TAG}_train" || exit 1

# 5. replay read-only, inference only; then zip (files at the zip root)
R=/tmp/${TAG}_replay; rm -rf "$R"; cp -r "$OUT" "$R"; chmod -R a-w "$R"
COMPET_SUBMISSION_DIR="$R" step replay 120m benchopt run tracks/bci_decoding \
    -d "BCI[study=$STUDY]" -s "$R/submission.py" --no-plot --no-html --output "${TAG}_replay"
python "$HOME/codabench/scripts/summarize_runs.py" \
    tracks/bci_decoding/outputs/"${TAG}"_*.parquet > "$LOGDIR/RESULTS.md" 2>&1
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
