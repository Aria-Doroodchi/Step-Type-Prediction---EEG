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
TAG=train_sealed_$STUDY
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
RECIPE_ALIGN=${RECIPE_ALIGN:-router-psd:riemann}
step validate 120m python "$A/sealed_run.py" --tag "$TAG" --study "$STUDY" \
    --models meanlr "$RECIPE_SPEC" --modes pooled persubject --aligns none "$RECIPE_ALIGN"
step personal 120m python "$A/sealed_personal.py" --tag "$TAG" --study "$STUDY" \
    --family "$RECIPE_SPEC" --align "$RECIPE_ALIGN"

# 4. benchopt training of the solver (needs the overlay registered in
#    bci_studies _OVERLAYS: see scripts/install_xsess_overlays.sh)
cd "$HOME/codabench/2026-competition"
OUT=tracks/bci_decoding/outputs/Riemann-Sealed
step train 180m benchopt run tracks/bci_decoding -d "BCI[study=$STUDY]" \
    -s ../solvers/bci_decoding/riemann_sealed.py -o "BCI-decoding[training=True]" \
    --no-plot --no-html --output "${TAG}_train" || exit 1

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
