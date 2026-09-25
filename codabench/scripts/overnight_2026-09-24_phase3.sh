#!/usr/bin/env bash
# Overnight 2026-09-24, phase 3.
#   1. train the frozen warm-up candidate EEGNet-StepType-WU1 on full Dreyer
#      with its DEFAULT parameters (= what Codabench will run),
#   2. replay it exactly like the platform (read-only copy, inference-only,
#      COMPET_SUBMISSION_DIR) and check the score matches,
#   3. zip it (files at the ZIP root) into ~/codabench/submissions/  (NOT uploaded),
#   4. EEGNet-StepType 200 epochs / patience 30, 3 seeds: do gains continue?
# Estimated ~3.5-4 h. Resumable like phases 1-2.

set -u
source "$HOME/codabench/env.sh" >/dev/null
cd "$HOME/codabench/2026-competition"
TAG=overnight_2026-09-24_p3
LOGDIR="$HOME/codabench/logs/$TAG"
mkdir -p "$LOGDIR" "$HOME/codabench/submissions"
S="$HOME/codabench/solvers/bci_decoding"
STATUS="$LOGDIR/STATUS.md"
[ -f "$STATUS" ] || printf '| time | event |\n|---|---|\n' > "$STATUS"
DATA='BCI[study=dreyer2023]'
OBJ='BCI-decoding[training=True]'
note() { echo "| $(date +%T) | $* |" >> "$STATUS"; }

step() {
  local name=$1 timeout=$2; shift 2
  if [ -f "$LOGDIR/$name.done" ]; then note "skip $name (done)"; return 0; fi
  note "START $name"
  benchopt run tracks/bci_decoding -d "$DATA" --no-plot --no-html \
      --timeout "$timeout" --output "${TAG}_$name" "$@" > "$LOGDIR/$name.log" 2>&1
  local rc=$?
  note "END $name rc=$rc"
  [ $rc -eq 0 ] && touch "$LOGDIR/$name.done"
  return $rc
}

score() {   # score <parquet name>  -> prints balanced accuracy
  python -c "import pandas as pd,sys; print(round(float(pd.read_parquet(sys.argv[1]).objective_balanced_accuracy.iloc[0]),4))" \
      "tracks/bci_decoding/outputs/$1.parquet"
}

# 1. train the candidate with its defaults (no parameter overrides!)
OUT=tracks/bci_decoding/outputs/EEGNet-StepType-WU1
[ -f "$LOGDIR/wu1_train.done" ] || rm -rf "$OUT"
if step wu1_train 150m -o "$OBJ" -s "$S/eegnet_steptype_wu1.py"; then
  note "wu1_train score $(score ${TAG}_wu1_train)"

  # 2. platform replay from a read-only copy, inference-only (no -o training)
  R=/tmp/wu1_replay; rm -rf "$R"; cp -r "$OUT" "$R"; chmod -R a-w "$R"
  ls -la "$R" >> "$LOGDIR/wu1_files.txt"
  export COMPET_SUBMISSION_DIR="$R"
  step wu1_replay 60m -s "$R/submission.py" \
      && note "wu1_replay score $(score ${TAG}_wu1_replay) (must equal wu1_train)"
  unset COMPET_SUBMISSION_DIR

  # 3. zip, files at the root
  Z="$HOME/codabench/submissions/eegnet_steptype_wu1_$(date +%F).zip"
  python - "$OUT" "$Z" <<'EOF'
import sys, zipfile, pathlib
src, dst = pathlib.Path(sys.argv[1]), sys.argv[2]
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(src.iterdir()):
        if f.is_file():
            z.write(f, arcname=f.name)
print(dst, [i.filename for i in zipfile.ZipFile(dst).infolist()])
EOF
  note "zipped: $(ls -la $HOME/codabench/submissions/ | tail -n +4 | awk '{print $5, $9}' | tr '\n' ' ')"
fi

# 4. longer training
step eegnet_200 240m -o "$OBJ" \
    -s "$S/eegnet_steptype.py[n_epochs=200,patience=30,seed=[33,34,35]]"

python "$HOME/codabench/scripts/summarize_runs.py" \
    tracks/bci_decoding/outputs/overnight_2026-09-24_*.parquet > "$LOGDIR/RESULTS_all.md" 2>&1
note "ALL DONE (RESULTS_all.md written)"
