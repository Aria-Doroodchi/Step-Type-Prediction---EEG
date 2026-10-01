#!/usr/bin/env bash
# Sprint 2026-10-01, Phase 1b: a deployment-test candidate for the warm-up
# (SUBMISSIONS.md "Recommended next upload"). Riemann-Sealed with the recipe's
# blocks (xblocks="bpt4", baked as a default like train_sealed.sh does) trained
# on the warm-up study Dreyer 2023, replayed inference-only from a read-only
# copy, zipped at the zip root. The score does not matter (Dreyer is
# cross-subject); the point is that the sklearn/pyriemann joblib loads and
# predicts on the scoring image. NOTHING IS UPLOADED.
#   wsl.exe bash -lc 'bash ~/codabench/scripts/sprint1001_deploytest.sh'
TAG=s1001
export XS_THREADS=${XS_THREADS:-10}
source "$HOME/codabench/scripts/sealed_lib.sh"
D="$LOGDIR/deploytest"; mkdir -p "$D"
OUT="$D/submission"
CAND="$D/riemann_sealed_cand.py"
DATASET="BCI[study=dreyer2023]"
sed -e 's/"xblocks": \[""\]/"xblocks": ["bpt4"]/' \
    -e 's/name = "Riemann-Sealed"/name = "Riemann-Sealed-Cand-deploytest"/' \
    "$HOME/codabench/solvers/bci_decoding/riemann_sealed.py" > "$CAND"
grep -qF '"xblocks": ["bpt4"]' "$CAND" \
  || { echo "| $(date +%T) | ERROR: xblocks not baked |" >> "$STATUS"; exit 1; }
cd "$HOME/codabench/2026-competition" || exit 1
[ -f "$LOGDIR/dt_train.done" ] || rm -f "$LOGDIR/dt_replay.done"
COMPET_SUBMISSION_DIR="$OUT" step dt_train 30 120m benchopt run tracks/bci_decoding \
    -d "$DATASET" -s "$CAND" -o "BCI-decoding[training=True]" \
    --no-plot --no-html --no-cache --output s1001_deploytest_train
[ -f "$LOGDIR/dt_train.done" ] || exit 1
R=/tmp/s1001_deploytest_replay; [ -d "$R" ] && chmod -R u+w "$R"; rm -rf "$R"
cp -r "$OUT" "$R"; chmod -R a-w "$R"
COMPET_SUBMISSION_DIR="$R" step dt_replay 15 60m benchopt run tracks/bci_decoding \
    -d "$DATASET" -s "$R/submission.py" --no-plot --no-html --no-cache \
    --output s1001_deploytest_replay
[ -f "$LOGDIR/dt_replay.done" ] || exit 1
saved() {
  grep -ao 'Saving result in: [^ ]*\.parquet' "$LOGDIR/$1.log" 2>/dev/null | tail -1 \
    | sed 's/^Saving result in: //'
}
python - "$(saved dt_train)" "$(saved dt_replay)" "$OUT" "$D" > "$D/RESULTS.md" 2>&1 <<'PY'
import pathlib, sys, zipfile, datetime
import joblib
import pandas as pd
pt, pr, out, d = sys.argv[1:]
tr = float(pd.read_parquet(pt)["objective_balanced_accuracy"].iloc[-1])
rp = float(pd.read_parquet(pr)["objective_balanced_accuracy"].iloc[-1])
same = "EQUAL" if tr == rp else "DIFFERENT"
print("# Deployment-test candidate (sprint 2026-10-01 Phase 1b; NOT uploaded)\n")
print(f"| check | value |\n|---|---|\n| benchopt train ({pathlib.Path(pt).name}) | {tr:.6f} |")
print(f"| read-only replay ({pathlib.Path(pr).name}) | {rp:.6f} ({same}) |")
out = pathlib.Path(out)
parts = joblib.load(out / "riemann_sealed.joblib")
types = {}


def walk(v, key):
    if isinstance(v, dict):
        for k, x in v.items():
            walk(x, f"{key}.{k}" if key else str(k))
    elif isinstance(v, (list, tuple)):
        for x in v:
            walk(x, key + "[]")
    else:
        t = type(v)
        types.setdefault(f"{t.__module__}.{t.__qualname__}", set()).add(key)


walk(parts, "")
print("\n## Object types in the joblib\n\n| type | keys |\n|---|---|")
for t in sorted(types):
    print(f"| `{t}` | {', '.join(sorted(types[t]))[:200]} |")
if same == "EQUAL":
    z = pathlib.Path(d) / f"riemann_sealed_deploytest_dreyer2023_{datetime.date.today()}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        for p in sorted(out.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                f.write(p, p.relative_to(out))
    with zipfile.ZipFile(z) as f:
        print(f"\n## Zip `{z.name}` ({z.stat().st_size / 2**20:.1f} MiB)\n")
        for i in f.infolist():
            print(f"- `{i.filename}` {i.file_size / 2**20:.2f} MiB")
else:
    print("\nNO ZIP: train != replay")
PY
echo "| $(date +%T) | deploytest summary written ($D/RESULTS.md) |" >> "$STATUS"
echo "| $(date +%T) | ALL DONE deploytest |" >> "$STATUS"
