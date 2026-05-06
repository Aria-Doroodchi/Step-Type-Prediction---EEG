# Workflow guide

How the reorganized pipeline actually works in day-to-day use.

---

## The mental model

```
configs/   ───►   src/eeg_steptype/   ───►   data/   ───►   outputs/
(what to run)     (the code)                 (cached state)  (results)
```

Three layers, decoupled:

1. **Configs are the source of truth.** All paths, participant lists,
   filter cutoffs, hyper-parameter grids, model choice — every knob is in
   `configs/default.yaml` (and per-participant override files). You don't
   edit code to change a run; you edit a YAML.
2. **Code is the engine.** `src/eeg_steptype/` is an installed package.
   Stages are pure functions: give them a config, they read inputs from
   `data/`, write outputs to `data/` or `outputs/`.
3. **Outputs are stamped and reproducible.** Every training run snapshots
   its config + git SHA next to its metrics, so any past result can be
   re-run.

The pipeline is **four stages**, each cached on disk so the slow ones
don't repeat:

```
[raw .bdf]  ──1─►  [cleaned epoch .fif]  ──2─►  [src CSV]  ──3─►  [features parquet]  ──4─►  [metrics CSV]
              preprocess              source-localize        feature-extract        train
              (~10 min/pid)           (~30 min/pid)          (~2 min/pid)           (hours, model-dep)
```

---

## First-time setup (once per machine)

```bash
# 1. Install the package
pip install -e .[dev]

# 2. Tell the pipeline where your raw data lives
cp configs/local.yaml.example configs/local.yaml
#    Edit the one line: paths.raw_root

# 3. Verify everything imports and the synthetic smoke pipeline runs
make test
#    ~30s; pure pytest, doesn't touch real data

# 4. End-to-end on 1 real participant with a tiny model
make smoke
#    ~1–2 min; runs preprocess → src → features → train(logistic) on P25
```

If `make test` and `make smoke` both pass, the pipeline is wired up
correctly on your machine.

---

## Daily workflow

### Scenario A — full cohort run from scratch

```bash
make all                       # = preprocess → src → features → train (xgb default)
```

This is overnight territory: ~30 participants × all four stages × the
default XGBoost grid. Each stage skips participants whose outputs already
exist, so killing it halfway and resuming `make all` later just continues
where it left off.

### Scenario B — try a different model, same features

```bash
make train MODEL=lstm          # or svm, logistic, xgb
```

Stage 4 only. Reads the cached feature parquets (stage 3 result) — does
NOT re-run preprocessing or LORETA. Each model run writes a new stamped
folder under `outputs/runs/`.

### Scenario C — change a hyper-parameter and re-train

```yaml
# Edit configs/default.yaml
modeling:
  rfecv:
    n_iterations: 10            # was 5
  shap_prune_quantile: 0.30     # was 0.20
```

```bash
make train MODEL=xgb
```

Same as Scenario B — features stay cached, only the model retrains.

### Scenario D — debug one participant

You notice P14's metrics look off; you suspect the ICA component pick
needs tweaking.

```yaml
# Edit configs/overrides/P14.yaml — uncomment the legacy block as a base
ica:
  manual_exclude: [0, 1, 5, 8, 12, 13]   # add some, drop others
```

```bash
# Re-run only that participant from preprocessing forward
python scripts/01_preprocess.py    --participants P14 --force
python scripts/03_extract_features.py --participants P14 --force
python scripts/04_train.py         --model xgb
```

`--force` tells the stage to overwrite the cached output for those pids.
The stage 4 train pass picks up the new P14 features automatically.

### Scenario E — add a brand-new participant (e.g. P40)

```bash
# 1. Drop the raw recording into your raw_root:
#    {raw_root}/P40/P40_CNV.bdf

# 2. Add the participant ID to the cohort:
#    edit configs/default.yaml → participants: [..., P40]

# 3. (Optional) If the recording needed manual surgery, write
#    configs/overrides/P40.yaml — see configs/overrides/README.md
#    for the schema (raw_assembly with cuts/concat, bads_extra, etc.)

# 4. Run only that participant through every stage:
python run.py --participants P40
```

If the participant has no override file, the pipeline uses the defaults
(single-file load, no crops, auto-detected bads, ICLabel auto-exclude,
autoreject thresholds).

### Scenario F — quick sanity check on changes you just made to the code

```bash
make test
```

This runs `tests/test_imports.py` (every module imports + 9 per-pid
override spot-checks) and `tests/test_smoke_pipeline.py` (synthetic-data
end-to-end through features → train). Sub-minute. Run this any time you
edit anything under `src/eeg_steptype/` to catch breakage early.

---

## How config flows through

All three YAMLs are deep-merged at load time:

```
configs/default.yaml   ←   configs/local.yaml   ←   configs/overrides/Pxx.yaml
   (committed)              (gitignored,                (per-participant,
   project defaults)         per-machine paths)          applied only when
                                                         processing Pxx)
```

So when the preprocessing stage processes P02, the config it sees is:

```python
defaults  +  your-machine paths  +  P02-specific overrides
```

…all merged into one dict. Right-hand wins on conflicts. Per-participant
overrides only apply when that participant is the active one — the cohort
loop reaches in and out of those overrides on each iteration.

**The key consequence:** every per-participant peculiarity (P02's two-file
concat, P08's two-window crop, P37's electrode swap) lives in exactly one
place and only affects that participant.

---

## What you get back

Every training run creates `outputs/runs/<run_id>/`:

```
outputs/runs/xgb_20260501_143022/
├── config.yaml      # full merged config snapshot — exactly what was used
├── git_sha.txt      # repo commit at run time
├── env.json         # python version, platform, argv
├── metrics.csv      # one row per participant: accuracy, AUC, best params
├── rollup.csv       # cohort totals
└── per_participant_accuracy.png   # (after running scripts/05_visualize.py)
```

To reproduce a result months later:

```bash
git checkout <git_sha from that run>
python run.py --config outputs/runs/xgb_20260501_143022/config.yaml
```

Same code, same config, same data → same result.

---

## Where things live (for poking around)

| You want to … | Look at |
|---|---|
| Change the participant list | `configs/default.yaml` → `participants:` |
| Change model hyperparameters | `configs/default.yaml` → `modeling.{xgb,svm,lstm}.param_grid` |
| Tweak filter cutoffs project-wide | `configs/default.yaml` → `preprocessing.filter` |
| Tweak filter cutoffs for one participant | `configs/overrides/Pxx.yaml` |
| Change which feature blocks compose | `configs/default.yaml` → `features.blocks` |
| Add a new model type | `src/eeg_steptype/models/<name>.py` + register in `train.py` |
| Add a new feature block | `src/eeg_steptype/features/<name>.py` + add to `assemble.py` |
| Change LORETA settings | `configs/default.yaml` → `source_localization` |
| Change automated ICA strictness | `configs/default.yaml` → `preprocessing.ica.iclabel_artifact_prob_threshold` |
| Override one participant's ICA picks | `configs/overrides/Pxx.yaml` → `ica.manual_exclude` / `ica.manual_keep` |
| See what got cached on disk | `data/interim/`, `data/src/`, `data/features/` |
| See past run results | `outputs/runs/` |

---

## Stage-by-stage cheat sheet

```bash
# Stage 1 — raw .bdf → cleaned epoch .fif (per pid, per condition)
python scripts/01_preprocess.py [--participants P01 P02] [--force]
#   reads: {raw_root}/{pid}/{pid}_CNV.bdf
#   writes: data/interim/epochs/{pid}_CNV_{One,Two}-epo.fif

# Stage 2 — cleaned epochs → source-localized CSV
python scripts/02_source_localize.py [--participants P01] [--force]
#   reads:  data/interim/epochs/...
#   writes: data/src/{pid}_{One,Two}_src.csv
#   caches: data/src/fwd/{pid}-fwd.fif

# Stage 3 — epochs (+ src CSV) → wide feature matrix
python scripts/03_extract_features.py [--participants P01] [--force]
#   reads:  data/interim/epochs/..., data/src/...
#   writes: data/features/{pid}_{One,Two}_features.parquet

# Stage 4 — features → trained model + metrics
python scripts/04_train.py --model xgb [--config configs/smoke.yaml] [--run-id custom]
#   reads:  data/features/...
#   writes: outputs/runs/<run_id>/

# Stage 5 — metrics CSV → plot
python scripts/05_visualize.py --run outputs/runs/xgb_20260501_143022
```

Each script reads `--config` (defaults to `configs/default.yaml`), merges
in `configs/local.yaml`, and per-participant overrides where applicable.

---

## Troubleshooting

| Symptom | Likely fix |
|---|---|
| `ModuleNotFoundError: eeg_steptype` | `pip install -e .` |
| `FileNotFoundError: ... .bdf` | `paths.raw_root` in `configs/local.yaml` is wrong |
| Stage skips a participant unexpectedly | Output already exists; pass `--force` to overwrite |
| `autoreject failed` warning | Falls back to threshold method; check the participant's override has `preprocessing.reject.thresholds` |
| ICA over- or under-rejecting | Adjust `preprocessing.ica.iclabel_artifact_prob_threshold` globally, or `ica.manual_exclude` per participant |
| Training takes too long | Run `make smoke` first; if that passes, the slow path is the GridSearchCV grid — shrink it in `configs/default.yaml` |
| Want to test a code change quickly | `make test` (1 min) before launching real runs |
