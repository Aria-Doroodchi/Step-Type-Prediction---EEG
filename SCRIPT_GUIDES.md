# Script Guides

This is the operator's reference for the EEG step-type pipeline. It covers
three things:

1. **Environment setup** — how to bring up the `.venv` from a fresh clone.
2. **Variables reference** — every CLI flag exposed by the entry-point
   scripts, and what each one does.
3. **Example commands** — categorized recipes for the runs you'll actually
   want to do, from a one-line smoke test to a parallel cohort training pass.

All commands assume you are at the repository root:

```powershell
cd "C:\Users\Ali D\Documents\ML"
```

---

## 1. Environment Setup (.venv)

The project ships a `pyproject.toml` that lists every required dependency.
The recommended setup is a per-clone virtual environment in `.venv/`.

### 1.1 First-time setup

PowerShell (Windows, this machine's default):

```powershell
# 1. Create a fresh virtual env in .venv/
python -m venv .venv

# 2. Activate it. After this, "python" and "pip" refer to .venv's copies.
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install the project in editable mode with all extras.
python -m pip install --upgrade pip
pip install -e ".[lstm,riemannian,dev]"
```

`-e` (editable) means your local source edits in `src/eeg_steptype/` take
effect immediately without reinstalling. The bracketed extras pull in the
optional dependency groups defined in `pyproject.toml`:

| Extra        | Adds                                | When to install                              |
|--------------|-------------------------------------|----------------------------------------------|
| `lstm`       | `tensorflow`, `scikeras`            | Required to run `--model lstm`.              |
| `riemannian` | `pyriemann`                         | Required for the Riemannian comparator stub. |
| `dev`        | `pytest`, `ruff`                    | Required to run the test suite and linter.   |

If you don't need a group, skip it: `pip install -e ".[dev]"` is enough for
running classical models and tests.

Bash / Linux / WSL equivalent:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[lstm,riemannian,dev]"
```

### 1.2 Day-to-day activation

Once `.venv/` exists, the only command you need at the start of each session
is the activation line:

```powershell
.\.venv\Scripts\Activate.ps1                  # PowerShell
# or
.\.venv\Scripts\activate.bat                  # cmd.exe
# or
source .venv/bin/activate                     # bash / zsh
```

After activation, `python --version` should report Python 3.10+ and
`where.exe python` (PowerShell) or `which python` (bash) should point inside
`.venv/`.

### 1.3 Configure your local paths

The pipeline reads raw `.bdf` data from a path that's user-specific. Override
it in `configs/local.yaml` (gitignored). If the file doesn't exist yet:

```powershell
# Minimum local.yaml: tell the pipeline where your raw data lives.
@"
paths:
  raw_root: "C:/Users/Ali D/Documents/ML_data/Participants"
"@ | Set-Content configs/local.yaml
```

Any other key you set in `configs/local.yaml` will be merged on top of
`configs/default.yaml` for every run, so it's also a good place to override
per-machine resource settings (e.g. `resources.n_jobs`) that you don't want
to retype on the CLI.

### 1.4 Verify the install

Run the import smoke test — it loads every package module without touching
real data and finishes in under a second:

```powershell
python -m pytest tests\test_imports.py -q
```

Then run the preflight check, which validates that the source-localization
assets and paths in your config resolve correctly:

```powershell
python scripts\00_preflight.py --config configs/default.yaml
```

Expected output ends with `Preflight OK` and a few resolved paths. If you see
a `FileNotFoundError` for `source_localization.*`, the message tells you
which key to set in `configs/local.yaml`.

### 1.5 Common setup issues

`meegkit` / `numpy 2.x` `ValueError: setting an array element with a sequence` —
already patched in `src/eeg_steptype/preprocessing/asr.py`. No action needed.

LSTM `ModuleNotFoundError: tensorflow` — you skipped the `lstm` extra. Re-run
`pip install -e ".[lstm]"`.

`ModuleNotFoundError: eeg_steptype` — you forgot the `-e .` install step.
Re-run `pip install -e .` from the repo root.

---

## 2. Variables Reference

Every flag accepted by the entry-point scripts. Most live on `run.py`; the
per-stage scripts (`01_preprocess.py`, `02_source_localize.py`,
`03_extract_features.py`, `04_train.py`, `05_visualize.py`) accept a strict
subset relevant to their stage.

### 2.1 Config / overlay flags

| Flag | Values | Default | Effect |
|---|---|---|---|
| `--config PATH` | path to a YAML file | `configs/default.yaml` (implicit) | Layered overlay on top of `default.yaml` and `local.yaml`. Right-most overlay wins on key conflicts. |
| `--speed-tier NAME` | `lightning`, `express`, `quick`, `riemannian` | (none) | Shortcut for `--config configs/<tier>.yaml`. Ignored if `--config` is also passed. `lightning` / `express` / `quick` are XGB-family speed trims; `riemannian` is a separate model + data path (epoch tensors instead of the flat parquet). See [§3.5](#35-speed-tiered-runs). |
| `--prediction-window NAME` | named window from `prediction_windows` in config, e.g. `late_cnv`, `full_cnv` | `late_cnv` | Overrides `features.min_time` / `features.max_time` for this run only. Affects feature extraction and training. |
| `--participant-override-mode MODE` | `raw_assembly_only`, `full`, `none` | `raw_assembly_only` | How aggressively to apply per-participant YAMLs from `configs/overrides/Pxx.yaml`. `raw_assembly_only` keeps preprocessing uniform across the cohort. `full` opts into every per-participant tweak. `none` ignores the override files entirely. |

### 2.2 Cohort selection

| Flag | Values | Default | Effect |
|---|---|---|---|
| `--participants P01 P02 ...` | space-separated participant IDs | `cfg["participants"]` (full cohort) | Restrict the run to a subset of participants. Applies to every stage including `train` (via `run.py`). |

### 2.3 Stage / model selection

| Flag | Values | Default | Effect |
|---|---|---|---|
| `--stages STAGE ...` | subset of `preprocess`, `src`, `features`, `train` | all four | Which pipeline stages to execute, in the listed order. Skipped stages assume their inputs already exist on disk. (`run.py` only.) |
| `--model NAME` | `xgb`, `svm`, `lstm`, `logistic`, `riemannian` | `cfg["modeling"]["default_model"]` if set (e.g. `riemannian` under `--speed-tier riemannian`), otherwise `xgb` | Which model factory to use during the `train` stage. See `src/eeg_steptype/models/README.md` for architectures. `riemannian` uses the epoch-tensor cache at `data/features/tensor/`, not the flat feature parquet. |
| `--channel-mode MODE` | `full`, `roi` | `cfg["channel_selection"]["mode"]` (default `full`) | Train on every electrode (`full`) or only the medial foot-motor ROI defined under `channel_selection.roi.channels` (`roi`). Always `full` for tensor-input models (`lstm`, future `cnn`). |
| `--cv-mode MODE` | `repeated_stratified`, `grouped`, `chronological` | `cfg["modeling"]["cv"]["mode"]` (default `repeated_stratified`) | Outer cross-validation strategy. `grouped` uses `block_id` so trials from the same recording block stay together. `chronological` is a no-shuffle temporal sanity check. |
| `--run-id NAME` | any string | auto-generated `<model>_<channel_mode>_<window>_<timestamp>` | Name of the output directory under `outputs/runs/`. Useful for resuming a run from per-participant CSV checkpoints. |

### 2.4 Resources / parallelism

| Flag | Values | Default | Effect |
|---|---|---|---|
| `--n-jobs N` | integer (negative = reserve that many cores) | `cfg["resources"]["n_jobs"]` (default `-8`) | Intra-participant thread budget. Honored by preprocess/src/features and by sequential training. Inside parallel-participants workers the inner threads are still pinned to 1. |
| `--parallel-participants N` | integer (negative = reserve that many cores) | `cfg["modeling"]["parallel"]["participants"]` (unset in `default.yaml` → sequential; `-8` for lightning/express, `-10` for quick) | Number of participants trained concurrently with joblib during the `train` stage. Follows the same negative-means-reserve convention as `--n-jobs`: `-8` on a 16-core box = 8 workers, on a 32-core box = 24 workers. Only affects the train stage; upstream stages remain sequential. |

### 2.5 Cache control

| Flag | Values | Default | Effect |
|---|---|---|---|
| `--force` | flag | off | Regenerate every stage's outputs from scratch, ignoring cached `.fif` / `.csv` / `.parquet` files. Stage-scoped: `--stages features --force` only rebuilds feature parquets. |

### 2.6 Visualization-only flags

`scripts/05_visualize.py` adds:

| Flag | Values | Default | Effect |
|---|---|---|---|
| `--run PATH` | path to a `outputs/runs/<id>/` directory | (required) | Which run's `metrics.csv` to plot. |

### 2.7 Flag availability matrix

Per-stage scripts accept only the flags relevant to their stage. `run.py`
accepts everything.

| Flag                           | `run.py` | `00_preflight` | `01_preprocess` | `02_source_localize` | `03_extract_features` | `04_train` | `05_visualize` |
|---|---|---|---|---|---|---|---|
| `--config`                     | yes | yes | yes | yes | yes | yes | yes |
| `--speed-tier`                 | yes | —   | —   | —   | —   | yes | — |
| `--prediction-window`          | yes | —   | —   | —   | yes | yes | — |
| `--participant-override-mode`  | yes | —   | yes | yes | yes | yes | yes |
| `--participants`               | yes | —   | yes | yes | yes | —   | — |
| `--stages`                     | yes | —   | —   | —   | —   | —   | — |
| `--model`                      | yes | —   | —   | —   | —   | yes | — |
| `--channel-mode`               | yes | —   | —   | —   | —   | yes | — |
| `--cv-mode`                    | yes | —   | —   | —   | —   | yes | — |
| `--run-id`                     | yes | —   | —   | —   | —   | yes | — |
| `--n-jobs`                     | yes | —   | —   | —   | —   | yes | — |
| `--parallel-participants`      | yes | —   | —   | —   | —   | yes | — |
| `--force`                      | yes | —   | yes | yes | yes | —   | — |
| `--run`                        | —   | —   | —   | —   | —   | —   | yes |

---

## 3. Example Commands

Every command below assumes the `.venv` is activated. If it isn't, prefix
`python` with the explicit path (`& '.\.venv\Scripts\python.exe' ...` in
PowerShell).

### 3.1 Health checks (no real data)

Import smoke test — verifies the package imports cleanly:

```powershell
python -m pytest tests\test_imports.py -q
```

Full smoke pipeline — runs the maintained stage flow on tiny synthetic
inputs in a temporary directory:

```powershell
python -m pytest tests\test_smoke_pipeline.py -q
```

Preflight on a real config — validates paths, source-localization assets,
and participant override files:

```powershell
python scripts\00_preflight.py --config configs/default.yaml
```

### 3.2 Single-participant runs

Fastest possible end-to-end pass — useful for "does my pipeline still work
on this participant" checks:

```powershell
python run.py --speed-tier lightning --participants P25 --model xgb
```

Single participant, near-research-grade training (10-15 min):

```powershell
python run.py --speed-tier quick --participants P25 --model xgb
```

Single participant, full default config (slow; use for archival runs):

```powershell
python run.py --participants P25 --model xgb
```

### 3.3 Multi-participant subsets

A handful of participants, balanced speed tier:

```powershell
python run.py --speed-tier express --participants P01 P02 P03 P04 P05 --model xgb
```

Subset, training stage only (when upstream stages are already cached):

```powershell
python run.py --speed-tier express --participants P01 P02 P03 P04 P05 --model xgb --stages train
```

### 3.4 Full-cohort runs

Full cohort with default research-grade settings:

```powershell
python run.py --config configs/default.yaml --model xgb
```

Full cohort, fast tier, parallel training:

```powershell
python run.py --speed-tier express --parallel-participants 8 --model xgb
```

Full cohort, train-only after data has been cached once:

```powershell
python run.py --speed-tier express --parallel-participants 8 --model xgb --stages train
```

### 3.5 Speed-tiered runs

The four overlay tiers (see `configs/README.md` for the full trade-off table):

```powershell
# Lightning -- ~1-3 min/participant, AUC drop ~5-10%, dev loop only
python run.py --speed-tier lightning --participants P25 --model xgb

# Express   -- ~4-8 min/participant, AUC drop ~2-4%, daily driver
python run.py --speed-tier express --participants P25 --model xgb

# Quick     -- ~10-15 min/participant, AUC drop <1.5%, near-research-grade
python run.py --speed-tier quick --participants P25 --model xgb

# Riemannian -- covariance + xDAWN + tangent-space + shrinkage LDA comparator.
# Different data path (epoch tensors instead of the flat feature parquet) and
# default window (full_cnv 0-2 s). Tier sets default_model: riemannian so the
# --model flag is optional.
python run.py --speed-tier riemannian --participants P25
```

### 3.6 Stage-by-stage runs

Useful when you want to debug a single stage or run stages on different
machines.

Preprocess only:

```powershell
python run.py --speed-tier express --participants P25 --stages preprocess
# Equivalent direct call:
python scripts\01_preprocess.py --participants P25 --config configs/default.yaml
```

Source localization only:

```powershell
python run.py --speed-tier express --participants P25 --stages src
# Direct:
python scripts\02_source_localize.py --participants P25
```

Feature extraction only:

```powershell
python run.py --speed-tier express --participants P25 --stages features
# Direct:
python scripts\03_extract_features.py --participants P25 --prediction-window late_cnv
```

Train only (assumes features already cached):

```powershell
python run.py --speed-tier express --participants P25 --stages train --model xgb
# Direct:
python scripts\04_train.py --speed-tier express --model xgb
```

### 3.7 Model comparator sweeps

Same data, different models:

```powershell
python run.py --speed-tier quick --participants P25 --model xgb     --stages train
python run.py --speed-tier quick --participants P25 --model svm     --stages train
python run.py --speed-tier quick --participants P25 --model logistic --stages train
python run.py --speed-tier quick --participants P25 --model lstm    --stages train
```

PowerShell loop equivalent:

```powershell
foreach ($m in @("xgb","svm","logistic")) {
    python run.py --speed-tier quick --model $m --stages train
}
```

### 3.8 Channel-set and ROI comparisons

Train on every electrode (default):

```powershell
python run.py --speed-tier express --participants P25 --model xgb --channel-mode full
```

Train on the medial foot-motor ROI only:

```powershell
python run.py --speed-tier express --participants P25 --model xgb --channel-mode roi
```

### 3.9 Prediction-window analyses

Late CNV (primary, default):

```powershell
python run.py --speed-tier express --participants P25 --prediction-window late_cnv
```

Full 0-2 s window (secondary analysis):

```powershell
python run.py --speed-tier express --participants P25 --prediction-window full_cnv
```

### 3.10 Cross-validation strategy switches

Repeated stratified (default — random splits):

```powershell
python run.py --speed-tier express --participants P25 --cv-mode repeated_stratified
```

Grouped by recording block — keeps trials from the same block together:

```powershell
python run.py --speed-tier express --participants P25 --cv-mode grouped
```

Chronological — no shuffle, train on earlier trials, test on later:

```powershell
python run.py --speed-tier express --participants P25 --cv-mode chronological
```

### 3.11 Participant override modes

Default — uniform preprocessing, only raw `.bdf` assembly per-participant:

```powershell
python run.py --speed-tier express --participants P02 --participant-override-mode raw_assembly_only
```

Full — opt into every per-participant tweak in `configs/overrides/Pxx.yaml`:

```powershell
python run.py --speed-tier express --participants P37 --participant-override-mode full
```

None — ignore participant YAMLs entirely:

```powershell
python run.py --speed-tier express --participants P25 --participant-override-mode none
```

### 3.12 Resource / parallelism tuning

Use a specific intra-participant thread count (reserve 8 cores):

```powershell
python run.py --speed-tier express --participants P25 --n-jobs -8 --model xgb
```

Train 6 participants concurrently (each pinned to 1 thread internally):

```powershell
python run.py --speed-tier express --parallel-participants 6 --model xgb
```

Force sequential training (one participant at a time, full thread budget):

```powershell
python run.py --speed-tier express --parallel-participants 1 --n-jobs -8 --model xgb
```

### 3.13 Forced rebuilds (cache busting)

Regenerate everything from raw `.bdf`:

```powershell
python run.py --speed-tier express --participants P25 --model xgb --force
```

Rebuild just the feature parquets for a participant (e.g. after a feature-
extraction bug fix):

```powershell
python run.py --speed-tier express --participants P25 --stages features --force
```

### 3.14 Visualization

Plot the per-participant accuracy chart for a completed run:

```powershell
python scripts\05_visualize.py --run outputs\runs\xgb_full_late_cnv_20260513_122000
```

### 3.15 Make shortcuts

If `make` is on your PATH, these one-liners are equivalent to the most
common commands:

```bash
make smoke-test       # tests/test_imports.py + tests/test_smoke_pipeline.py
make smoke-test-two   # tests/test_smoke_pipeline.py::test_two_participant_full_workflow_smoke
make preflight        # scripts/00_preflight.py with configs/default.yaml
make full-xgb         # preflight + run.py --config configs/default.yaml --model xgb
```

Pass `OVERRIDE_MODE=full` to `make full-xgb` to opt into per-participant
tuning (`make full-xgb OVERRIDE_MODE=full`).

---

## 4. Watching Progress During A Run

The training driver logs a startup banner (total outer folds and inner-grid
size), a "starting fold X/Y" line at the top of each outer fold, and a
"done in T.Ts" line when the fold finishes. For the tensor-input path
(Riemannian, future CNN) it also logs the loaded tensor shape and per-class
sample count before the CV loop begins.

**Live progress requires sequential training.** When `--parallel-participants`
resolves to a worker count greater than 1, joblib's `LokyBackend` captures
each worker's stdout and stderr and only releases it back to the parent
process when the worker returns. So for a *single-participant* run you'll see
the MNE epoch-read messages from before the worker started, then nothing
until the participant finishes. Two easy ways out:

```powershell
# Force sequential training -- workers stream live
python run.py --speed-tier riemannian --participants P25 --parallel-participants 1

# Or set the same thing via the per-run config: parallel.participants = 1
```

For *cohort* runs you can keep the parallel pool — each participant's log
batch arrives as that worker finishes. It's only the single-participant case
where parallelism hides the entire run from view.

**Per-candidate progress from sklearn.** If you want the inner hyperparameter
search to log progress as well (one line per candidate × inner-CV-fold pair),
set `modeling.search.verbose` in your config. Defaults to 0 (silent).

```yaml
# In configs/local.yaml, or temporarily in the active tier overlay:
modeling:
  search:
    verbose: 1     # 1 = per-candidate, 3 = per-candidate × inner-fold
```

**Diagnosing a slow Riemannian fold.** The Riemannian pipeline's per-fold cost
is roughly: covariance estimation on `n_epochs × n_channels × n_times`
floats, then xDAWN fitting (eigendecomposition + spatial filter learning),
then tangent-space projection (matrix logarithm per epoch), then shrinkage
LDA. On a single participant with the default `nfilter × covariance estimator`
grid (6 candidates), 5×5 outer CV, and 3 inner folds, that's roughly 450 inner
fits + 25 outer refits — 15–25 minutes is typical even on a fast workstation.
The new per-fold logs let you see the per-fold timing and extrapolate.

---

## 5. Output Locations

A successful run writes to:

```text
data/interim/epochs/<pid>_<cond>_cleaned-epo.fif       # preprocess output
data/src/<pid>_<cond>_src.csv                          # source-localization output
data/features/<pid>_<cond>_features_t<min>-<max>.parquet  # tabular feature cache (xgb/svm/logistic/lstm)
data/features/tensor/<pid>_<cond>_epochs_t<min>-<max>.npz # epoch-tensor cache (riemannian, future cnn)
outputs/qc/<pid>.html                                  # per-participant QC report
outputs/runs/<run_id>/participants/<pid>_metrics.csv   # per-participant checkpoint
outputs/runs/<run_id>/metrics.csv                      # cohort metrics
outputs/runs/<run_id>/rollup.csv                       # cohort rollup
outputs/runs/<run_id>/config.yaml                      # snapshot of the merged config
```

Per-participant checkpoint CSVs make every run resumable: re-running the
same command with the same `--run-id` will skip participants that already
have a `metrics.csv` in `participants/`.

---

## 6. Cached Outputs and Cross-Run Reuse

Every pipeline stage writes its outputs to disk and, on the next run, checks
whether the file already exists before recomputing. This means a *lot* of
work is reusable when you switch models or rerun a participant. This section
catalogs exactly what each run saves, which artifacts are portable across
models, and how to take advantage of the cached portions to save time.

### 6.1 What each stage saves, and to whom it belongs

| Stage                | Artifact path                                                     | Model-specific? | Reused when changing model?                    |
|----------------------|-------------------------------------------------------------------|-----------------|------------------------------------------------|
| 1. Preprocess        | `data/interim/epochs/<pid>_CNV_<cond>-epo.fif`                    | No              | Yes — every model reads the same epoch file.  |
| 2. Source localize   | `data/src/<pid>_<cond>_src.csv`                                   | No              | Yes — both classical and Riemannian use it.   |
| 3a. Features (flat)  | `data/features/<pid>_<cond>_features_t<min>-<max>.parquet`        | Partially       | Yes for `xgb` / `svm` / `logistic` / `lstm`.  |
| 3b. Features (tensor)| `data/features/tensor/<pid>_<cond>_epochs_t<min>-<max>.npz`       | Partially       | Yes for `riemannian` (and future `cnn`).      |
| Preprocess QC        | `outputs/qc/<pid>.html`                                           | No              | Yes — written once, valid for every model.    |
| 4. Train (checkpoint)| `outputs/runs/<run_id>/participants/<pid>_metrics.csv`            | **Yes**         | **No** — tied to one model × `--run-id`.      |
| 4. Train (cohort)    | `outputs/runs/<run_id>/metrics.csv`, `rollup.csv`                 | **Yes**         | **No** — comparison artifacts only.           |
| 4. Reproducibility   | `outputs/runs/<run_id>/config.yaml`, `git_sha.txt`, `env.json`    | **Yes**         | **No** — per-run snapshot.                    |

So three of the four stages (preprocess, source localization, feature
extraction) cache outputs that are entirely reusable across model choices.
The fourth stage (training) is the only one that has to rerun every time
you change the model, the window, the CV settings, or any of the other
modeling knobs.

### 6.2 Two flavors of feature cache, side by side

The features stage has two output flavors because tensor-input models
(Riemannian and future CNN) need raw `(n_epochs × n_channels × n_times)`
data, while classical models work on the flat per-bin aggregates:

| Cache file                                                              | Built by stage 3 for…              | Consumed by                           |
|-------------------------------------------------------------------------|------------------------------------|---------------------------------------|
| `data/features/<pid>_<cond>_features_t<min>-<max>.parquet`              | `assemble.build_for_participant`   | `xgb`, `svm`, `logistic`, `lstm`      |
| `data/features/tensor/<pid>_<cond>_epochs_t<min>-<max>.npz`             | `tensor.build_tensor_for_participant` | `riemannian` (and future `cnn`)   |

Both are keyed by the prediction window (`_t<min>-<max>`), so the same
participant can have, for example, `features_t1p0-2p0.parquet` for the
default late-CNV window *and* `epochs_t0p0-2p0.npz` for the Riemannian
full-CNV window coexisting on disk without colliding. The two caches are
independent: switching from `xgb` to `svm` reuses the parquet without
needing the tensor, and switching from `xgb` to `riemannian` triggers a
lazy build of the tensor on first use without invalidating the parquet.

### 6.3 Skipping cached stages when switching models

Stages are skipped automatically when their output file already exists.
You don't have to do anything special — the default code path checks
`out.exists() and not force` at the top of every stage builder. The only
question is which stages you bother to *invoke* when you re-run.

The minimal re-run pattern when switching models on already-processed
participants:

```powershell
# First run: build everything end-to-end for a participant.
python run.py --speed-tier express --participants P25 --model xgb

# Switch to SVM (or logistic, or LSTM) -- only training has to run.
# Stages 1-3 short-circuit on the cached parquet.
python run.py --speed-tier express --participants P25 --model svm --stages train

# Switch to Riemannian -- still only training, but the tensor cache will be
# built on first use of build_tensor_for_participant inside the train stage.
python run.py --speed-tier riemannian --participants P25 --stages train
```

If you'd rather build the tensor cache up front (e.g. for a cohort run so
the first parallel worker doesn't get charged the full build), invoke the
features stage explicitly first:

```powershell
python run.py --speed-tier riemannian --participants P25 --stages features
python run.py --speed-tier riemannian --participants P25 --stages train
```

For a full comparator sweep on one participant, the data path is built
once and reused across all four classical models, with only training
charging real wall time per model:

```powershell
# Pay for stages 1-3 once.
python run.py --speed-tier express --participants P25 --model xgb

# Each subsequent model only pays for training.
foreach ($m in @("svm","logistic","lstm")) {
    python run.py --speed-tier express --participants P25 --model $m --stages train
}

# Riemannian builds its tensor cache on first use, then trains.
python run.py --speed-tier riemannian --participants P25 --stages train
```

### 6.4 Resuming an interrupted cohort run

Per-participant checkpoint CSVs at
`outputs/runs/<run_id>/participants/<pid>_metrics.csv` make every cohort
run resumable — but only for the *same model* and the *same run id*. If a
30-participant XGB run crashes after participant 18, you can pick up
where it left off:

```powershell
# Original run -- interrupted partway through.
python run.py --speed-tier express --model xgb --run-id sweep_xgb_v1

# Resume. Skips the 18 participants whose <pid>_metrics.csv already exists
# under outputs/runs/sweep_xgb_v1/participants/.
python run.py --speed-tier express --model xgb --run-id sweep_xgb_v1
```

Two important caveats. First, the checkpoint reuse is **per model, per
run-id**. If you change the `--model` flag you must also change the
`--run-id`; otherwise the new run will load the old model's checkpoints
and write them into `metrics.csv` mislabeled. Second, the checkpoints
don't capture intermediate fold state inside one participant — if the
crash happened mid-participant, that participant is redone from scratch.

### 6.5 What invalidates a cache

Stages short-circuit on file existence, not on content hash, so a cache
is "valid" as long as its output file is at the expected path. Things
that *do* invalidate a cache and force a rebuild:

The `--force` flag — short-circuits the existence check for whichever
stages you invoke, e.g. `python run.py --stages features --force` to
rebuild the parquet after fixing a bug in feature extraction.

A different prediction window — the parquet and tensor filenames include
the window range (`_t1p0-2p0.parquet`, `_t0p0-2p0.npz`). Switching from
`late_cnv` to `full_cnv` writes to a *different* filename, so both
windows can coexist on disk; you just pay for the second one on first
use.

A change to the preprocessing config that doesn't change the path — e.g.
flipping `ica.method`, `bads.method`, or any other preprocessing
parameter. The output path doesn't encode the config, so the cache will
silently reuse the old `.fif` even though your config changed. When this
happens, pass `--force` to rebuild from scratch.

A change to `participant_overrides.mode` from `raw_assembly_only` to
`full` — the same caveat applies. If your participant YAML carries
non-trivial preprocessing overrides, switching modes changes what gets
applied without changing the output path, so use `--force`.

### 6.6 Summary in one paragraph

For practical purposes: the heavy upstream stages (cleaning, source
localization, feature extraction) cache to disk and are reused across
every model and across runs, with two independent feature flavors
(`parquet` for tabular models, `npz` for tensor models). The training
stage is the only thing that has to rerun when you change the model,
window, or CV settings, and even that can be resumed mid-cohort by
reusing the same `--run-id`. After your first end-to-end pass on a
participant, every subsequent model on that participant only pays for
the training stage — typically minutes instead of an hour.

---

## 7. What Each Speed Tier Runs (and Skips)

This section is a process-level reference. Each speed tier disables a
specific subset of the training pipeline's feature-selection and
hyperparameter-search stages — they aren't "all or nothing" presets. The
tables below show exactly which steps each tier runs vs. skips, organized
by category. For the higher-level "when to use which" overview, see
`configs/README.md`.

`default` columns refer to `configs/default.yaml` (the research-grade slow
path; no tier flag). The four tier configs (`lightning.yaml`,
`express.yaml`, `quick.yaml`, `riemannian.yaml`) layer on top of it and
only the listed keys are overridden.

### 7.1 Per-fold feature-selection stages

These five steps run sequentially inside every outer CV fold of the
*tabular* training path (XGB / SVM / Logistic / LSTM). The tensor path
used by Riemannian skips all of them — it has no flat feature vector to
select from.

| Stage                           | default              | quick                  | express                 | lightning           | riemannian (tensor) |
|---------------------------------|----------------------|------------------------|-------------------------|---------------------|---------------------|
| 1. Correlation drop             | runs (θ=0.90)        | runs (θ=0.90)          | runs (θ=0.95)           | runs (θ=0.95)       | **skipped**         |
| 2. SelectKBest (ANOVA F)        | runs (k=500)         | runs (k=300)           | runs (k=200)            | runs (k=100)        | **skipped**         |
| 3. Iterated RFECV (XGB only)    | runs (5 iters)       | runs (2 iters)         | runs (1 iter)           | **skipped**         | **skipped**         |
| 4. Gain-prune + refit (XGB)     | runs                 | runs                   | runs                    | **skipped**         | **skipped**         |
| 5. SHAP-prune + refit (XGB)     | runs (q=0.20)        | runs (q=0.20)          | **skipped**             | **skipped**         | **skipped**         |

A note about the XGB-only steps: even on the tiers where they're enabled,
they're gated at the model level by `factory["rfecv_base"]`,
`factory["supports_gain"]`, and `factory["supports_shap"]`. So an SVM,
Logistic, or LSTM run will still skip RFECV/gain/SHAP regardless of tier
because those models lack the underlying capability (e.g. SVM has no
`feature_importances_`). The tier setting just makes the config
self-consistent.

For Riemannian specifically, every step is "skipped" because the tensor
training path in `_fit_score_split_tensor` doesn't have a flat feature
matrix to operate on — covariance matrices and tangent-space projections
happen *inside* the model pipeline, not as separate selection stages.

### 7.2 Hyperparameter search

| Aspect                          | default                       | quick                      | express                    | lightning                  | riemannian                 |
|---------------------------------|-------------------------------|----------------------------|----------------------------|----------------------------|----------------------------|
| Search method                   | auto (halving for XGB)        | halving random             | halving random             | grid                       | grid                       |
| Search `n_iter` (halving only)  | 100 candidates                | 50 candidates              | 25 candidates              | n/a (small grid)           | n/a (small grid)           |
| XGB `n_estimators` ceiling      | 1000                          | 600                        | 400                        | 200                        | n/a                        |
| Halving budget (min → max)      | 100 → 1000, factor 3          | 100 → 600, factor 3        | 50 → 400, factor 3         | n/a                        | n/a                        |
| Effective XGB grid size         | ~10 000 combos                | ~2 000 combos              | ~250 combos                | 2 combos                   | n/a                        |
| Effective SVM grid size         | ~150 combos                   | (inherits default)         | 12 combos                  | 1 combo                    | n/a                        |
| Effective Logistic grid size    | n/a (smoke-tier default)      | 6 combos                   | 4 combos                   | 2 combos                   | n/a                        |
| Effective Riemannian grid       | n/a                           | n/a                        | n/a                        | n/a                        | 6 combos (nfilter × cov)   |

### 7.3 Cross-validation

| Aspect                          | default               | quick                | express             | lightning           | riemannian          |
|---------------------------------|-----------------------|----------------------|---------------------|---------------------|---------------------|
| Outer CV n_splits               | 5                     | 5                    | 5                   | 3                   | 5                   |
| Outer CV n_repeats              | 20                    | 5                    | 2                   | 1                   | 5                   |
| Total outer folds per participant | 100 + chronological | 25                   | 10                  | 3                   | 25                  |
| Inner CV splits                 | 3                     | 3                    | 2                   | 2                   | 3                   |
| Chronological sanity check      | runs (~5 extra folds) | **skipped**          | **skipped**         | **skipped**         | **skipped**         |

### 7.4 Other tier-level defaults

| Aspect                          | default               | quick               | express             | lightning           | riemannian            |
|---------------------------------|-----------------------|---------------------|---------------------|---------------------|------------------------|
| Default model when `--model` omitted | xgb              | xgb (inherited)     | xgb (inherited)     | xgb (inherited)     | **riemannian**         |
| Default prediction window       | late_cnv (1.0-2.0 s)  | late_cnv            | late_cnv            | late_cnv            | **full_cnv (0-2.0 s)** |
| `resources.n_jobs`              | -8                    | -8 (inherited)      | -8 (inherited)      | -8 (inherited)      | -8 (inherited)         |
| `modeling.parallel.participants`| unset → sequential    | **-10**             | **-8**              | **-8**              | **-8**                 |
| Channel mode default            | full                  | full (inherited)    | full (inherited)    | full (inherited)    | full (forced)          |
| `stamp_runs`                    | true                  | true                | true                | true                | true                   |

### 7.5 Approximate wall time on a 16-core box

These are rough single-participant estimates assuming features are already
cached. Cohort runs scale roughly as `time_per_participant ×
n_participants / parallel.participants` once the worker pool is saturated.

| Tier                | Single participant   | Full 29-participant cohort (parallel) |
|---------------------|----------------------|---------------------------------------|
| `default` (XGB)     | 60-180 min           | 6-12 hours sequentially, or 1-3 hours with `--parallel-participants -8` |
| `quick`             | 10-15 min            | 35-60 min                              |
| `express`           | 4-8 min              | 15-30 min                              |
| `lightning`         | 1-3 min              | 4-10 min                               |
| `riemannian`        | 15-25 min (one-time tensor build adds ~30 s/participant on top) | 35-90 min |

### 7.6 At-a-glance: which stages survive in each tier

```text
default:    [corr] [k-best] [RFECV] [search] [gain-prune+refit] [SHAP-prune+refit] · 5×20 outer · chronological check
quick:      [corr] [k-best] [RFECV] [search] [gain-prune+refit] [SHAP-prune+refit] · 5×5  outer
express:    [corr] [k-best] [RFECV] [search] [gain-prune+refit]                    · 5×2  outer
lightning:  [corr] [k-best]         [search]                                       · 3×1  outer
riemannian: (tensor path: no FS stages; just [search])                             · 5×5  outer
```

The tiers can be read left-to-right as progressively trimming stages from
the right end of the tabular pipeline. Riemannian is in its own column
because it walks a different code path entirely.
