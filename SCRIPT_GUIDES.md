# Script Guides

These commands assume you are running from the repository root:

```powershell
cd C:\Users\Aria\Desktop\ML_V2
```

If `python` is not on PATH, use your installed interpreter explicitly:

```powershell
& 'C:\Users\Aria\AppData\Local\Programs\Python\Python312\python.exe' <args>
```

For example:

```powershell
& 'C:\Users\Aria\AppData\Local\Programs\Python\Python312\python.exe' -m pytest -q
```

## 1. Run The Smoke Tests

Purpose: quick confidence check that imports, config loading, synthetic feature extraction, training, and workflow smoke tests still run.

PowerShell:

```powershell
& 'C:\Users\Aria\AppData\Local\Programs\Python\Python312\python.exe' -m pytest tests\test_imports.py tests\test_smoke_pipeline.py -q
```

Make:

```bash
make smoke-test
```

Expected result:

```text
tests pass with possible warnings from tiny synthetic data
```

## 2. Run The Two-Participant End-To-End Smoke Test

Purpose: run the maintained stage flow on two synthetic participants:

```text
preprocess -> source localization -> feature assembly -> training
```

This test writes all outputs to a temporary pytest directory, so it will not touch your real `data/` or `outputs/` folders.

PowerShell:

```powershell
& 'C:\Users\Aria\AppData\Local\Programs\Python\Python312\python.exe' -m pytest tests\test_smoke_pipeline.py::test_two_participant_full_workflow_smoke -q
```

Make:

```bash
make smoke-test-two
```

Expected result:

```text
1 passed
```

Warnings from ICLabel, CSD, or constant synthetic features are acceptable here; this test is checking stage wiring and file creation, not real EEG validity.

## 3. Run A Full XGB Pipeline

Purpose: run the real data workflow with the default XGBoost model:

```text
raw .bdf -> cleaned epochs -> source CSVs -> feature parquets -> XGB metrics
```

Before running this, make sure `configs/local.yaml` exists and points to your raw data folder:

```yaml
paths:
  raw_root: "C:/path/to/Participants"
```

Also make sure the source-localization assets are available. By default the
pipeline expects these files in the repo root:

```text
fsaverage-src.fif
fsaverage-bem-sol.fif
fsaverage-trans.fif
```

Check this before launching the full workflow:

```powershell
& 'C:\Users\Aria\AppData\Local\Programs\Python\Python312\python.exe' scripts\00_preflight.py --config configs/default.yaml
```

Make:

```bash
make preflight
```

Full workflow, default uniform configs:

```powershell
& 'C:\Users\Aria\AppData\Local\Programs\Python\Python312\python.exe' run.py --config configs/default.yaml --model xgb
```

Make:

```bash
make full-xgb
```

`make full-xgb` is a shortcut for the same full XGB run, with `make preflight`
executed first. "Default uniform configs" means participant override YAMLs are
used only for manual raw `.bdf` crops/appends; preprocessing/modeling parameters
stay uniform unless you pass `OVERRIDE_MODE=full`.

Useful variants:

```powershell
# Re-run only selected participants
& 'C:\Users\Aria\AppData\Local\Programs\Python\Python312\python.exe' run.py --config configs/default.yaml --model xgb --participants P25 P26

# Run ROI-only training features for interpretability comparison
& 'C:\Users\Aria\AppData\Local\Programs\Python\Python312\python.exe' run.py --config configs/default.yaml --model xgb --channel-mode roi

# Opt into all participant-specific override tuning
& 'C:\Users\Aria\AppData\Local\Programs\Python\Python312\python.exe' run.py --config configs/default.yaml --model xgb --participant-override-mode full

# Force regeneration of stage outputs
& 'C:\Users\Aria\AppData\Local\Programs\Python\Python312\python.exe' run.py --config configs/default.yaml --model xgb --force
```

Outputs:

```text
data/interim/epochs/
data/src/
data/features/
outputs/qc/
outputs/runs/<run_id>/metrics.csv
outputs/runs/<run_id>/rollup.csv
```

The default XGB path uses nested repeated CV and `HalvingRandomSearchCV`, so a full cohort run can still take a long time. For a first real-data check, run one or two participants before launching the full cohort.
