# Tests

Two layers of smoke tests, plus pytest fixtures.

## `test_imports.py` — static checks (~1 s)

Imports every module in the `eeg_steptype` package and verifies:

- Package layout is sound (no broken imports, no syntax errors).
- `configs/smoke.yaml` loads and the expected keys are present.
- Per-participant overrides apply correctly. Spot-checks:
  - **P02** — multi-file `raw_assembly` (concat).
  - **P08** — two crop windows in `raw_assembly`.
  - **P37** — `montage_mapping_override` swaps `B17`↔`B22`.

Run first; if this fails, nothing else can.

```bash
pytest tests/test_imports.py -q
```

## `test_smoke_pipeline.py` — end-to-end (~30–60 s)

Generates a synthetic 10-channel "BioSemi" Epochs object directly in
`tmp_path`, then runs:

1. `features.assemble.build_for_participant` — feature engineering on real
   `mne.Epochs`, including the Morlet TFR and the slope/amplitude blocks.
2. `models.train.run` with `--model logistic` — full feature-selection
   schedule (corr drop → KBest → GridSearchCV) on a tiny logistic-regression
   estimator.

Asserts that the metrics DataFrame comes back populated with the expected
columns. The synthetic data is shaped so the two conditions are weakly
separable — accuracy isn't checked, only that the pipeline runs.

The real `01_preprocess.py` stage isn't covered here because it requires
`.bdf` raw files outside the repo. To smoke-test that stage on real data:

```bash
make smoke    # runs configs/smoke.yaml end-to-end on the first participant
```

## Running everything

```bash
make test      # pytest -q
```
