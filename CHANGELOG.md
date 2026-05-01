# Changelog

## 2026-05-01 — Global preprocessing profiles

Extracted preprocessing settings out of `configs/default.yaml` into named
profile files under `configs/preprocessing/<name>.yaml`. The active
profile is selected by `preprocessing_profile:` in the cohort config, by
`--preprocessing-profile` on the CLI, or by an explicit argument to
`load_config()`. Per-participant overrides still layer on top.

Why: A/B-comparing preprocessing variants for ML tuning. Hold the model
fixed, swap the profile, retrain, and the run snapshot in
`outputs/runs/<id>/config.yaml` records exactly which profile produced
the result.

### Added
- `configs/preprocessing/default.yaml` — the cohort default (PyPREP bads
  + ICLabel @ p>0.9 + autoreject + 0.1–40 Hz bandpass).
- `configs/preprocessing/smoke.yaml` — fast variant for end-to-end smoke
  runs (no PyPREP, half-size ICA, threshold rejection, narrow band).
- `configs/preprocessing/README.md` — schema + workflow for adding new
  profiles.
- `--preprocessing-profile` CLI flag on `run.py` and `scripts/01_preprocess.py`.
- `eeg_steptype.config.list_preprocessing_profiles()`.
- 6 new pytest cases in `tests/test_imports.py` covering profile loading,
  arg-override, missing-profile error, and per-participant overrides
  beating the profile.

### Changed
- `configs/default.yaml` — the inline `preprocessing:` block is gone;
  replaced with `preprocessing_profile: default`.
- `configs/smoke.yaml` — adds `preprocessing_profile: smoke`.
- `eeg_steptype.config.load_config()` — accepts a new
  `preprocessing_profile` keyword arg; merges layers in the order
  `profile → cohort → local`; stamps the resolved profile name into the
  returned dict so it appears in run snapshots.

---

## 2026-05-01 — Pipeline reorganization

Moved from a folder of stand-alone scripts to a config-driven, installable
package. Old code stays on disk for reference but is gitignored.

### Added

- **`src/eeg_steptype/`** — installable package (`pip install -e .`):
  - `preprocessing/` — automated raw → epoch pipeline using PyPREP
    (bad-channel detection), `mne-icalabel` (conservative ICA component
    classification at p > 0.9), and `autoreject` (per-channel rejection
    thresholds). Replaces the per-participant scripts at
    `bad_interpolated/Pxx/Pxx_CNV.py`.
  - `source_localization/` — eLORETA pipeline. Hoists `noise_cov`,
    `forward`, and `inverse_operator` out of the per-epoch loop (they
    were rebuilt for every epoch in the old `SRC_writer.py`); caches
    `forward` per participant.
  - `features/` — amplitude, slopes, PSD (Morlet) extraction. Caches the
    wide feature matrix to parquet so model runs no longer re-read `.fif`.
  - `models/` — feature selection (correlation drop / SelectKBest /
    iterated RFECV / gain prune / SHAP prune) and classifier factories
    for XGBoost, SVM, LSTM, and logistic regression. The shared
    per-participant fit/eval driver in `train.py` replaces the duplicated
    inline loops in `CNV_XGB_4.3.py`, `CNV_LSTM_3.py`, `CNV_ML_SVM_1.py`.
- **`configs/`** — single source of truth for all paths and hyper-parameters:
  - `default.yaml` — committed defaults.
  - `local.yaml.example` — template for per-machine path overrides.
  - `smoke.yaml` — tiny end-to-end check (1 participant, logistic
    regression, shrunk grids).
  - `overrides/Pxx.yaml` × 34 — per-participant tweaks. Manual cuts and
    appends from each original `Pxx_CNV.py` are preserved declaratively
    (e.g. P02 multi-file concat, P08 two-window crop, P14/P19/P23 single
    crop, P37 cut+concat with B17/B22 electrode swap, P03 extended ICA
    training window). Lab-flagged bad channels and the legacy hand-tuned
    ICA-exclude lists / rejection thresholds are also captured (legacy
    values commented for fallback).
- **`scripts/01_preprocess.py`...`05_visualize.py`** — thin per-stage CLIs.
- **`run.py`** — single-process driver: `python run.py --stages …`.
- **`Makefile`** — `make install / smoke / test / preprocess / src /
  features / train MODEL=xgb`.
- **`tests/`** — `test_imports.py` (every module imports + override
  spot-checks) and `test_smoke_pipeline.py` (synthetic-data end-to-end
  run in <60 s).
- **`pyproject.toml`** — installable package metadata.
- **`REORG_PROPOSAL.md`** — design doc this layout was built from.

### Changed

- `requirements.txt` — added `pyprep`, `mne-icalabel`, `autoreject`,
  `pyyaml`, `pyarrow`, `scikeras`.
- `README.md` — rewritten around the new layout, quick-start, and
  reproducibility model.
- `.gitignore` — gitignores legacy folders (`01_preprocessing/`,
  `02_models/{archive,lstm,svm,xgboost}/`, `03_visualization/python/`,
  `sandbox/`, `_repo_export/`) and new pipeline data
  (`data/interim/`, `data/features/`, `data/src/`, `outputs/runs/`).
  Per-machine `configs/local.yaml` is gitignored; `local.yaml.example`
  is committed.

### Preserved

- All R-side code at `02_models/R/` and `03_visualization/R/` is
  untouched and still tracked.
- Original per-participant preprocessing scripts under
  `bad_interpolated/Pxx/Pxx_CNV.py` are unchanged in their lab folder
  (outside this repo).

### Behavioral notes

- ICA component selection is now automated (ICLabel @ p > 0.9, conservative).
  Each override YAML keeps the original hand-picked exclude list as a
  commented fallback in case the auto-classifier under-flags a participant.
- Epoch rejection is now `autoreject` by default. The original per-condition
  voltage thresholds (e.g. `One: 48e-6`, `Two: 51.5e-6` for P25) are kept
  as commented fallbacks per participant.
- Final filter bandpass default changed to `[0.1, 40]` Hz (the modal value
  across the cohort). Participants whose original script used a different
  bandpass have it set explicitly in their override (P05/P08/P10/P11/P12/
  P16/P17/P21/P24/P25/P28/P29/P30/P31/P37).
- Every training run writes a stamped folder under `outputs/runs/<id>/`
  containing the full config snapshot, git SHA, and metrics — any past
  result can be reproduced from those three files.
