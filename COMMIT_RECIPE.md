# Commit recipe — landing the reorganization

The legacy folders (`01_preprocessing/`, `02_models/{archive,lstm,svm,xgboost}/`,
`03_visualization/python/`, `sandbox/`, `_repo_export/`) are *currently
tracked* in your git index. Adding them to `.gitignore` does not untrack
them on its own — you also need `git rm --cached` to drop them from the
index while leaving the files on disk.

Run from the repo root:

```bash
# 1. Sanity check: see what's about to change.
git status

# 2. Untrack the legacy folders (keeps the files on disk).
git rm -r --cached \
  01_preprocessing \
  02_models/archive \
  02_models/lstm \
  02_models/svm \
  02_models/xgboost \
  03_visualization/python \
  sandbox \
  _repo_export

# 3. Stage the new pipeline + the gitignore + the changelog.
git add .gitignore CHANGELOG.md REORG_PROPOSAL.md
git add pyproject.toml requirements.txt Makefile run.py README.md
git add configs/ src/ scripts/ tests/

# 4. Commit.
git commit -m "Reorganize into installable package with config-driven pipeline

- src/eeg_steptype/ installable package (preprocessing, source_localization,
  features, models, viz)
- configs/ as single source of truth: default.yaml, local.yaml.example,
  smoke.yaml, and 34 per-participant Pxx.yaml overrides preserving manual
  cuts/appends, electrode swaps, and lab-flagged bads
- scripts/01_..05_ thin per-stage CLIs; run.py single-process driver;
  Makefile orchestration
- Automated preprocessing via pyprep (bads), mne-icalabel (ICA, conservative
  p>0.9), autoreject (epoch rejection)
- LORETA: forward/noise_cov/inverse_operator hoisted out of per-epoch loop
- Feature matrices cached to parquet
- tests/ with import smoke and synthetic-data end-to-end pipeline test
- Legacy 01_preprocessing/, 02_models/{archive,lstm,svm,xgboost}/,
  03_visualization/python/, sandbox/ untracked but kept on disk
- See CHANGELOG.md and REORG_PROPOSAL.md for details"
```

## What stays tracked

Kept under version control because they're not part of the Python
reorganization:

- `02_models/R/` — R-side modeling code (CNV_model.R, CNV_XGB_ch.Rmd)
- `03_visualization/R/` — R-side viz scripts
- `data/README.md`
- `ML.Rproj`

## After the commit

`make test` to confirm the new package imports and the smoke pipeline
runs on synthetic data:

```bash
pip install -e .[dev]
make test
```

If that's green, push:

```bash
git push
```
