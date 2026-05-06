# Step Type Prediction — EEG

Machine-learning pipeline for predicting **step type** — *straight* (`One`)
vs *diagonal* (`Two`) — from EEG signals recorded during a stepping task.
Part of an MSc thesis project.

Features come from electrode-level amplitudes, power spectral density
(Morlet TFR), and source-space activity reconstructed via eLORETA.

---

## Quick start

```bash
# 1. Install (editable)
pip install -e .

# 2. Point the pipeline at your raw data folder
cp configs/local.yaml.example configs/local.yaml
# …edit `paths.raw_root` in configs/local.yaml…

# 3. Smoke test (~30 s — checks the pipeline end-to-end on synthetic data)
make test

# 4. Full run (preprocessing → src → features → XGBoost training)
make all                        # all stages, default model = xgb
make train MODEL=lstm           # just the training stage with a different model
make all OVERRIDE_MODE=full     # opt into participant-specific fine-tuning
make train CHANNEL_MODE=roi     # train on medial foot-motor ROI features
make train PREDICTION_WINDOW=full_cnv  # secondary full-window analysis
```

See [`SCRIPT_GUIDES.md`](SCRIPT_GUIDES.md) for copy-paste commands covering
the standard smoke tests, the two-participant end-to-end smoke test, and a
full XGBoost pipeline run.

Stage-by-stage runs:

```bash
python scripts/01_preprocess.py            # raw .bdf → cleaned epochs .fif
python scripts/02_source_localize.py       # epochs → per-participant src CSV
python scripts/03_extract_features.py      # epochs+src → cached parquet
python scripts/04_train.py --model xgb     # features → metrics CSV
python scripts/05_visualize.py --run outputs/runs/<run_id>
```

Or in one Python process:

```bash
python run.py --config configs/default.yaml --model xgb
python run.py --stages features train --participants P25 P26 --model logistic
python run.py --config configs/default.yaml --participant-override-mode full
python scripts/04_train.py --model xgb --channel-mode roi
python scripts/03_extract_features.py --prediction-window full_cnv
python scripts/04_train.py --model xgb --prediction-window full_cnv
```

---

## Repository layout

```
.
├── configs/
│   ├── default.yaml          # all knobs: paths, participants, params, grids
│   ├── local.yaml.example    # per-machine override (commit local.yaml as gitignored)
│   ├── smoke.yaml            # tiny config for end-to-end smoke runs
│   └── overrides/            # one YAML per participant — preserves manual cuts/appends
│       ├── P01.yaml … P39.yaml
│       └── README.md
│
├── src/eeg_steptype/         # importable package (pip install -e .)
│   ├── config.py             # YAML loading + per-participant merge
│   ├── io.py                 # standard path layout
│   ├── logging_utils.py      # logger + run-stamping
│   ├── preprocessing/        # raw .bdf → cleaned epochs
│   │   ├── montage.py        ├── load.py        ├── bads.py
│   │   ├── filter.py         ├── reference.py   ├── ica.py
│   │   ├── events.py         ├── epoching.py    ├── reject.py
│   │   └── pipeline.py
│   ├── source_localization/  # epochs → src CSV (cached forward+inverse)
│   │   ├── forward.py  ├── inverse.py  ├── labels.py  └── pipeline.py
│   ├── features/             # amplitude, slopes, PSD, assemble → parquet
│   │   ├── amplitude.py  ├── slopes.py  ├── psd.py  └── assemble.py
│   ├── models/               # feature selection + classifier factories
│   │   ├── feature_selection.py     # corr / KBest / RFECV / gain / SHAP
│   │   ├── xgb.py  ├── svm.py  ├── lstm.py  ├── logistic.py
│   │   ├── train.py          # generic per-participant fit/eval driver
│   │   └── evaluate.py       # confusion matrix + cohort rollup
│   └── viz/                  # plots
│
├── scripts/                  # thin per-stage CLI orchestrators
│   ├── 01_preprocess.py        02_source_localize.py
│   ├── 03_extract_features.py  04_train.py
│   └── 05_visualize.py
│
├── tests/                    # smoke tests
│   ├── test_imports.py       # every module imports cleanly
│   ├── test_smoke_pipeline.py# synthetic-data end-to-end run
│   └── conftest.py
│
├── data/                     # gitignored
│   ├── interim/epochs/         cleaned .fif
│   ├── src/                    per-participant src CSVs
│   └── features/               cached feature parquets
│
├── outputs/
│   ├── runs/<run_id>/          metrics.csv, rollup.csv, config.yaml, git_sha.txt
│   ├── qc/                     per-participant preprocessing reports
│   └── figs/                   topomaps, brain plots
│
├── run.py                    # single-process pipeline driver
├── Makefile                  # `make smoke`, `make preprocess`, `make train MODEL=…`
├── pyproject.toml            # installable package
├── requirements.txt
└── REORG_PROPOSAL.md         # design doc this layout was built from
```

---

## Configuration

Three YAMLs are deep-merged at load time, in this order:

1. **`configs/default.yaml`** — committed; project defaults (paths,
   participant list, all hyper-parameters and grids).
2. **`configs/local.yaml`** — gitignored; per-machine overrides. The only
   thing most users need to set is `paths.raw_root`.
3. **`configs/overrides/Pxx.yaml`** — applied on top *only when that
   participant is being processed*. Default runs apply only `raw_assembly`
   from these files, so cohort preprocessing stays uniform except for manual
   raw `.bdf` crops/appends that cannot be automated reliably. Full
   participant-specific tuning remains available by setting
   `participant_overrides.mode: full` or passing
   `--participant-override-mode full`. See
   [`configs/overrides/README.md`](configs/overrides/README.md) for the full
   schema.

**Example — P02 had two raw files concatenated:**

```yaml
# configs/overrides/P02.yaml
raw_assembly:
  files:
    - "P02/P02_CNV.bdf"
    - "P02/P02_CNV_2.bdf"
```

**Example — P08 had two crop windows from one file:**

```yaml
raw_assembly:
  files:
    - { path: "P08/P08_CNV.bdf", tmin: 72.0,  tmax: 135.0  }
    - { path: "P08/P08_CNV.bdf", tmin: 215.0, tmax: 1100.0 }
```

---

## Pipeline

```
raw .bdf  ──►  01_preprocess          (ZapLine, PyPREP bads, ASR, CAR→Picard ICA→CSD, autoreject)
           ──►  02_source_localize    (cached forward + inverse, eLORETA)
           ──►  03_extract_features   (amplitude, slopes, PSD → parquet)
           ──►  04_train              (corr → KBest → RFECV → gain → SHAP → GridSearch)
           ──►  05_visualize
```

What each stage produces:

| Stage | Inputs | Outputs |
|---|---|---|
| 01 preprocess  | `{raw_root}/Pxx/Pxx_CNV.bdf` | `data/interim/epochs/Pxx_CNV_{One,Two}-epo.fif` (CSD-referenced) |
| 02 src         | epoch .fif                   | `data/src/Pxx_{One,Two}_src.csv` |
| 03 features    | epoch .fif + src CSV         | `data/features/Pxx_{One,Two}_features.parquet` |
| 04 train       | feature parquets             | `outputs/runs/<run_id>/{metrics.csv, rollup.csv, config.yaml, git_sha.txt}` |
| 05 visualize   | metrics.csv                  | `outputs/runs/<run_id>/per_participant_accuracy.png` |

Every stage is **idempotent**: re-running a stage skips participants whose
output already exists, unless `--force` is passed.

Training can compare two channel configurations without rebuilding features:
`--channel-mode full` keeps the full feature parquet, while
`--channel-mode roi` restricts electrode amplitude, slope, and PSD features to
the medial foot-motor cluster declared in `configs/default.yaml`. Source-space
features and metadata are left intact. Future Riemannian/CNN model paths keep
all channels by design.

The primary prediction window is late CNV, `1.0-2.0 s`, where foot-motor
preparation is expected to be most discriminative. Feature cache filenames are
window-aware, so `late_cnv` and secondary `full_cnv` runs do not reuse each
other's parquets. Secondary settings for cropped-training augmentation and a
sliding-window AUC time-course are recorded in `configs/default.yaml` for
follow-up analyses.

Future Riemannian/SCP comparators are scaffolded but not used by the current
XGBoost path. `configs/default.yaml` records an xDAWN-covariance tangent-space
path with OAS covariance, broadband covariance tangent-space features, and
mu/beta FBCSP-style log-variance features. An opt-in `cnv_benchmark` feature
block computes 250 ms mean-amplitude bins over the 9 medial motor channels for
a shrinkage-LDA benchmark.

---

## Smoke testing

```bash
make test       # pytest: imports + synthetic-data pipeline (~30–60 s)
make smoke      # end-to-end run on configs/smoke.yaml (1 participant, logistic)
```

`configs/smoke.yaml` shrinks every expensive knob:

- 1 participant, 1 condition pair
- 4 time bins instead of 16
- Two frequency bands (Theta, Alpha) instead of five
- `n_iterations=1` for RFECV (vs 5)
- Tiny GridSearchCV grid for logistic regression (`C ∈ {0.1, 1.0}`)
- SHAP pruning disabled
- Forward solution caching off

A full XGBoost run on 30 participants takes hours; `make smoke` with
logistic regression on 1 participant takes well under a minute.

---

## Reproducibility

Every training run writes a stamped folder under `outputs/runs/<run_id>/`:

- `config.yaml` — full merged config snapshot
- `git_sha.txt` — repo commit at run time
- `env.json`   — Python version, platform, argv
- `metrics.csv` — per-participant scores
- `rollup.csv`  — cohort totals

This means a result can be reproduced by checking out the recorded git SHA
and running `python run.py --config <runs/.../config.yaml>`.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .[dev,lstm]         # editable install + extras
```

The optional `lstm` extra pulls in TensorFlow + scikeras (large download);
omit it if you only run XGBoost / SVM / logistic.

### R side

The legacy R scripts in `02_models/R/` and `03_visualization/R/` are kept
for compatibility but aren't part of the new automated pipeline.

---

## Migration notes

This layout was built from the proposal in
[`REORG_PROPOSAL.md`](REORG_PROPOSAL.md), which documents the rationale for
each module split and the rejected alternatives. The 7-phase migration plan
in that document is fully applied.

The old per-participant preprocessing scripts at
`bad_interpolated/Pxx/Pxx_CNV.py` were translated module-by-module into
`src/eeg_steptype/preprocessing/` plus 33 YAML override files preserving
every hand-tuned parameter (cuts, appends, channel swaps, bads,
ICA-exclude lists, rejection thresholds). The legacy hand-tuned values are
kept as commented provenance inside each override, while default runs use
uniform AutoReject-local epoch repair/rejection.

## License

TBD
