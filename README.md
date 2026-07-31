# Single-Trial Movement-Intent Decoding from EEG

A single-trial EEG decoding pipeline that predicts an upcoming movement —
a *straight* (`One`) vs *diagonal* (`Two`) step — from the pre-movement
preparation signal (the contingent negative variation, CNV). In
brain–computer-interface terms this is a **motor-intent decoding** problem:
read the cortical preparation that precedes a movement and classify what the
movement will be, one trial at a time.

The repository implements a complete decoding stack — artifact-robust
preprocessing, cortical source reconstruction, and a model zoo spanning
classical ML through an attention-based CNN — with per-participant nested
cross-validation and fully reproducible, stamped runs. Features come from
electrode-level amplitudes, power spectral density (Morlet TFR), and
source-space activity reconstructed via eLORETA.

Originally developed as an MSc neuroscience thesis on movement planning.

> **Scope.** This is an *offline* decoder: trained and evaluated on recorded
> EEG, single-trial and per-participant. It is not a real-time / closed-loop
> system — it is the same signal-processing and modelling stack a motor BCI
> relies on, applied retrospectively.

---

## The decoding stack

Mapping the repository onto the stages a motor-BCI decoding team would recognise:

| Decoding stage | What this pipeline does |
|---|---|
| **Signal conditioning** | ZapLine line-noise removal, PyPREP bad-channel detection, ASR, common-average → Picard ICA, current-source-density (CSD) referencing, AutoReject epoch repair |
| **Neural source estimation** | eLORETA cortical source reconstruction (cached forward + inverse operators) |
| **Feature extraction** | pre-movement amplitudes & slopes, Morlet time-frequency PSD across bands, source-space activity; Riemannian (xDAWN-covariance tangent-space) and FBCSP-style mu/beta log-variance features are scaffolded alongside |
| **Decoders** | XGBoost, SVM, logistic regression, LSTM, and hybrid attention CNNs (EEGNet / EEGNeXt — multi-scale temporal stem + squeeze-and-excitation channel attention + residual separable blocks) |
| **Model selection** | corr → KBest → RFECV → gain → SHAP feature pruning, GridSearch, per-participant **nested** cross-validation |
| **Evaluation** | single-trial AUC / accuracy, a sliding-window AUC time-course, cohort roll-ups, and reproducible git-stamped runs |

---

## Quick start

```bash
# 1. Install (editable)
pip install -e .

# 2. Link large generated artifacts to OneDrive
make sync-setup

# 3. Point the pipeline at your raw data folder
cp configs/local.yaml.example configs/local.yaml
# …edit `paths.raw_root` in configs/local.yaml…

# 4. Smoke test (~30 s — checks the pipeline end-to-end on synthetic data)
make test

# 5. Full run (preprocessing → src → features → XGBoost training)
make all                        # all stages, default model = xgb
make train MODEL=lstm           # just the training stage with a different model
make all OVERRIDE_MODE=full     # opt into participant-specific fine-tuning
make train CHANNEL_MODE=roi     # train on medial foot-motor ROI features
make train PREDICTION_WINDOW=full_cnv  # secondary full-window analysis
```

See [`SCRIPT_GUIDES.md`](SCRIPT_GUIDES.md) for copy-paste commands covering
the standard smoke tests, the two-participant end-to-end smoke test, and a
full XGBoost pipeline run.

Large generated artifacts are kept out of Git and synced through OneDrive.
`make sync-setup` links these ignored folders into the OneDrive artifact store:
`data/interim`, `data/features`, `data/src`, `outputs/runs`, and `outputs/qc`.
On a new machine, clone the repo, let OneDrive finish syncing, then run
`make sync-setup` once before running the pipeline. To override the OneDrive
location, set `ML_V2_ONEDRIVE_ROOT` or run
`powershell -ExecutionPolicy Bypass -File scripts/setup_onedrive_artifacts.ps1 -OneDriveRoot "C:\path\to\ML_V2"`.

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
features and metadata are left intact. Tensor and hybrid neural model paths
keep all channels by design; CNN/EEGNet/EEGNeXt also fuse the raw tensor with the
XGB-style feature parquet when their tabular branch is enabled. `eegnext` is a
more sophisticated hybrid CNN (multi-scale temporal stem + squeeze-and-excitation
channel attention + residual separable blocks); run it with
`python run.py --speed-tier eegnext --participants P25 --stages train`.

The default prediction window for **every model** is full CNV, `0.0-2.0 s`,
which carries the most discriminative signal across the cohort. The narrower
late-CNV window (`1.0-2.0 s`, foot-motor preparation) is retained as a secondary
comparison window — select it per run with `--prediction-window late_cnv`.
Feature cache filenames are window-aware, so `full_cnv` and `late_cnv` runs do
not reuse each other's parquets. Cropped-training augmentation and a
sliding-window AUC time-course are recorded under the primary window in
`configs/default.yaml` for follow-up analyses.

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

Released under the [PolyForm Noncommercial License 1.0.0](LICENSE): free to
use, modify, and redistribute **for noncommercial purposes** — including
research, teaching, personal study, and use by academic, nonprofit,
public-research, and government organizations (regardless of funding source) —
provided the copyright and license notices are kept intact. **Commercial use is
not granted by this license.**

Copyright (c) 2026 Ali Doroodchi

## Citation

If you use this software, its models, or its results in academic work, please
cite it. GitHub's **"Cite this repository"** sidebar — generated from
[`CITATION.cff`](CITATION.cff) — exports ready-made APA and BibTeX entries.
BibTeX example:

```bibtex
@software{doroodchi_steptype_eeg_2026,
  author  = {Doroodchi, Ali},
  title   = {Step-Type Prediction from EEG Signals},
  year    = {2026},
  version = {2.5.0},
  url     = {https://github.com/Aria-Doroodchi/Step-Type-Prediction---EEG}
}
```
