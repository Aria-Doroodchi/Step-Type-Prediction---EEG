# Step Type Prediction — EEG

Machine-learning pipeline for predicting **step type** — *straight* vs
*diagonal* — from EEG signals recorded during a stepping task. Part of
an MSc thesis project.

The classification target is the binary condition label (`One` =
straight, `Two` = diagonal) attached to each epoch. Features are drawn
from electrode-level amplitudes, power spectral density (PSD), and
source-space activity reconstructed via eLORETA.

---

## Repository layout

```
.
├── 01_preprocessing/        # Build features from the raw .fif epoch files
│   ├── CNV_epoch_extraction.py     # → master epoch CSV
│   └── SRC_writer.py               # → per-participant source-localized CSVs
│
├── 02_models/               # Classifiers
│   ├── xgboost/
│   │   ├── CNV_XGB_4.3.py          ← current XGBoost model
│   │   └── archive/                # earlier iterations + VERSIONS.md
│   ├── lstm/
│   │   ├── CNV_LSTM_3.py           ← current LSTM model
│   │   └── archive/                # earlier iterations + VERSIONS.md
│   ├── svm/
│   │   └── CNV_ML_SVM_1.py
│   ├── R/
│   │   ├── CNV_model.R             # XGBoost in R
│   │   └── CNV_XGB_ch.Rmd          # knitted analysis report
│   └── archive/                    # early electrode-level baselines + VERSIONS.md
│
├── 03_visualization/        # Plots: topomaps, source brain graphs, results
│   ├── python/
│   └── R/
│
├── sandbox/                 # Ad-hoc / exploratory
│   └── test.py
│
├── data/                    # Data files live here locally; gitignored
│   └── README.md            # what's expected, how to regenerate
│
├── outputs/
│   ├── figs/                # Generated figures (LORETA, topomaps)
│   └── reports/             # Knitted HTML reports
│
├── ML.Rproj                 # RStudio project file
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

## Pipeline

```
raw .fif epochs (Participants/) ──► 01_preprocessing/CNV_epoch_extraction.py
                                  ──► 01_preprocessing/SRC_writer.py

                                            │
                                            ▼

                              02_models/{xgboost, lstm, svm, R}/

                                            │
                                            ▼

                              03_visualization/{python, R}/
                              outputs/figs/, outputs/reports/
```

## Version history of the model scripts

Each model family was developed iteratively. The "current" version sits
at the top of its `02_models/<family>/` folder, and earlier numbered
versions live in `archive/` alongside a `VERSIONS.md` that summarizes
what changed between consecutive versions:

- [`02_models/xgboost/archive/VERSIONS.md`](02_models/xgboost/archive/VERSIONS.md)
- [`02_models/lstm/archive/VERSIONS.md`](02_models/lstm/archive/VERSIONS.md)
- [`02_models/archive/VERSIONS.md`](02_models/archive/VERSIONS.md)
  (early electrode-level baselines that preceded the XGBoost / LSTM
  split)

For exact line-by-line diffs use `git log -p <file>` or
`diff -u <a> <b>`.

## Setup

### Python

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### R

Open `ML.Rproj` in RStudio. The R modeling scripts use `xgboost`,
`caret`, `tidyverse`, `Matrix`, `plotly`, `gt`, `pracma`, `reshape2`,
`ggstatsplot`. Install via `install.packages(...)` as needed.

### Data

Data files are not in the repo. See [`data/README.md`](data/README.md)
for what's expected and how to regenerate it from the raw `.fif` epochs.

## Known portability issue

The scripts currently hard-code Windows OneDrive paths of the form
`C:/Users/Aria/OneDrive - The University of Western Ontario/...`. These
need to be edited before the scripts will run on a different machine.
A future cleanup pass should replace them with a single configurable
project root (config file or environment variable).

## License

TBD
