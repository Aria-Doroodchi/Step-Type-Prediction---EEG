# Reorganization & Automation Proposal — Step Type Prediction (EEG)

**Purpose.** Improve reproducibility, eliminate hard-coded paths, deduplicate
code that currently lives in every script, and turn an ad-hoc set of
copy-pasted versions into an importable Python package with a clear pipeline.

This document answers the three questions you raised, then lays out a
concrete target layout, a stage-by-stage breakdown for LORETA and the
machine-learning modules, an orchestration recommendation, and a
phased migration plan.

---

## Q1. Is the current folder structure appropriate? How does it compare to typical ML projects?

**Short answer.** The numeric `01_preprocessing / 02_models / 03_visualization`
layout is workflow-oriented and reasonable for a thesis project. It mirrors
the *stages* of the pipeline rather than the *kind of code*, which is a
common pattern in research repos. But it's missing two things that most
mature ML projects have:

1. **A `src/` package of importable, reusable code** separate from the
   "scripts you run." Right now every `.py` file is a giant top-to-bottom
   script that re-defines participant lists, paths, frequency bands, channel
   names, etc. There is no shared code — only copy-paste.
2. **A configuration layer** so that paths, participants, frequency bands,
   bin width, hyper-parameter grids, etc. live in one YAML/TOML file rather
   than being duplicated across `01_preprocessing/CNV_epoch_extraction.py`,
   `01_preprocessing/SRC_writer.py`, `02_models/xgboost/CNV_XGB_4.3.py`,
   `02_models/lstm/CNV_LSTM_3.py`, and `02_models/svm/CNV_ML_SVM_1.py`.

**How peer projects do it.** Two widely used templates worth knowing:

- **Cookiecutter Data Science** — splits code into `src/{data,features,models,visualization}/`
  with thin `scripts/` on top. Heavily used in academic ML.
- **Kedro** (or **Snakemake**, **DVC**, **MLflow Projects**) — pipeline
  frameworks that explicitly track input/output files between stages and
  only re-run what changed. Heavier to learn but excellent for
  reproducibility.

**Recommendation.** Keep the workflow-oriented top-level folders, but split
out an importable `src/eeg_steptype/` package. The numeric folders become
*thin orchestrator scripts* that import from the package. This is the
"hybrid" pattern and is the most common in modern ML research codebases.

### Proposed layout

```
eeg-steptype/
├── configs/
│   ├── default.yaml              # paths, participants, freq bands, bin width, model grids
│   └── local.yaml.example        # per-machine overrides (gitignored copy → local.yaml)
│
├── src/eeg_steptype/             # importable package: `pip install -e .`
│   ├── __init__.py
│   ├── config.py                 # load YAML → dataclass; resolve paths
│   ├── io.py                     # read_epochs, save/load CSVs, path helpers
│   ├── logging_utils.py          # set up logging + run-id stamps
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── filter.py             # bandpass, notch
│   │   ├── ica.py                # ICA fit + automated component selection
│   │   ├── interpolation.py      # bad-channel detection + interpolation
│   │   ├── epoching.py           # event extraction, epoching, baseline
│   │   ├── reject.py             # automated artifact rejection (autoreject etc.)
│   │   └── pipeline.py           # orchestrate raw → cleaned epochs per participant
│   │
│   ├── source_localization/
│   │   ├── __init__.py
│   │   ├── forward.py            # build / cache fwd model (once per subject)
│   │   ├── inverse.py            # noise cov + inverse operator
│   │   ├── labels.py             # parcellation, label time courses
│   │   └── pipeline.py           # per-participant LORETA → per-epoch CSV
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── amplitude.py          # binned mean amplitude (current epoch_wide block)
│   │   ├── slopes.py             # per-bin linear-trend slopes
│   │   ├── psd.py                # Morlet TFR → band power
│   │   └── assemble.py           # join feature blocks → wide DataFrame
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── feature_selection.py  # corr-drop, SelectKBest, RFECV, gain prune, SHAP prune
│   │   ├── xgb.py                # XGB model factory + param grid
│   │   ├── lstm.py               # LSTM architecture factory
│   │   ├── svm.py                # SVM factory
│   │   ├── train.py              # generic per-participant fit/eval loop
│   │   └── evaluate.py           # confusion matrix, AUC, per-participant rollup
│   │
│   └── viz/
│       ├── __init__.py
│       ├── topomaps.py
│       ├── brain.py              # source-space brain graphs
│       └── results.py            # SHAP plots, accuracy bar charts
│
├── scripts/                      # thin orchestrators — these are what you "run"
│   ├── 01_preprocess.py          # raw .fif → cleaned epochs
│   ├── 02_source_localize.py     # epochs → src CSVs
│   ├── 03_extract_features.py    # epochs + src → feature matrix
│   ├── 04_train.py               # --model xgb|lstm|svm → metrics + artifacts
│   └── 05_visualize.py           # outputs → figures
│
├── data/                         # gitignored
│   ├── raw/                      # symlink or copy of raw .fif
│   ├── interim/                  # cleaned epochs
│   ├── features/                 # per-participant feature CSVs
│   └── src/                      # per-participant source-localized CSVs
│
├── outputs/
│   ├── runs/<run_id>/            # per-run: metrics CSVs, SHAP plots, config snapshot, git SHA
│   ├── figs/                     # publication-ready figures
│   └── reports/                  # knitted HTML reports
│
├── tests/                        # pytest smoke tests on synthetic data
│   ├── test_features.py
│   └── test_pipeline.py
│
├── R/                            # R-side analysis (kept separate)
│   ├── CNV_model.R
│   ├── CNV_XGB_ch.Rmd
│   └── viz/
│
├── pyproject.toml                # makes `eeg_steptype` installable
├── requirements.txt
├── Makefile                      # `make preprocess`, `make features`, `make train MODEL=xgb`
├── .gitignore                    # ignore .Rproj.user, .RData, .Rhistory, data/, outputs/runs/
└── README.md
```

---

## Q2. How should the preprocessing, LORETA, and ML modules be split?

### Preprocessing (informed by `P25_CNV.py`)

Your sample script `P25_CNV.py` is a **per-participant** script with five
hand-tuned values that change between participants. These are the manual
steps the new module needs to either automate or hoist into config:

| Manual step in `P25_CNV.py` | Lines | What's hand-tuned | Automation strategy |
|---|---|---|---|
| Bad-channel list | 56 (`extend(['P10','PO8'])`) | Per-participant channel list | **PyPREP** `NoisyChannels` or **MNE** `find_bad_channels_lof` for automatic detection. Keep a per-participant override list in config for the ones the algorithm misses. |
| ICA component exclude list | 77 (`[0,1,2,3,6,7,9,12,...]`) | Hand-picked indices after visually inspecting components | **`mne-icalabel`** automatic component classification (eye / heart / muscle / line-noise / channel-noise / brain / other). Auto-exclude everything classified non-brain above a probability threshold. Keep an override hook for edge cases. |
| Per-condition rejection threshold (`eeg=48e-6` / `51.5e-6`) | 133-134 | Two ad-hoc voltage thresholds | **`autoreject.AutoReject`** computes per-channel thresholds via cross-validation. Removes the hand-tuning entirely. |
| Drop-rate window (15% ± a couple) | 136-140 | Hand target on *how many* epochs to reject | Replaced by autoreject's principled approach. The drop-rate target becomes a sanity check / log message, not an input. |
| Channel mapping (`A1→Fp1`, ...) | 32-44 | Hard-coded 64-line dict | Move into `src/eeg_steptype/preprocessing/montage.py` as a constant. Same montage for every participant. |
| Reference channels list | 23-29 | Hard-coded `['A1', ..., 'Status']` | Same — move into `montage.py`. |
| Channel name `Status → Stim` | 43 | Stimulus channel name | Config. |
| Trigger codes 256, 512, 96 | 98, 100, 114, 116 | Hard-coded condition codes | Move into config: `events: {One: 256, Two: 512, response: 96}`. |
| ICA training window (`crop(50, 100)`) | 69 | 50 s window of "clean" data | Config-level `ica.train_window: [50, 100]`. |
| Filter / notch / sampling | 63, 69, 82 | `60×n` Hz, 1 Hz HP for ICA, 0.15-36 Hz final | Config: `filter: {bandpass: [0.15, 36], notch: 60, ica_hp: 1}`. |

The new preprocessing modules and what each owns:

```
src/eeg_steptype/preprocessing/
├── montage.py        # CHANNEL_MAPPING, PICK_CHANNELS, REF, montage = 'biosemi64'
├── load.py           # read_raw_bdf(participant) → Raw with picks/rename/montage applied
├── bads.py           # detect_bad_channels(raw) → list[str]; uses PyPREP/LOF
├── filter.py         # apply_notch(raw, freqs); apply_bandpass(raw, l, h)
├── reference.py      # apply_average_reference(raw)
├── ica.py            # fit_ica(raw, train_window, n_components);
│                     # auto_classify_components(ica, raw)  ← uses mne-icalabel
│                     # apply_ica(ica, raw)
├── events.py         # find_step_events(raw, codes={'One':256,'Two':512,'response':96})
│                     # → dict[condition → ndarray] (your existing 256/96 pairing logic)
├── epoching.py       # build_epochs(raw, events, tmin, tmax, baseline)
├── reject.py         # autoreject_epochs(epochs) → cleaned epochs + log
└── pipeline.py       # run(participant_id, config) → writes
                      #   data/interim/Pxx_CNV_{One,Two}-epo.fif
```

`scripts/01_preprocess.py` then becomes:

```python
# scripts/01_preprocess.py
from eeg_steptype.config import load_config
from eeg_steptype.preprocessing import pipeline

cfg = load_config()
for pid in cfg.participants:
    pipeline.run(pid, cfg)        # idempotent: skips if output exists & --force not set
```

**The key automation wins** are bads detection (PyPREP), ICA classification
(`mne-icalabel`), and rejection thresholds (`autoreject`). All three are
mature, well-cited tools that replace exactly the steps where your sample
script has hand-coded values per participant. Add them to
`requirements.txt`:

```
pyprep>=0.4
mne-icalabel>=0.6
autoreject>=0.4
```

A per-participant override file (`configs/overrides/P25.yaml`) handles
the rare case where the auto-classifier misses something — but for ~30
participants you should be hand-overriding maybe 3-5 of them, not all 30.

The preprocessing pipeline should also write a **QC report** per
participant — a small HTML or PNG dump showing the raw PSD, dropped
channels, ICA components classified as artifacts, and final epoch
drop-log — so a human can spot-check without re-running the script.

### LORETA

The current `01_preprocessing/SRC_writer.py` does the full inverse-modeling
pipeline inside one nested loop. It also has a **major efficiency bug**:

> Lines 73–106: for each *epoch*, it recomputes `noise_cov`,
> `make_forward_solution`, and `make_inverse_operator`. The forward and
> inverse models depend only on the sensor montage and the head model, not
> on the trial — they should be computed **once per participant** and
> reused across all epochs. This alone is probably an order-of-magnitude
> speedup.

Suggested split:

| Module | Responsibility |
|---|---|
| `forward.py` | `build_forward(info, src, bem, trans)` → caches `fsaverage-fwd.fif` per participant; reuses if present. |
| `inverse.py` | `build_inverse(info, fwd, noise_cov, snr=2)` → returns `InverseOperator`. Computed once. |
| `labels.py` | `load_parcellation(parc='aparc.a2009s')`, `extract_label_courses(stc, labels, src)`. |
| `pipeline.py` | Loop: for each participant × condition → load epochs → build fwd (cached) → noise cov → inv op → loop epochs → `apply_inverse(evoked, inv_op)` → label time courses → bin → write `data/src/Pxx_<cond>_src.csv`. |

The orchestrator script becomes `scripts/02_source_localize.py`, which
just calls `source_localization.pipeline.run(config)`.

### Machine learning

The current `CNV_XGB_4.3.py` is 932 lines and bundles **six distinct
concerns** into one file: data wrangling, feature extraction, three layers
of feature selection (correlation drop / SelectKBest / RFECV / gain-pruning /
SHAP), GridSearchCV, evaluation, and CSV reporting. Worse, the same data
wrangling appears almost verbatim in `CNV_LSTM_3.py` and `CNV_ML_SVM_1.py`.

Suggested split:

| Module | Responsibility | Notes |
|---|---|---|
| `features/amplitude.py` | `binned_mean(epochs, bin_n)` | Lifted from CNV_XGB_4.3.py lines 188–218. Reused by all three model lines. |
| `features/slopes.py` | `binned_slopes(epochs, bin_n)` | Lines 150–181. |
| `features/psd.py` | `band_power(epochs, freqs, bands, bin_n)` | Lines 224–303. |
| `features/assemble.py` | `build_feature_matrix(participant, condition, config) → DataFrame` | Joins amp + psd + slopes + src; the heavy lifting that all model scripts redo. **Cache result to `data/features/Pxx_<cond>_features.parquet`** so model training is fast. |
| `models/feature_selection.py` | `drop_correlated`, `select_kbest`, `rfecv_iterated`, `gain_prune`, `shap_prune` | Each returns the surviving column list. Currently all inline in CNV_XGB_4.3.py. |
| `models/xgb.py` | `make_xgb(scale_pos_weight)`, `XGB_PARAM_GRID` | Just the model factory + grid. |
| `models/lstm.py` | `make_lstm(input_shape, ...)` | Architecture only. |
| `models/svm.py` | `make_svm()`, `SVM_PARAM_GRID` |  |
| `models/train.py` | `train_per_participant(model_factory, param_grid, fs_steps, X, y) → fitted_model, metrics` | The shared participant-loop logic. |
| `models/evaluate.py` | `participant_metrics(y_true, y_pred, y_proba) → dict` and `aggregate(rows) → DataFrame` |  |

`scripts/04_train.py` then becomes a ~50-line orchestrator that picks the
right model factory based on `--model xgb|lstm|svm`, applies the same
feature-selection pipeline, runs the participant loop, and writes a single
results CSV plus model artifacts to `outputs/runs/<run_id>/`.

The R modeling scripts stay in their own `R/` folder with their own
`R/config.R` exposing the same paths/participants. (Optional follow-up:
adopt `{targets}` for an R-side equivalent of Snakemake.)

---

## Q3. How should the scripts work together? Single controller, or something else?

You have three reasonable patterns. From simplest to most rigorous:

### Option A — Config + Makefile (recommended starting point)

A single `configs/default.yaml` is the source of truth. Each
`scripts/0X_*.py` is a CLI entry point that loads the config and runs one
stage. A `Makefile` chains them so `make all` runs the pipeline end-to-end,
and `make train MODEL=xgb` runs just one stage.

```yaml
# configs/default.yaml
paths:
  raw_root: "C:/Users/Aria/OneDrive - .../Participants"
  data_dir: "./data"
  outputs_dir: "./outputs"

participants: [P01, P02, P03, P05, ...]
conditions: [One, Two]

preprocessing:
  bandpass: [0.5, 40]
  notch: 60
  ica_n_components: 0.99

features:
  bin_n: 0.125
  freq_bands:
    Delta: [0.5, 4]
    Theta: [4, 8]
    Alpha: [8, 13]
    Beta:  [13, 30]
    Gamma: [30, 40]

models:
  xgb:
    param_grid: { max_depth: [2,4,8,16], learning_rate: [0.01,0.03,0.05], ... }
  lstm:
    epochs: 50
    batch_size: 32
```

```makefile
# Makefile
preprocess:
	python scripts/01_preprocess.py --config configs/default.yaml

src: preprocess
	python scripts/02_source_localize.py --config configs/default.yaml

features: src
	python scripts/03_extract_features.py --config configs/default.yaml

train: features
	python scripts/04_train.py --model $(MODEL) --config configs/default.yaml

all: train

.PHONY: preprocess src features train all
```

**Why this is the right starting point:** lowest learning curve, no new
dependencies, each stage stays runnable on its own, and Make caches by
file timestamp so unchanged stages don't re-run.

### Option B — Snakemake / DVC pipeline (recommended once stable)

Same idea as Make, but the pipeline runner *understands data dependencies*.
You declare which files each stage produces and consumes; Snakemake or
DVC re-runs only the stages whose inputs changed. DVC adds data
versioning so you can reproduce a result months later.

This is what most research-reproducibility-focused labs eventually adopt.
I'd defer it until the package layout is in place.

### Option C — Single Python controller (`run.py`)

A `run.py` at the root that imports each stage's `pipeline.run(config)`
function and calls them in sequence. Simpler than Make for someone who
prefers everything in Python, but loses the per-stage caching. Fine if
you mostly run end-to-end.

```python
# run.py
import argparse
from eeg_steptype.config import load_config
from eeg_steptype.preprocessing import pipeline as preprocess
from eeg_steptype.source_localization import pipeline as src_loc
from eeg_steptype.features import assemble
from eeg_steptype.models import train

STAGES = {
    "preprocess": preprocess.run,
    "source":     src_loc.run,
    "features":   assemble.run,
    "train":      train.run,
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--stages", nargs="+", default=list(STAGES))
    p.add_argument("--model",  default="xgb")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg["model"] = args.model
    for s in args.stages:
        STAGES[s](cfg)
```

**My recommendation:** start with **Option A (Config + Makefile)**, with a
`run.py` (Option C) on top as a convenience for "run everything in
Python." Adopt Snakemake/DVC (Option B) only when the lab decides
reproducibility-by-content-hash is worth the overhead.

---

## Cross-cutting recommendations

These apply regardless of which orchestration option you pick.

1. **Eliminate the hard-coded `C:/Users/Aria/...` paths.** All paths come
   from `configs/default.yaml`. Per-machine overrides go in
   `configs/local.yaml` (gitignored). This is the single biggest
   portability win.

2. **Cache feature matrices to disk.** `CNV_XGB_4.3.py` currently
   recomputes amplitude / slopes / PSD from `.fif` on every run. Save
   them to `data/features/Pxx_<cond>_features.parquet` once; reload in
   model scripts. (You already do this for source localization — extend
   the same pattern.)

3. **Fix the LORETA inner-loop inefficiency.** Build `noise_cov`, `fwd`,
   and `inv_op` once per participant, not per epoch. (Big speedup.)

4. **Replace `print()` with `logging`.** One config flag controls
   verbosity; logs go to `outputs/runs/<run_id>/run.log`.

5. **Stamp every run.** On entry, `04_train.py` writes
   `outputs/runs/<run_id>/{config.yaml, git_sha.txt, env.txt, metrics.csv}`.
   Lets you reproduce any historical result.

6. **Make `eeg_steptype` an installable package.** `pip install -e .`
   means any script anywhere can `from eeg_steptype.features import ...`
   without `sys.path` hacks. This is what enables "scripts call upon each
   other and import from each other."

7. **Smoke tests.** `tests/test_pipeline.py` runs the full pipeline on
   one or two synthetic participants in <30 s. Catches regressions before
   you discover them mid-run on 30 participants.

8. **Clean `.gitignore`.** Add `.Rproj.user/`, `.RData`, `.Rhistory`,
   `data/`, `outputs/runs/`, `*.fif`, `*.rds`. Several of these are
   currently sitting in the working copy.

9. **Archive aggressively.** Keep the `archive/` folders and the
   `VERSIONS.md` files — they are excellent. Once the package layout
   stabilizes, the `archive/` folders become git history rather than
   live files (move them out of the working tree).

---

## Phased migration plan

You don't need to do this in one pass. Suggested order:

**Phase 1 — Configuration (1 sitting).**
Create `configs/default.yaml`, write `src/eeg_steptype/config.py` to load
it, and replace the `os.chdir(...)` and participant-list literals in the
existing scripts with config reads. No file moves yet. Immediate win:
the scripts run on any machine.

**Phase 2 — Extract shared feature code (1–2 sittings).**
Move the amplitude / slopes / PSD blocks out of `CNV_XGB_4.3.py` into
`src/eeg_steptype/features/`. Have `CNV_XGB_4.3.py`, `CNV_LSTM_3.py`, and
`CNV_ML_SVM_1.py` all import from there. Big de-duplication win.

**Phase 3 — Cache feature matrices (1 sitting).**
Add `scripts/03_extract_features.py` that writes per-participant
parquet files. Model scripts now read parquet instead of recomputing.

**Phase 4 — Fix LORETA + extract source-localization code (1 sitting).**
Move `SRC_writer.py` logic into `src/eeg_steptype/source_localization/`
and pull `noise_cov` / `fwd` / `inv_op` out of the per-epoch loop.

**Phase 5 — Extract feature-selection + model factories (1–2 sittings).**
Move correlation drop, SelectKBest, RFECV, gain prune, SHAP prune into
`models/feature_selection.py`. Move XGB / LSTM / SVM factories into
their own files. `scripts/04_train.py` becomes a thin orchestrator.

**Phase 6 — Makefile + run.py (1 sitting).**
Add the orchestration layer.

**Phase 7 — Tests + CI (optional).**
Smoke tests on synthetic data. GitHub Actions runs them on every push.

---

## Open questions for you

Before any code moves, a few decisions to make:

1. **Configuration format** — YAML is most common in ML research;
   alternatives are TOML (cleaner syntax, native to `pyproject.toml`) or
   Hydra (overkill for this size of project but excellent for sweep-style
   experiments). Default recommendation: YAML.
2. **Auto-ICA tolerance** — `mne-icalabel` returns probabilities per
   component class. What probability threshold do you want for
   auto-exclusion? Conservative (only exclude if `p_artifact > 0.9`) keeps
   more variance but lets some artifact through; aggressive (`p_brain <
   0.5`) is closer to your current hand-picked behaviour on `P25`. I'd
   suggest starting at `p_artifact > 0.8` and tuning against a few hand-
   labelled participants.
3. **Per-participant overrides** — keep them in `configs/overrides/Pxx.yaml`
   (one file per participant), or in a single `configs/overrides.yaml`
   keyed by participant ID? One-file-per-participant scales better and
   diffs more cleanly in git.
4. **Orchestration** — confirm Option A (Config + Makefile + thin
   `run.py`) is the right starting point, or do you want me to sketch
   the Snakemake equivalent in detail?
