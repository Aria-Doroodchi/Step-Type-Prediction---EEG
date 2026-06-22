# Changelog

## 2026-06-12 — Rich-pooling sub-loop: cheap ANOVA pre-filter (`modeling.pre_kbest`)

Sub-loop on `perf/agentic-improvements` testing cross-subject **partial pooling on the
RICH ~12k-feature set** (the prior loop ran only on the 2.3k fast set). One code change
unblocks it; everything else is config + docs.

### Added

- **`modeling.pre_kbest`** (default `null` = legacy/off) — a cheap univariate ANOVA top-K
  pre-filter in `models/train._fit_score_split`, fit on the **train fold only** and applied
  **before** the correlation drop. The correlation drop builds a dense `p x p` matrix per
  fold (~1.2 GB at 12.3k cols, ~8 GB at 31.7k) — the binding wall-clock cost of the rich
  path, made ~20x heavier by pooling. Cutting `p` with a linear-time top-K first collapses
  it (XGB_MODEL_SUMMARY §4). Leakage-safe (reuses `feature_selection.select_kbest`); the
  null default leaves the per-participant path byte-identical. Legacy/opt-out: leave unset
  or `null`. Enable with an int, e.g. `modeling.pre_kbest: 2000`.
- **`configs/pooling_compare_rich.yaml`** — rich pooled comparison overlay: blocks
  `amplitude(0.125, mean) + slopes + psd` (drops `src`+`cnv_benchmark` so all 20 subjects
  run without the eLORETA b0.125 caches that P35/P39 lack), fresh `cache_tag: rich_nosrc_0125`,
  the light pooled funnel (k_best 150, stability n_subsamples 8 / n_lambda 5), and
  `pre_kbest: 2000`.
- **`configs/pooling_rich.yaml`** — committed train-time overlay (the **recommended HONEST
  rich-pooled config**): the rich no-src frame + light funnel + `pre_kbest: 2000` +
  `modeling.pooling.mode: partial`. Use `scripts/04_train.py --model xgb --config
  configs/pooling_rich.yaml`. Legacy/opt-out: `modeling.pooling.mode: per_participant`.

### Result — CONFIRMED rich partial pooling (20-subject cohort, run `r_rich_conf20`)

| arm | cohort AUC | gap |
|---|---|---|
| recorded rich per-participant (heavy funnel + src, 5×20 CV) | 0.655 | +0.169 |
| per_participant (matched: no-src, light funnel) | 0.5990 | +0.1978 |
| **partial (rich pooled)** | **0.6376** | **−0.0385** |

- **Paired partial − per_participant: +0.0386 AUC** (SE 0.033, t=1.17, 11/20 up) — clears the
  +0.03 bar; t=1.17 ⇒ real but **not** statistically significant (like the fast set's +0.031).
- **Gap collapses +0.1978 → −0.0385** — the robust headline; reproduces at 8 and 20 subjects,
  fast and rich. vs the recorded rich 0.655, the pooled 0.6376 is ~flat (−0.017, within noise)
  but **honest**. Pooling makes the project's best-AUC region trustworthy.
- The 8-subject screen was *pessimistic* (+0.0156) — its subset is enriched for strong subjects;
  the full cohort showed the real lift (shrinkage helps the harder subjects, drags the stars).
- `pre_kbest` global default stays `null` (tractability lever, AUC-neutral). `pooling.mode`
  global default stays `per_participant` (paradigm preservation). `src` not re-added (lower-EV).
- Full write-up: `outputs/perf_loop/RICH_POOLING_SUMMARY.md`; numbers in `LEDGER.md`. 120 tests green.

## 2026-06-11 — Perf loop concluded: plateau after Round 4 (v2.4.1)

The agentic perf-improvement loop reached its stopping condition (3 consecutive no-win
rounds). No code/behavior change — documentation and results only.

### Results

- **Rounds 2–4 produced no further XGB win.** Looser feature funnel (−0.039),
  richer/deeper search (+0.020 screen → +0.001 at cohort scale), and Legendre shape
  features (+0.008) are all null at the 20-subject scale. The pooled XGB is at its ceiling
  on the 2.3k fast feature set. Every promising 8-subject screen lift shrank toward zero on
  the 20-subject confirm — the confirm step killed 2 false positives.
- **CNN confirmation baseline** established: cohort AUC 0.5675 (gap +0.013, 18 subjects).
  No CNN improvement candidate was screened (cost-prohibitive in-session).
- Final: **XGB 0.5674 → 0.5957** (gap +0.166 → −0.014) via Round-1 partial pooling.

### Added / updated

- **`outputs/perf_loop/SUMMARY.md`** — baseline→final, ranked table of all 8 changes tried,
  the winner's key + legacy override, and recommended next steps.
- **`XGB_MODEL_SUMMARY.md` §3.5, `MODELS.md` §7** — pooling section updated with the
  confirmed 20-subject numbers (was "re-run to confirm magnitudes").
- `LEDGER.md` — Rounds 2–4 results and the loop-complete summary.

## 2026-06-10 — Cross-subject partial pooling wired into the train entrypoint (perf loop, v2.4.0)

Agentic perf-improvement loop, Round 1. Cross-subject **partial pooling** confirmed
on the full 20-subject cohort as the strongest lever on the inner-vs-outer overfit gap.

### Added

- **`models/train.py` — `run()` now routes on `modeling.pooling.mode`.** `partial`/`full`
  dispatch the pooled workflow (`models.pooling`) from the normal `04_train.py` path
  (previously reachable only via `scripts/09_pooling_comparison.py`). Tabular models only;
  **tensor models (cnn/eegnet/eegnext) and <2-subject cohorts auto-fall back to
  per_participant**, so the toggle is safe for neural runs and smoke configs.
- **`configs/pooling.yaml`** — committed overlay whose default *is* the improved behavior
  (`modeling.pooling.mode: partial`). Layer on any tier:
  `python scripts/04_train.py --model xgb --config configs/pooling.yaml`.
- **`outputs/perf_loop/`** — the loop's ledger (`LEDGER.md`, source of truth) and
  screen→confirm harness (`aggregate.py`, `screen.sh`, …). `*.log` are gitignored (90 MB+).

### Changed / new config key

- **`modeling.pooling.mode`** — now an explicit, documented key in `configs/default.yaml`.
  - **Default (legacy):** `per_participant` — one model per subject on ~80 epochs (the
    per-subject paradigm; global default unchanged, preserves chronological check + tests).
  - **Improved (confirmed):** `partial` — each subject's train split + all other subjects'
    epochs, same test folds (paired), subject-grouped inner CV. Enable via `configs/pooling.yaml`
    or set the key directly. `full` = leave-one-subject-out transfer.

### Results (20-subject cohort, 2.3k fast feature set, express CV; `r1_pool_confirm20`)

| mode | cohort AUC | overfit gap |
|---|---|---|
| per_participant (baseline) | 0.5646 | +0.173 |
| **partial** | **0.5957** | **−0.014** |
| full | 0.5882 | −0.012 |

Partial vs per_participant (paired, same folds): **+0.031 AUC** (t=1.27, not significant —
the AUC lift is modest/noisy) and a **robust gap collapse (+0.173 → −0.014)**. Pooling
strictly dominates the objective (AUC not worse, guardrail far better). Default kept at
`per_participant` as a deliberate judgment call (a paradigm flip is disproportionate to a
t=1.27 lift); `partial` is the one-line, confirmed-better cross-subject option. 118 tests pass.

## 2026-06-08 — EEGNeXt sophisticated hybrid CNN

### Added

- **`models/eegnext.py`** — a more sophisticated CNN built on the EEGNet-lite
  block, registered as the `eegnext` model / `--speed-tier eegnext`
  (`configs/eegnext.yaml`). Three upgrades over `cnn`/`eegnet`: a **multi-scale
  temporal stem** (parallel temporal convs at several kernel lengths), **squeeze-
  and-excitation channel attention**, and **residual separable blocks**. Keeps
  the hybrid tensor + tabular fusion (`require_source: true`) and the full-CNV
  window. TensorFlow imports stay deferred so the package still imports without
  TF.

### Changed

- **`models/train.py`** — `eegnext` added to `MODEL_FACTORIES` and
  `NEURAL_HYBRID_MODELS`; the two neural-model branch checks now key off
  `NEURAL_HYBRID_MODELS` instead of a hardcoded `{"cnn", "eegnet"}` set so future
  hybrid models slot in automatically.
- **`models/normalization.py`** — routes `eegnext` to its fold-local
  exponential-moving standardizer.
- **`run.py`, `scripts/04_train.py`, `scripts/07_feature_informativeness.py`,
  `scripts/08_tensor_model_diagnostics.py`** — `eegnext` added to the
  `SPEED_TIERS` maps and the tensor / full-CNV model sets.
- **`scripts/06_compare_runs.py`** — `eegnext` added to the screening
  diagnostics: the `--default-tier` choices, the single-tier/tensor-model
  classification sets, and the run-name model/tier inference fallbacks
  (`eegnext` ordered before `eegnet` so the more specific token wins). `eegnext`
  now has full parity with `cnn`/`eegnet` across the performance recorders and
  diagnostic tools (per-run metrics, screening, occlusion).

### Tests

- **`tests/test_imports.py`** — added `eegnet`/`eegnext`/`lstm` to the import
  smoke list (previously only `cnn` was covered) and a
  `test_eegnext_has_full_recorder_and_diagnostic_parity` guard that locks the
  registry, forced-full-channel, normalizer, and diagnostic-script wiring so the
  parity cannot silently regress.

## 2026-05-29 — Shape-decomposition features + stability selection

Two changes targeting (a) information lost when amplitude time courses are
collapsed to per-bin means, and (b) the data-starved 5× 2-fold RFECV at the
small per-participant trial counts.

### Added

- **`features/basis.py`** — shape-decomposition (basis-expansion) features that
  describe each channel's time course by a few coefficients instead of per-bin
  means:
  - `polynomial_basis_features` — orthogonal Legendre/Chebyshev coefficients
    (per-epoch, leakage-free). c0 = level, c1 = CNV-ramp slope, c2 = curvature.
  - `bspline_basis_features` — clamped least-squares B-spline coefficients
    (per-epoch, leakage-free) for localized deflections.
  - `FunctionalPCABasis` — data-driven functional PCA over per-channel amplitude
    bins, implemented as a scikit-learn transformer so it is fit on the training
    fold only (leakage-safe), wired into `models.train` via
    `modeling.feature_selection.fpca`.
  - Opt-in via `features.blocks: [..., basis]`; configured under `features.basis`.
- **`feature_selection.stability_select`** — complementary-pairs stability
  selection (Shah & Samworth 2013) with an elastic-net logistic base. Robust at
  small trial counts, model-agnostic (logistic/svm/xgb), and carries a
  false-discovery bound. Now the **default** in-fold selector
  (`modeling.feature_selection.method: stability`).
- **`tests/test_basis_features.py`, `tests/test_stability_select.py`** — unit
  tests for the basis math and the selector (synthetic data, no MNE needed).

### Changed

- `models/train.py` step 3 now dispatches on `modeling.feature_selection.method`
  (`stability` | `rfecv` | `none`) and applies the optional in-fold fPCA before
  selection. Stability selection replaces iterated RFECV as the default; RFECV
  is retained for comparison runs. ROI channel parsing recognises the new
  `poly_/bspl_/fpca_` columns.
- `configs/default.yaml`, `configs/smoke.yaml` — added `features.basis` and
  `modeling.feature_selection` stanzas; RFECV marked legacy.

### Notes

- Shape decomposition and stability selection compose: orthogonal/fPCA features
  are uncorrelated, which is exactly what makes the elastic-net selector's
  selection frequencies stable.

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
