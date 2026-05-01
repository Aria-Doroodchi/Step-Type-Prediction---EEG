# Early ML iterations — version history

These are the earliest electrode-level baseline scripts (the `CNV_ML_*`
series) that preceded the dedicated XGBoost and LSTM lines. They live in
this archive because their direct descendants are the current
`02_models/xgboost/CNV_XGB_4.3.py` and `02_models/lstm/CNV_LSTM_3.py`.

The whole series ran from early November to early December 2025. The
version numbers in filenames are the intended ordering; file mtimes are
sometimes slightly out of order because earlier scripts got re-edited
during development.

| Version | Date (mtime) | Lines | Notes |
|---|---|---:|---|
| `CNV_ML_draft.py`   | 2025-11-18 | 227 | Earliest sketch — feature-selection scaffolding only |
| `CNV_ML.py`         | 2025-11-08 | 488 | First end-to-end pipeline (XGBoost + Keras) |
| `CNV_ML_2.py`       | 2025-11-07 | 308 | Slimmed to Keras-only |
| `CNV_ML_3.py`       | 2025-11-12 | 347 | Reintroduces XGBoost; adds RFECV + K-fold |
| `CNV_ML_3.1.py`     | 2025-11-18 | 396 | Bumps `n_estimators` 1000 → 2000 |
| `CNV_ML_3.2.py`     | 2025-11-18 | 412 | Adds `SelectKBest` |
| `CNV_ML_3.3.py`     | 2025-12-03 | 460 | Adds explicit feature-selection region |
| `CNV_ML_3.4.py`     | 2025-12-01 | 119 | Stub focused only on LORETA loading |
| `CNV_ML_3.4.1.py`   | 2025-12-03 | 564 | Full pipeline rebuilt on top of source localization |

---

## CNV_ML_draft — sketch

Only 227 lines. Imports SHAP and matplotlib but no model class yet.
Regions: setting up the environment → data wrangling → PSD → feature
selection. Acts as a planning sketch rather than a runnable pipeline.

## CNV_ML — first complete pipeline

`+345 / -84` from the draft, 40% similar. First fully end-to-end script:
loads data, computes PSD, fits both an XGBoost classifier
(`n_estimators=2000`) and a Keras model (`epochs=200`), runs SHAP. Region
structure: setting up the environment → data wrangling → PSD → ML.

## CNV_ML → CNV_ML_2

`+286 / -466`, **6% similar** — essentially a rewrite. XGBoost is dropped;
script focuses on a Keras-only path. Adds an explicit `epochs` region.
Significantly shorter (488 → 308 lines).

## CNV_ML_2 → CNV_ML_3

`+141 / -102`, 63% similar.

- XGBoost is reintroduced alongside Keras.
- Adds `RFECV` for feature selection.
- Adds K-fold cross-validation.
- `n_estimators` reduced to **1000**.

## CNV_ML_3 → CNV_ML_3.1

`+91 / -42`, 82% similar — minor.

- Restores `n_estimators` to **2000**.
- Small tweaks to data wrangling and feature handling.

## CNV_ML_3.1 → CNV_ML_3.2

`+21 / -5`, **97% similar** — a small-but-targeted change.

- Adds `SelectKBest` to the feature-selection toolkit (now alongside
  `RFECV`). This combination becomes standard in all later scripts.

## CNV_ML_3.2 → CNV_ML_3.3

`+117 / -69`, 79% similar.

- Adds an explicit `# region feature selection` block (previously feature
  selection was inline in the ML region).
- General cleanup of the data-wrangling code.

## CNV_ML_3.3 → CNV_ML_3.4

`+75 / -416`, **15% similar** — a deliberate prune.

- Drops XGBoost, Keras, RFECV, SelectKBest, K-fold — *everything*. Down to
  119 lines.
- Adds the `LORETA` flag (eLORETA source localization).
- This file is best read as a focused experimental stub: rebuild the
  pipeline starting from source-localized features instead of electrode
  amplitude / PSD.

## CNV_ML_3.4 → CNV_ML_3.4.1 — final in this line

`+515 / -70`, **14% similar** — the stub from `3.4` is expanded back into
a full pipeline.

- Rebuilds the full toolkit (`xgboost`, `keras`, `shap`, `RFECV`,
  `SelectKBest`, K-fold) on top of the source-localization layer.
- Region structure: setting up the environment → data wrangling →
  **source localization** → electrode amplitude → PSD → feature selection.
  This three-feature-block layout (source / electrode / PSD) is the
  structural ancestor of `02_models/xgboost/archive/CNV_XGB_4.py`.

After `3.4.1` the development line forks: the XGBoost-focused path
becomes the `CNV_XGB_4.x` series, and the LSTM-focused path becomes the
`LSTM_2.x` / `CNV_LSTM_*` series.

---

*To see a precise line-by-line diff between any two versions:*

```bash
diff -u CNV_ML_3.1.py CNV_ML_3.2.py | less
```
