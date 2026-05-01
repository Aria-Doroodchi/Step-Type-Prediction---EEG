# XGBoost — version history

The current XGBoost model lives one folder up at
[`../CNV_XGB_4.3.py`](../CNV_XGB_4.3.py). This folder preserves the previous
iterations from the development line. Notes below summarize the substantive
changes between consecutive versions; line counts and similarity scores are
from a `difflib.SequenceMatcher` pass over the actual files.

| Version | Date (mtime) | Lines | Notes |
|---|---|---:|---|
| `CNV_XGB_4.py`   (v4.0) | 2025-12-16 | 590 | First production XGBoost run |
| `CNV_XGB_4.1.py` (v4.1) | 2025-12-19 | 625 | Restructure + parameter tightening |
| `CNV_XGB_4.2.py` (v4.2) | 2025-12-21 | 784 | Pulls source-localized features from disk |
| `../CNV_XGB_4.3.py` (current) | 2025-12-22 | 932 | Latest; additive over v4.2 |

---

## v4.0 — initial XGBoost pipeline (2025-12-16)

The starting point of the XGBoost line. Pipeline:

- Loads epoch data and runs source localization in-script (eLORETA).
- Builds three feature blocks: electrode amplitude, PSD, and source-space
  activity.
- Feature selection via `SelectKBest`, `RFECV`, and `GridSearchCV`.
- XGBoost classifier with `n_estimators` up to **2000**, K-fold CV.
- SHAP analysis for feature importance.

## v4.0 → v4.1 (2025-12-19)

`+152 / -117` lines, 78% similar to v4.0.

- Introduced an explicit `# region variables and parameters` preamble so all
  knobs (participant list, bin width, feature counts) live in one place.
- Reduced `n_estimators` ceiling from **2000 → 1000** (faster training, fewer
  diminishing-return rounds).
- Tightened `data wrangling` and `feature selection` regions; the
  `electrode amplitude` and `PSD` sub-regions from v4.0 were merged into the
  general data-wrangling block.

## v4.1 → v4.2 (2025-12-21)

`+267 / -108` lines, 73% similar — the largest restructure in this line.

- Stops re-running source localization inside the script. Instead reads the
  pre-computed per-participant CSVs from `../../../src/Pxx_{One,Two}_src.csv`
  (produced by `01_preprocessing/SRC_writer.py`). This is what the `uses_src`
  flag picks up.
- The source-localization region is removed entirely; data wrangling now
  joins electrode amplitude / PSD with the loaded source-space CSVs.
- Net effect: much faster iteration, since per-participant LORETA only has
  to be computed once and cached on disk.

## v4.2 → v4.3 (2025-12-22) — current

`+174 / -26` lines, 88% similar — almost entirely additive.

- Same overall pipeline as v4.2, with extensions to the analysis layer
  (additional SHAP plots, more diagnostic output, more results bookkeeping).
- This is the version promoted to the parent folder as the canonical model.

---

*To see a precise line-by-line diff between any two versions:*

```bash
diff -u archive/CNV_XGB_4.1.py archive/CNV_XGB_4.2.py | less
```
