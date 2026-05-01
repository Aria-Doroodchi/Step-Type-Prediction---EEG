# LSTM — version history

The current LSTM model lives one folder up at
[`../CNV_LSTM_3.py`](../CNV_LSTM_3.py). This folder preserves the earlier
iterations. The LSTM line went through a heavy mid-development refactor and
then a deliberate simplification.

| Version | Date (mtime) | Lines | Notes |
|---|---|---:|---|
| `CNV_ML_LSTM_1.py`  | 2025-12-06 | 544 | First combined CNV+LSTM script |
| `LSTM_2.py`         | 2025-12-04 | 558 | Standalone LSTM rewrite |
| `LSTM_2.1.py`       | 2025-12-04 | 646 | Adds pruning + hyperparameter tuning |
| `CNV_LSTM_2.2.py`   | 2025-12-05 | 736 | Adds memory monitoring + parallel runs |
| `../CNV_LSTM_3.py` (current) | 2025-12-08 | 519 | Major simplification |

> Note: file mtimes don't strictly match the version-number ordering — the
> `2.x` rewrite was started before `CNV_ML_LSTM_1` was finalized. The
> version numbers reflect the intended development order and should be read
> in numeric order.

---

## CNV_ML_LSTM_1 — first LSTM attempt

A combined script that included data wrangling, PSD computation, feature
selection, and an LSTM trained with Keras. `epochs=200`, K-fold CV, SHAP
analysis. Region structure: data wrangling → epochs → PSD → feature
selection and LSTM preparation.

## CNV_ML_LSTM_1 → LSTM_2.0

`+208 / -194`, 64% similar — this was a near-rewrite that split the LSTM
work into its own focused script. Same core stack (Keras LSTM, K-fold,
SHAP), but the data-wrangling preamble is dropped in favor of a leaner
structure organized purely around feature extraction (`epochs`, `PSD`).

## LSTM_2.0 → LSTM_2.1

`+115 / -27`, 88% similar — mostly additive.

- Adds a dedicated `# region pruning function` for trial pruning during
  training.
- Adds `# region hyperparameter tuning` with explicit search space.
- Adds `# region main & parallel` to support multi-process runs.
- Restructures into Setup → feature extraction → feature selection →
  hyperparameter tuning → main.

## LSTM_2.1 → CNV_LSTM_2.2

`+541 / -451`, **28% similar** — a deep restructure despite the small
version bump.

- Adds `# region Memory Monitoring Setup` (likely in response to OOM during
  earlier parallel runs).
- Pulls all knobs into a `# region global parameters` block.
- Reworks pruning / hyperparameter tuning / parallelization scaffolding.
- Net result is a more modular script with explicit memory and parameter
  control planes.

## CNV_LSTM_2.2 → CNV_LSTM_3 — current

`+447 / -664`, **11% similar** — this is essentially a rewrite. The
direction of travel reverses: the script was intentionally simplified.

- Drops the pruning / parallel / heavy-hyperparameter scaffolding from
  `2.1` and `2.2`.
- Returns to a flatter region layout: data wrangling → epochs → PSD →
  feature selection and LSTM preparation. (Closer in shape to
  `CNV_ML_LSTM_1` than to `2.2`.)
- `epochs` reduced from **200 → 20** for the Keras training loop.
- Adds `GridSearchCV` to the toolkit (was previously RFECV-driven).

---

*To see a precise line-by-line diff between any two versions:*

```bash
diff -u archive/LSTM_2.1.py archive/CNV_LSTM_2.2.py | less
```
