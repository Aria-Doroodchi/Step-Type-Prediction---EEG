# Models

This directory contains the model factories and the per-participant training
driver. The driver (`train.py`) is generic: it picks a factory by name, applies
a configurable feature-selection schedule, runs nested cross-validation, and
returns fold-level metrics.

```
train.py                — generic train/eval driver (nested CV, refits, parallel)
feature_selection.py    — correlation drop, k-best, RFECV, gain prune, SHAP prune
evaluate.py             — per-participant metrics + cohort rollup
normalization.py        — wraps factories in StandardScaler etc. when required
xgb.py                  — XGBoost factory (primary model)
svm.py                  — sklearn SVC factory (comparator)
logistic.py             — sklearn LogisticRegression factory (baseline / smoke)
lstm.py                 — Keras Sequential BiLSTM factory (deep comparator)
riemannian.py           — covariance-based comparator (XdawnCovariances + MDM)
cnn.py                  — convolutional comparator (Braindecode-style)
```

Every model is registered in `MODEL_FACTORIES` in `train.py`. Pick one on the
CLI with `--model {xgb,svm,lstm,logistic}`.

## Training driver — what `train.py` does

For each participant the driver runs nested CV:

1. **Outer CV** — `RepeatedStratifiedKFold` (default) or `StratifiedGroupKFold`
   (grouped) or chronological `KFold`. Configured under `modeling.cv`.
2. Inside each outer fold:
   1. **Correlation drop** — drop one of every |corr|>θ pair (cheap).
   2. **ANOVA F-test** — keep the top-K features.
   3. **Iterated RFECV** *(XGB only; togglable)* — N runs of RFECV with the
      XGB RFECV base estimator, take the union of consistently-kept features
      and the top 80% by mean importance.
   4. **Hyperparameter search** — `HalvingRandomSearchCV` for XGB,
      `GridSearchCV` for others. Inner CV is `StratifiedKFold`.
   5. **Gain prune** *(togglable)* — drop features with zero (or low) XGB
      gain importance, then refit the search on the smaller column set.
   6. **SHAP prune** *(togglable)* — drop the bottom quantile by mean |SHAP|,
      then refit one more time.
3. **Score** the final estimator on the held-out fold; record metrics.

Per-participant metrics are checkpointed to
`outputs/runs/<run_id>/participants/<pid>_metrics.csv`. The cohort metrics CSV
and rollup are written to the run root when all participants finish.

### Parallel participants

When `modeling.parallel.participants > 1` (or `--parallel-participants N` is
passed on the CLI), the driver dispatches participants over a joblib `loky`
pool. Inside each worker the inner XGBoost / sklearn `n_jobs` is pinned to 1
so the joblib pool doesn't oversubscribe the CPU. This is the biggest single
wall-time win once the per-participant work is already trimmed by a speed
tier (see `configs/README.md`).

### Feature-selection toggles

| Config key                          | Default | Effect when `false`                                  |
|-------------------------------------|---------|------------------------------------------------------|
| `modeling.rfecv.enabled`            | `true`  | Skip the iterated RFECV pass entirely.               |
| `modeling.gain_prune.enabled`       | `true`  | Skip gain-prune column subset and its refit.         |
| `modeling.gain_prune.refit`         | `true`  | Compute gain importances but don't refit afterwards. |
| `modeling.shap_prune.enabled`       | derived | Skip the SHAP-prune column subset and its refit.     |
| `modeling.shap_prune.refit`         | `true`  | Don't refit after SHAP pruning.                      |
| `modeling.shap_prune.quantile`      | `0.2`   | Threshold for dropping low-|SHAP| features.          |

Legacy key `modeling.shap_prune_quantile` is still honored if the new
`shap_prune` block isn't provided.

---

## Model architectures

### XGBoost (`xgb.py`) — primary

Gradient-boosted decision trees via `xgboost.XGBClassifier`.

- **Objective**: `binary:logistic`.
- **Tree method**: `hist` (fast histogram-based splits).
- **Class imbalance**: per-fold `scale_pos_weight = neg/pos`.
- **n_estimators**: up to `modeling.xgb.n_estimators` (default 1000; tier configs
  trim this to 200/400/600). Halving search uses `n_estimators` as its
  successive-halving resource budget.
- **Search grid** (`modeling.xgb.param_grid`): `max_depth`, `min_child_weight`,
  `reg_alpha`, `gamma`, `reg_lambda`, `colsample_bytree`, `colsample_bylevel`,
  `learning_rate`, `subsample`.
- **RFECV base** (`make_rfecv_base`): smaller XGB (800 trees, depth 4,
  learning_rate 0.05) used inside `RFECV` to keep that step affordable.

Why it's the primary model: XGBoost handles wide, mixed-scale tabular features
(amplitude bins, slopes, band powers, source-space activations) without
hand-tuned preprocessing, supports `feature_importances_` and SHAP for the
prune stages, and ships with native multithreading.

### SVM (`svm.py`) — comparator

`sklearn.svm.SVC` with `probability=True` and `class_weight="balanced"`.

- **Kernels**: `rbf`, `linear`, `poly` (grid-selected).
- **Hyperparameters**: `C`, `gamma`, `degree` for poly.
- **Preprocessing**: wrapped in `StandardScaler` via `normalization.maybe_wrap_estimator`.
- **Feature selection**: correlation drop and ANOVA k-best only — no RFECV,
  gain, or SHAP (no `feature_importances_`).

Good for confirming whether the gains XGB sees are tree-specific.

### Logistic regression (`logistic.py`) — baseline / smoke

`sklearn.linear_model.LogisticRegression` with `solver="liblinear"`,
`class_weight="balanced"`, `max_iter=2000`.

- **Hyperparameters**: just `C`.
- **Preprocessing**: `StandardScaler` wrapper.
- **Purpose**: smoke-test target (fast end-to-end pipeline check) and linear
  baseline for the comparator table.

### Bidirectional LSTM (`lstm.py`) — deep comparator

Built lazily so the package imports without TensorFlow installed.

```
Input(shape=(n_timesteps, n_features))
Bidirectional(LSTM(units, return_sequences=False))
Dropout(dropout)
Dense(32, activation="relu")
Dense(1, activation="sigmoid")
```

- **Optimizer**: `adam`, loss `binary_crossentropy`.
- **Hyperparameters**: `units` and `dropout` (grid via `modeling.lstm.units_grid`
  / `dropout_grid`).
- **Training**: `epochs=50`, `batch_size=32` by default.
- **Wrapper**: `scikeras.wrappers.KerasClassifier` so it plugs into the same
  GridSearchCV path as the classical models.

The current driver passes one timestep per feature (legacy CNV_LSTM_3.py
behaviour). True per-timestep windowing is left for a future revision.

### Riemannian (`riemannian.py`) — covariance-based comparator

Tensor-input pipeline that runs directly on `(n_epochs, n_channels, n_times)`
rather than the flat feature parquet. Architecture:

```
RiemannianFeatureUnion(
    xDAWN covariance        -> tangent space  (nfilter sweeps 2/4/6),
    broadband covariance    -> tangent space,
    FBCSP log-variance      (Mu, Beta bands),
) -> concatenate -> BalancedShrinkageLDA (priors=[0.5, 0.5])
```

- **Covariance estimator**: OAS shrinkage by default (`modeling.riemannian.covariance_estimator`); the search grid also sweeps Ledoit-Wolf (`lwf`).
- **xDAWN**: spatial filter tuned for ERP/SCP shapes -- a good fit for the CNV. `nfilter` swept 2/4/6 in the grid.
- **FBCSP block**: log-variance features for Mu (8-13 Hz) and Beta (13-30 Hz). Filtering itself is left to the eventual tensor-data preprocessing path; the block captures the fold-local contract today.
- **Final classifier**: `BalancedShrinkageLDA` — `sklearn`'s LDA with `solver=lsqr`, `shrinkage=auto`, `priors=[0.5, 0.5]` for minor class imbalance.
- **Feature selection**: none. Correlation drop, k-best, RFECV, gain-prune, SHAP-prune are all skipped by the tensor-input training path — they're undefined on `(n_epochs, n_channels, n_times)` input.
- **Data path**: reads cached `.npz` tensors from `data/features/tensor/`, built on first use by `features.tensor.build_tensor_for_participant`. The classical feature parquet is never touched.
- **Window**: defaults to `full_cnv` (0-2 s) via the `riemannian` overlay because covariance structure changes meaningfully across the whole motor-preparation interval.

Activate end-to-end with:

```bash
python run.py --speed-tier riemannian              # tier sets default_model: riemannian
python run.py --speed-tier riemannian --model riemannian   # explicit form
```

### CNN — future comparator

Stub in `cnn.py`. The config carries placeholder hyperparameters under
`modeling.cnn` but the factory isn't yet wired into `MODEL_FACTORIES`. When
it is, it'll share the tensor-input data path with the Riemannian model
(same `data/features/tensor/<pid>_<cond>_*.npz` cache).

---

## Running

```bash
# Primary model, default settings:
python scripts/04_train.py --model xgb

# Single participant, lightning tier:
python scripts/04_train.py --model xgb --speed-tier lightning \
    --config configs/single.yaml

# Full cohort, parallel, balanced speed:
python scripts/04_train.py --model xgb --speed-tier express \
    --parallel-participants 8

# Comparator sweep at quick tier:
for m in xgb svm logistic; do
  python scripts/04_train.py --model $m --speed-tier quick
done
```

See `configs/README.md` for speed-tier details and the AUC trade-off table.
