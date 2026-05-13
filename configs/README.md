# Speed tiers

The training stage in this pipeline is by far the most expensive: every participant
runs nested CV with iterated RFECV, a HalvingRandomSearch hyperparameter sweep,
gain-prune refit, and a SHAP-prune refit. At default settings a single participant
can take an hour or more on a workstation, and the full cohort is an overnight job.

To make the workflow usable for day-to-day iteration we ship three "speed tier"
overlay configs. Each one layers on top of `default.yaml` (the slow, research-grade
config remains the authority for final results). Tiers are exposed as a CLI flag
that picks the matching YAML; you can also point `--config` at the tier file
directly.

```bash
# Recommended day-to-day workflow:
python scripts/04_train.py --model xgb --speed-tier express

# Single-participant smoke loop:
python scripts/04_train.py --model xgb --speed-tier lightning \
    --config configs/single.yaml          # P25 only, lightning settings

# Closest-to-research, multi-participant parallel run:
python run.py --stages train --speed-tier quick --parallel-participants 6
```

## Tier comparison

| Tier         | Model       | Time/participant | Expected AUC drop | RFECV               | Hyperparameter search                              | Gain prune refit | SHAP prune refit | Outer CV       |
|--------------|-------------|------------------|-------------------|---------------------|----------------------------------------------------|------------------|------------------|----------------|
| `lightning`  | xgb / svm / logistic | ~1-3 min  | 5-10%             | **skipped**         | grid (≤2 candidates), `n_estimators=200`           | **skipped**      | **skipped**      | 3 folds × 1    |
| `express`    | xgb / svm / logistic | ~4-8 min  | 2-4%              | 1 iter, step 0.2    | halving random, 25 iters, `n_estimators≤400`      | kept             | **skipped**      | 5 folds × 2    |
| `quick`      | xgb / svm / logistic | ~10-15 min| <1.5%             | 2 iters, step 0.1   | halving random, 50 iters, `n_estimators≤600`      | kept             | kept (1 pass)    | 5 folds × 5    |
| `riemannian` | riemannian (auto) | ~0.5-2 min | (separate baseline) | n/a (tensor input) | grid over `features__nfilter` × covariance estimator | n/a              | n/a              | 5 folds × 5    |
| `default`    | xgb (and others, slow) | ~60-180 min | (baseline)    | 5 iters, step 0.05  | halving random, 100 iters, `n_estimators≤1000`    | kept             | kept (1 pass)    | 5 folds × 20   |

The "time/participant" figures assume a typical multi-core workstation with
parallel-participant execution disabled (single-participant timing). On the full
cohort, total wall time scales as `≈ time_per_participant × n_participants /
parallel.participants` once the joblib pool is saturated.

### When to use which

- **lightning** — pure dev loop. You are debugging a feature path, sanity-checking
  a code change, or generating mock metrics to validate downstream plotting. Do
  not report these numbers; the AUC drop varies a lot fold to fold because the CV
  is so shallow.
- **express** — daily driver. Numbers are stable enough to compare configurations
  (e.g. ROI vs full channels), pick prediction windows, or screen feature blocks
  before a final run.
- **quick** — pre-final-run. Almost matches default behavior except for the
  CV-repeat budget. Good for sharing preliminary results internally.
- **riemannian** — covariance-based comparator. Different model family (not a
  speed-trimmed XGB), different data path (epoch tensors at
  `data/features/tensor/`, not the flat parquet), and different default window
  (`full_cnv` 0-2 s). Use as a comparator to the tree/kernel models, not as a
  faster XGB. The "AUC drop" column doesn't apply because there's no XGB
  baseline being approximated.
- **default.yaml** — leave this untouched for the final, archived run that gets
  written up.

## What each tier changes

All three tiers obey the same set of `modeling.*` toggles:

- `modeling.rfecv.enabled` — skip the iterated RFECV pass entirely (lightning).
- `modeling.gain_prune.enabled` / `modeling.gain_prune.refit` — keep the prune
  but skip the refit (lightning), or run both (express, quick).
- `modeling.shap_prune.enabled` / `modeling.shap_prune.refit` / `quantile` —
  skip SHAP (lightning, express) or do one pass (quick, default).
- `modeling.parallel.participants` — number of participants to train concurrently
  with joblib. Follows the project's negative-means-reserve convention: `-8`
  resolves to "all available cores except 8". Tier defaults are `-8` for
  lightning/express and `-10` for quick (SHAP raises peak RAM, so quick
  reserves two extra cores' worth of headroom). When the resolved count is
  > 1, inner XGBoost/sklearn workers are pinned to one thread inside each
  subprocess to avoid oversubscription.

The XGB hyperparameter grids in each tier are shrunk in lockstep with the search
budget so that the halving / grid search remains balanced. SVM and logistic-
regression grids are similarly trimmed.

## Adding your own tier

Tiers are plain overlay YAMLs. The deep-merge in `eeg_steptype/config.py` only
overrides the keys you specify, so a custom tier can be three lines if all you
want to do is, say, disable SHAP from `default.yaml`:

```yaml
# configs/my_tier.yaml
modeling:
  shap_prune:
    enabled: false
```

Then either:

```bash
python run.py --stages train --config configs/my_tier.yaml
```

or register it in `SPEED_TIERS` in `run.py` and `scripts/04_train.py` to expose
it as a `--speed-tier` choice.

## Related files

- `configs/default.yaml` — research-grade baseline (slow path).
- `configs/smoke.yaml` — end-to-end pipeline smoke test in seconds.
- `configs/single.yaml` — one-participant convenience overlay (`participants: [P25]`).
- `src/eeg_steptype/models/README.md` — model architectures and feature-selection layers.
