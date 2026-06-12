# Closing the inner-vs-outer overfitting gap (XGBoost)

_Companion to [`XGB_MODEL_SUMMARY.md`](../XGB_MODEL_SUMMARY.md). Focus: the
**+0.17 to +0.24 AUC gap** between the inner-CV score the hyperparameter search
sees and the held-out outer-fold AUC actually reported._

**Goal (per discussion):** both *honest numbers* (a non-optimistic inner
estimate) **and** *better generalization* (higher held-out AUC). Runtime is
explicitly out of scope for now.

---

## 1. Why the gap exists here

The gap is not one bug; it is the sum of four things, all amplified by the data
shape — **~80 epochs per subject against ~26k–32k features (`p ≫ n` by
200–400×)**, one model fit **per participant**:

| Source | Mechanism in this repo |
|---|---|
| **Winner's curse** | `inner_best_score` is the **max** over ~25–100 search candidates (`HalvingRandomSearchCV`). The maximum of many noisy CV estimates is upward-biased — the more candidates, the bigger the optimism. |
| **Tiny inner folds** | Inner CV is `StratifiedKFold(2–3)` on a ~64-epoch training fold → ~21–32 epochs per inner-validation. AUC on ~30 samples is extremely noisy, so the search tunes partly to noise. |
| **Selection inside a thin fold** | The funnel (correlation-drop → k-best 500 → stability-selection → gain-prune) is refit per fold, but on ~64 epochs the *selected feature set itself* is high-variance; the model then looks good on the same fold that picked the features. |
| **Metric mismatch** | The inner search optimizes **accuracy** (`scoring: null` → `"accuracy"` in `_make_search_cv`), while the headline is **AUC**. Comparing inner-accuracy to outer-AUC muddies the "gap" number even before any real overfitting. |

The through-line: **with ~80 epochs there is too little data both to tune and to
estimate honestly.** That is also why *more* cross-validation does not help — see
§2.

---

## 2. CV protocol — fewer, larger, grouped folds

> Your instinct is right: at this epoch count a *large* number of CV sets is
> counter-productive. Each extra split shrinks the test fold, and a noisier
> per-fold estimate is exactly what the inner search overfits to.

**2.1 Stop optimizing on micro-folds.** `5-split` outer → ~16-epoch test folds;
`2–3-split` inner → ~21–32-epoch validation. Recommendation: **3-fold outer, 2-
fold inner** at this n. `RepeatedStratifiedKFold` with **more repeats but fewer
splits** keeps the *estimate* stable (averaging over repeats) without starving
each fold. The repeats reduce variance of the *mean*; they do **not** reduce the
optimism of the *max* (that is the grid's job, §3).

**2.2 Honor the trial structure.** Epochs from the same stepping block are
correlated; random stratified folds can put near-duplicate epochs in train and
test, inflating both inner and outer scores. The repo already supports
`modeling.cv.mode: grouped` (`StratifiedGroupKFold` on `block_id`) — using it
makes the outer estimate honest and usually *widens* the measured gap (because
it removes leakage that was propping up the outer score), which is the correct,
non-self-deceiving baseline to improve from.

**2.3 Nested CV must stay nested.** The inner CV is what the search sees; the
gap is only meaningful if the inner folds never touch the outer test fold. That
holds today for the funnel (fit on train only) — keep it that way for any new
step (calibration, fPCA, etc.).

---

## 3. Regularization & the search grid — kill the winner's curse

**3.1 Shrink and curate the grid.** The current grid is large
(`max_depth ∈ {2,4,8,16}`, 6 `gamma`, 4 `reg_lambda`, …). On ~80 epochs,
`max_depth ≥ 8` can only memorize. Concretely:

- Cap depth: `max_depth ∈ {2, 3}`; `min_child_weight ∈ {3, 5, 10}` (force leaves
  to cover real support).
- Push regularization: `reg_lambda ∈ {5, 10, 20}`, `gamma ∈ {0.5, 1, 2}`,
  `colsample_bytree ∈ {0.4, 0.6}`, `subsample ∈ {0.6, 0.8}`.
- **Fewer candidates** (`search.n_iter`): the optimism grows with the number of
  draws, so a tighter, better-reasoned grid both regularizes *and* reduces the
  winner's curse.

**3.2 Apply the "1-SE rule."** Instead of picking the single best candidate,
pick the **simplest model within 1 standard error of the best** inner score.
This is the standard antidote to selection optimism and directly shrinks the
gap. (Implementable as a custom `refit` callable passed to the search.)

**3.3 Fix the depth/rounds coupling.** With halving on `n_estimators`, also lower
`learning_rate` and let early-stopping choose the rounds, rather than tuning a
huge `n_estimators` ceiling.

---

## 4. Feature-selection funnel — fewer survivors for `p ≫ n`

**4.1 Tighten the caps.** `k_best=500` then stability-selection `max_features≈150`
still leaves ~150 features for ~64 training epochs (≈2.3 epochs/feature). Drop to
`k_best ∈ {100, 200}` and stability `max_features ∈ {20, 40}`. Fewer survivors =
less room to fit fold-specific noise = smaller gap.

**4.2 Prefer leakage-free shape features over raw bins.** The repo already has
`basis` (Legendre/B-spline per-epoch coefficients) and in-fold `fpca`. Replacing
thousands of correlated per-bin amplitudes with a handful of shape coefficients
collapses `p` by 1–2 orders of magnitude with little signal loss — the cleanest
structural fix for `p ≫ n`.

**4.3 Make selection stability the metric.** Stability-selection already returns
selection probabilities; require a higher `threshold` (e.g. 0.7) so only
features that survive across subsamples are kept. Unstable features are exactly
the ones that widen the gap.

---

## 5. Evaluation hygiene & calibration

**5.1 Align the inner metric to AUC.** Set `modeling.scoring: roc_auc` so the
search optimizes the same quantity you report. The inner-vs-outer gap then
compares like with like (today it is inner-*accuracy* vs outer-*AUC*).

**5.2 Optimize a proper scoring rule.** Tuning on **log-loss** or **Brier**
(rather than accuracy/AUC) rewards well-calibrated probabilities and is less
prone to the threshold artifacts that inflate small-sample accuracy.

**5.3 Calibrate inside the nested CV.** Wrap the final estimator in
`CalibratedClassifierCV` (isotonic/Platt) fit on the *training* fold only. This
mainly fixes calibration, but combined with a proper scoring rule it reduces the
optimism the search can exploit.

**5.4 Always report the gap.** Treat `inner_best_score − outer_auc` as a
first-class diagnostic in every run rollup (it is already in `metrics.csv`), so
regressions in honesty are visible immediately.

---

## 6. The biggest lever — share data across subjects (implemented)

Items 2–5 trim optimism within the ~80-epoch budget. The **largest** lever is to
escape that budget: pool epochs across subjects (`n ≈ 80 → ≈ 1,520`). More data
makes the inner estimate stable *and* lifts generalization — it attacks both
halves of the goal at once.

Two workflows were added (`src/eeg_steptype/models/pooling.py`), reusing the
**exact** in-fold funnel + nested search, so the only thing that changes between
them and the baseline is *how data is shared*:

| Workflow | Training data for held-out subject *s* | Test set | Question it answers |
|---|---|---|---|
| `per_participant` (baseline) | *s*'s own train split only (~64 ep) | *s*'s held-out fold | within-subject (status quo) |
| **`partial`** | *s*'s train split **+ all other subjects** | *s*'s held-out fold | global prior + local adaptation |
| **`full`** | **all other subjects only** | all of *s* | pure cross-subject transfer (leave-one-subject-out) |

`per_participant` and `partial` share **identical test folds**, so their
difference is a clean paired estimate of what pooling buys. Both pooled modes use
**subject-grouped inner CV** (`StratifiedGroupKFold`, via the new `groups`
argument threaded into `train._fit_score_split`) so tuning never peeks across the
train/test subject boundary — without this, pooling's inner score would itself be
optimistic and defeat the purpose.

Run it:

```bash
python scripts/09_pooling_comparison.py --config configs/pooling_compare.yaml
```

### Empirical comparison

`xgb`, 8-subject subset (P30, P02, P15, P13, P25, P07, P12, P08), reduced
feature set (~2.3k cols) for tractability — the **relative** gap is the point,
not the absolute AUC. Source: `outputs/runs/pooling_compare_demo/pooling_summary.csv`.

| mode | folds | held-out AUC | inner-CV | **gap (inner − outer)** |
|---|---|---|---|---|
| `per_participant` (baseline) | 32 | 0.567 | 0.744 | **+0.177** |
| `full` (leave-subject-out) | 8 | 0.626 | 0.611 | **−0.015** |
| `partial` (prior + local) | 32 | **0.673** | 0.637 | **−0.036** |

**Both goals achieved at once:**

- **The gap collapses** from **+0.177 to ≈0** (slightly negative). The
  subject-grouped inner CV is no longer optimistic — with ~1.5k pooled epochs the
  inner estimate is stable and, if anything, mildly *conservative* vs the
  held-out fold. That is the honest direction.
- **Held-out AUC rises.** `partial` shares the *exact same test folds* as the
  baseline, so its **+0.106 AUC** (0.567 → 0.673) is a clean paired gain — adding
  other subjects' epochs to each subject's training set is pure information gain.
  `full` (pure transfer, no target data) also beats the baseline (0.626) on a
  larger per-fold test set.

Caveat: 8 subjects and a reduced feature set make the per-subject AUC noisy
(`test_auc_sd ≈ 0.19`); the gap-collapse and the partial-pooling AUC lift are the
robust takeaways. Re-run on the full cohort/feature set
(`--config configs/default.yaml --participants …`) to confirm magnitudes.

**Verdict:** of every lever in this document, pooling is the only one that
*both* removes the optimism **and** raises generalization. **`partial` is the
recommended default** (best AUC, honest gap, subject-specific output retained);
`full` is the right tool when you specifically need to quantify cross-subject
transfer.

---

## 7. Recommended order of attack

1. **Align the inner metric to AUC / a proper scoring rule** (§5.1–5.2) — one
   config line; makes the gap *mean* something before you chase it.
2. **Fewer, larger, grouped folds** (§2) — stop manufacturing noise the search
   overfits.
3. **Tighten the grid + 1-SE rule** (§3) — directly removes the winner's curse.
4. **Tighten the funnel / shape features** (§4) — shrink `p` toward `n`.
5. **Adopt pooling** (§6) — the structural fix; `partial` if you need
   subject-specific output, `full` to quantify transfer. Use the comparison
   script to decide per the numbers.

Items 1–4 make the *reported* number trustworthy on the existing per-subject
design; item 5 is the one most likely to **raise** held-out AUC, because it is
the only one that adds information rather than just removing optimism.
