# XGBoost Model — Status Summary

_EEG **step-type** classification — straight (`One`) vs diagonal (`Two`) — from
CNV signals recorded during a stepping task. MSc thesis project._

**Scope:** this document is a focused status report on the **XGBoost model only**
(the project's designated primary model). For the full model roster and the
comparators (SVM, logistic, BiLSTM, Riemannian, CNN, EEGNet, shrinkage-LDA) see
[`MODELS.md`](MODELS.md).

**Status:** living document · **Compiled:** 2026-06-08 · **Owner:** Ali

**Performance source:** recorded full nested-CV runs under `outputs/runs/*xgb*`
(screening runs 2026-05-14 → 2026-05-29), re-aggregated for this report, plus a
fresh same-session reproduction (see [§3.4](#34-fresh-reproduction-this-session)).
AUC = 0.50 is chance.

---

## Contents

- [1. Overall architecture](#1-overall-architecture)
- [2. Tunable characteristics](#2-tunable-characteristics)
- [3. Current performance](#3-current-performance)
- [4. Potential next steps](#4-potential-next-steps-for-improvement)
- [Appendix — how performance was measured](#appendix--how-performance-was-measured)

---

## 1. Overall architecture

XGBoost is a **gradient-boosted decision-tree** ensemble
(`XGBClassifier`, `objective=binary:logistic`, `tree_method=hist`). Trees are
added sequentially; each new tree fits the residual error of the ensemble so
far, and `learning_rate` shrinks each tree's contribution so more trees can be
added before overfitting. It is the project's primary model because it ingests
wide, mixed-scale tabular features without hand-tuned preprocessing, exposes
`feature_importances_` and SHAP for the prune stages, and multithreads natively.

```
features ──► tree 1 ─(residuals)─► tree 2 ─(residuals)─► … ─► tree N
                 └────────────── Σ · learning_rate ──────────────┘ ──► sigmoid ──► P(diagonal)
```

**What the model actually sits inside.** "The XGBoost model" is really
*model + per-fold feature-selection funnel + nested-CV protocol*, fit **once per
participant** (subjects are never pooled). That scaffold is the dominant
architectural fact, because of the data shape it has to cope with:

| Aspect | Value | Why it matters |
|---|---|---|
| Samples per participant | **~80 epochs** (e.g. P30: 40 `One` + 40 `Two`) | Tiny |
| Raw feature columns | **~16,700** (late window) / **~31,700** (full window) | Huge |
| Regime | **p ≫ n by ~200–400×** | Drives everything: the funnel, the regularization grid, and the overfit gap in §3 |

Inputs are the **tabular feature parquet** `(n_epochs, n_features)`: binned
electrode amplitudes (multiple bin widths × {mean, std, min, max, median}),
slopes, PSD band-powers (Morlet TFR), and eLORETA source-space activations.
Class imbalance is handled per fold via `scale_pos_weight = neg/pos`.

### The per-fold feature-selection funnel

Every stage is fit on the **training fold only** and applied to the test fold —
this is what keeps the held-out AUC honest (and the inner-vs-outer gap
meaningful). Order and current defaults (from `models/feature_selection.py` and
the express screening config):

| # | Stage | Default | Effect |
|---|---|---|---|
| 1 | Correlation drop | `|r| > 0.9` | Removes one of every near-duplicate pair |
| 2 | ANOVA F-test k-best | `k = 500` | Univariate top-500 of ~17k–32k |
| 3 | **Stability selection** (elastic-net, complementary-pairs, Shah–Samworth) | default selector, `≤ 150` features kept, prob. threshold 0.6 | Model-agnostic, robust at low trial counts, carries a false-discovery bound — **replaced** the legacy iterated RFECV |
| 4 | Gain prune + refit | `enabled`, drop zero-gain features then re-search | XGB-specific; trims to features the trees actually used |
| 5 | SHAP prune + refit | opt-in (off in express tier) | Drops bottom-quantile features by mean \|SHAP\| |

Legacy iterated **RFECV** (step 3 alternative) is retained for comparison runs
but is no longer the default.

### Nested cross-validation

- **Outer:** `RepeatedStratifiedKFold` — 5 splits × 20 repeats for the full
  publication tier (5 × 2 in the express screening tier). A no-shuffle
  chronological check can run alongside to catch temporal leakage.
- **Inner:** `StratifiedKFold` (2–3 splits) drives the hyperparameter search.
- **Search:** `HalvingRandomSearchCV` with `n_estimators` as the
  successive-halving *resource* (e.g. 50→400 trees, factor 3), so XGB can afford
  a far larger grid than the grid-searched comparators.

---

## 2. Tunable characteristics

### 2a. Model hyperparameters (the search grid)

Verbatim from `configs/default.yaml` (`modeling.xgb.param_grid`):

| Hyperparameter | Grid | What tuning it does |
|---|---|---|
| `n_estimators` | up to 1000 (halving resource) | Boosting rounds. More = lower bias, higher overfit risk; paired with `learning_rate`. |
| `learning_rate` | `0.01, 0.03, 0.05` | Shrinks each tree's step. Lower = needs more trees but generalizes better. |
| `max_depth` | `2, 4, 8, 16` | Max interactions per tree. **The main bias–variance knob** — shallow underfits, deep overfits. |
| `min_child_weight` | `1` | Min summed instance weight per leaf. Higher = more conservative splits. |
| `gamma` | `0, 0.1, 0.3, 0.5, 1, 2` | Min loss reduction to split. Higher = prunes weak splits → simpler trees. |
| `reg_alpha` (L1) | `0, 0.1, 0.5` | L1 penalty → sparsity; drives weak features to zero. |
| `reg_lambda` (L2) | `1, 3, 5, 10` | L2 penalty → smooths/shrinks all leaf weights. |
| `subsample` | `0.6, 0.8, 1.0` | Row fraction per tree; `<1` adds randomness → variance reduction. |
| `colsample_bytree` / `bylevel` | `0.6, 0.7` / `0.6, 0.8` | Column fraction per tree / per level. Decorrelates trees on wide feature sets. |
| `scale_pos_weight` | computed `neg/pos` | Balances the per-fold loss across classes (not searched). |

> In the `p ≫ n` regime here, the **regularization / sampling knobs**
> (`gamma`, `reg_lambda`, `colsample_*`, `subsample`, and a shallow `max_depth`)
> matter more than raw capacity — they are what hold back the inner-CV
> overfitting documented in §3.

### 2b. Pipeline-level knobs that move XGB more than its own grid

These live outside the model factory but are the dials that have actually
changed results so far:

| Knob | Where | Observed leverage |
|---|---|---|
| **Prediction window** | `features.min_time/max_time` (`late_cnv` 1–2 s vs `full_cnv` 0–2 s) | **Largest single effect** — +0.09 to +0.10 cohort AUC (per-subject up to +0.45). |
| Binning recipe | `configs/binning/*` (pyramid/rich/stats, 0.0625–0.5 s) | Within the late window, **barely matters** (0.558–0.568). |
| Channel mode | `--channel-mode full` vs `roi` (9 medial motor) | Tests the foot-motor hypothesis directly; not yet a headline. |
| Feature-selection toggles | `modeling.{rfecv,gain_prune,shap_prune,feature_selection}` | Isolate each funnel stage; default = stability + gain prune. |
| Search budget / CV repeats | speed tiers (`lightning`→`default`) | Trades wall-time vs thoroughness; XGB shows the only positive tier-response slope. |

---

## 3. Current performance

> **Headline:** XGBoost is the **best classical model in the project**, and the
> only one clearly above chance on the primary late window. But (a) its absolute
> AUC is modest (~0.56 late / ~0.65 full), (b) it **overfits the inner CV by
> +0.17 to +0.24 AUC**, and (c) the **prediction window matters more than any
> tuning done so far**.

### 3.1 Cohort AUC by configuration (recorded full nested-CV runs, n = 20)

| Window | Binning | Test AUC | Inner-CV AUC | Overfit gap | Source run |
|---|---|---|---|---|---|
| **late_cnv** (1–2 s) | stats_pyramid_core | **0.558** | 0.794 | **+0.236** | `bin_late_cnv_stats_pyramid_core_xgb` |
| late_cnv | pyramid_mean_core | 0.568 | 0.783 | +0.215 | `bin_late_cnv_pyramid_mean_core_xgb` |
| late_cnv | rich_mean_0125 | 0.568 | 0.778 | +0.210 | `bin_late_cnv_rich_mean_0125_xgb` |
| late_cnv | stats_0125 | 0.564 | 0.783 | +0.220 | `bin_late_cnv_stats_0125_xgb` |
| late_cnv | pyramid_mean_fine | 0.559 | 0.777 | +0.218 | `bin_late_cnv_pyramid_mean_fine_xgb` |
| **full_cnv** (0–2 s) | rich_mean_0125 | **0.655** | 0.824 | **+0.169** | `bin_full_cnv_rich_mean_0125_xgb` |

Smaller early-cohort screens corroborate the level: rich-feature n=11 → 0.654,
express n=8 → 0.578, lightning n=8 → 0.561.

**Reading it:**
- **Window dominates binning.** Five late-window binning recipes span only
  0.558–0.568 — the ceiling is set by the *window*, not the binning. Moving to
  the full window lifts XGB to ~0.655 (+~0.09).
- **Persistent overfitting.** The inner search always looks far better
  (0.78–0.82) than the held-out fold (0.56–0.65). The gap is smaller on the full
  window (+0.17) than the late window (+0.21–0.24) — more real signal leaves less
  room for the search to fit noise.

### 3.2 Per-participant heterogeneity (full_cnv, rich_mean_0125)

Signal is concentrated in a handful of subjects. **12 / 20 participants clear
0.60 AUC; only 2 fall below chance.**

| Tier | Participants (full-CNV AUC) |
|---|---|
| Strong (≥ 0.80) | **P02 0.97**, **P30 0.92**, **P15 0.81** |
| Good (0.65–0.79) | P13 0.75, P12 0.72, P25 0.72, P07 0.72, P24 0.71, P01 0.65 |
| Marginal (0.50–0.64) | P06 0.63, P19 0.62, P11 0.61, P23 0.58, P10 0.57, P39 0.57, P14 0.56, P05 0.55, P35 0.55 |
| Below chance | P08 0.48, P03 0.42 |

The **window effect is also per-subject**: P15 jumps +0.45 AUC (0.36 → 0.81) and
P02 +0.09 (0.87 → 0.97) from late → full, while P03/P08 stay hard in every
configuration. Scattered rankings like this are the main argument for
**per-participant model selection or an ensemble** rather than one global model.

### 3.3 The five screening diagnostics, for XGB

| Diagnostic | What it measures | XGB result |
|---|---|---|
| D1 — mean AUC ± CI | accuracy | Best classical model; 0.56 late / 0.65 full. Others near chance on late. |
| D2 — tier-response slope | does more budget help? | **Only model with a positive slope** (+0.016) — headroom remains. |
| D3 — across-fold variance | stability | Moderate (~0.10–0.16 SD), comparable to peers. |
| D4 — inner-vs-outer gap | overfitting | **+0.17 to +0.24** — overfits (Riemannian alone generalizes, but at chance AUC). |
| D5 — per-participant rank | homogeneity | Most rank-1 finishes of any model; rankings scattered. |

### 3.4 Fresh reproduction (this session)

The **authoritative current-performance measure is §3.1** — recorded full
nested-CV runs (5 splits × 20 repeats, n = 20). Those are more complete than any
quick re-run, and they were re-aggregated from the raw per-fold `metrics.csv`
for this report (not copied from the prior summaries).

A same-session express-tier re-run was launched to (a) reproduce the late-window
number and (b) fill the one genuine gap — the `full_cnv / stats_pyramid_core`
XGB run that was configured but never aggregated
(`outputs/runs/bin_full_cnv_stats_pyramid_core_xgb/` has a config but no
`metrics.csv`). It surfaced a **practical finding worth recording**:

> On the **full window the feature matrix is ~31,700 columns** for
> `stats_pyramid_core` (vs ~80 epochs). The funnel's first step — the
> correlation drop — builds a dense ~31.7k × 31.7k correlation matrix (**~8 GB**)
> **per outer fold**, so a single participant's 10 express folds run to *tens of
> minutes*. This is the binding wall-clock cost of the full-window XGB path and
> is the concrete motivation for next-step #6 (shrink the feature explosion: ROI
> restriction, a cheaper pre-filter before correlation drop, or a lower-variance
> binning).
>
> **Implemented (2026-06-12):** the "cheaper pre-filter before correlation drop"
> is now `modeling.pre_kbest` — a train-fold-only ANOVA top-K applied *before* the
> correlation drop (default `null` = off; e.g. `2000`). It collapses the quadratic
> corr-matrix cost and is what made the rich-feature pooling confirm tractable
> (§3.5). AUC-neutral (the downstream funnel reproduces its selection); enable it
> on wide-feature runs via [`configs/pooling_rich.yaml`](configs/pooling_rich.yaml).

Because §3.1 already shows that **binning barely moves XGB within a window**
(five late recipes span only 0.558–0.568), the `full_cnv / stats_pyramid_core`
number is expected to track the recorded `full_cnv` AUC (~0.65) rather than
reveal anything new — so the authoritative §3.1 numbers stand as the current
performance regardless of the re-run's completion.

### 3.5 Pooling experiment — directly attacking the overfitting gap

The +0.17–0.24 gap (D4 / §3.1) is driven by fitting one model per subject on
~80 epochs against ~26–32k features. The strongest remedy is to **share data
across subjects**. Three workflows were implemented
(`src/eeg_steptype/models/pooling.py`) — all reusing the *exact* in-fold
feature-selection funnel and nested search, so the only thing that differs is
how data is shared — and compared on one shared feature frame:

- **`per_participant`** (baseline): train on the subject's own training split only.
- **`partial`** (global prior + local adapt): the subject's own training split
  **plus all other subjects' epochs**; tested on the subject's held-out fold
  (same test folds as the baseline → a clean paired comparison).
- **`full`** (leave-one-subject-out transfer): train on **all other subjects
  only**, test the entire held-out subject.

Both pooled modes use **subject-grouped inner CV** so hyperparameter tuning never
peeks across the train/test subject boundary (otherwise the inner score would
itself be optimistic).

**Result** (`xgb`, 8-subject subset, reduced ~2.3k-feature set for tractability —
the *relative* gap is the point, not the absolute AUC;
`outputs/runs/pooling_compare_demo/`):

| mode | folds | held-out AUC | inner-CV | **gap (inner − outer)** |
|---|---|---|---|---|
| `per_participant` (baseline) | 32 | 0.567 | 0.744 | **+0.177** |
| `full` (leave-subject-out) | 8 | 0.626 | 0.611 | **−0.015** |
| `partial` (prior + local) | 32 | **0.673** | 0.637 | **−0.036** |

**Pooling achieves both goals at once:**
- **The gap collapses** from **+0.177 to ≈0**. With ~1.5k pooled epochs the
  (grouped) inner estimate is stable and, if anything, mildly *conservative* vs
  the held-out fold — the honest direction.
- **Held-out AUC rises.** `partial` shares the baseline's exact test folds, so its
  **+0.106 AUC** (0.567 → 0.673) is a clean paired gain; `full` (no target data
  at all) still beats the baseline at 0.626.

Caveat: 8 subjects + reduced features make per-subject AUC noisy
(`test_auc_sd ≈ 0.19`); the gap-collapse and partial-pooling lift are the robust
takeaways — re-run on the full cohort/feature set to confirm magnitudes.

**Confirmed on the full 20-subject cohort** (perf loop, `r1_pool_confirm20`):

| mode | cohort AUC | gap |
|---|---|---|
| per_participant (baseline) | 0.5646 | +0.173 |
| **partial** | **0.5957** | **−0.014** |
| full | 0.5882 | −0.012 |

The **gap collapse reproduces robustly** (+0.173 → −0.014); the AUC lift **shrinks**
from the 8-subject +0.106 to **+0.031 paired** (t=1.27, not significant) at cohort scale —
real but modest. Partial pooling is now a confirmed, one-line opt-in
(`modeling.pooling.mode: partial`, committed overlay [`configs/pooling.yaml`](configs/pooling.yaml);
default stays `per_participant`). A subsequent 4-round perf loop found **no further XGB win**
(looser funnel, richer search, and Legendre shape features are all null at cohort scale) —
the pooled model is at its feature-set ceiling. Full rationale and the other (non-pooling)
gap remedies are in [`docs/OVERFITTING_GAP_SOLUTIONS.md`](docs/OVERFITTING_GAP_SOLUTIONS.md);
reproduce with `python scripts/09_pooling_comparison.py --config configs/pooling_compare.yaml`.

**Confirmed on the RICH feature set too** (rich-pooling sub-loop, run `r_rich_conf20`,
20 subjects, `amplitude0.125+slopes+psd` ≈ 9.7k cols — the recorded rich recipe minus
`src`+`cnv_benchmark`, with a new `modeling.pre_kbest` ANOVA pre-filter making it tractable):

| mode | cohort AUC | gap |
|---|---|---|
| per_participant (matched arm) | 0.5990 | +0.1978 |
| **partial** | **0.6376** | **−0.0385** |

Paired **+0.0386 AUC** (t=1.17, n.s.) and the **gap collapses +0.198 → −0.039**. vs the
recorded rich per-participant **0.655 / +0.169** (heavier funnel + src + 5×20 CV) the pooled
0.6376 is ~flat (−0.017, within noise) but now **honest** — pooling makes the project's
best-AUC region trustworthy. The pattern matches the fast set (modest, non-significant AUC
lift; robust gap collapse), now at the higher rich operating point — the rich features and
pooling are largely **complementary**, not redundant. Recommended rich config:
[`configs/pooling_rich.yaml`](configs/pooling_rich.yaml); full write-up in
[`outputs/perf_loop/RICH_POOLING_SUMMARY.md`](outputs/perf_loop/RICH_POOLING_SUMMARY.md).

---

## 4. Potential next steps for improvement

Ordered by expected payoff. Items 1–3 are about **trusting and lifting the
number we have**; items 4–6 are about **getting more out of XGB specifically**.

1. **Resolve the window question — and likely promote the full window.**
   Switching late → full CNV is the biggest lever seen (+0.09 cohort, up to
   +0.45 per subject) and dwarfs any tuning. First **confirm it is not a
   leakage/labeling artifact** (e.g. that the 0–1 s segment isn't carrying
   pre-stimulus or response information it shouldn't), then consider making full
   CNV — or a data-driven wider window — the primary analysis. *Highest leverage,
   do this first.*

2. **Close the inner-vs-outer overfitting gap (+0.17 to +0.24).** The number we
   report is the held-out one, but a search that looks 0.2 AUC better than
   reality is fragile. Within-design fixes: shrink/centre the grid toward the
   regularization knobs (smaller `max_depth` ceiling, stronger
   `reg_lambda`/`gamma`, lower `colsample`), tighten the stability-selection
   feature cap, align the inner search metric to AUC, and add **probability
   calibration** (`CalibratedClassifierCV` inside the nested CV). **The structural
   fix is cross-subject pooling — now implemented and validated in §3.5: it
   collapses the gap to ≈0 *and* raises held-out AUC (+0.106 for `partial`).**
   See [`docs/OVERFITTING_GAP_SOLUTIONS.md`](docs/OVERFITTING_GAP_SOLUTIONS.md).

3. **Make the full-window comparison apples-to-apples.** Complete the partial
   `full_cnv` runs across all binnings and the full cohort (the in-progress
   gap-fill in §3.4 starts this) so the window claim rests on matched cohorts and
   recipes, not mixed ones.

4. **Per-participant model selection / ensembling.** With 12/20 above 0.6, two
   stars (P02, P30 ~0.92–0.97), and two persistently hard subjects (P03, P08),
   one global model is leaving signal on the table. Try (a) picking the best
   model per subject via inner CV, or (b) a soft-voting ensemble of XGB + the
   comparators. The scattered D5 rankings predict this pays off.

5. **Tune XGB on the *right* window with a budget that matches its tier-response.**
   XGB is the only model with a positive tier-response slope, so a fuller search
   (more `n_iter`, the full 5×20 CV) is justified here — but run it on the full
   window, where the gap is smaller and the signal is real, not on the late
   window where it sits near 0.56.

6. **Feature-side experiments, holding the model fixed.** (a) Restrict to the
   9-channel medial-motor ROI (`--channel-mode roi`) to test the foot-motor
   hypothesis and shrink p≫n; (b) run the cheap **shrinkage-LDA CNV benchmark**
   as a floor — if a 9-channel ERP reading matches tuned XGB, that reframes the
   whole effort; (c) ablate the funnel stages (gain/SHAP/stability on/off) to see
   which actually earns its place at these trial counts.

### Decision aid

| If you want… | Then… |
|---|---|
| The single biggest AUC gain | Window first (step 1), not tuning |
| A number you can defend in the thesis | Calibration + gap mitigation (step 2) |
| To beat one global model | Per-participant selection / ensemble (step 4) |
| Interpretability | Stay with XGB (gain/SHAP) — keep the prune stages |

---

## Appendix — how performance was measured

- **Authoritative numbers (§3.1–3.3):** re-aggregated from recorded
  `outputs/runs/*xgb*/metrics.csv`, restricted to the primary
  `repeated_stratified` outer folds (the chronological-check rows are excluded).
  Per-participant AUC = mean over that participant's outer folds; cohort AUC =
  mean over participants.
- **Overfit gap** = `inner_best_score` (inner-CV score of the selected
  hyperparameters) − held-out test AUC, averaged over folds.
- **Fresh reproduction (§3.4):** `scripts/_xgb_perf_snapshot.py`, express tier
  (5 × 2 outer, inner 2-fold, `HalvingRandomSearchCV` `n_iter=25`,
  stability-selection + gain-prune), reusing the saved express config so the
  protocol matches the recorded screening runs.
- **Pooling comparison (§3.5):** `scripts/09_pooling_comparison.py` with
  `configs/pooling_compare.yaml` (8-subject subset, ~2.3k-feature set, all three
  modes on one shared pooled frame, subject-grouped inner CV for the pooled
  modes). Gap = `inner_best_score − held-out AUC`, averaged over folds.
- **Reproduce a recorded run** from its stamped folder:
  `python run.py --config outputs/runs/<run_id>/config.yaml --model xgb`.
