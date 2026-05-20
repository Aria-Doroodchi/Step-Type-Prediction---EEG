# Model Screening Results

_Generated 2026-05-16 15:42:21 by `scripts/06_compare_runs.py`_

## Overview

- **Models compared:** logistic, riemannian, svm, xgb
- **Tiers:** express, riemannian
- **Participants:** P01, P02, P03, P05, P06, P07, P08, P10, P11, P12, P13, P14, P15, P19, P23, P24, P25, P30, P35, P39 (20 total)
- **Total runs aggregated:** 4

All five diagnostics are computed on the **Express** tier (primary tier; for Riemannian: the `riemannian` tier — its single config). Diagnostic 2 (tier-response slope) additionally consumes the **Lightning** tier runs for the three classical models.

## Diagnostic 1 — Mean test AUC ± 95% CI

Per-fold ROC-AUC averaged across all CV folds × participants, with a Wald 95% CI. Higher mean is better; tighter CI means more consistent estimates.

| model | tier | n_folds | mean_auc | ci95 | sd |
|---|---|---|---|---|---|
| logistic | express | 200 | 0.4569 | ± 0.0174 | 0.1256 |
| riemannian | riemannian | 500 | 0.5316 | ± 0.0119 | 0.1360 |
| svm | express | 200 | 0.5335 | ± 0.0218 | 0.1570 |
| xgb | express | 200 | 0.5678 | ± 0.0244 | 0.1757 |

## Diagnostic 2 — Tier-response slope

Mean Express AUC minus mean Lightning AUC for the same model. Positive slope means the model rewards heavier optimization budget (more CV repeats, RFECV pass, gain-prune refit). A near-zero slope suggests the model is already near its ceiling; switching model family will pay off more than further tuning. Riemannian has only one tier and is therefore not slope-comparable here.

| model | express_auc | lightning_auc | slope (Express − Lightning) | interpretation |
|---|---|---|---|---|
| logistic | 0.4569 | n/a | n/a | missing tier run |
| riemannian | n/a | n/a | n/a | n/a (single-tier model) |
| svm | 0.5335 | n/a | n/a | missing tier run |
| xgb | 0.5678 | n/a | n/a | missing tier run |

## Diagnostic 3 — Across-fold AUC variance

Standard deviation of per-fold AUC within each participant, averaged across participants. Lower means the model gives more stable fold-to-fold predictions. A high `max_within_participant_sd` flags a participant whose AUC swings wildly between folds — usually a data-quality issue or a fundamentally hard subgroup.

| model | tier | mean_within_participant_sd | max_within_participant_sd | n_participants |
|---|---|---|---|---|
| logistic | express | 0.1104 | 0.1635 | 20 |
| riemannian | riemannian | 0.1241 | 0.1541 | 20 |
| svm | express | 0.1441 | 0.2188 | 20 |
| xgb | express | 0.1230 | 0.1863 | 20 |

## Diagnostic 4 — Inner-vs-outer gap

Mean of (`inner_best_score` − `overall_accuracy`) across folds. Small absolute gap means the hyperparameter search generalizes — the inner CV's optimism matches the outer held-out score. A large positive gap (≥ 0.05) means the model overfits the inner search and further tuning will mostly chase noise. Negative gap means the inner CV was unusually pessimistic relative to the outer fold (can happen when inner folds are small).

| model | tier | mean_gap | median_gap | max_gap | interpretation |
|---|---|---|---|---|---|
| logistic | express | +0.2845 | +0.2870 | +0.7031 | overfits inner CV |
| riemannian | riemannian | +0.0364 | +0.0326 | +0.4710 | mild optimism |
| svm | express | +0.2540 | +0.2500 | +0.6064 | overfits inner CV |
| xgb | express | +0.2290 | +0.2344 | +0.5938 | overfits inner CV |

## Diagnostic 5 — Per-participant model ranking

For each participant, the four models are ranked by mean AUC on the primary tier. The table below counts how often each model finished in each rank. Concentrated rankings (one model usually #1) = homogeneous signal across the cohort. Scattered rankings (each model wins for some subset) = heterogeneous signal — per-participant model selection or an ensemble may help more than tuning one model.

| model | rank_1 | rank_2 | rank_3 | rank_4 |
|---|---|---|---|---|
| logistic | 0 | 3 | 5 | 12 |
| riemannian | 8 | 3 | 6 | 3 |
| svm | 3 | 10 | 6 | 1 |
| xgb | 9 | 4 | 3 | 4 |

### Per-participant AUC matrix

| participant | logistic | riemannian | svm | xgb |
|---|---|---|---|---|
| P01 | 0.3797 | 0.5425 | 0.4250 | 0.5375 |
| P02 | 0.5047 | 0.4816 | 0.7672 | 0.8719 |
| P03 | 0.5281 | 0.5491 | 0.5437 | 0.4094 |
| P05 | 0.5422 | 0.6216 | 0.4930 | 0.4297 |
| P06 | 0.3984 | 0.4319 | 0.4531 | 0.6578 |
| P07 | 0.3891 | 0.5078 | 0.5469 | 0.6672 |
| P08 | 0.4496 | 0.6311 | 0.5233 | 0.5042 |
| P10 | 0.3531 | 0.5425 | 0.5062 | 0.4984 |
| P11 | 0.5172 | 0.4284 | 0.5297 | 0.4328 |
| P12 | 0.4344 | 0.4834 | 0.5594 | 0.5984 |
| P13 | 0.5047 | 0.5453 | 0.5250 | 0.5625 |
| P14 | 0.3922 | 0.4872 | 0.5148 | 0.5641 |
| P15 | 0.4531 | 0.4941 | 0.5734 | 0.4328 |
| P19 | 0.5406 | 0.5084 | 0.5180 | 0.5844 |
| P23 | 0.3920 | 0.6171 | 0.4984 | 0.5917 |
| P24 | 0.5312 | 0.5431 | 0.5859 | 0.5609 |
| P25 | 0.4344 | 0.4834 | 0.5594 | 0.5984 |
| P30 | 0.5500 | 0.5919 | 0.5328 | 0.8875 |
| P35 | 0.3953 | 0.6053 | 0.5109 | 0.5250 |
| P39 | 0.4484 | 0.5363 | 0.5031 | 0.4422 |

## How to read this report

A model has **high potential** (worth further tuning) when its Diagnostic-2 slope is large AND its Diagnostic-4 gap is small. That combination means optimization buys you real generalizing gains. A model with a big slope but a big gap is overfitting its hyperparameter search; additional tuning will mostly improve inner CV without moving outer performance. A model with a small slope is already near its ceiling — switch family rather than tune. A model with high Diagnostic-3 variance is unstable, and any apparent improvement from tuning could be noise.

## Source runs aggregated

| run_id | model | tier | n_rows | n_participants |
|---|---|---|---|---|
| bin_late_cnv_pyramid_mean_core_logistic | logistic | express | 200 | 20 |
| bin_late_cnv_riemannian | riemannian | riemannian | 500 | 20 |
| bin_late_cnv_pyramid_mean_core_svm | svm | express | 200 | 20 |
| bin_late_cnv_pyramid_mean_core_xgb | xgb | express | 200 | 20 |

