# Model Screening Results

_Generated 2026-05-16 18:22:40 by `scripts/06_compare_runs.py`_

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
| logistic | express | 200 | 0.4620 | ± 0.0193 | 0.1391 |
| riemannian | riemannian | 500 | 0.5316 | ± 0.0119 | 0.1360 |
| svm | express | 200 | 0.5188 | ± 0.0215 | 0.1552 |
| xgb | express | 200 | 0.5576 | ± 0.0243 | 0.1755 |

## Diagnostic 2 — Tier-response slope

Mean Express AUC minus mean Lightning AUC for the same model. Positive slope means the model rewards heavier optimization budget (more CV repeats, RFECV pass, gain-prune refit). A near-zero slope suggests the model is already near its ceiling; switching model family will pay off more than further tuning. Riemannian has only one tier and is therefore not slope-comparable here.

| model | express_auc | lightning_auc | slope (Express − Lightning) | interpretation |
|---|---|---|---|---|
| logistic | 0.4620 | n/a | n/a | missing tier run |
| riemannian | n/a | n/a | n/a | n/a (single-tier model) |
| svm | 0.5188 | n/a | n/a | missing tier run |
| xgb | 0.5576 | n/a | n/a | missing tier run |

## Diagnostic 3 — Across-fold AUC variance

Standard deviation of per-fold AUC within each participant, averaged across participants. Lower means the model gives more stable fold-to-fold predictions. A high `max_within_participant_sd` flags a participant whose AUC swings wildly between folds — usually a data-quality issue or a fundamentally hard subgroup.

| model | tier | mean_within_participant_sd | max_within_participant_sd | n_participants |
|---|---|---|---|---|
| logistic | express | 0.1170 | 0.1626 | 20 |
| riemannian | riemannian | 0.1241 | 0.1541 | 20 |
| svm | express | 0.1411 | 0.2048 | 20 |
| xgb | express | 0.1223 | 0.2029 | 20 |

## Diagnostic 4 — Inner-vs-outer gap

Mean of (`inner_best_score` − `overall_accuracy`) across folds. Small absolute gap means the hyperparameter search generalizes — the inner CV's optimism matches the outer held-out score. A large positive gap (≥ 0.05) means the model overfits the inner search and further tuning will mostly chase noise. Negative gap means the inner CV was unusually pessimistic relative to the outer fold (can happen when inner folds are small).

| model | tier | mean_gap | median_gap | max_gap | interpretation |
|---|---|---|---|---|---|
| logistic | express | +0.2774 | +0.2969 | +0.6438 | overfits inner CV |
| riemannian | riemannian | +0.0364 | +0.0326 | +0.4710 | mild optimism |
| svm | express | +0.2705 | +0.2656 | +0.6594 | overfits inner CV |
| xgb | express | +0.2441 | +0.2500 | +0.6562 | overfits inner CV |

## Diagnostic 5 — Per-participant model ranking

For each participant, the four models are ranked by mean AUC on the primary tier. The table below counts how often each model finished in each rank. Concentrated rankings (one model usually #1) = homogeneous signal across the cohort. Scattered rankings (each model wins for some subset) = heterogeneous signal — per-participant model selection or an ensemble may help more than tuning one model.

| model | rank_1 | rank_2 | rank_3 | rank_4 |
|---|---|---|---|---|
| logistic | 2 | 4 | 1 | 13 |
| riemannian | 7 | 4 | 6 | 3 |
| svm | 2 | 8 | 8 | 2 |
| xgb | 9 | 4 | 5 | 2 |

### Per-participant AUC matrix

| participant | logistic | riemannian | svm | xgb |
|---|---|---|---|---|
| P01 | 0.3906 | 0.5425 | 0.4578 | 0.5031 |
| P02 | 0.5047 | 0.4816 | 0.7672 | 0.8719 |
| P03 | 0.5109 | 0.5491 | 0.4891 | 0.4141 |
| P05 | 0.6016 | 0.6216 | 0.5406 | 0.4406 |
| P06 | 0.4156 | 0.4319 | 0.4953 | 0.6359 |
| P07 | 0.3688 | 0.5078 | 0.4594 | 0.6547 |
| P08 | 0.4708 | 0.6311 | 0.4714 | 0.4901 |
| P10 | 0.3516 | 0.5425 | 0.5680 | 0.4234 |
| P11 | 0.5109 | 0.4284 | 0.5070 | 0.4891 |
| P12 | 0.4281 | 0.4834 | 0.5406 | 0.5969 |
| P13 | 0.5578 | 0.5453 | 0.4930 | 0.6109 |
| P14 | 0.4172 | 0.4872 | 0.5461 | 0.4984 |
| P15 | 0.3547 | 0.4941 | 0.4625 | 0.3625 |
| P19 | 0.5906 | 0.5084 | 0.5586 | 0.5188 |
| P23 | 0.3913 | 0.6171 | 0.3963 | 0.5451 |
| P24 | 0.5453 | 0.5431 | 0.4867 | 0.5750 |
| P25 | 0.4281 | 0.4834 | 0.5406 | 0.5969 |
| P30 | 0.5469 | 0.5919 | 0.5695 | 0.8734 |
| P35 | 0.4047 | 0.6053 | 0.5172 | 0.5141 |
| P39 | 0.4500 | 0.5363 | 0.5086 | 0.5375 |

## How to read this report

A model has **high potential** (worth further tuning) when its Diagnostic-2 slope is large AND its Diagnostic-4 gap is small. That combination means optimization buys you real generalizing gains. A model with a big slope but a big gap is overfitting its hyperparameter search; additional tuning will mostly improve inner CV without moving outer performance. A model with a small slope is already near its ceiling — switch family rather than tune. A model with high Diagnostic-3 variance is unstable, and any apparent improvement from tuning could be noise.

## Source runs aggregated

| run_id | model | tier | n_rows | n_participants |
|---|---|---|---|---|
| bin_late_cnv_stats_pyramid_core_logistic | logistic | express | 200 | 20 |
| bin_late_cnv_riemannian | riemannian | riemannian | 500 | 20 |
| bin_late_cnv_stats_pyramid_core_svm | svm | express | 200 | 20 |
| bin_late_cnv_stats_pyramid_core_xgb | xgb | express | 200 | 20 |

