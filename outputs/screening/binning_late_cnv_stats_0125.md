# Model Screening Results

_Generated 2026-05-16 15:03:06 by `scripts/06_compare_runs.py`_

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
| logistic | express | 200 | 0.4601 | ± 0.0178 | 0.1287 |
| riemannian | riemannian | 500 | 0.5316 | ± 0.0119 | 0.1360 |
| svm | express | 200 | 0.5292 | ± 0.0214 | 0.1543 |
| xgb | express | 200 | 0.5635 | ± 0.0241 | 0.1742 |

## Diagnostic 2 — Tier-response slope

Mean Express AUC minus mean Lightning AUC for the same model. Positive slope means the model rewards heavier optimization budget (more CV repeats, RFECV pass, gain-prune refit). A near-zero slope suggests the model is already near its ceiling; switching model family will pay off more than further tuning. Riemannian has only one tier and is therefore not slope-comparable here.

| model | express_auc | lightning_auc | slope (Express − Lightning) | interpretation |
|---|---|---|---|---|
| logistic | 0.4601 | n/a | n/a | missing tier run |
| riemannian | n/a | n/a | n/a | n/a (single-tier model) |
| svm | 0.5292 | n/a | n/a | missing tier run |
| xgb | 0.5635 | n/a | n/a | missing tier run |

## Diagnostic 3 — Across-fold AUC variance

Standard deviation of per-fold AUC within each participant, averaged across participants. Lower means the model gives more stable fold-to-fold predictions. A high `max_within_participant_sd` flags a participant whose AUC swings wildly between folds — usually a data-quality issue or a fundamentally hard subgroup.

| model | tier | mean_within_participant_sd | max_within_participant_sd | n_participants |
|---|---|---|---|---|
| logistic | express | 0.1154 | 0.1672 | 20 |
| riemannian | riemannian | 0.1241 | 0.1541 | 20 |
| svm | express | 0.1407 | 0.2172 | 20 |
| xgb | express | 0.1229 | 0.1606 | 20 |

## Diagnostic 4 — Inner-vs-outer gap

Mean of (`inner_best_score` − `overall_accuracy`) across folds. Small absolute gap means the hyperparameter search generalizes — the inner CV's optimism matches the outer held-out score. A large positive gap (≥ 0.05) means the model overfits the inner search and further tuning will mostly chase noise. Negative gap means the inner CV was unusually pessimistic relative to the outer fold (can happen when inner folds are small).

| model | tier | mean_gap | median_gap | max_gap | interpretation |
|---|---|---|---|---|---|
| logistic | express | +0.2821 | +0.2812 | +0.7188 | overfits inner CV |
| riemannian | riemannian | +0.0364 | +0.0326 | +0.4710 | mild optimism |
| svm | express | +0.2629 | +0.2566 | +0.6406 | overfits inner CV |
| xgb | express | +0.2347 | +0.2344 | +0.6250 | overfits inner CV |

## Diagnostic 5 — Per-participant model ranking

For each participant, the four models are ranked by mean AUC on the primary tier. The table below counts how often each model finished in each rank. Concentrated rankings (one model usually #1) = homogeneous signal across the cohort. Scattered rankings (each model wins for some subset) = heterogeneous signal — per-participant model selection or an ensemble may help more than tuning one model.

| model | rank_1 | rank_2 | rank_3 | rank_4 |
|---|---|---|---|---|
| logistic | 0 | 2 | 6 | 12 |
| riemannian | 9 | 2 | 6 | 3 |
| svm | 3 | 11 | 4 | 2 |
| xgb | 8 | 5 | 4 | 3 |

### Per-participant AUC matrix

| participant | logistic | riemannian | svm | xgb |
|---|---|---|---|---|
| P01 | 0.3984 | 0.5425 | 0.5383 | 0.4688 |
| P02 | 0.5047 | 0.4816 | 0.7672 | 0.8719 |
| P03 | 0.5203 | 0.5491 | 0.5383 | 0.4031 |
| P05 | 0.5547 | 0.6216 | 0.4828 | 0.4344 |
| P06 | 0.3937 | 0.4319 | 0.5055 | 0.6438 |
| P07 | 0.3859 | 0.5078 | 0.5289 | 0.6750 |
| P08 | 0.4594 | 0.6311 | 0.4900 | 0.5443 |
| P10 | 0.3656 | 0.5425 | 0.5250 | 0.4750 |
| P11 | 0.5016 | 0.4284 | 0.5180 | 0.4562 |
| P12 | 0.4437 | 0.4834 | 0.5828 | 0.6156 |
| P13 | 0.4969 | 0.5453 | 0.4680 | 0.5797 |
| P14 | 0.4094 | 0.4872 | 0.5211 | 0.5297 |
| P15 | 0.4391 | 0.4941 | 0.4938 | 0.4250 |
| P19 | 0.5344 | 0.5084 | 0.5484 | 0.5375 |
| P23 | 0.4205 | 0.6171 | 0.4964 | 0.5801 |
| P24 | 0.5328 | 0.5431 | 0.5766 | 0.5687 |
| P25 | 0.4437 | 0.4834 | 0.5828 | 0.6156 |
| P30 | 0.5609 | 0.5919 | 0.5031 | 0.8703 |
| P35 | 0.4000 | 0.6053 | 0.4437 | 0.5141 |
| P39 | 0.4359 | 0.5363 | 0.4742 | 0.4609 |

## How to read this report

A model has **high potential** (worth further tuning) when its Diagnostic-2 slope is large AND its Diagnostic-4 gap is small. That combination means optimization buys you real generalizing gains. A model with a big slope but a big gap is overfitting its hyperparameter search; additional tuning will mostly improve inner CV without moving outer performance. A model with a small slope is already near its ceiling — switch family rather than tune. A model with high Diagnostic-3 variance is unstable, and any apparent improvement from tuning could be noise.

## Source runs aggregated

| run_id | model | tier | n_rows | n_participants |
|---|---|---|---|---|
| bin_late_cnv_stats_0125_logistic | logistic | express | 200 | 20 |
| bin_late_cnv_riemannian | riemannian | riemannian | 500 | 20 |
| bin_late_cnv_stats_0125_svm | svm | express | 200 | 20 |
| bin_late_cnv_stats_0125_xgb | xgb | express | 200 | 20 |

