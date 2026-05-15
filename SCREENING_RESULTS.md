# Model Screening Results

_Generated 2026-05-15 15:56:38 by `scripts/06_compare_runs.py`_

## Overview

- **Models compared:** logistic, riemannian, svm, xgb
- **Tiers:** express, riemannian
- **Participants:** P07, P08, P11, P12, P19, P23, P24, P25, P30, P35, P39 (11 total)
- **Total runs aggregated:** 4

All five diagnostics are computed on the **Express** tier (primary tier; for Riemannian: the `riemannian` tier — its single config). Diagnostic 2 (tier-response slope) additionally consumes the **Lightning** tier runs for the three classical models.

## Diagnostic 1 — Mean test AUC ± 95% CI

Per-fold ROC-AUC averaged across all CV folds × participants, with a Wald 95% CI. Higher mean is better; tighter CI means more consistent estimates.

| model | tier | n_folds | mean_auc | ci95 | sd |
|---|---|---|---|---|---|
| logistic | express | 110 | 0.6485 | ± 0.0291 | 0.1557 |
| riemannian | riemannian | 275 | 0.4888 | ± 0.0176 | 0.1490 |
| svm | express | 110 | 0.6103 | ± 0.0326 | 0.1743 |
| xgb | express | 110 | 0.6535 | ± 0.0308 | 0.1647 |

## Diagnostic 2 — Tier-response slope

Mean Express AUC minus mean Lightning AUC for the same model. Positive slope means the model rewards heavier optimization budget (more CV repeats, RFECV pass, gain-prune refit). A near-zero slope suggests the model is already near its ceiling; switching model family will pay off more than further tuning. Riemannian has only one tier and is therefore not slope-comparable here.

| model | express_auc | lightning_auc | slope (Express − Lightning) | interpretation |
|---|---|---|---|---|
| logistic | 0.6485 | n/a | n/a | missing tier run |
| riemannian | n/a | n/a | n/a | n/a (single-tier model) |
| svm | 0.6103 | n/a | n/a | missing tier run |
| xgb | 0.6535 | n/a | n/a | missing tier run |

## Diagnostic 3 — Across-fold AUC variance

Standard deviation of per-fold AUC within each participant, averaged across participants. Lower means the model gives more stable fold-to-fold predictions. A high `max_within_participant_sd` flags a participant whose AUC swings wildly between folds — usually a data-quality issue or a fundamentally hard subgroup.

| model | tier | mean_within_participant_sd | max_within_participant_sd | n_participants |
|---|---|---|---|---|
| logistic | express | 0.1029 | 0.1557 | 11 |
| riemannian | riemannian | 0.1253 | 0.1512 | 11 |
| svm | express | 0.1296 | 0.2029 | 11 |
| xgb | express | 0.1159 | 0.1981 | 11 |

## Diagnostic 4 — Inner-vs-outer gap

Mean of (`inner_best_score` − `overall_accuracy`) across folds. Small absolute gap means the hyperparameter search generalizes — the inner CV's optimism matches the outer held-out score. A large positive gap (≥ 0.05) means the model overfits the inner search and further tuning will mostly chase noise. Negative gap means the inner CV was unusually pessimistic relative to the outer fold (can happen when inner folds are small).

| model | tier | mean_gap | median_gap | max_gap | interpretation |
|---|---|---|---|---|---|
| logistic | express | +0.2522 | +0.2629 | +0.6406 | overfits inner CV |
| riemannian | riemannian | +0.0551 | +0.0613 | +0.4395 | overfits inner CV |
| svm | express | +0.2591 | +0.2656 | +0.6409 | overfits inner CV |
| xgb | express | +0.1939 | +0.1935 | +0.5761 | overfits inner CV |

## Diagnostic 5 — Per-participant model ranking

For each participant, the four models are ranked by mean AUC on the primary tier. The table below counts how often each model finished in each rank. Concentrated rankings (one model usually #1) = homogeneous signal across the cohort. Scattered rankings (each model wins for some subset) = heterogeneous signal — per-participant model selection or an ensemble may help more than tuning one model.

| model | rank_1 | rank_2 | rank_3 | rank_4 |
|---|---|---|---|---|
| logistic | 5 | 4 | 1 | 1 |
| riemannian | 1 | 1 | 1 | 8 |
| svm | 0 | 6 | 3 | 2 |
| xgb | 5 | 0 | 6 | 0 |

### Per-participant AUC matrix

| participant | logistic | riemannian | svm | xgb |
|---|---|---|---|---|
| P07 | 0.6813 | 0.4516 | 0.5203 | 0.7219 |
| P08 | 0.4502 | 0.6071 | 0.4972 | 0.4821 |
| P11 | 0.5563 | 0.3628 | 0.4719 | 0.6062 |
| P12 | 0.7969 | 0.4263 | 0.7570 | 0.7219 |
| P19 | 0.5516 | 0.5038 | 0.5727 | 0.6203 |
| P23 | 0.6453 | 0.6205 | 0.5750 | 0.5752 |
| P24 | 0.7469 | 0.4603 | 0.7398 | 0.7047 |
| P25 | 0.7969 | 0.4263 | 0.7570 | 0.7219 |
| P30 | 0.7641 | 0.6106 | 0.7547 | 0.9234 |
| P35 | 0.6469 | 0.4291 | 0.6062 | 0.5453 |
| P39 | 0.4969 | 0.4791 | 0.4609 | 0.5656 |

## How to read this report

A model has **high potential** (worth further tuning) when its Diagnostic-2 slope is large AND its Diagnostic-4 gap is small. That combination means optimization buys you real generalizing gains. A model with a big slope but a big gap is overfitting its hyperparameter search; additional tuning will mostly improve inner CV without moving outer performance. A model with a small slope is already near its ceiling — switch family rather than tune. A model with high Diagnostic-3 variance is unstable, and any apparent improvement from tuning could be noise.

## Source runs aggregated

| run_id | model | tier | n_rows | n_participants |
|---|---|---|---|---|
| screen_logistic_richfeats | logistic | express | 110 | 11 |
| screen_riemannian_richfeats | riemannian | riemannian | 275 | 11 |
| screen_svm_richfeats | svm | express | 110 | 11 |
| screen_xgb_richfeats | xgb | express | 110 | 11 |

