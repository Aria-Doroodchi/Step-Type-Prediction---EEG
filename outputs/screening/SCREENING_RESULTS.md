# Model Screening Results

_Generated 2026-05-14 16:29:52 by `scripts/06_compare_runs.py`_

## Overview

- **Models compared:** logistic, riemannian, svm, xgb
- **Tiers:** express, lightning, riemannian
- **Participants:** P08, P11, P19, P23, P24, P25, P30, P39 (8 total)
- **Total runs aggregated:** 7

All five diagnostics are computed on the **Express** tier (primary tier; for Riemannian: the `riemannian` tier — its single config). Diagnostic 2 (tier-response slope) additionally consumes the **Lightning** tier runs for the three classical models.

## Diagnostic 1 — Mean test AUC ± 95% CI

Per-fold ROC-AUC averaged across all CV folds × participants, with a Wald 95% CI. Higher mean is better; tighter CI means more consistent estimates.

| model | tier | n_folds | mean_auc | ci95 | sd |
|---|---|---|---|---|---|
| logistic | express | 80 | 0.4822 | ± 0.0268 | 0.1223 |
| logistic | lightning | 24 | 0.4755 | ± 0.0361 | 0.0903 |
| riemannian | riemannian | 200 | 0.5088 | ± 0.0212 | 0.1530 |
| svm | express | 80 | 0.4905 | ± 0.0266 | 0.1212 |
| svm | lightning | 24 | 0.4802 | ± 0.0358 | 0.0894 |
| xgb | express | 80 | 0.5777 | ± 0.0402 | 0.1833 |
| xgb | lightning | 24 | 0.5614 | ± 0.0534 | 0.1333 |

## Diagnostic 2 — Tier-response slope

Mean Express AUC minus mean Lightning AUC for the same model. Positive slope means the model rewards heavier optimization budget (more CV repeats, RFECV pass, gain-prune refit). A near-zero slope suggests the model is already near its ceiling; switching model family will pay off more than further tuning. Riemannian has only one tier and is therefore not slope-comparable here.

| model | express_auc | lightning_auc | slope (Express − Lightning) | interpretation |
|---|---|---|---|---|
| logistic | 0.4822 | 0.4755 | +0.0068 | near ceiling — flat response |
| riemannian | n/a | n/a | n/a | n/a (single-tier model) |
| svm | 0.4905 | 0.4802 | +0.0104 | near ceiling — flat response |
| xgb | 0.5777 | 0.5614 | +0.0163 | moderate headroom |

## Diagnostic 3 — Across-fold AUC variance

Standard deviation of per-fold AUC within each participant, averaged across participants. Lower means the model gives more stable fold-to-fold predictions. A high `max_within_participant_sd` flags a participant whose AUC swings wildly between folds — usually a data-quality issue or a fundamentally hard subgroup.

| model | tier | mean_within_participant_sd | max_within_participant_sd | n_participants |
|---|---|---|---|---|
| logistic | express | 0.1083 | 0.1478 | 8 |
| riemannian | riemannian | 0.1255 | 0.1512 | 8 |
| svm | express | 0.1199 | 0.1721 | 8 |
| xgb | express | 0.1209 | 0.1539 | 8 |

## Diagnostic 4 — Inner-vs-outer gap

Mean of (`inner_best_score` − `overall_accuracy`) across folds. Small absolute gap means the hyperparameter search generalizes — the inner CV's optimism matches the outer held-out score. A large positive gap (≥ 0.05) means the model overfits the inner search and further tuning will mostly chase noise. Negative gap means the inner CV was unusually pessimistic relative to the outer fold (can happen when inner folds are small).

| model | tier | mean_gap | median_gap | max_gap | interpretation |
|---|---|---|---|---|---|
| logistic | express | +0.2825 | +0.2812 | +0.6772 | overfits inner CV |
| riemannian | riemannian | +0.0578 | +0.0626 | +0.4380 | overfits inner CV |
| svm | express | +0.2584 | +0.2500 | +0.6052 | overfits inner CV |
| xgb | express | +0.2031 | +0.2031 | +0.5781 | overfits inner CV |

## Diagnostic 5 — Per-participant model ranking

For each participant, the four models are ranked by mean AUC on the primary tier. The table below counts how often each model finished in each rank. Concentrated rankings (one model usually #1) = homogeneous signal across the cohort. Scattered rankings (each model wins for some subset) = heterogeneous signal — per-participant model selection or an ensemble may help more than tuning one model.

| model | rank_1 | rank_2 | rank_3 | rank_4 |
|---|---|---|---|---|
| logistic | 1 | 2 | 3 | 2 |
| riemannian | 2 | 1 | 2 | 3 |
| svm | 0 | 3 | 2 | 3 |
| xgb | 5 | 2 | 1 | 0 |

### Per-participant AUC matrix

| participant | logistic | riemannian | svm | xgb |
|---|---|---|---|---|
| P08 | 0.4331 | 0.6071 | 0.4293 | 0.4918 |
| P11 | 0.5094 | 0.3628 | 0.4750 | 0.3828 |
| P19 | 0.5406 | 0.5038 | 0.4992 | 0.5859 |
| P23 | 0.3670 | 0.6205 | 0.4809 | 0.6036 |
| P24 | 0.5500 | 0.4603 | 0.5328 | 0.6094 |
| P25 | 0.4656 | 0.4263 | 0.5172 | 0.5312 |
| P30 | 0.5484 | 0.6106 | 0.5062 | 0.8984 |
| P39 | 0.4437 | 0.4791 | 0.4836 | 0.5188 |

## How to read this report

A model has **high potential** (worth further tuning) when its Diagnostic-2 slope is large AND its Diagnostic-4 gap is small. That combination means optimization buys you real generalizing gains. A model with a big slope but a big gap is overfitting its hyperparameter search; additional tuning will mostly improve inner CV without moving outer performance. A model with a small slope is already near its ceiling — switch family rather than tune. A model with high Diagnostic-3 variance is unstable, and any apparent improvement from tuning could be noise.

## Source runs aggregated

| run_id | model | tier | n_rows | n_participants |
|---|---|---|---|---|
| screen_logistic_express | logistic | express | 80 | 8 |
| screen_logistic_lightning | logistic | lightning | 24 | 8 |
| screen_riemannian | riemannian | riemannian | 200 | 8 |
| screen_svm_express | svm | express | 80 | 8 |
| screen_svm_lightning | svm | lightning | 24 | 8 |
| screen_xgb_express | xgb | express | 80 | 8 |
| screen_xgb_lightning | xgb | lightning | 24 | 8 |

