# Model Screening Results

_Generated 2026-05-16 16:31:50 by `scripts/06_compare_runs.py`_

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
| logistic | express | 200 | 0.4566 | ± 0.0175 | 0.1262 |
| riemannian | riemannian | 500 | 0.5316 | ± 0.0119 | 0.1360 |
| svm | express | 200 | 0.5237 | ± 0.0210 | 0.1518 |
| xgb | express | 200 | 0.5588 | ± 0.0237 | 0.1711 |

## Diagnostic 2 — Tier-response slope

Mean Express AUC minus mean Lightning AUC for the same model. Positive slope means the model rewards heavier optimization budget (more CV repeats, RFECV pass, gain-prune refit). A near-zero slope suggests the model is already near its ceiling; switching model family will pay off more than further tuning. Riemannian has only one tier and is therefore not slope-comparable here.

| model | express_auc | lightning_auc | slope (Express − Lightning) | interpretation |
|---|---|---|---|---|
| logistic | 0.4566 | n/a | n/a | missing tier run |
| riemannian | n/a | n/a | n/a | n/a (single-tier model) |
| svm | 0.5237 | n/a | n/a | missing tier run |
| xgb | 0.5588 | n/a | n/a | missing tier run |

## Diagnostic 3 — Across-fold AUC variance

Standard deviation of per-fold AUC within each participant, averaged across participants. Lower means the model gives more stable fold-to-fold predictions. A high `max_within_participant_sd` flags a participant whose AUC swings wildly between folds — usually a data-quality issue or a fundamentally hard subgroup.

| model | tier | mean_within_participant_sd | max_within_participant_sd | n_participants |
|---|---|---|---|---|
| logistic | express | 0.1102 | 0.1591 | 20 |
| riemannian | riemannian | 0.1241 | 0.1541 | 20 |
| svm | express | 0.1387 | 0.2162 | 20 |
| xgb | express | 0.1203 | 0.1934 | 20 |

## Diagnostic 4 — Inner-vs-outer gap

Mean of (`inner_best_score` − `overall_accuracy`) across folds. Small absolute gap means the hyperparameter search generalizes — the inner CV's optimism matches the outer held-out score. A large positive gap (≥ 0.05) means the model overfits the inner search and further tuning will mostly chase noise. Negative gap means the inner CV was unusually pessimistic relative to the outer fold (can happen when inner folds are small).

| model | tier | mean_gap | median_gap | max_gap | interpretation |
|---|---|---|---|---|---|
| logistic | express | +0.2896 | +0.2969 | +0.7188 | overfits inner CV |
| riemannian | riemannian | +0.0364 | +0.0326 | +0.4710 | mild optimism |
| svm | express | +0.2676 | +0.2656 | +0.6406 | overfits inner CV |
| xgb | express | +0.2317 | +0.2402 | +0.6719 | overfits inner CV |

## Diagnostic 5 — Per-participant model ranking

For each participant, the four models are ranked by mean AUC on the primary tier. The table below counts how often each model finished in each rank. Concentrated rankings (one model usually #1) = homogeneous signal across the cohort. Scattered rankings (each model wins for some subset) = heterogeneous signal — per-participant model selection or an ensemble may help more than tuning one model.

| model | rank_1 | rank_2 | rank_3 | rank_4 |
|---|---|---|---|---|
| logistic | 1 | 2 | 5 | 12 |
| riemannian | 9 | 3 | 5 | 3 |
| svm | 2 | 8 | 7 | 3 |
| xgb | 8 | 7 | 3 | 2 |

### Per-participant AUC matrix

| participant | logistic | riemannian | svm | xgb |
|---|---|---|---|---|
| P01 | 0.3750 | 0.5425 | 0.4797 | 0.4766 |
| P02 | 0.5047 | 0.4816 | 0.7672 | 0.8719 |
| P03 | 0.4984 | 0.5491 | 0.5062 | 0.3953 |
| P05 | 0.5813 | 0.6216 | 0.4977 | 0.4578 |
| P06 | 0.3875 | 0.4319 | 0.4797 | 0.6469 |
| P07 | 0.3812 | 0.5078 | 0.5945 | 0.6516 |
| P08 | 0.4523 | 0.6311 | 0.4815 | 0.5014 |
| P10 | 0.3609 | 0.5425 | 0.4789 | 0.4844 |
| P11 | 0.5016 | 0.4284 | 0.4820 | 0.4656 |
| P12 | 0.4375 | 0.4834 | 0.5594 | 0.5859 |
| P13 | 0.5094 | 0.5453 | 0.4977 | 0.5312 |
| P14 | 0.4203 | 0.4872 | 0.5336 | 0.5141 |
| P15 | 0.4266 | 0.4941 | 0.5594 | 0.4375 |
| P19 | 0.5234 | 0.5084 | 0.5164 | 0.5859 |
| P23 | 0.4067 | 0.6171 | 0.4455 | 0.5531 |
| P24 | 0.5344 | 0.5431 | 0.5094 | 0.5672 |
| P25 | 0.4375 | 0.4834 | 0.5594 | 0.5859 |
| P30 | 0.5469 | 0.5919 | 0.5406 | 0.8688 |
| P35 | 0.3937 | 0.6053 | 0.5188 | 0.5297 |
| P39 | 0.4531 | 0.5363 | 0.4656 | 0.4656 |

## How to read this report

A model has **high potential** (worth further tuning) when its Diagnostic-2 slope is large AND its Diagnostic-4 gap is small. That combination means optimization buys you real generalizing gains. A model with a big slope but a big gap is overfitting its hyperparameter search; additional tuning will mostly improve inner CV without moving outer performance. A model with a small slope is already near its ceiling — switch family rather than tune. A model with high Diagnostic-3 variance is unstable, and any apparent improvement from tuning could be noise.

## Source runs aggregated

| run_id | model | tier | n_rows | n_participants |
|---|---|---|---|---|
| bin_late_cnv_pyramid_mean_fine_logistic | logistic | express | 200 | 20 |
| bin_late_cnv_riemannian | riemannian | riemannian | 500 | 20 |
| bin_late_cnv_pyramid_mean_fine_svm | svm | express | 200 | 20 |
| bin_late_cnv_pyramid_mean_fine_xgb | xgb | express | 200 | 20 |

