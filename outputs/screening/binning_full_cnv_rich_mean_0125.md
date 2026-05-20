# Model Screening Results

_Generated 2026-05-16 14:20:39 by `scripts/06_compare_runs.py`_

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
| logistic | express | 200 | 0.6499 | ± 0.0211 | 0.1525 |
| riemannian | riemannian | 500 | 0.5066 | ± 0.0127 | 0.1446 |
| svm | express | 200 | 0.6224 | ± 0.0265 | 0.1910 |
| xgb | express | 200 | 0.6548 | ± 0.0248 | 0.1793 |

## Diagnostic 2 — Tier-response slope

Mean Express AUC minus mean Lightning AUC for the same model. Positive slope means the model rewards heavier optimization budget (more CV repeats, RFECV pass, gain-prune refit). A near-zero slope suggests the model is already near its ceiling; switching model family will pay off more than further tuning. Riemannian has only one tier and is therefore not slope-comparable here.

| model | express_auc | lightning_auc | slope (Express − Lightning) | interpretation |
|---|---|---|---|---|
| logistic | 0.6499 | n/a | n/a | missing tier run |
| riemannian | n/a | n/a | n/a | n/a (single-tier model) |
| svm | 0.6224 | n/a | n/a | missing tier run |
| xgb | 0.6548 | n/a | n/a | missing tier run |

## Diagnostic 3 — Across-fold AUC variance

Standard deviation of per-fold AUC within each participant, averaged across participants. Lower means the model gives more stable fold-to-fold predictions. A high `max_within_participant_sd` flags a participant whose AUC swings wildly between folds — usually a data-quality issue or a fundamentally hard subgroup.

| model | tier | mean_within_participant_sd | max_within_participant_sd | n_participants |
|---|---|---|---|---|
| logistic | express | 0.1031 | 0.1591 | 20 |
| riemannian | riemannian | 0.1254 | 0.1558 | 20 |
| svm | express | 0.1403 | 0.2882 | 20 |
| xgb | express | 0.1191 | 0.1981 | 20 |

## Diagnostic 4 — Inner-vs-outer gap

Mean of (`inner_best_score` − `overall_accuracy`) across folds. Small absolute gap means the hyperparameter search generalizes — the inner CV's optimism matches the outer held-out score. A large positive gap (≥ 0.05) means the model overfits the inner search and further tuning will mostly chase noise. Negative gap means the inner CV was unusually pessimistic relative to the outer fold (can happen when inner folds are small).

| model | tier | mean_gap | median_gap | max_gap | interpretation |
|---|---|---|---|---|---|
| logistic | express | +0.2264 | +0.2344 | +0.6406 | overfits inner CV |
| riemannian | riemannian | +0.0479 | +0.0468 | +0.4395 | mild optimism |
| svm | express | +0.2448 | +0.2656 | +0.6409 | overfits inner CV |
| xgb | express | +0.2069 | +0.2004 | +0.5761 | overfits inner CV |

## Diagnostic 5 — Per-participant model ranking

For each participant, the four models are ranked by mean AUC on the primary tier. The table below counts how often each model finished in each rank. Concentrated rankings (one model usually #1) = homogeneous signal across the cohort. Scattered rankings (each model wins for some subset) = heterogeneous signal — per-participant model selection or an ensemble may help more than tuning one model.

| model | rank_1 | rank_2 | rank_3 | rank_4 |
|---|---|---|---|---|
| logistic | 10 | 6 | 2 | 2 |
| riemannian | 3 | 1 | 3 | 13 |
| svm | 0 | 11 | 5 | 4 |
| xgb | 7 | 2 | 10 | 1 |

### Per-participant AUC matrix

| participant | logistic | riemannian | svm | xgb |
|---|---|---|---|---|
| P01 | 0.6750 | 0.5241 | 0.6719 | 0.6531 |
| P02 | 0.5000 | 0.5088 | 0.9344 | 0.9656 |
| P03 | 0.5641 | 0.5681 | 0.5461 | 0.4234 |
| P05 | 0.6797 | 0.5203 | 0.5867 | 0.5484 |
| P06 | 0.7953 | 0.4222 | 0.7797 | 0.6281 |
| P07 | 0.6813 | 0.4516 | 0.5203 | 0.7219 |
| P08 | 0.4502 | 0.6071 | 0.4972 | 0.4821 |
| P10 | 0.5000 | 0.5822 | 0.4578 | 0.5734 |
| P11 | 0.5563 | 0.3628 | 0.4719 | 0.6062 |
| P12 | 0.7969 | 0.4263 | 0.7570 | 0.7219 |
| P13 | 0.7672 | 0.6128 | 0.5391 | 0.7516 |
| P14 | 0.7109 | 0.5347 | 0.6000 | 0.5547 |
| P15 | 0.6734 | 0.4813 | 0.6188 | 0.8094 |
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
| bin_full_cnv_rich_mean_0125_logistic | logistic | express | 200 | 20 |
| bin_full_cnv_riemannian | riemannian | riemannian | 500 | 20 |
| bin_full_cnv_rich_mean_0125_svm | svm | express | 200 | 20 |
| bin_full_cnv_rich_mean_0125_xgb | xgb | express | 200 | 20 |

