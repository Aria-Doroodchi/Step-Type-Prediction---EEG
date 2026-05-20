# Model Screening Results

_Generated 2026-05-16 21:07:44 by `scripts/06_compare_runs.py`_

## Overview

- **Models compared:** riemannian, svm
- **Tiers:** express, riemannian
- **Participants:** P01, P02, P03, P05, P06, P07, P08, P10, P11, P12, P13, P14, P15, P19, P23, P24, P25, P30, P35, P39 (20 total)
- **Total runs aggregated:** 2

All five diagnostics are computed on the **Express** tier (primary tier; for Riemannian: the `riemannian` tier — its single config). Diagnostic 2 (tier-response slope) additionally consumes the **Lightning** tier runs for the three classical models.

## Diagnostic 1 — Mean test AUC ± 95% CI

Per-fold ROC-AUC averaged across all CV folds × participants, with a Wald 95% CI. Higher mean is better; tighter CI means more consistent estimates.

| model | tier | n_folds | mean_auc | ci95 | sd |
|---|---|---|---|---|---|
| riemannian | riemannian | 500 | 0.5066 | ± 0.0127 | 0.1446 |
| svm | express | 70 | 0.6762 | ± 0.0397 | 0.1695 |

## Diagnostic 2 — Tier-response slope

Mean Express AUC minus mean Lightning AUC for the same model. Positive slope means the model rewards heavier optimization budget (more CV repeats, RFECV pass, gain-prune refit). A near-zero slope suggests the model is already near its ceiling; switching model family will pay off more than further tuning. Riemannian has only one tier and is therefore not slope-comparable here.

| model | express_auc | lightning_auc | slope (Express − Lightning) | interpretation |
|---|---|---|---|---|
| riemannian | n/a | n/a | n/a | n/a (single-tier model) |
| svm | 0.6762 | n/a | n/a | missing tier run |

## Diagnostic 3 — Across-fold AUC variance

Standard deviation of per-fold AUC within each participant, averaged across participants. Lower means the model gives more stable fold-to-fold predictions. A high `max_within_participant_sd` flags a participant whose AUC swings wildly between folds — usually a data-quality issue or a fundamentally hard subgroup.

| model | tier | mean_within_participant_sd | max_within_participant_sd | n_participants |
|---|---|---|---|---|
| riemannian | riemannian | 0.1254 | 0.1558 | 20 |
| svm | express | 0.1558 | 0.1993 | 7 |

## Diagnostic 4 — Inner-vs-outer gap

Mean of (`inner_best_score` − `overall_accuracy`) across folds. Small absolute gap means the hyperparameter search generalizes — the inner CV's optimism matches the outer held-out score. A large positive gap (≥ 0.05) means the model overfits the inner search and further tuning will mostly chase noise. Negative gap means the inner CV was unusually pessimistic relative to the outer fold (can happen when inner folds are small).

| model | tier | mean_gap | median_gap | max_gap | interpretation |
|---|---|---|---|---|---|
| riemannian | riemannian | +0.0479 | +0.0468 | +0.4395 | mild optimism |
| svm | express | +0.2154 | +0.2266 | +0.4844 | overfits inner CV |

## Diagnostic 5 — Per-participant model ranking

For each participant, the four models are ranked by mean AUC on the primary tier. The table below counts how often each model finished in each rank. Concentrated rankings (one model usually #1) = homogeneous signal across the cohort. Scattered rankings (each model wins for some subset) = heterogeneous signal — per-participant model selection or an ensemble may help more than tuning one model.

| model | rank_1 | rank_2 |
|---|---|---|
| riemannian | 13 | 7 |
| svm | 7 | 0 |

### Per-participant AUC matrix

| participant | riemannian | svm |
|---|---|---|
| P01 | 0.5241 | 0.6687 |
| P02 | 0.5088 | n/a |
| P03 | 0.5681 | n/a |
| P05 | 0.5203 | n/a |
| P06 | 0.4222 | 0.7812 |
| P07 | 0.4516 | 0.5742 |
| P08 | 0.6071 | n/a |
| P10 | 0.5822 | n/a |
| P11 | 0.3628 | n/a |
| P12 | 0.4263 | n/a |
| P13 | 0.6128 | n/a |
| P14 | 0.5347 | 0.6172 |
| P15 | 0.4813 | 0.6875 |
| P19 | 0.5038 | n/a |
| P23 | 0.6205 | n/a |
| P24 | 0.4603 | n/a |
| P25 | 0.4263 | n/a |
| P30 | 0.6106 | 0.7797 |
| P35 | 0.4291 | 0.6250 |
| P39 | 0.4791 | n/a |

## How to read this report

A model has **high potential** (worth further tuning) when its Diagnostic-2 slope is large AND its Diagnostic-4 gap is small. That combination means optimization buys you real generalizing gains. A model with a big slope but a big gap is overfitting its hyperparameter search; additional tuning will mostly improve inner CV without moving outer performance. A model with a small slope is already near its ceiling — switch family rather than tune. A model with high Diagnostic-3 variance is unstable, and any apparent improvement from tuning could be noise.

## Source runs aggregated

| run_id | model | tier | n_rows | n_participants |
|---|---|---|---|---|
| bin_full_cnv_riemannian | riemannian | riemannian | 500 | 20 |
| bin_full_cnv_stats_pyramid_core_svm | svm | express | 70 | 7 |

