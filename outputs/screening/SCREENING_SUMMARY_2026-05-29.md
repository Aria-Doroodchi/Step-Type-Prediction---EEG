# Screening Summary — Step-Type EEG Classification

_Compiled 2026-05-29. Synthesizes the 9 screening reports in `outputs/screening/`
(generated 2026-05-14 → 2026-05-17 by `scripts/06_compare_runs.py`)._

## Context

The project predicts **step type** — straight (`One`) vs diagonal (`Two`) — from
EEG recorded during a stepping task (MSc thesis). Features are electrode
amplitudes, PSD (Morlet TFR), and eLORETA source activity. Four model families
are screened — **logistic, SVM, XGBoost, and a Riemannian tangent-space
pipeline** — using five diagnostics: mean test AUC ± 95% CI (D1), tier-response
slope (D2), across-fold variance (D3), inner-vs-outer overfitting gap (D4), and
per-participant model ranking (D5). The primary prediction window is **late CNV
(1.0–2.0 s)**; **full CNV** is a secondary window. AUC near 0.50 is chance.

The 9 reports fall into three groups: two early cohort screens (8 and 11
participants) and seven feature-binning comparisons on the frozen 20-participant
cohort.

## Headline finding

**The full-CNV window beats the late-CNV window by a wide margin**, even though
late CNV is the designated primary window. On full CNV, logistic and XGBoost
reach ~0.65 AUC; on every late-CNV binning variant they sit at ~0.46
(logistic) and ~0.56–0.57 (XGBoost). This is the single largest effect across
all runs and is worth following up before further model tuning.

A second, consistent pattern: **the three classical models all overfit the
inner CV** (D4 gap +0.20 to +0.29 — the inner search looks far better than the
held-out fold), while the **Riemannian pipeline is the only one that
generalizes** (gap ~0.04–0.06) — but it does so at essentially chance AUC
(~0.51–0.53). So no model is both well-calibrated and accurate yet.

## Mean test AUC by configuration (Diagnostic 1, Express tier)

Best model per row in **bold**. All 20-participant runs unless noted.

| Report (date) | window / features | logistic | riemannian | svm | xgb |
|---|---|---|---|---|---|
| SCREENING_RESULTS (05-14, n=8) | early cohort | 0.4822 | 0.5088 | 0.4905 | **0.5777** |
| SCREENING_RESULTS_RICHFEATS (05-15, n=11) | early cohort | 0.6485 | 0.4888 | 0.6103 | **0.6535** |
| bin_late_cnv_pyramid_mean_core | late CNV | 0.4569 | 0.5316 | 0.5335 | **0.5678** |
| bin_late_cnv_pyramid_mean_fine | late CNV | 0.4566 | 0.5316 | 0.5237 | **0.5588** |
| bin_late_cnv_rich_mean_0125 | late CNV | 0.4539 | 0.5316 | 0.5267 | **0.5679** |
| bin_late_cnv_stats_0125 | late CNV | 0.4601 | 0.5316 | 0.5292 | **0.5635** |
| bin_late_cnv_stats_pyramid_core | late CNV | 0.4620 | 0.5316 | 0.5188 | **0.5576** |
| bin_full_cnv_rich_mean_0125 | full CNV | 0.6499 | 0.5066 | 0.6224 | **0.6548** |
| bin_full_cnv_stats_pyramid_core | full CNV (svm n=7) | — | 0.5066 | **0.6762** | — |

Notes: the last row is a partial run (SVM on 7 participants only; logistic/xgb
not aggregated), so its 0.6762 is not directly comparable to the full-cohort
rows. Riemannian's late-CNV AUC (0.5316) is identical across the five late-CNV
reports because they all reuse the same single `bin_late_cnv_riemannian` run.

## What the diagnostics say

**Within late CNV, the binning recipe barely matters.** Across the five late-CNV
variants (pyramid-mean core/fine, rich-mean 0.125, stats 0.125, stats-pyramid
core), XGBoost moves only between 0.5576 and 0.5679 and logistic between 0.4539
and 0.4620. No binning scheme rescues the late-CNV window — the ceiling is set by
the window, not the binning.

**XGBoost is the most reliable classical winner.** In D5 per-participant
rankings it takes the most rank-1 finishes in nearly every report (e.g. 10/20 on
late_cnv_stats_0125, 9/20 on full_cnv_rich), and it shows the only positive
tier-response slope worth noting in the early 8-participant screen (+0.0163,
"moderate headroom") while logistic and SVM were already flat (near ceiling).

**Riemannian is the calibrated-but-flat option.** Lowest D4 gap everywhere, and
it actually leads the late-CNV per-participant rankings (8–9 rank-1 finishes)
because the classical models collapse below chance on late CNV — but its mean
AUC never clears ~0.53.

**Variance (D3) is moderate and similar across models** (~0.10–0.16 mean
within-participant SD); SVM is the most volatile (max SD up to 0.29 on
full_cnv_rich). No model is unstable enough to dismiss on variance alone.

## Per-participant signal is heterogeneous

A few participants carry most of the discriminable signal:

- **P30** — strongest throughout; XGBoost reaches 0.87–0.92 in nearly every run.
- **P02** — near-chance on late CNV but jumps to SVM 0.77 / XGBoost 0.87–0.97 on full CNV, the clearest window effect of any subject.
- **P12, P24, P25** — consistently good for logistic/SVM in the rich-feature cohort screen (~0.74–0.80).
- **P08, P11, P39** — persistently hard; classical models hover at or below chance, and Riemannian is often their best (P08 ~0.61–0.63).

The scattered rankings (D5) point to genuinely heterogeneous signal across the
cohort, which suggests **per-participant model selection or an ensemble** may pay
off more than squeezing one global model.

## Suggested next steps

1. **Investigate why full CNV outperforms the primary late-CNV window** — this is the most actionable result. Confirm it isn't a leakage/labeling artifact, then consider promoting full CNV (or a wider window) for the main analysis.
2. **Center further tuning on XGBoost**, the most consistent classical winner with the only non-flat tier slope; deprioritize logistic/SVM tuning on late CNV (flat and below chance).
3. **Address the inner-vs-outer overfitting gap** in the classical pipelines (lighter hyperparameter search, nested-CV calibration) before trusting express-tier AUCs.
4. **Complete the partial runs** — `bin_full_cnv_stats_pyramid_core` only has SVM on 7 participants; run logistic/xgb on the full cohort to make the full-CNV comparison apples-to-apples.
5. **Try per-participant model selection / ensembling** given the heterogeneous D5 rankings, rather than assuming one global model fits all subjects.

## Source files summarized

`SCREENING_RESULTS.md`, `SCREENING_RESULTS_RICHFEATS.md`,
`binning_late_cnv_pyramid_mean_core.md`, `binning_late_cnv_pyramid_mean_fine.md`,
`binning_late_cnv_rich_mean_0125.md`, `binning_late_cnv_stats_0125.md`,
`binning_late_cnv_stats_pyramid_core.md`, `binning_full_cnv_rich_mean_0125.md`,
`binning_full_cnv_stats_pyramid_core.md`.
