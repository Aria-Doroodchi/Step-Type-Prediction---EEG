# Agentic perf-improvement loop — SUMMARY

Branch `perf/agentic-improvements`. Goal: autonomously raise EEG step-type model
performance (XGB + CNN), maximizing cohort held-out AUC under a gap guardrail
(inner−outer overfit gap must not worsen by >+0.03). Source of truth: [`LEDGER.md`](LEDGER.md).

## Headline

**Cross-subject partial pooling is the one lever that moved the numbers** — and mostly
on *honesty*: it collapses the inner-vs-outer overfit gap from +0.17 to ≈0 while modestly
lifting held-out AUC. Four rounds in, the XGB model is at its ceiling on the working
feature set; every within-feature/model lever (funnel caps, search richness, shape
features) is null at cohort scale. The loop stopped on the plateau rule (3 consecutive
no-win rounds).

## Baseline → final

| Model | scope | baseline AUC | baseline gap | FINAL AUC | FINAL gap | Δ AUC |
|---|---|---|---|---|---|---|
| **XGB** | 20-subject confirmation (pooled partial) | 0.5674 | +0.166 | **0.5957** | **−0.014** | **+0.028** |
| CNN | 8-subject iteration | 0.5644 | +0.070 | 0.5644 | +0.070 | — |
| CNN | 18-subject confirmation | — | — | 0.5675 | +0.013 | — |

AUC = held-out test AUC (mean over participants of mean outer-fold AUC, repeated_stratified
only). 0.50 = chance. All XGB numbers on the standard 2.3k "fast" feature set
(amplitude+slopes, bin 0.25 s, full-CNV 0–2 s), express CV.

## Every change tried (ranked)

| # | Round | Change (lever) | Model | Screen (8-subj) | Confirm (20-subj) | Verdict | Why |
|---|---|---|---|---|---|---|---|
| 1 | 1 | **Partial cross-subject pooling** | xgb | +0.086 paired | **+0.031 paired** (t=1.27); gap +0.17→−0.01 | **WIN → promoted** | Escapes the ~80-epoch p≫n budget; gap collapses, AUC up (modest). |
| 2 | 1 | Tighten reg. grid (depth≤3, ↑L2/γ) | xgb | −0.006; gap −0.045 | — | mild positive | Gap↓ but AUC-neutral; not an AUC win. |
| 3 | 1 | Align inner metric to roc_auc | xgb | −0.007 | — | reject | Slightly hurts AUC. |
| 4 | 1 | Tighten feature funnel | xgb | +0.003 | — | reject | Within noise. |
| 5 | 1 | Stability threshold 0.6→0.7 | xgb | −0.004 | — | reject | Within noise. |
| 6 | 2 | Loosen funnel for pooled n | xgb | **−0.039** (t−1.18) | — | reject | Tight funnel still better even pooled; cap isn't the bottleneck. |
| 7 | 2 | Richer/deeper search for pooled n | xgb | +0.020 (t1.54, 6/8↑) | **+0.001** (17/20, t0.10) | reject | Best screen signal, but null at cohort scale (subset noise). |
| 8 | 3 | +Legendre shape features (+324 cols) | xgb | +0.008 (t0.61) | — | reject | Redundant with amplitude/slope bins (same time-domain info). |

Wins: **1**. Mild positives (gap-only, recorded): 2. Rejected: 6.

## Winner — config key + legacy override

| Win | Key | Default (improved) | Legacy value | How to use |
|---|---|---|---|---|
| Partial pooling | `modeling.pooling.mode` | exposed via committed overlay **`configs/pooling.yaml`** (= `partial`) | `per_participant` (the global default in `configs/default.yaml`, retained) | `python scripts/04_train.py --model xgb --config configs/pooling.yaml` |

**Judgment call (flagged for review):** the global `default.yaml` default was kept at
`per_participant`, NOT flipped to `partial`. A silent flip would convert the project's
per-subject paradigm to cross-subject pooling, make `--speed-tier express` pooled, bypass
the chronological-leakage check, and break the per-participant unit tests — disproportionate
to a t=1.27 AUC lift. Partial pooling is instead a confirmed, one-line, documented opt-in.
Flip the default if you want it project-wide. Tabular models only; tensor models
(cnn/eegnet/eegnext) and <2-subject cohorts auto-fall back to per-participant.

## What the loop established

- **Pooling's value is mostly honesty.** The gap collapse (+0.17→−0.01) is robust and
  reproducible (matches the documented demo); the AUC lift (+0.028..0.031) is real but
  modest and not statistically significant (t=1.27). Reported numbers are now trustworthy.
- **The pooled XGB is at its feature-set ceiling.** Funnel caps, search richness, and shape
  features are all null at cohort scale. Subset (8-subj) screens are badly optimistic — every
  promising screen lift shrank toward zero on the 20-subject confirm (the confirm step earned
  its keep, killing 2 false positives: richer-grid and the original 8-subj pooling magnitude).
- **CNN** baselines: iter 0.5644 (gap +0.070), confirm 0.5675 (gap +0.013). No CNN
  improvement candidate was screened — each CNN screen costs ≈ the entire XGB loop's compute,
  and XGB had the cheaper headroom. The CNN gap is already small, so only AUC-raising levers remain.

## Recommended next steps

1. **Decide the pooling default.** If cross-subject pooling is acceptable for the thesis,
   flip `configs/default.yaml` `modeling.pooling.mode: partial` (and pin the per-participant
   unit tests). The gap honesty alone may justify it.
2. **Try pooling on the RICH feature set.** The whole loop ran on the tractable 2.3k set;
   per-participant on the 12.3k rich set already reaches ~0.655. Pooling there (heavier, was
   out of in-session scope) is the most likely real AUC gain left.
3. **A dedicated CNN sub-loop.** Screen dropout / temporal-kernel / fusion-unit on the
   8-subject set (budget ~1 h per candidate). The CNN's small gap means there is room to add
   capacity if AUC keeps up.
4. **Per-participant model selection / ensembling** (XGB + EEGNet), given the scattered
   per-subject rankings — a different axis than pooling.
5. **Disable system sleep for overnight runs** — a sleep at 10:17 interrupted the Round 4
   confirm at 17/20 (script-09 is not checkpointed; `04_train.py` is).

## Reproducibility

Run ids cited throughout (`outputs/runs/<id>/`). Harness: `screen.sh` (per-participant
tiers), `round2_pool.sh` (pooled screens), `aggregate.py` (task-exact cohort AUC + gap),
`scripts/09_pooling_comparison.py` (pooled per_participant/partial/full on one matched frame).
Neural runs use `.venv312` (Py 3.12 + TF); classical use `.venv` (Py 3.14). Set
`PYTHONUTF8=1` to avoid a cp1252 logging crash on the θ glyph.
