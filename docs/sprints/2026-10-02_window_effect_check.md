# Sprint brief: is the full-CNV window effect real? (2026-10-02 → 2026-10-05 05:00)

Plan of record. Started Fri 2026-10-02 ~16:30 EDT. Deadline Mon 2026-10-05 05:00;
wrap-up reserve from Mon 01:00. User away; no questions expected.

## Why this goal (value check)

The user left the area open. Candidates considered:

| Candidate | Verdict |
|---|---|
| Track 2 sealed recipe | Saturated: real data not released, every remaining lever measured (SEALED_RECIPE, DIRECTIONS § 4). Nothing to gain until release day. |
| Riemann + EEGNet ensemble | Already measured: +0.8 (LOG 2026-09-25); EEGNet now 6–12 points behind on the sealed-like stand-ins. Not worth a day. |
| **Thesis: is the full-CNV (0–2 s) vs late (1–2 s) AUC gap (+0.09) a real effect?** | MODELS.md § 8 item 1, called "the single highest-leverage move", still ⬜. The headline 0.714 is a public number (README). Epochs are t0 = condition trigger; headline CV is shuffled repeated stratified CV on ~80 epochs per person with partial pooling; no chronological result exists. Cheap (hours), needs nothing from the user. **Chosen.** |
| Warm-up high accuracy | Low priority per user. The 0.9+ leaderboard entries use Dreyer2023 as "external pre-training data" (which contains the test subjects 61–81) or per-batch statistics; the best honest cross-subject number in the literature is 0.74. A model trained on the test subjects' public labelled recordings would be a leak: **not built**. Legit tries only, ≤ 3, after the thesis check. |

## Verified context

- Epochs: `data/interim/epochs/Pxx_CNV_{One,Two}-epo.fif`, 34 participants, ~40 epochs per condition,
  1024 Hz, 65 ch, tmin -0.1, tmax 2.0, baseline (None, 0). Conditions are interleaved in time (not blocked).
- Headline: `outputs/runs/xgb_full_full_cnv_20260612_093700` AUC 0.714 (30 participants, partial pooling,
  repeated stratified CV). No chronological-CV metrics were recorded (`final_statistics/metrics.csv`).
- Frozen: that run folder, delivered reports. Nothing here modifies them.

## Goal

Decide, with pre-registered rules, whether (a) shuffled CV overstates the full-window result
(temporal leakage), and (b) the early part of the window (0–0.4 s, sensory response to the trigger) carries
the gain. Report-only: no change to the headline; the README gets a caveat only if a rule fires.

## Method (script `scripts/diagnostics/10_window_effect_check.py`)

Per participant, simple fast model so the question is about the data, not the model: mean amplitude per
channel in 125 ms bins (= `rich_mean_0125` family), z-scored in-fold, shrinkage LDA (Ledoit-Wolf, solver lsqr),
AUC. Same epochs and 5-fold CV for each cell:
1. Windows: full 0–2, late 1–2, early 0–0.5, mid 0.5–1.0, baseline -0.1–0 (as 1 bin), plus sliding 250 ms windows.
2. CV scheme: shuffled repeated stratified (the headline scheme, 5 × 10) vs chronological (contiguous folds
   in recording order, no shuffle; train-test gap of one epoch purged).
3. Permutation null: 200 label shuffles on the full window (shuffled CV) per participant → null mean AUC.
4. Cross-subject (LOSO) time-resolved decoding on the same bins: a sensory/trigger response is consistent across
   people; a motor-preparation CNV is not necessarily.

## Pre-registered decision rules (cohort mean AUC over participants; participant-level bootstrap 95 % CI)

- **R1 temporal leakage:** full-window AUC shuffled − chronological > 0.04 with CI excluding 0 → FLAG
  "shuffled CV is optimistic". ≤ 0.02 → NO LEAKAGE. In between → inconclusive.
- **R2 baseline:** baseline (-0.1–0) shuffled AUC > permutation-null mean + 0.03 → FLAG (drift or leakage
  between epochs). Otherwise clean.
- **R3 window effect:** (full − late) AUC under *chronological* CV: > +0.03 with CI excluding 0 → effect
  SURVIVES; otherwise → effect NOT CONFIRMED.
- **R4 early response:** early (0–0.5 s) LOSO AUC ≥ 0.60 → the trigger evokes a consistent sensory response
  that differs by condition (cue effect); report how much of full − late it explains (full minus early-removed 0.5–2 s).

## Keep-jobs-alive

Per-participant results appended to a CSV after each participant (resumable: skip finished participants).
Estimates: epoch load ~2 s each; the per-participant grid ≈ 20–40 s → whole cohort ≈ 20–30 min; LOSO ≈ 5 min.
Heartbeat line per participant. Run in `.venv` with `PYTHONUTF8=1`. Overrun rule: > 1.5× → diagnose.

## Phases and budget

| Phase | Est. | Window |
|---|---|---|
| 1 script + smoke (3 participants) | 30 min | Fri evening |
| 2 full run + analysis + write-up (`docs/sprints/…_RESULTS.md`, LEDGER-style entry) | 1.5 h | Fri night |
| 3 (only if R1/R3 fire) follow-up with the real XGB pipeline in chronological mode on cached features | ≤ 4 h | Sat |
| 4 warm-up honest tries (≤ 3; stop if best honest val < +0.01 over WU1 0.82 local) | ≤ 12 h | Sat–Sun |
| 5 wrap-up (DIRECTIONS update, HANDOFF, LOG, memory, commits, summary) | from Mon 01:00; ≥ 1 h | Mon |

Launch cutoff: no job starts unless 1.5× its ETA ends before Mon 01:00.

## Deliverable

`docs/sprints/2026-10-02_window_effect_check_RESULTS.md` with the four rules' outcomes, table and figure;
CHANGELOG/README caveat only if a rule fires (on the feature branch, no merge); `codabench/DIRECTIONS.md`
step 4 updated.

## Contingencies

- Epoch files missing or participant mismatch → report which, run on what exists.
- Rules inconclusive → say so; do not rerun until something passes.
- Standing constraints: no uploads/merges/tags/PRs; push to `personal` after each commit.
