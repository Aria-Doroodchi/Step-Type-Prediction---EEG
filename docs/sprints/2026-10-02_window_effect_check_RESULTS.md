# Results: is the full-CNV window effect real? (2026-10-02)

Plan: [2026-10-02_window_effect_check.md](2026-10-02_window_effect_check.md). Script:
`scripts/diagnostics/10_window_effect_check.py`. Raw output (gitignored): `outputs/diagnostics/window_effect/`
(`within.csv`, `perm.csv`, `loso.csv`, `RESULTS.md`, `run.log`).
Run 16:04-16:54 on 2026-10-02: estimate 30 min, actual 50 min (first launch used 100 permutations, 200 s per participant;
cut to 20 after the first participant, then 81-85 s each; my estimate error, not a stall).

34 participants (the headline run used 30), ~80 epochs each (40 per condition), CSD epochs, 64 channels.
Model: mean amplitude per channel per 125 ms bin, z-scored, shrinkage LDA. This is deliberately simpler than the
XGB headline, so absolute AUCs are lower than 0.714 (that used richer features and partial pooling).

## Rules (pre-registered in the brief)

| Rule | Result | Outcome |
|---|---|---|
| R1 shuffled vs chronological CV, full window | -0.004 [-0.022, 0.014] | **No temporal leakage** |
| R2 pre-trigger baseline (-0.1 to 0 s) | 0.502 vs permutation null 0.498 | **Clean** |
| R3 full - late under chronological CV | +0.098 [0.072, 0.126] | **Effect is real (survives)** |
| R4 early 0-0.5 s cross-subject AUC | 0.675 [0.637, 0.712] (threshold 0.60) | **Trigger-locked early response FLAG** |

## What it means

1. Shuffled CV is not inflating the numbers. Chronological folds give the same AUC (full 0.578 vs 0.583).
2. The full-vs-late gap is real, but **it is carried entirely by the first half second after t = 0**.

| window | within-person AUC (chrono) | cross-subject AUC (LOSO) |
|---|---|---|
| early 0-0.5 s | 0.633 [0.602, 0.666] | 0.675 [0.637, 0.712] |
| mid 0.5-1.0 s | 0.547 | 0.518 |
| late 1-2 s | 0.485 | 0.497 |
| 0.5-2 s together | 0.522 | 0.521 |
| full 0-2 s | 0.583 | 0.651 |

3. From 0.5 s on, the data hold no detectable step-type information with this feature set (late window is at chance).
   The early window predicts the step type in people the model never saw, so the signal is the same across people. A
   consistent, early, cross-person signal fits a stimulus-locked response to whatever happens at t = 0 (the repo does
   not say what the One / Two trigger marks). It does not fit individual motor preparation.
4. So the 0.714 headline is not leakage between epochs, but "CNV" is probably the wrong description of what is being
   decoded, unless t = 0 is something with the same appearance for both step types (then it would be an early
   preparation response; see below).

## Caveats

- Linear model on amplitude bins only. The headline XGB also uses slopes/PSD/source features; its late-window AUC
  (about 0.57 in the README) is above this model's 0.49, so late-window information may exist in features that
  amplitude bins miss. The early-window dominance is not tested for XGB.
- ~80 epochs per person: per-person AUCs are noisy (SD of a person's AUC ~0.08); cohort means are what is reported.
- 34 participants here vs 30 in the headline.

## Needs the user

What does the One/Two trigger (256 / 512) mark at t = 0: a cue whose appearance differs between straight and
diagonal (then the early response is sensory and "predicting the step" is really reading the cue), or an event
that looks the same for both (then 0-0.5 s is genuine early preparation)? This decides how the README should word
the finding. No README or headline number was changed.

## Next (only if wanted)

Rerun the XGB pipeline with the window `0.5-2.0` (excluding the first 0.5 s) to see what the headline model does
without the early response (~3-4 h).

## Follow-up: the same split with XGB (cached `rich_mean_0125` features, 20 participants, 18:34-18:50, ETA 30 min)

`scripts/diagnostics/11_xgb_window_split.py`: fixed shallow XGB (150 trees, depth 3), 5-fold x3 repeated stratified CV,
columns selected by their `_bin_k` suffix (125 ms bins). Per-participant AUC, cohort mean:

| columns | AUC |
|---|---|
| bins 0-3 (0-0.5 s) | **0.689** |
| all 16 bins (0-2 s) | 0.653 |
| bins 8-15 (1-2 s, late) | 0.580 |
| bins 4-15 (0.5-2 s) | 0.568 |

All - late = +0.074 [0.039, 0.111]; early-only - everything-after-0.5 s = +0.121 [0.073, 0.175] (participant bootstrap).
So the XGB feature set shows the same thing as the linear model: the first half second is the best part of the window
(adding later bins dilutes it), and the late window keeps a small signal (0.58) that amplitude bins alone did not show.
Caveats: 20 of the 30-34 participants have this cache; not nested CV; fixed hyper-parameters; per-person only (no pooling).
