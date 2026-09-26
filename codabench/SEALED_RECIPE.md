# Sealed-phase recipe (Track 2, cross-session, 3-class)

Evidence-ranked training recipe for the Track 2 sealed phase (Oct 28 – Nov 21),
built over the weekend of 2026-09-25/27 on within-subject, cross-session proxies,
because the Graz + BrainHero training data is not released yet. All numbers are
**cell-averaged balanced accuracy** on each subject's *last* session, the
sealed metric without the context dimension. Logs, timings and every decision
are in [LOG.md](LOG.md) (2026-09-25 entries); raw tables in
`logs/sealed_p*/RESULTS.md`; code in `analysis/` and `scripts/sealed_*.sh`.

## 1. The recipe

**Riemann-Sealed** (`solvers/bci_decoding/riemann_sealed.py`): the
Riemann-StepType feature union (xDAWN covariances, broadband tangent space,
log-variance **and** the 4-band filter-bank tangent space) with shrinkage LDA,
made cross-session-aware in three steps that need **no subject or session id at
prediction time**:

1. A **subject router** (log-PSD 1–45 Hz per channel, shrinkage LDA over the
   calibration subjects) names the subject of each test window. It identifies
   the subject of a later-session window 87–100 % of the time on the proxies.
   Windows it is unsure about (below the 1st percentile of its training
   confidence) fall back to the pooled model.
2. **Per-subject whitening** of every window with its (routed) subject's
   Riemannian-mean reference from the calibration data.
3. **blend_calib personalisation**: the pooled LDA's probabilities are blended
   with those of the routed subject's own LDA (fitted on the shared features),
   P = w·P_pooled + (1−w)·P_subject. w is chosen on the calibration data only
   (on the release: leave-one-calibration-session-out; on the proxies it came
   out 0, 0.5 or 0.75).

**Conditional upgrade, pending the organisers' answer:** *online per-subject
re-centring* (`adapt="online"`). Each routed subject's whitening reference is the
mean covariance of the last 64 test windows routed to it. This is the single
largest lever measured: **+3.5 to +7.5 points** on top of the clean recipe, at
or above what the (unattainable) oracle session re-centring gives. It uses
unlabelled test windows, which the rules neither allow nor forbid explicitly (§ 5,
step 1).

Not in the recipe: EEGNet (6–12 points below Riemann in every comparison, with
or without last-layer calibration or alignment); cross-dataset pre-training
(§ 2, Phase 5); train-only alignment (hurts); batch-level statistics (fragile,
rule-dependent, superseded by online re-centring).

## 2. Evidence

Riemann-StepType, xDAWN + filter bank unless stated; single deterministic fits
(Riemann) or mean ± SD over 3 seeds (neural). Differences under ~2 points are
noise on these 4–9-subject datasets.

**Phase 1 — base family and pooling** (no alignment):

| Model | Tangermann pooled / per-subj. (4 cl.) | Scherer pooled / per-subj. (5 cl.) | Zhou pooled / per-subj. (3 cl.) |
|---|---|---|---|
| MeanLogReg | 0.258 / 0.270 | 0.218 / 0.193 | 0.435 / 0.457 |
| Riemann xDAWN + FB | 0.639 / **0.775** | 0.259 / **0.320** | **0.772** / 0.708 |
| Riemann FB, no xDAWN | 0.544 / 0.726 | 0.277 / 0.301 | 0.728 / 0.667 |
| braindecode EEGNet | 0.622 ± 0.026 / 0.395 ± 0.008 | 0.295 ± 0.014 / 0.213 ± 0.016 | 0.592 ± 0.073 / 0.477 ± 0.017 |
| EEGNet-StepType | 0.644 ± 0.037 / 0.546 ± 0.065 | 0.281 ± 0.010 / 0.261 ± 0.015 | 0.754 ± 0.063 / 0.561 ± 0.003 |

**Phase 2 — alignment without ids** (per-subject / pooled):

| Proxy | none | oracle (test session's stats) | train-only | router (clean) | online-64 (rule-dep.) |
|---|---|---|---|---|---|
| Tangermann | 0.775 / 0.639 | 0.824 / 0.726 | 0.727 / 0.588 | 0.780 / 0.687 | **0.831** / 0.729 |
| Scherer | 0.320 / 0.259 | 0.383 / 0.293 | 0.303 / 0.283 | 0.322 / 0.277 | **0.385** / 0.301 |
| Zhou | 0.708 / 0.772 | 0.763 / 0.742 | 0.732 / 0.770 | 0.688 / 0.767 | **0.783** / 0.747 |

Router accuracy (per window, cross-session): Tangermann 0.965, Scherer 0.866,
Zhou 1.000; Scherer 3-class 0.897. Why the clean router barely helps per-subject
models: whitening a subject's train and test windows with the same reference is
nearly invisible to a tangent space at that subject's own mean (affine
invariance). Only test-session statistics fix *session* drift.

**Phase 3 — pooling vs personalisation** (router ids; clean router alignment /
online):

| Proxy | pooled | calib | blend | **blend_calib** | per-subject |
|---|---|---|---|---|---|
| Tangermann | 0.687 / 0.729 | 0.792 / 0.824 | 0.792 / 0.836 | **0.802 / 0.837** | 0.777 / 0.820 |
| Scherer | 0.277 / 0.301 | 0.322 / 0.375 | 0.304 / 0.363 | **0.322 / 0.375** | 0.302 / 0.362 |
| Zhou | 0.767 / 0.747 | 0.708 / 0.778 | 0.757 / 0.802 | **0.778** / 0.797 | 0.688 / 0.783 |
| Scherer 3-class (Phase 4) | 0.452 / 0.485 | 0.473 / 0.558 | 0.499 / 0.544 | 0.486 / **0.561** | 0.486 / 0.546 |

EEGNet-StepType, router alignment, pooled / last-layer calibration (router ids):
Tangermann 0.653 / 0.678, Scherer 0.295 / 0.303, Zhou 0.704 / 0.729.

**Phase 4 — the sealed-like classes (WORD, SUB, HAND) and feature blocks**
(Scherer 3-class, per-subject; none / router / online): all blocks 0.493 / 0.499 /
0.573; no xDAWN 0.491 / 0.476 / 0.556; filter bank only 0.484 / 0.463 / 0.553;
xDAWN only 0.419 / 0.472 / 0.483; MeanLogReg 0.336 (chance 0.333). Zyma
arithmetic vs rest, cross-subject: FB only 0.734, all 0.723, MeanLogReg 0.512;
0.76–0.78 with per-person re-centring.

**Phase 5 — cross-dataset pre-training** (EEGNet-StepType pre-trained on Dreyer +
Tangermann + Zhou, 27,776 windows, 11 shared channels; fine-tuned on Scherer
3-class; 3 seeds):

| | per-subject | pooled |
|---|---|---|
| from scratch, 11 ch | 0.349 ± 0.017 | 0.438 ± 0.028 |
| pre-trained, 11 ch | 0.410 ± 0.005 | 0.364 ± 0.033 |
| pre-trained (aligned), online / router | _pending_ | _pending_ |
| Riemann FB TS, 11 ch, reference Scherer vs MI + Scherer | 0.436 vs 0.432 | 0.436 vs 0.436 |
| *reference: Riemann-Sealed recipe, 30 ch (clean)* | *0.486 (blend_calib)* | *0.452* |

## 3. Untested until the Graz + BrainHero data arrives

What the proxies cannot tell us, and the ablation that settles each point once
the data is released. The first run on the new data should use the **ten fully
labelled training participants as an internal replica of the sealed split**:
calibrate on their sessions 1–3, score their sessions 4–6 with the cell metric.
The organisers' split has the same structure for the ten evaluation participants.

| Open point | Why the proxies can't answer it | Ablation on the released data |
|---|---|---|
| **EMG (2) and EOG (2) channels** | no proxy has them; the default EEG pick drops them anyway | EEG only vs EEG + EOG vs EEG + EMG vs all 47, cross-session on the replica split. Keep a non-EEG channel only if it helps *and* the gain holds on sessions 4–6. The thesis trap: an "easy" class separated by gross muscle or eye activity that drifts across days. Watch word association (speech-like muscle activity) and calculation (eye movements) |
| **500 Hz, 43 EEG channels** | proxies are 14–30 ch at 120 Hz | Check what the sealed loader delivers (NeuralBench resamples EEG tasks to 120 Hz so far). If 500 Hz arrives: time one fit + one full predict pass (60 min A100 budget, our code is CPU numpy), and compare the router's log-PSD fingerprint at 1–45 Hz vs 1–100 Hz |
| **"Context" cells (Graz vs BrainHero)** | proxies have one context | The harness already scores subject × session × context cells if the cache has a context column (`xsess_cache.py` saves `context`/`condition`/`paradigm`). Compare alignment per subject vs per (subject, context) and a router over (subject, context) pairs |
| **Three calibration sessions per evaluation participant** | Tangermann and Scherer have one; only Zhou has two training sessions | Choose the blend weight (and router threshold) by leave-one-calibration-session-out. Offline training with session ids also allows per-(subject, session) training alignment: on Zhou it added +2.4 per-subject (train-only, Riemannian) |
| **Ten training participants with all six sessions** | no proxy has extra fully labelled people | Pooled component on all 20 participants' labelled data vs evaluation participants only; router over the 10 evaluation subjects vs all 20 |
| **xDAWN and class-specific cues** | Scherer's window starts at the visual cue | EDA: evoked response to the cue per class; ablate xDAWN on the replica split |
| **Test window order and batch composition** | proxies are in recording order by construction | The clean recipe routes per window and does not care. The online upgrade needs windows of one subject to arrive near each other; the Dreyer test loader delivers recording order (81 % of batches single-subject) |
| **Class balance per cell** | proxies are balanced | Report per-cell class counts in the EDA; uniform LDA priors already match the balanced-accuracy metric |

## 4. Reproduce end to end

```bash
bash ~/codabench/scripts/train_sealed.sh <data_home> <study> <modality/task> [overlay]
```

Steps (resumable, logs in `logs/train_sealed_<study>/`): cache every window with
subject/session/run(/context) → validate the recipe in the harness (MeanLogReg
floor, pooled vs per-subject, none vs router alignment) → Phase 3 personalisation
(chooses the blend weight on calibration data) → bake the chosen settings into a
candidate copy of the solver as its defaults (Codabench runs defaults) → benchopt
training on the study overlay → read-only inference replay (must reproduce the
score) → zip into the log folder. It never uploads and never writes
`codabench/submissions/`. `ADAPT=online` switches on the conditional upgrade.

## 5. Next steps

_(filled in after Phase 5 and the end-to-end check)_
