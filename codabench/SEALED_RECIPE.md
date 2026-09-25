# Sealed-phase recipe (Track 2, cross-session, 3-class) — DRAFT

Status: draft, being filled in during the 2026-09-25/27 weekend runs. Sections 1,
2 and 5 get their numbers from `logs/sealed_p*/RESULTS.md` as the phases finish.

## 1. The recipe

_(pending Phases 2–5)_

## 2. Evidence

_(pending: one table per phase, per dataset, cell-averaged and pooled, seeds)_

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
| **"Context" cells (Graz vs BrainHero)** | proxies have one context | Score per context. Compare alignment per (subject) vs per (subject, context) in training, and a router over (subject, context) pairs if contexts differ in spectrum |
| **Three calibration sessions per evaluation participant** | Tangermann and Scherer have one; only Zhou has two training sessions | Leave-one-calibration-session-out validation becomes possible: choose the blend weight, EEGNet epoch count and router threshold on held-out *sessions*, not windows |
| **Ten training participants with all six sessions** | no proxy has extra fully labelled people | Pooled model on all 20 participants' labelled data vs evaluation participants only; router over 10 evaluation subjects vs all 20 |
| **Test window order and batch composition** | proxies are in recording order by construction | Nothing to do if the recipe routes per window (it does); batch-level statistics stay out of the recipe unless the organisers confirm they are allowed |
| **Class balance per cell** | proxies are balanced | Report per-cell class counts in the EDA; uniform LDA priors already match the balanced-accuracy metric |

## 4. Reproduce end to end

`scripts/train_sealed.sh <data_dir> <study>` _(skeleton, see the script)_.

## 5. Next steps

_(pending)_
