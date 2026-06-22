# New module: foot-sole electrical-stimulation EEG — Task 1: trigger-delta calibration

## Context
This is a sibling module to the existing EEG step-type pipeline in
`src/eeg_steptype/`. The task and cues are IDENTICAL to the main study
(straight = "One" / diagonal = "Two").

The ONE experimental difference: in these trials, electrical stimulation was
delivered to the sole of the foot during step planning. These stimuli are
expected to elicit short-latency somatosensory evoked potentials (SEPs).

### Data
Raw files live in the SAME per-participant folder as the existing CNV files,
under `paths.raw_root` (`{raw_root}/Pxx/...`). `Pxx` is the participant ID
(e.g. `P25`); filenames embed that ID:
  - `Pxx/Pxx_Stim.bdf`     — the stimulation task (primary input for this module),
                             e.g. `P25/P25_Stim.bdf`
  - `Pxx/Pxx_Standing.bdf` — standing task, e.g. `P25/P25_Standing.bdf`
The current (pre-refactor) analysis script is `Pxx/Pxx_Stim.py` (e.g.
`P25/P25_Stim.py`), in that same directory. Read it first — it contains the
hand-set trigger deltas and the existing event-handling logic this module must
reproduce and improve on.

### Trigger scheme — LOCKED, do not rename or remap
The event-code naming MUST stay exactly:
  - `256`  — "One" audio prompt (straight)
  - `512`  — "Two" audio prompt (diagonal)
  - `1024` — stimulation trigger
This differs from the main pipeline's `events:` block (which uses 256/512 plus a
response code 96); the stim module needs its own stim-events config.

CRITICAL disambiguation requirement:
  - `1024` ALWAYS corresponds to the electrical stimulation trigger.
  - But `1024` events occur after BOTH `256` (One) and `512` (Two) prompts, so
    each `1024` must be attributed to the prompt that preceded it (One-stim vs
    Two-stim). Build the pairing logic explicitly (analogous to the existing
    `_pair()` in `events.py`, which walks the event stream and binds each event
    to the last active condition). Get this attribution provably correct — it
    is the backbone of every downstream analysis.

### The trigger problem
The hardware `1024` triggers did not fire at the true stimulus instant. The
researcher corrected them by hand, per-participant, by visual inspection, using
per-condition SAMPLE offsets like:

    Control_trig_diff = 210
    One_trig_diff     = 263
    Two_trig_diff     = 263

These are in SAMPLES, so they scale with each recording's sampling frequency,
and were set individually per participant.

NOTE — audio-output triggers: the triggering system also automatically logged
the OUTPUT AUDIO of each prompt. That audio onset may be a useful independent
timing anchor (it is hardware-generated, not hand-corrected) — worth exploring
as a cross-check or alternate reference for the stim timing.

## The question to answer (Task 1)
The recording equipment and stimulator setup were identical across all
participants and trials. So:

1. Can these per-participant, hand-set `1024` trigger deltas be replaced by a
   SINGLE uniform offset (expressed in TIME, then converted to samples per
   recording) that yields equal-or-better alignment than the hand-set values?
2. If yes, find that number as accurately as possible, with an uncertainty
   estimate, and recommend whether to adopt it.

This is a measurement/calibration question, NOT a modeling question. Do not
touch the classifier. The deliverable is a defensible offset (or a defensible
"no, it must stay per-participant") backed by evidence.

## What "accurate" means here — anchor on physiology, not eyeballing
The true stimulus onset is recoverable from the data itself, because the
electrical pulse and its evoked response are stereotyped and stimulus-locked.
Investigate these signals, in roughly this priority:

A. **Stimulation artifact.** An electrical pulse to the foot usually leaves a
   sharp, high-amplitude deflection in the EEG (and EMG/reference channels) at
   the TRUE instant of stimulation. If present, this is the ground truth: detect
   it per trial (threshold / matched filter / cross-correlation) and measure the
   offset between the recorded `1024` trigger and the true onset. A tight,
   consistent offset distribution across trials and participants is the
   strongest possible evidence for a single uniform delta.

B. **SEP morphology.** Foot-sole / tibial somatosensory stimulation produces a
   short-latency cortical SEP near the vertex (Cz region), with fixed-latency
   early components. The correct alignment is the one that makes the trial-
   averaged SEP SHARPEST: maximal early-component peak-to-peak amplitude,
   minimal across-trial latency jitter, maximal inter-trial phase coherence.
   Cross-correlate each participant's (or trial's) response against a SEP
   template, or sweep the candidate delta and maximize an SEP-quality metric.

C. **Audio-output trigger (cross-check).** Use the auto-logged prompt-audio
   onset as an independent, hardware-timed reference to validate the stim offset
   and detect any participant where the `1024`/audio relationship is anomalous.

## How to investigate
First ESTABLISH THE FACTS before optimizing — write them to the ledger:
  - Confirm the data location and read `Pxx/Pxx_Stim.py` to extract every
    hand-set delta and the existing event logic.
  - The sampling frequency of these recordings (and whether it varies by
    participant). Convert the example deltas (210, 263 samples) to ms at that
    sfreq and sanity-check against plausible SEP latency.
  - Verify the `256`/`512`/`1024` codes and that every `1024` can be correctly
    attributed to its preceding One/Two prompt. Report counts per condition.
  - HOW the `1024` trigger failed relative to the true stim — the shape of the
    failure determines whether a uniform shift can possibly work:
      * constant offset  → uniform delta is exactly right
      * random jitter / drops → uniform delta cannot fully fix it; per-trial
        artifact-based realignment is needed instead. Say so if you find this.
  - Whether a stim artifact is actually visible in these channels.

Then, for the alignment estimate:
  - Estimate each participant's data-driven optimal delta (via A and/or B) WITH
    an uncertainty (e.g. bootstrap over trials).
  - Compare three alignments head-to-head on an SEP-quality metric: (i) the
    researcher's hand-set deltas, (ii) per-participant data-driven optima,
    (iii) a single pooled uniform delta.
  - Decide uniformity statistically: if the between-participant spread of the
    optima is within the within-participant uncertainty, a single delta is
    justified. Report the pooled delta + CI in BOTH ms and samples (and the
    per-sfreq sample value if sfreq varies).
  - Quantify the cost of being wrong: how much does SEP quality degrade per ms
    of misalignment? This tells you how precise the number must be.

## Working conventions (match the existing perf-loop)
- Work on a new branch, e.g. `feat/stim-module` (do NOT commit to main).
- Keep `outputs/stim_module/LEDGER.md` as the single source of truth: log every
  fact established, every candidate delta tried, the metric, and the verdict.
  Append-only, timestamped, with the git SHA.
- Every change behind a toggle / new config block — do not mutate the existing
  pipeline's behavior. Add a stim-events config (codes 256/512/1024) analogous
  to the `events:` block in `configs/default.yaml`, with the hand-set
  per-participant deltas preserved as commented provenance.
- Reuse existing machinery wherever possible: the `Stim`-channel reader,
  `events.py` (especially the `_pair()` walk for binding 1024→prompt),
  `epoching.py`, and the per-participant override scheme in `configs/overrides/`.
  Don't re-implement what the main pipeline already does.
- Produce diagnostic figures: grand-average SEP before/after alignment, the
  offset-distribution histogram, the delta-sweep curve, the per-participant
  optimum-vs-hand-set scatter, and (if used) the audio-vs-1024 offset plot.
  Save under `outputs/stim_module/figs/`.
- Smoke-test on 1–2 participants before running the cohort.

## Deliverable for Task 1
A short report in `outputs/stim_module/LEDGER.md` (and a figure set) answering:
1. What signal gives ground truth here (artifact, SEP, audio, or neither)?
2. Is a single uniform delta justified? Yes/No, with the statistical basis.
3. If yes: the recommended number (ms + samples, with CI) and the config change
   to adopt it. If no: why, and the per-trial realignment approach to use instead.
Stop and report before changing any downstream preprocessing.
