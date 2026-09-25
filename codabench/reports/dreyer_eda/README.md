# Dreyer 2023: what the Track 2 warm-up data looks like

Exploratory analysis of exactly what a Track 2 solver receives: NeuralBench's
preprocessed windows, not the raw files. Script: `codabench/analysis/dreyer_eda.py`
(runtime 11 min on CPU; re-runs in ~6 min using the cache).
Numbers: `summary.json`. Generated 2026-09-24.

**Data at a glance:** 87 participants, **20,792 windows**, each **27 EEG
channels × 480 samples = 4 s at 120 Hz**, starting at the cue. Left/right hand
imagery, perfectly balanced (10,394 / 10,398); 240 windows per person (one has
160). Split by person: 52 train / 14 val / 21 test (participants 61–81 = the
Codabench warm-up test set). Labels: 0 = left hand, 1 = right hand (confirmed by
the physiology in fig. 06).

---

## 1. The EEG recording: figures 00, 01, 02

- **What the model sees** (01): robust-scaled traces (each channel of each
  recording is divided by its own spread), so absolute µV amplitude is gone. A
  single window is dominated by **slow drifts shared by all channels**. The
  right-hand example shows a large common wave in the first 0.7 s and then a
  downward drift on every electrode.
  → **Common-average reference (CAR) removes exactly that**, which is why CAR
  helped both models in the preprocessing sweep.
- **Spectrum** (02): a normal EEG spectrum with a **1/f slope**, an
  **alpha/mu peak near 10 Hz** (clear in most people, absent in some), and a
  **flat noise floor above ~35 Hz**. Visible artifacts: the **50 Hz notch** and
  a sharp roll-off near **60 Hz**, where resampling to 120 Hz cuts everything off.
  Anything above ~40 Hz is mostly noise and muscle.

## 2. Features that are likely useful: figures 04, 05, 06, 07

Two independent signals separate left from right:

| Signal | Where / when | Strength | Figure |
|---|---|---|---|
| **Slow lateralised potential** (< 4 Hz waveform) | left hemisphere more **negative** for right-hand imagery, right hemisphere more positive; peak **FC5/C5/C3**, from ~0.5 s to 4 s | strongest: peak d = −0.33, same sign in 79 % of people; **cross-subject accuracy 0.77** (all 27 ch, simple LDA) | 04 right, 05 |
| **Mu desynchronisation** (10–13 Hz power) | right-hand imagery → **less mu over C3**, more over C4 (and the mirror image for left); from ~0.6 s to the end | classic motor-imagery ERD: C3 d = −0.10, C4 d = +0.12; cross-subject **0.64** | 04 left, 06 |
| Beta power (13–30 Hz) | weaker, mostly right hemisphere | cross-subject **0.61** | 04, 07 |
| Delta/theta **power** (0.5–8 Hz) | — | near chance (0.52–0.59): the slow information is in the **waveform shape, not power** | 07 |

- **Both classes show two large cue responses**, at ~0.4 s and ~1.6 s (05),
  most likely the arrow appearing and disappearing. They're identical for left
  and right, so they carry no information. A model can ignore them, or a filter
  can remove them.
- **The two signals are complementary.** People who decode well from one are
  not the ones who decode well from the other (09, r = −0.02). Combining slow
  waveform + mu/beta features should beat either alone.

### The eye-movement check (fig. 07)

Question: is the strong < 4 Hz information just horizontal eye movements toward
the arrow? Test: decode from channel groups only (cross-subject, train+val →
test):

| Channels | slow waveform | mu power |
|---|---|---|
| frontal F3 Fz F4 | 0.67 | 0.52 |
| **fronto-central FC** | **0.73** | 0.59 |
| central C | 0.70 | **0.62** |
| centro-parietal / parietal | 0.66 | 0.59 |
| all 27 | **0.77** | 0.64 |

Eye-movement leakage would be strongest at the **front** and fade backwards.
Instead the slow signal peaks **fronto-centrally over the motor strip** and is
still strong over parietal channels, and its topography (05) is a
left-negative / right-positive gradient centred on FC5/C5. That looks like a
**lateralised cortical slow potential** (a readiness-potential-like contralateral
negativity) more than an ocular artifact.
**Not ruled out:** the EOG channels are stripped before we see the data, and
the most lateral frontal sites (F7/F8) aren't recorded. So a partial ocular
contribution can't be excluded.

**For the competition:** this signal is real and useful for the **warm-up**
(same Dreyer paradigm). The sealed task (motor imagery vs calculation vs word
association) is **not a left/right pair**, so a lateralised slow potential
probably won't carry over. Mu/beta and broader spectral/spatial features
are the safer investment.

## 3. Amount of noise: figures 02, 03

- **Artifact windows** (03 top-left): per channel, the share of windows with an
  excursion beyond 6 robust units. A few people have 5–7 % on most channels
  (whole-head events: movement, electrodes), most are around 0–2 %. No
  consistently bad channel across the cohort.
- **High-frequency (30–45 Hz) power** (03 top-right): varies by about ±8 dB
  **between people** (vertical stripes = whole-head, person-level differences
  in muscle tension or electrode quality). The lateral channels (FC5/FC6,
  C5/C6, CP5/CP6) run hotter in some people, which is typical of jaw/temporal
  muscle.
- **Outlier windows per person** (03 bottom): median **3.3 %**, maximum
  **8.75 %**. Test subjects aren't noisier than training subjects. Clipping at
  ±20 is essentially absent (median 0 %, max 2.2 %).
- **Overall:** moderate, typical research-grade EEG. Not worth hand-cleaning
  (you couldn't at scoring time anyway). Dropping extreme **training** windows
  is a cheap option.

## 4. Trial-to-trial variability: figure 08

- **Every trial of one median person** (08 left): single trials are dominated
  by slow drifts and offsets (horizontal bands) far larger than the class
  difference. Left and right blocks are hard to tell apart by eye.
- **Single-trial values of the strongest feature** (08 right, 8 people from
  worst to best): the two class clouds **overlap heavily** in everyone. The
  medians are shifted by a fraction of the spread (d ≈ 0.3), and a handful of
  extreme trials sit far outside.
- **Implication:** any single feature is weak; accuracy comes from pooling
  many weak cues across channels, time and frequency. Averaging many trials
  (training on lots of people) helps more than per-trial cleverness.

## 5. Between-person variability: figure 09

- **How decodable each person is** (09 top, within-person 5-fold CV):
  - slow-waveform LDA median **0.725**, mu/beta Riemann median **0.696**;
  - 92 % / 85 % of people are above chance (> 0.563), ranging from **~0.55 to
    0.98**. Large "BCI-inefficiency" spread, as usual for motor imagery.
- **Complementarity** (09 bottom-right): r = −0.02. Many people are good at one
  signal and poor at the other → combine both.
- **Individual alpha peak** (09 bottom-middle): median **10.5 Hz**, 10–90 %
  range **9.0–12.2 Hz**. A fixed 8–13 Hz band covers most people, but the
  individual peak shifts which band is best per person.
- **Test people look like training people** (09 bottom-left, covariance map):
  the test subjects are scattered among the training subjects, with no
  systematic shift between splits.
- **Pooled training works:** cross-subject slow-waveform decoding (0.77)
  **beats** the median within-person decoder (0.725). With ~190 training trials
  per person, pooling 66 people outweighs person-specificity here.

## 6. Other things that should influence decisions

1. **Scaling destroys absolute amplitude and distorts spatial physics.** Each
   channel of each recording is scaled separately. Laplacian / source-space
   methods assume consistent units across channels; this likely explains why the
   Laplacian lagged CAR. Prefer data-driven spatial filters (CSP, tangent space,
   learned convolutions) over physics-based ones.
2. **The warm-up (Dreyer) is not the sealed task.** Dreyer is cross-*subject*
   left/right imagery; the sealed phase is cross-*session*, same people, three
   different mental tasks. The strongest Dreyer cue (lateralised slow
   potential) is the least likely to transfer. Don't over-tune to the warm-up
   leaderboard.
3. **Cue-locked windows.** Windows start at the cue, so the cue responses
   (0.4 s, 1.6 s) and any cue-driven eye movement are always inside the window.
4. **No baseline period** exists in the windows, so baseline-relative ERD
   measures are impossible. Features must be relative (between channels, bands
   or time points within the window).
5. **Headroom estimate for the warm-up:** simple linear models already reach
   0.77 cross-subject. A combined slow + mu/beta model is the obvious next
   benchmark to beat.
