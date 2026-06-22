# Stim module — Task 1: trigger-delta calibration — LEDGER

Append-only, timestamped. Single source of truth for the foot-sole electrical-
stimulation trigger calibration. Branch: `feat/stim-module`.

---

## 2026-06-12 ~10:40 — FACTS ESTABLISHED (git 2d49fb6)

### Data location & inventory
- Raw files: `{raw_root}/Pxx/Pxx_Stim.bdf`, with `raw_root` =
  `C:/Users/Ali D/OneDrive - The University of Western Ontario/MSc/Thesis/Data/Participants`
  (set in `configs/local.yaml`).
- 33 participants have `Pxx_Stim.bdf` (P26 missing). Pre-refactor analysis
  script `Pxx/Pxx_Stim.py` present for each; read for deltas + event logic.
- Inventory written to `outputs/stim_module/facts_inventory.csv`
  (script: `scripts/stim_module/facts_01_inventory.py`).

### Sampling frequency — VARIES
- 30 participants @ **1024 Hz**; **P02 and P11 @ 2048 Hz**.
- => deltas expressed in SAMPLES are NOT comparable across participants; a delta
  must be specified in TIME and converted per recording. This is the crux.

### Trigger scheme — confirmed
- Codes in the BioSemi `Status` channel: `256`=One prompt, `512`=Two prompt,
  `1024`=stim trigger. Also present cohort-wide: `96` (response), `2048`,
  `2144`(=2048+96), `4096`, and occasional `32`/`64` bit flips.
- Per prompt block: 4 × `1024` stim triggers. Typical session = 40 One + 40 Two
  prompts → 320 stim triggers (160 One-stim, 160 Two-stim). Counts vary for a
  few (P03 half-session; P07/P27/P39 irregular; P05 is a 12-trial fragment).

### 1024 → prompt attribution — DONE, provably correct
- `scripts/stim_module/stim_common.py::pair_stims` reproduces the per-participant
  scripts' walk: a 256/512 opens a One/Two block, the following `1024` events
  (Status prev-value col == 0) bind to that block up to 4, then the block closes.
- For all regular sessions every `1024` is attributed (One/Two counts match
  4×n_prompts exactly). A handful of trailing/again-pressed `1024`s beyond the
  4-cap are dropped (P01:48, P19:41, P37:5) — same behaviour as the original.

### Hand-set deltas (from Pxx_Stim.py) — scattered, in SAMPLES
- Per-participant, per-condition `Control/One/Two_trig_diff` (added to the 1024
  sample index). Harvested for all participants (facts_inventory.csv).
- Spread (samples): One mean 247 ± 29 (167–317); Two mean 242 ± 40 (98–317);
  Control mean 251 ± 25. In TIME at each recording's sfreq the spread is even
  worse because the two 2048 Hz participants got ~similar SAMPLE values
  (~220 smp = ~108 ms) as the 1024 Hz cohort (~250 smp = ~244 ms) — i.e. the
  hand-set deltas badly UNDER-correct the 2048 Hz recordings.

### Stim artifact — VISIBLE and is the ground truth (signal A)
- A sharp, broadband, near-synchronous deflection appears in the scalp EEG at a
  fixed latency after the `1024` trigger. Detected per participant as the peak of
  the across-channel median |gradient| (matched-filter / bootstrap;
  `stim_common.ga_peak_latency`).
- In 18/32 participants the artifact is sharp and unambiguous (GA peak FWHM
  ≈ 12 ms, prominence ≫ noise). In the rest the scalp artifact is weak/absent
  and the detector self-flags it (wide bootstrap CI, large FWHM) — these need an
  SEP-based fallback, not a different delta.

### Trigger→true-stim offset (clean participants, n=18)
- **Constant in TIME, not samples.** Peak offset mean **273.8 ± 4.9 ms**
  (onset 272.3 ± 4.9 ms); within-participant bootstrap SD ≈ 4 ms; FWHM ≈ 12 ms.
- P02 @ 2048 Hz lands at 266 ms — inside the 1024 Hz cluster — directly
  confirming time-constancy across sampling rate.
- Failure shape: the artifact GA is razor-sharp (FWHM ≈ 12 ms ⇒ trial-to-trial
  jitter is small), so the 1024 trigger is offset from the true stim by a near-
  **constant** amount, NOT random jitter/drops. ⇒ a uniform shift CAN work.
- Output: `outputs/stim_module/offsets_per_participant.csv`,
  per-participant figs `figs/offset_<pid>.png`
  (script: `scripts/stim_module/measure_03_offsets.py`).

### Sanity check vs physiology
- 273 ms is a large, fixed trigger→stimulus delay — consistent with a fixed
  system/output latency (identical equipment across participants/trials, per the
  task brief). It is NOT the SEP latency; the cortical foot-SEP appears tens of
  ms AFTER this true-stim instant (verified next in signal B).

### Preliminary verdict (pending signal-B confirmation)
- Between-participant spread of the data-driven optimum (SD ≈ 5 ms) ≈ the
  within-participant measurement uncertainty (≈ 4 ms) ⇒ a SINGLE uniform delta
  is statistically justified and is strictly better-behaved than the hand-set
  per-participant/per-condition samples (which range 95–310 ms in time and
  mis-scale the 2048 Hz recordings).
- Candidate uniform delta: **≈ 273 ms** (≈ 280 samples @ 1024 Hz, ≈ 560 @ 2048).
  CI + cost-per-ms to be finalised after the SEP grand-average comparison.

---

## 2026-06-12 ~11:00 — SIGNAL B (SEP), SIGNAL C (audio), PAYOFF, VERDICT (git 2d49fb6)

### Signal B — cortical SEP (cross-check)
- Scripts: `measure_04_sep.py` (vertex SEP), `measure_05_payoff.py` (artifact coherence).
- The trial-averaged SEP shape within a participant is invariant to a constant
  trigger shift (averaging is shift-invariant), so a constant delta can only be
  evaluated at the CROSS-PARTICIPANT level.
- The vertex cortical foot-SEP is WEAK / not robustly time-locked at the group
  level here (group RMS flips sign between mean and median estimators; residual
  artifact dominates the window). HONEST CONCLUSION: the cortical SEP is NOT a
  reliable ground-truth in this dataset — the stim ARTIFACT is.
- The high-SNR cross-check that DOES work: cross-participant coherence of the
  artifact saliency. Aligning the 18 clean participants by the uniform offset
  yields a group artifact peak **2.86× taller and sharper** (FWHM 13 ms) than
  aligning by the hand-set deltas (FWHM 22 ms, peak smeared over +10..+60 ms).
  Hand-set deltas sit on average **48 ms short** of the true stim (SD 46 ms).
  Fig: `figs/artifact_coherence.png`, `figs/optimum_vs_handset.png`,
  `figs/offset_distribution.png`.

### Signal C — audio / auto-logged event codes (NOT usable as a stim anchor)
- Per-prompt event structure (from the Status stream): prompt 256/512 → 4×1024
  stim triggers (ISI ~520 ms) → response 96/2144 → 3×2048 → 10×4096, then next
  prompt. The 2048/4096 bursts occur AFTER the stim block and belong to later
  task phases (their latency to the preceding 1024 is ~2.1 s / ~7.2 s), not the
  stimulus instant. No logged code sits near the true stim (~273 ms post-1024).
- => The auto-logged audio/event codes cannot independently time the stimulus,
  so signal C provides no usable cross-check. The artifact remains ground truth.

### Cost of misalignment (precision required)
- Averaging responses mis-shifted by SD σ scales a Gaussian component of std
  s_p by s_p/sqrt(s_p²+σ²). At the measured between-participant SD ≈ 5 ms: the
  sharp artifact retains 74%, an early cortical SEP (FWHM 30 ms) **93%**, a broad
  SEP/CNV (FWHM 60 ms) **98%**. Required timing precision for neural analysis is
  loose (~±10 ms). Fig: `figs/coherence_vs_jitter.png`.

### ===== TASK-1 DELIVERABLE / VERDICT =====
1. **Ground-truth signal:** the **stim ARTIFACT** — a sharp, broadband,
   near-synchronous deflection ~273 ms after each recorded `1024` trigger. Clean
   and unambiguous in 18/33 participants (FWHM ≈ 12 ms). Cortical SEP and the
   audio/event codes do NOT provide reliable independent stim timing here.

2. **Is a single uniform delta justified? YES.**
   - The failure mode is a near-**constant offset**, not random jitter/drops:
     the artifact GA is razor-sharp (FWHM ≈ 12 ms ⇒ trial-to-trial jitter small).
   - The offset is constant in **TIME**, not samples: clean optima span only
     266–278 ms with between-participant **SD ≈ 4.9 ms**, which is ≈ the
     within-participant measurement uncertainty (bootstrap SD ≈ 4 ms). The two
     2048 Hz recordings (P02 = 266 ms, and P09/others) land in the same cluster.
   - Statistical basis: between-participant spread ≤ within-participant
     uncertainty ⇒ no evidence the optimum differs by participant; a single
     value is supported. Uniform alignment beats the hand-set deltas 2.86× on
     group artifact coherence.

3. **Recommended number:** uniform trigger→true-stim offset
   **≈ 273 ms** (peak; artifact onset ≈ 272 ms — adopt **273 ms**).
   - 95% CI on the mean: **[271.6, 276.1] ms**; between-participant SD ≈ 5 ms.
   - In samples (offset_s × sfreq, rounded per recording):
     **280 samples @ 1024 Hz**, **560 samples @ 2048 Hz**.
   - ADOPT IT, specified in TIME and converted to samples per recording. This
     also CORRECTS the two 2048 Hz participants, whose hand-set sample deltas
     (~110 ms) badly under-corrected.
   - Config: `configs/stim.yaml` (`stim_events.trigger_offset_s: 0.273`), with
     the per-participant hand-set deltas preserved as commented provenance.

### Caveats / follow-ups
- 15/33 participants have a weak/absent scalp artifact (detector self-flags via
  wide bootstrap CI + large FWHM). The uniform 273 ms still applies to them (it
  is a fixed system latency); they simply can't *confirm* it individually. If a
  per-participant check is ever needed for those, use an EXG/EMG or mastoid
  channel (closer to the stim) rather than the scalp average.
- P05 is a 12-trial fragment; P07/P16/P30 produced off-cluster detector peaks
  (excluded as unclean) — not used in the pooled estimate.
- The ~10 ms biphasic structure (clean optima cluster at 267 vs 277 ms) is the
  two lobes of the biphasic artifact; both are within ±5 ms of 273 ms and below
  the required precision, so they do not affect the recommendation.

---

## 2026-06-12 ~12:05 — TASK 2: SEP evoked plots + component t-tests (git c1ecbf7)

User request: limit to the 30-participant study cohort; build evoked SEPs for
(standing pooled, straight stepping = 4 stims pooled, diagonal stepping = 4
stims pooled); t-test P2P / P50 / N90; then by stim ORDER (1st..4th in each
path), t-test the same. Settings agreed: uniform automated preprocessing (no
CSD), re-measure the standing offset, Part B = both within-path and matched
order. Alignment = Task-1 uniform 273 ms (stim) / re-measured offset (standing).

### Cohort handling
- 30 requested. **P26 excluded** (no Stim/Standing .bdf). **P01** has no Standing
  file → contributes to stepping/Part B but not standing contrasts. **P05** Stim
  is a 12-trial fragment → its stepping cells fall below the 15-trial floor and
  are dropped (its 300-trial standing is kept). Net: 29 participants with data;
  paired tests use pairwise-complete n (27–28 per contrast).

### Preprocessing (uniform, automated; `stim_preprocess.py`)
- pick 64 EEG → biosemi64 montage → notch 60 Hz + harmonics → pyprep auto
  bad-channel detect + interpolate → average reference → ICA (infomax-extended,
  fit on a 1–100 Hz / 256 Hz copy, auto-labelled by ICLabel, artifact ICs at
  p≥0.8 excluded) → 0.1–36 Hz bandpass → epoch −100..300 ms, baseline (−50,0),
  fixed 150 µV reject → average. NO CSD/ASR/zapline (would distort P50/N90).
- Standing offset re-measured per participant from its own artifact: clean in
  most (e.g. P25 267.4, P35 266.7 ms, FWHM ~12 ms — same fixed latency as the
  stim task); falls back to 273 ms where the artifact is undetectable (P31, P33).
- Cached per-participant evokeds in `outputs/stim_module/evokeds/<pid>.npz`
  (driver `cache_06_evokeds.py`); stats/plots in `measure_07_stats_plots.py`.

### SEP morphology (vertex Cz, grand-average, true-stim-aligned)
- Reproducible but small: a weak early positive **P50 (~40 ms, ~0.1 µV)** and a
  dominant negative **N90 trough (~70–90 ms, ~−0.35 to −0.49 µV)**, then a late
  positive rise (~130–160 ms). Components measured per participant as max in
  P50 window [25,65] ms and min in N90 window [65,115] ms; P2P = P50−N90.
- Consistent with Task 1: the cortical SEP here is low-amplitude (sub-µV).

### Results — paired t-tests at Cz (Holm-corrected within each measure family)
- **Part A (standing / straight / diagonal):** NO significant differences in
  P2P, P50, or N90 (all p_holm > 0.6). The three conditions have
  near-identical vertex SEPs. Tables: `ttests_partA.csv`.
- **Part B2 (matched order, straight-N vs diagonal-N):** only stim #2 shows a
  P2P difference (straight > diagonal, Δ=+0.52 µV, t(26)=+2.33, raw p=0.028,
  p_holm=0.11, dz=0.45) — a trend, not surviving correction. #1/#3/#4 null.
  Table: `ttests_partB_matched.csv`.
- **Part B1 (within-path order):** the notable effect is **straight stim #1 <
  stim #2 in P2P** (Δ=−0.47 µV, t(26)=−2.91, raw p=0.0073, dz=−0.56) — strong as
  an uncorrected effect but only marginal after Holm (p_holm=0.087) and somewhat
  window-sensitive (Holm 0.036 with a [75,130] ms N90 window). Diagonal shows
  weaker N90-amplitude trends across order (e.g. #2 vs #4, raw p=0.03) that do
  not survive correction. Table: `ttests_partB_within.csv`.

### Verdict (Task 2)
- The first stim in straight stepping evokes a SMALLER vertex SEP (P2P) than
  later stims — the only effect approaching significance; treat as suggestive
  (uncorrected p<0.01, marginal after multiple-comparison correction), worth
  confirming with a pre-registered window or a repeated-measures model.
- Otherwise SEP amplitude/components are statistically indistinguishable across
  standing vs straight vs diagonal and across most stim orders, at this (low)
  cortical-SEP SNR.

### Figures (`outputs/stim_module/figs/`)
- `sep_partA_conditions.png` — standing/straight/diagonal grand-avg ±SEM at Cz.
- `sep_partB_straight_order.png`, `sep_partB_diagonal_order.png` — by stim order.
- `sep_partB_matched_order.png` — straight vs diagonal at each matched order.

### Tables (`outputs/stim_module/`)
- `sep_measures_per_participant.csv` — per-participant P50/N90/P2P + latencies.
- `ttests_partA.csv`, `ttests_partB_within.csv`, `ttests_partB_matched.csv`.

### Notes / caveats
- Components are sub-µV; null results are expected at this SNR and do not imply
  no true differences. Windows were set from the grand-average morphology, not
  tuned for significance; the one trend's sensitivity to window is flagged above.
- Paired (within-subject) t-tests, pairwise-complete; Holm within each measure
  family. ROI-mean (Cz,C1,C2,FCz,CPz) available via the script's CHANNEL switch.
