# New branch: 3-state EEG classification — standing vs straight-step vs diagonal-step

## Mission
Build a new analysis branch that classifies a participant's **motor state** from EEG —
**standing** vs **straight stepping** vs **diagonal stepping** — using per-participant
nested cross-validation. It is the 3-class sibling of the existing binary CNV step-type
pipeline (`src/eeg_steptype/`, straight `One` vs diagonal `Two`). Mirror that pipeline's
architecture exactly; only the **dependent variable** (2 classes → 3 states) and the
**independent variables** (CNV-window features **+ a new foot-sole SEP feature block**)
change.

Work on a new branch off `main`: **`feat/state-classification`**. Do NOT modify the CNV
pipeline (`src/eeg_steptype/`) — create a parallel sibling package. Keep
`outputs/state_module/LEDGER.md` as the append-only, timestamped, git-SHA-stamped single
source of truth, exactly like `outputs/stim_module/LEDGER.md` and `outputs/perf_loop/LEDGER.md`.

---

## Context — what already exists and what to reuse

**The CNV pipeline to mirror — `src/eeg_steptype/`** (installable package, `run.py` driver,
thin `scripts/01–05`, deep-merged `configs/`, per-participant `configs/overrides/Pxx.yaml`):
- `preprocessing/` — automated, objective: ZapLine → PyPREP bad-channel detect/interp →
  ASR → provisional CAR → dual-filter Picard ICA (fit on 1 Hz copy, apply to 0.1–40 Hz
  copy) + ICLabel auto-exclude → undo CAR → CSD → events → epoch → AutoReject.
- `source_localization/` — cached forward + eLORETA → per-epoch label time-courses CSV.
- `features/` — `amplitude`, `slopes`, `psd` (Morlet TFR), `src` blocks → wide parquet
  (`features/assemble.py`; blocks are pluggable — append a DataFrame keyed on `epoch`).
- `models/` — `train.py` (generic per-participant nested-CV driver), `feature_selection.py`
  (correlation drop → KBest → stability selection), `evaluate.py`, model factories
  (`xgb`, `svm`, `logistic`, `lstm`, `riemannian`, `cnn`, `eegnet`, `eegnext`), `pooling.py`
  (cross-subject partial/full pooling).
- `io.py` — **all path builders hardcode `_CNV_` in filenames** (e.g.
  `{pid}_CNV_{cond}-epo.fif`). The sibling package needs its own `io.py` with state-task
  paths and 3 conditions.

**The stim module to reuse — `scripts/stim_module/` (branch work already merged to `main`):**
- `stim_common.py` — `load_stim_raw()` (reads `Pxx_Stim.bdf`, renames A1–B32→10-20 via
  `MAPPING`, sets biosemi64 montage), `pair_stims()` (binds each `1024` e-stim to its
  preceding `256`/`512` prompt), `ga_peak_latency()` / `detect_artifact_latency()` (stim
  artifact onset), `bootstrap_ci()`.
- `stim_preprocess.py` — uniform automated **SEP** preprocessing (notch → PyPREP interp →
  average reference → Picard ICA + ICLabel → 0.1–36 Hz → epoch around the offset-corrected
  `1024` → fixed-threshold reject → average). **Deliberately NO CSD** (CSD distorts the
  P50/N90 SEP morphology). `_epoch_average()` applies the trigger offset and epochs.
- `configs/stim.yaml` — `stim_events` (256/512/1024), and the **Task-1 calibrated uniform
  trigger→true-stim offset `trigger_offset_s: 0.273`** (273 ms, CI [271.6, 276.1]), to be
  applied per recording as `round(0.273 * sfreq)` samples. Per-participant hand-set sample
  deltas are preserved there as commented provenance.
- SEP findings (stim module Task 2): vertex foot-SEP is sub-µV — weak **P50 (~40–50 ms)**,
  dominant **N90 (~75–90 ms)**; measured at `VERTEX = [Cz, C1, C2, FCz, CPz]` over a
  ~15–130 ms window. Reuse this measurement code; do not reinvent it.

**The researcher's own precedent scripts** (in `{raw_root}/Pxx/`, READ THEM FIRST):
- `Pxx_Stim_2s.py` — already extracts the **stepping 2 s window**: pairs `256→96` and
  `512→96` and epochs `tmin=-0.1, tmax=2.0` (it literally calls this "Stim's CNV"), AND
  extracts the `1024`-locked SEPs. It also shows the **manual per-participant steps to
  preserve**: raw crop/concat (e.g. P01 `crop(40,575)+crop(585,1125)` then
  `concatenate_raws`), the A1–B32→10-20 `mapping`, manual `bads`, and trigger deltas.
- `Pxx_Stim.py` — the SEP/standing precedent: loads BOTH `Pxx_Stim.bdf` (One/Two) and
  `Pxx_Standing.bdf` (Control = standing); per-condition trigger deltas; SEP epoching
  `-0.05..0.5 s`.
- `Pxx_CNV.py` — the original CNV preprocessing (montage, `256/512`+`96` pairing, `-0.1..2.0`).

---

## Data — verified against the raw `.bdf` (do not re-derive; confirm on smoke participants)

`raw_root` = `C:/Users/Ali D/OneDrive - The University of Western Ontario/MSc/Thesis/Data/Participants`
(set in `configs/local.yaml`). Files live at `{raw_root}/Pxx/Pxx_{Stim,Standing}.bdf`, same
per-participant folder layout as the CNV files.

**Trigger schemes (BioSemi `Status` channel) — verified on the full cohort:**

| Condition | Source file | Triggers present | Epoch definition (DECIDED) |
|---|---|---|---|
| **standing** | `Pxx_Standing.bdf` | **only `1024`** e-stim (~300 @ ISI ≈0.52 s, ~175 s). **No `256`/`512` cues.** | **Random 2 s epochs** drawn from the continuous recording. |
| **straight** | `Pxx_Stim.bdf`, `256`/One | `256` prompt (×40) → 4× `1024` + a `96` response | **The cue→response 2 s window**: pair `256→96`, epoch `tmin=-0.1, tmax=2.0` (the `Pxx_Stim_2s.py` "Stim-CNV"). |
| **diagonal** | `Pxx_Stim.bdf`, `512`/Two | `512` prompt (×40) → 4× `1024` + a `96` response | Same, pairing `512→96`. |

- Sampling rate is **mostly 1024 Hz; P02, P06, P11, P29 are 2048 Hz** — never express
  timing in samples without converting per recording's `sfreq`.
- `P01` and `P26` have **no Standing file**; `P26` has **no Stim file**. A few standing
  recordings are longer (P07/P13/P17/P29). Handle missing conditions gracefully (a
  participant with <3 conditions is dropped from the 3-class run, logged, not crashed).
- **Note on the brief's wording vs the data:** the brief said "stepping has no auditory
  cues; standing has 2." The trigger channel is the opposite — standing is uncued, stepping
  is cued. The resolution adopted above: stepping uses its cue→response window; standing,
  having no cue structure, is cut into random 2 s epochs. (Confirmed with the user.)

**Both** the stepping window AND the standing window contain foot-sole e-stims, so the SEP
block is available for all three conditions.

---

## Dependent & independent variables

**Dependent (label):** one of `{standing, straight, diagonal}` per epoch. Internally map
`straight ↔ One/256`, `diagonal ↔ Two/512`, `standing ↔ Control`. A **single multiclass
(3-way) model per participant** (not one-vs-rest).

**Independent (features), per epoch:**
1. The CNV-window feature blocks, mirrored from `eeg_steptype/features/`: `amplitude`,
   `slopes`, `psd` (Morlet), and `src` (eLORETA source-space) — computed over the 2 s window
   (`0.0–2.0 s`, bin width as in CNV). All four are on by default, exactly as the CNV branch
   (`blocks: [amplitude, slopes, psd, src]`).
2. **A NEW `sep` feature block** (per-epoch): from the foot-sole e-stims that fall inside
   that epoch, compute vertex SEP summaries at `[Cz, C1, C2, FCz, CPz]`. Reuse the stim
   module's measurement code; do not reinvent the windowing.

   **How the window is set around each electrical-stim timing** (mirror
   `scripts/stim_module/stim_preprocess.py`):
   - **Lock to the true stim, not the recorded trigger.** Shift every `1024` event sample by
     the Task-1 offset: `corrected_sample = recorded_1024_sample + round(trigger_offset_s ×
     sfreq)` with `trigger_offset_s = 0.273` (so ~280 samples @ 1024 Hz, ~560 @ 2048 Hz).
     The true stim is then at t = 0.
   - **Per-stim epoch:** `tmin = −0.10 s, tmax = +0.30 s` around the corrected stim, baseline
     `(−0.05, 0.0)` (the 50 ms immediately before the true stim). These are the stim module's
     `TMIN/TMAX/BASELINE`. The 400 ms window is shorter than the ~520 ms inter-stim interval,
     so successive e-stims don't contaminate each other's window.
   - **Common grid:** resample/interp each epoch onto a fixed 1024 Hz ms axis so 1024 Hz and
     2048 Hz participants share one time base; fixed peak-to-peak reject (`eeg = 150e-6`)
     drops artifact-blown trials.
   - **Read components after the artifact.** The sharp stim artifact sits at t = 0, so read
     only from **≥15 ms**: **P50 (~40–50 ms)** and **N90 (~75–90 ms)** peak amplitude +
     latency, and **peak-to-peak / RMS over 15–130 ms**. These are fixed physiological windows
     (not data-driven → no leakage).
   - **Per-epoch aggregation (leakage-safe):** for each 2 s analysis epoch, average the
     baselined SEP windows of the e-stims whose corrected time falls inside that epoch (≈4 for
     stepping and standing alike) into one cleaner per-epoch SEP, then read the components from
     it. **Per epoch, never a per-condition mean** (see validity trap #1).

---

## What must stay INTACT (manual, per-participant) vs. what must be AUTOMATED

Per the rules — preserve only the irreducibly manual raw-assembly steps, exactly as the CNV
branch does (in `configs/overrides/Pxx.yaml`, applied only when that participant is
processed):
- **KEEP (manual, per-participant, in overrides):** raw `.bdf` **trimming & binding**
  (crop windows / concatenation, e.g. P01's two-segment crop), **electrode renaming**
  (A1–B32 → 10-20 via the fixed `MAPPING`), **event categorization** (the
  `256/512→96` cue→response pairing and `256/512→1024` e-stim attribution), and the
  **trigger-movement step** (e-stim trigger correction must still happen).
- **AUTOMATE & make objective (cohort-uniform, like CNV):** everything else — bad-channel
  detection (PyPREP, not hand-typed bads), ICA component exclusion (ICLabel, not hand-picked
  indices), referencing, filtering, rejection (AutoReject). No per-participant hand-tuned
  ICA exclude lists or bads in default runs (preserve the originals as commented provenance
  only, mirroring the CNV migration).
- **Trigger movement, made objective:** use the stim module's **uniform 273 ms** offset
  (`round(0.273 * sfreq)` per recording) as the default, replacing the scattered hand-set
  per-participant sample deltas (Task 1 proved this is equal-or-better and fixes the
  mis-scaled 2048 Hz files). Keep the hand-set deltas as commented provenance + an opt-in
  fallback toggle. (Flagged assumption — see bottom.)

---

## Architecture — new sibling package mirroring the CNV branch

Create `src/eeg_statetype/` as a parallel of `src/eeg_steptype/` (rename freely if you
prefer, e.g. `eeg_state`). Reuse CNV modules by import where behavior is identical; fork a
module only where the dependent/independent variables force a change.

| CNV file | New file | Change required |
|---|---|---|
| `preprocessing/load.py` | reuse / thin wrapper | raw assembly from `Pxx_Stim.bdf` + `Pxx_Standing.bdf` (two source files); per-participant crop/concat via overrides. |
| `preprocessing/events.py` | `events_state.py` | 3-way event building: `256→96` (straight), `512→96` (diagonal) cue→response windows from Stim; **random 2 s epoch onsets** from Standing's continuous `1024` stream; plus e-stim attribution for the SEP block. |
| `preprocessing/pipeline.py` | `pipeline.py` | orchestrate per condition across both files; emit CSD epochs (window features) **and** a non-CSD avg-ref branch (SEP + source). |
| `source_localization/` | reuse | cached forward + eLORETA on the non-CSD avg-ref branch; `src` block on by default. |
| `features/{amplitude,slopes,psd,assemble}.py` | reuse + add `features/sep.py` | new `sep` block; register it in `assemble.py`'s block dispatch. |
| `models/train.py` | `train.py` | **binary → multiclass** (see below). |
| `models/evaluate.py` | `evaluate.py` | **binary → 3-class** metrics. |
| `models/xgb.py` + config | `xgb.py` + config | `objective: multi:softprob`, `eval_metric: mlogloss`, `num_class: 3`. |
| `io.py` | `io.py` | state-task paths (`_Stim`/`_Standing`, 3 conditions), own cache namespace. |
| `config.py`, `logging_utils.py` | reuse | unchanged. |
| `run.py`, `scripts/01–05` | mirror | `scripts/state_*` or a `run_state.py` driver. |
| `configs/default.yaml` + `overrides/` | `configs/state/…` | `conditions: [standing, straight, diagonal]`, `state_events` (256/512/96/1024 + 273 ms offset), per-participant raw-assembly overrides. |

### Binary → multiclass: the exact code points to generalize
- `train.py`: label map `{"One":0,"Two":1}` → `{"standing":0,"straight":1,"diagonal":2}`
  (tabular path ~L168 and tensor path ~L251); `predict_proba(X_test)[:, 1]` /
  `(proba>=0.5)` → `predict_proba` (n×3) + `argmax` (both `_fit_score_split` and
  `_fit_score_split_tensor`); `_scale_pos_weight` → per-class `sample_weight`
  (`compute_sample_weight("balanced", y_train)`); inner search `scoring` →
  `roc_auc_ovr` (or `f1_macro`). `RepeatedStratifiedKFold` already supports >2 classes.
- `evaluate.py`: `participant_metrics` `confusion_matrix(labels=[0,1])` + tn/fp/fn/tp +
  binary `roc_auc_score` → **3×3 confusion**, per-class recall (`accuracy_standing/straight/diagonal`),
  **macro one-vs-rest AUC** (`roc_auc_score(y, proba, multi_class="ovr", average="macro")`),
  overall accuracy, macro-F1. Update `cohort_rollup`/`cv_rollup` (which hardcode
  `total_One/Two`) to the 3 classes.
- xgb factory + config: multiclass objective/metric/`num_class` as above.

---

## Build plan (the loop) — phased, checkpoint between phases

> Discipline (from `CLAUDE.md`): **estimate an ETA before every long task, time-log the
> actual (logs carry `HH:MM:SS`), and investigate any overrun >1.5–2× or stall** — don't
> wait passively. Smoke-test on 1–2 participants before any cohort run.

**Phase 0 — Facts & scaffold.** Branch off `main`. Confirm the trigger inventory on 2–3
participants (incl. one 2048 Hz, e.g. P02/P11) and that `pair_stims` / the `256/512→96`
pairing reproduce the expected counts (40/condition stepping; ~85 candidate 2 s standing
windows). Stand up the empty `src/eeg_statetype/` package + `configs/state/` + LEDGER.
Write the established facts to the LEDGER. **Checkpoint.**

**Phase 1 — Preprocessing (automated, objective).** Implement the dual-output pipeline:
(a) the CNV automated chain (ZapLine→PyPREP→ASR→ICA/ICLabel→CSD→AutoReject) producing CSD
epochs for the window blocks, and (b) a non-CSD avg-ref branch for the SEP block (and
source). Standing → random 2 s epochs (fixed seed; non-overlapping or random onsets, count
configurable). Stepping → cue→response 2 s windows. Apply the 273 ms e-stim offset for SEP.
QC report per participant (bads, ICA excludes, epoch counts per condition, **e-stim count
per epoch**). Smoke on 1–2 participants. **Checkpoint.**

**Phase 2 — Source localization + features incl. the SEP block.** Run the source stage
(cached forward + eLORETA, on the non-CSD avg-ref epochs) since `src` is a default block.
Reuse amplitude/slopes/psd/src; implement `features/sep.py` (per-epoch vertex P50/N90/P2P
amplitude+latency from in-epoch e-stims) and register it. Default
`blocks: [amplitude, slopes, psd, src, sep]` (CNV's four blocks + the new SEP block). Verify
the wide parquet builds and the SEP columns are **per-epoch, not per-condition-constant**
(see validity traps). eLORETA per epoch is expensive — estimate/time-log it. **Checkpoint.**

**Phase 3 — 3-class modeling baseline.** Generalize `train.py`/`evaluate.py`/`xgb` to
multiclass. Run per-participant nested CV (mirror CNV: `repeated_stratified`, 5×20 outer,
3 inner; stability selection in-fold) with `xgb` (and `logistic` for a fast smoke). Report
per-participant + cohort **macro-OVR AUC, overall accuracy, 3×3 confusion**, with the
chance line (1/3) and the inner-vs-outer overfit gap. Write baseline to LEDGER. **Checkpoint.**

**Phase 4 — Performance loop (optional, mirror `perf/agentic-improvements`).** If asked to
push accuracy: iterate behind toggles — SCREEN on ~8 participants, CONFIRM on ~20; objective
= cohort held-out macro-OVR AUC, guardrail = overfit gap not worse than +0.03 vs baseline;
candidates e.g. SEP-only vs window-only vs combined (ablation), partial pooling
(`pooling.mode: partial`), `src` block on/off, bin width, channel ROI. Every result to the
LEDGER (timestamp + git SHA + verdict). Tag releases (semver, remote `personal`).

---

## Validity traps to avoid (call these out explicitly in the LEDGER)
1. **SEP leakage.** SEP features MUST be computed per-epoch from that epoch's own e-stims.
   A per-condition mean SEP broadcast to all epochs of a condition is **constant within
   class = perfect label leakage**. Per-epoch SEPs (≈4 stims) are noisy but honest.
2. **E-stim rhythm as a trivial standing↔stepping cue.** Standing has continuous ~0.52 s
   e-stims; stepping has 4 clustered stims then a gap. The e-stim **artifact rhythm** in the
   window features could let the model separate standing from stepping without any brain
   signal. Mitigate (interpolate/blank the stim-artifact intervals before window-feature
   extraction, or document it) and sanity-check with a stim-artifact-only control.
3. **Class imbalance.** Standing epoch count is a free parameter; balance it to the ~40
   stepping epochs/condition (or use `class_weight`/`sample_weight`), and stratify CV.
4. **Repo gotchas (from `CLAUDE.md` / memory):** always run with `PYTHONUTF8=1` (a θ glyph
   crashes cp1252 logging); `--config <missing-file>` silently falls back to the 30-subject
   default — verify participant + feature-column counts in the first log lines; the feature
   cache key omits the block list — bump `features.cache_tag` when changing blocks; venvs:
   `.venv` (Py 3.14, classical XGB/sklearn), `.venv312` (Py 3.12 + TF, neural only).

---

## Working conventions
- New branch off `main`; never commit to `main`. Don't commit large logs/artifacts (they
  sync via OneDrive; see `.gitignore`).
- `outputs/state_module/LEDGER.md` = single source of truth: append-only, timestamped,
  git-SHA-stamped; log every fact, candidate, metric, and verdict.
- Every change behind a config toggle / new block; do not mutate `src/eeg_steptype/`.
- Reuse existing machinery (stim `load_stim_raw`/`pair_stims`/SEP code, CNV preprocessing,
  feature blocks, `config.py` override merge, nested-CV driver) — don't re-implement.
- Diagnostic figures under `outputs/state_module/figs/`: per-condition grand-average window
  ERP, SEP grand-average, 3×3 confusion heatmap, per-participant accuracy/AUC vs chance,
  feature-importance / SEP-vs-window ablation.
- Smoke-test (1–2 participants, `logistic`) before every cohort run.

---

## Deliverables
1. `src/eeg_statetype/` + `configs/state/` + `scripts/state_*` (or `run_state.py`), mirroring
   the CNV branch, producing per-participant cleaned epochs → features (with SEP block) →
   3-class nested-CV metrics.
2. A baseline 3-class result (cohort macro-OVR AUC + overall accuracy + 3×3 confusion, vs the
   1/3 chance line and with the overfit gap), written to `outputs/state_module/LEDGER.md`.
3. The ablation answering: **does the SEP block add discriminative signal beyond the CNV
   window features, and which of the three states are separable?**

---

## Assumptions baked in — flag in the LEDGER and confirm/adjust if wrong
- Package/branch names: `src/eeg_statetype/`, `feat/state-classification`.
- E-stim trigger correction uses the **uniform 273 ms** offset (objective; Task-1 result),
  not the per-participant hand-set deltas (kept as provenance + fallback toggle).
- Dual preprocessing branch: **CSD** for window electrode features (mirrors CNV), **non-CSD
  avg-ref** for the SEP block (stim module found CSD distorts SEPs).
- Default feature blocks `[amplitude, slopes, psd, src, sep]` — CNV's four blocks (incl.
  `src`/eLORETA on by default) plus the new per-epoch SEP block. The source stage runs as
  part of the standard pipeline; budget for per-epoch eLORETA cost.
- Standing 2 s epochs: balanced to stepping counts (~40/condition), fixed seed.
- Stepping window locked to the paired `96` response, `tmin=-0.1, tmax=2.0` (mirrors
  `Pxx_Stim_2s.py` and the CNV window).
- Primary metric: per-participant macro one-vs-rest AUC (+ accuracy + 3×3 confusion);
  per-participant nested CV mirrors CNV (`repeated_stratified` 5×20 / inner 3).
