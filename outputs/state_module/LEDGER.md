# State module — 3-class motor-state classification — LEDGER

Append-only, timestamped, git-SHA-stamped. Single source of truth for the
3-state EEG classifier (**standing** vs **straight** step vs **diagonal** step),
per-participant nested CV. Sibling of the binary CNV step-type pipeline
(`src/eeg_steptype/`). Branch: `feat/state-classification`.

Mirrors the discipline of `outputs/stim_module/LEDGER.md` and
`outputs/perf_loop/LEDGER.md`. Every fact, candidate, metric, and verdict is
logged here with a timestamp + git SHA.

---

## 2026-06-22 ~18:05 — PHASE 0: FACTS ESTABLISHED + SCAFFOLD (git d6317fc)

Branch `feat/state-classification` created off `main` (d6317fc). The CNV
pipeline `src/eeg_steptype/` is **not** modified; the state module is a parallel
sibling that reuses CNV/stim machinery by import.

### Data location & cohort
- Raw files: `{raw_root}/Pxx/Pxx_{Stim,Standing}.bdf`, `raw_root` =
  `C:/Users/Ali D/OneDrive - The University of Western Ontario/MSc/Thesis/Data/Participants`
  (`configs/local.yaml`). Verified by direct scan (read-only, no preload).
- **File availability (34 participant dirs):**
  - BOTH `Stim`+`Standing` (32): P02 P03 P04 P05 P06 P07 P08 P09 P10 P11 P12 P13
    P14 P15 P16 P17 P18 P19 P21 P22 P23 P24 P25 P27 P28 P29 P30 P31 P33 P35 P37 P39
  - ONLY `Stim` (1): **P01** (no Standing → cannot do 3-class → dropped).
  - NEITHER (1): **P26** (dropped).
- **State cohort (DECIDED, default):** the CNV-vetted cohort ∩ has-both-files =
  **28**: P02 P03 P05 P06 P07 P08 P10 P11 P12 P13 P14 P15 P16 P18 P19 P21 P23 P24
  P25 P27 P28 P29 P30 P31 P33 P35 P37 P39.
  - **FLAG:** P04 P09 P17 P22 have both files but were excluded by the CNV branch
    (`configs/default.yaml`, reason not recorded — possibly CNV-specific). Held
    out of the default state cohort for parity; can be added (they are valid
    Stim+Standing recordings). Decision pending user confirmation.
  - **P05 caveat:** `P05_Stim.bdf` is a **40 s fragment** (2×256, 1×512, 12×1024)
    → only ~2 straight + ~1 diagonal stepping epochs. Effectively unusable for
    the stepping classes. Kept in the list; the per-participant min-epoch guard
    will drop it (logged, not crashed). Standing is fine (300 stims).

### Sampling frequency — VARIES PER FILE (critical)
- Most recordings 1024 Hz. 2048 Hz outliers, **verified per file**:
  - `Stim` @ 2048 Hz: P02, P11.
  - `Standing` @ 2048 Hz: **P02, P06, P11, P29.**
- **P06 and P29 have a 1024 Hz `Stim` file but a 2048 Hz `Standing` file.**
  ⇒ sfreq is a property of the *recording*, not the participant. Never express
  timing in samples without reading each file's `sfreq`. The 273 ms e-stim offset
  must be applied as `round(0.273 * sfreq)` **per file** (so ~280 samples @ 1024,
  ~560 @ 2048). This is the same crux the stim module flagged, now per-file.

### Trigger schemes — verified on P01/P02/P03/P05/P06/P07/P11/P29 (Status channel)
| Condition | File | Triggers | Epoch (DECIDED) |
|---|---|---|---|
| **standing** | `Standing.bdf` | **only `1024`** (~300 @ ISI **0.520 s**, ~172–177 s); P07 317 stims/191 s; P29 480 stims/279 s (2048 Hz). **No 256/512/96.** | random/tiled **2 s** windows |
| **straight** | `Stim.bdf` 256/One | 40× `256` → `96` → 4× `1024` → … | pair `256→96`, epoch `[-0.1, +2.0]` |
| **diagonal** | `Stim.bdf` 512/Two | 40× `512` → `96` → 4× `1024` → … | pair `512→96`, epoch `[-0.1, +2.0]` |
- `pair_stims` (stim module) reproduces 160 One-stims + 160 Two-stims (80 at each
  within-prompt order 1–4) for regular sessions; cue→`96` pairing gives 40+40.
  Irregular: P03 half (20+20), P05 fragment (2+1), P07 (38+44).

### KEY TIMING FINDING — e-stims land mostly AFTER the 2 s stepping window
Verified event order within a stepping prompt (P29, P03), relative to `256` cue:
```
256 cue   +0.000 s
 96       +0.55  s   <- the paired response (events.py _pair locks here)
1024      +1.60  s     (+1.05 s rel. 96)
1024      +2.12  s     (+1.57 s rel. 96)
1024      +2.64  s     (+2.09 s rel. 96)  <- past +2.0 window edge
1024      +3.16  s     (+2.61 s rel. 96)  <- past +2.0 window edge
 96       +4.09  s
```
- The 4 foot-sole e-stims of a trial occur **~1.0–2.6 s after the paired `96`**.
  Locked to that `96` with `[-0.1, +2.0]`, only **~2** e-stims fall strictly
  inside the window — **not 4** as the brief estimated.
- **Resolution for the SEP block attribution (DECIDED):**
  - **Stepping:** attribute each trial's **4 e-stims to that trial's analysis
    epoch by prompt** (via `pair_stims` walk), not by strict time-containment.
    This matches the brief's intent (≈4 e-stims/epoch), gives a cleaner per-epoch
    SEP, and is **still leakage-safe** — each epoch's SEP comes only from *its
    own trial's* e-stims (no cross-epoch broadcast).
  - **Standing:** no prompt structure → attribute by **time-containment** (e-stims
    whose corrected sample ∈ the window); ISI 0.52 s ⇒ ~4 per 2 s window.
  - QC will report the **actual e-stim count per epoch** per condition (validity
    trap #1 / #2 monitoring). The per-epoch e-stim *count* is **not** emitted as a
    model feature (it would be a trivial standing↔stepping cue — trap #2).

### Validity traps (carried from the brief; monitored here)
1. **SEP leakage** — SEP features are per-epoch from that epoch's own e-stims
   only; never a per-condition mean broadcast. (Design above enforces this.)
2. **E-stim rhythm cue** — standing = continuous 0.52 s e-stims; stepping = a
   cluster then a gap. The artifact rhythm in window features could separate
   standing↔stepping with no brain signal. Mitigation plan: blank/interpolate the
   stim-artifact intervals before window-feature extraction + a stim-artifact-only
   control. To implement in Phase 1–2; flagged loudly.
3. **Class imbalance** — standing epoch count is a free parameter; balance to the
   stepping count (~40/condition; fewer for half/irregular sessions), fixed seed,
   stratified CV, plus `sample_weight="balanced"`.
4. **Repo gotchas** — run with `PYTHONUTF8=1`; verify participant + feature-column
   counts in first log lines; bump `features.cache_tag` when blocks change; venvs
   `.venv` (Py 3.14 classical) / `.venv312` (Py 3.12 + TF).

### Assumptions baked in (flag + confirm)
- Package `src/eeg_statetype/`, branch `feat/state-classification` (created).
- Conditions `[standing, straight, diagonal]`; internal label map
  `{standing:0, straight:1, diagonal:2}` (straight↔One/256, diagonal↔Two/512).
- E-stim trigger correction = uniform **273 ms** (`round(0.273*sfreq)` per file);
  hand-set deltas kept as provenance + opt-in fallback (`apply_uniform_offset`).
- Dual preprocessing: **CSD** for window electrode features (mirrors CNV);
  **non-CSD avg-ref** for the SEP block + source localization.
- Default feature blocks `[amplitude, slopes, psd, src, sep]`.
- Standing 2 s epochs balanced to stepping counts, fixed seed.
- Stepping window locked to paired `96`, `[-0.1, +2.0]` (mirrors `Pxx_Stim_2s.py`
  + CNV). SEP per-epoch window `[-0.1, +0.3]`, baseline `(-0.05, 0)`, common
  1024 Hz grid; components read ≥15 ms: P50 (40–50 ms), N90 (75–90 ms),
  P2P + RMS (15–130 ms) at vertex `[Cz, C1, C2, FCz, CPz]`.
- Primary metric: per-participant macro-OVR AUC (+ overall accuracy + 3×3
  confusion), vs the 1/3 chance line + inner-vs-outer overfit gap. Nested CV
  mirrors CNV (`repeated_stratified` 5×20 outer / 3 inner, stability selection
  in-fold).

### Phase 0 deliverables (this entry)
- `scripts/state_module/facts_01_inventory.py` (read-only trigger/sfreq audit).
- `configs/state/default.yaml` (3 conditions, `state_events`, 273 ms offset,
  blocks incl. `sep`, cohort).
- `src/eeg_statetype/` scaffold: `io.py` (state paths/cache namespace),
  `config.py` (state config loader), package `__init__`s.
- This LEDGER.

### Next (Phase 1, pending checkpoint)
Preprocessing: dual-branch (CSD window epochs + non-CSD avg-ref source & SEP
epochs); standing random 2 s epochs (fixed seed, balanced); stepping cue→response
windows; 273 ms e-stim offset; raw-assembly overrides harvested from the precedent
`Pxx_Stim_2s.py`/`Pxx_Stim.py` crop/concat; QC incl. e-stim-count-per-epoch.
Smoke on 1–2 participants (incl. a 2048 Hz Standing, e.g. P06) before any cohort run.

---

## 2026-06-23 — USER DECISIONS (checkpoint after Phase 0)
- **Cohort = all 32 with both files** (include P04, P09, P17, P22 that the CNV
  branch dropped; no documented reason was found in-repo). `configs/state/default.yaml`
  updated to 32. P05 still auto-dropped (40 s `Stim` fragment) by the min-epoch guard.
- **Build Phases 1–3 continuously**, smoke-testing each stage; pause only for the
  compute-heavy cohort runs or on breakage.

---

## 2026-06-23 ~10:05 — PHASES 1–3 IMPLEMENTED + SMOKE-VALIDATED (git d6317fc, uncommitted)

All three phases are implemented and validated end-to-end on smoke participants.
Code (state package): `preprocessing/{events_state,load,epoching,pipeline}.py`,
`features/{sep,assemble}.py`, `source_localization/pipeline.py`,
`models/{feature_selection,xgb,logistic,evaluate,train}.py`, `viz/results.py`,
`run_state.py`; 15 `configs/state/overrides/*.yaml`; `configs/state/{default,smoke}.yaml`.

### Phase 1 — preprocessing (validated on P06; P03 split-file in progress)
- **P06 done in 12.9 min.** Per-file sfreq handled: P06 Stim @1024 (offset 280
  smp), Standing @2048 (offset 559 smp). 3-segment Standing crop applied.
- **Per-prompt e-stim attribution gives exactly 4.00 e-stims/stepping-epoch**
  (straight 40 epochs/160 estims, diagonal 40/160); standing 3.77/window.
- Epoch counts (P06): standing 63, straight 40, diagonal 40; AutoReject dropped 0.
- SEP epochs kept 240/245 (150 µV reject). Dual branches (CSD window epochs +
  non-CSD avg-ref source & SEP epochs) all written. Stim-artifact blanking
  (±12 ms) applied to window-feature data (trap #2 mitigation).
- ICA on the full ~1080 s Stim recording is the cost driver: **~495 s** (vs 97 s
  for the short Standing file).

### Phase 2 — features (validated on P06)
- src: per-epoch eLORETA **~3–5 s/epoch**, 4951 src cols/epoch. **FLAG: mean
  per-epoch variance-explained is low (5–12%)** — expected for single-trial
  eLORETA but means `src` may add little; the on/off ablation will decide.
- **SEP block = 36 features/epoch** (5 vertex × {p50,n90 amp+lat, p2p, rms} + vtx
  mean). **Trap #1 verified: SEP features vary WITHIN each condition**
  (e.g. `sep_vtx_p2p` within-cond std 2.0–3.5), i.e. genuinely per-epoch, not a
  per-condition constant. Only 3/143 epochs needed fill (no in-epoch e-stim).
- Wide feature matrix: **19,560 cols/epoch** ([amplitude, slopes, psd, src, sep]).

### Phase 3 — 3-class modeling (validated on P06, smoke config: reduced folds)
- Fixed: CNV logistic uses `liblinear` (binary-only) → state `logistic.py` uses
  `lbfgs` (multinomial). xgb uses `multi:softprob`.
- Balancing: standing subsampled to stepping count (63→40) → 40/40/40, 120 epochs.
- **logistic:** macro-OVR AUC **0.66**, acc 0.36 (chance 0.33), overfit gap 0.11.
- **xgb:** macro-OVR AUC **0.86**, acc **0.68**, macro-F1 0.67, overfit gap **0.02**
  (honest). Per-class recall: standing 0.975, straight 0.55, diagonal 0.51.
- Metrics: 3×3 confusion + per-class recall + macro-OVR AUC + macro-F1 + overfit
  gap vs chance (acc 1/3, AUC 0.5). All wired in `evaluate.py`.

### Honest framing / validity (to verify at cohort scale + ablation)
- **Standing is easily separated (0.975 recall) — partly confounded by stim
  rhythm (trap #2).** Standing = continuous 0.52 s e-stims; stepping = 4 clustered.
  Artifacts are blanked, but the *blank pattern* itself differs by condition, so
  window-feature standing↔stepping separation is not fully clean. **`straight` vs
  `diagonal` is the confound-free comparison** (both stepping, identical stim
  structure) — and is the harder part (~0.5 recall each), mirroring the CNV
  binary task. The SEP-vs-window ablation + a stim-artifact-only control (Phase 4)
  will quantify the confound.

### Cohort run — NOT yet run (the compute-heavy step; pausing here per directive)
Naive sequential cohort ETA (32 participants): preprocess ~7 h (ICA-bound) +
src ~6 h + features ~1.5 h + train (parallelizable). Optimizations available:
ICA-fit downsampling (≈4× faster ICA), capping standing windows before src,
joblib participant parallelism. To be decided at the checkpoint.

---

## 2026-06-24 ~09:25 — INCIDENT: 22 h SCREEN stall — root-caused + fixed (git 555cb8a)

### What happened
The first SCREEN build (launched 06-23 10:23) **silently stalled for ~22 h**. The
python process died mid-ICA on **P05's Stim file** (10:42:46) and the background
shell hung without emitting a completion event, so no notification fired and the
run was waited on blind. P02/P03 preprocessed fine; P06 had faithful epochs from
the smoke; src/features never started.

### Root cause — Picard ICA on the long Stim recordings (NOT memory/sleep)
- 44 GB RAM free → **not OOM**; no kernel-power sleep and no app-error event at
  the freeze. The downsampling WAS applied to every file.
- But ICA fit time on the ~1080 s Stim files was wildly variable and convergence-
  bound, not sample-bound: P02 Stim 64 s, **P03 Stim 520 s** (rank 62),
  **P05 Stim hung indefinitely** (rank 58). Standing files (~135 s) were all fine
  (15–26 s). ⇒ Picard converges slowly/unstably on the long recordings.

### Fix (validated)
1. **Fit ICA on a bounded central segment** (`preprocessing.ica.fit_max_duration_s
   = 180`) + cap iterations (`max_iter = 200`). The researcher's own precedent
   scripts fit ICA on a ~50 s crop; the unmixing is still applied full-res. **P05
   Stim ICA: hang → 14.8 s; P05 now fully preprocesses in 3.2 min** (straight 40 /
   diagonal 40 / standing 50, 4.00 e-stims/epoch — P05 is usable via Stim_1.bdf).
2. **Process-isolated cohort runner** (`scripts/state_module/run_cohort.py`): each
   (participant, stage) in its own subprocess with per-stage timeouts
   (preprocess 1200 s / src 3000 s / features 900 s), resumable, writing a
   heartbeat to `outputs/state_module/logs/cohort_progress.txt`. A future hang is
   now killed by the timeout and the batch continues; progress is monitored live.

### Lesson
One long-lived process for the whole cohort is fragile + unobservable. Long runs
must be process-isolated, timeout-guarded, and actively monitored (not waited on
via a single completion notification).

### Now running (monitored): robust SCREEN build (8 participants, fast config).
