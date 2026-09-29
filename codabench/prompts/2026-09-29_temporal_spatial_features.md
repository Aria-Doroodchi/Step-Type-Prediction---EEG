# Sprint 2026-09-29: temporal and spatial features for the sealed recipe

Time frame: Tue 2026-09-29 17:46 → **Wed 2026-09-30 05:45** EDT (the user said
"12 hours"). The user said "find a worthy target and start a workflow" and
suggested *"investigating temporal features, and improving spatial features so
the differences in brain activation can be captured"*. The value check (§ 3)
kept that focus and applied it to the Track 2 sealed recipe. The user is not
expected to answer. Build and review run as multi-agent Workflow calls; compute
runs as watched, resumable lanes. Decide each next step from the previous result
with the rules below. Never let a job go stale (§ 6).

## 1. Context (verified 2026-09-29 17:50)

- Workstream: Codabench Track 2 (`codabench/`). Branch `feat/codabench-track2`,
  commit prefix `codabench:`. Read first: `SEALED_RECIPE.md` (§ 1, § 2 Phase 4,
  § 5), `HANDOFF.md`, `LOG.md` 2026-09-25 "Phase 4".
- Environment: WSL Ubuntu, `source ~/codabench/env.sh`, CPU only (20 threads,
  31 GB RAM). Proxy caches are in `~/neuralbench/xsess_cache/{scherer2015,
  tangermann2012,zhou2016,zyma2019}`, all at 120 Hz with 4 s windows (Zyma 5 s).
  The mock caches (`mock_sealed_*`) are synthetic and are used for sizing and
  gates only, never for feature decisions.
- What counts: the **sealed phase** (Oct 28 – Nov 21). Its classes are the 3
  mental tasks: hand MI, calculation and word association. The nearest proxy is
  **Scherer 2015 restricted to WORD / SUB / HAND** (cache labels 0,1,3). It has
  9 subjects, 30 ch, one training session and one test session each, and 120
  windows per session. The tracks page still said "coming soon" at 17:50 today.
- Power: sleep is disabled on AC and DC (read 17:52; not changed).
- **Already measured, do not repeat** (SEALED_RECIPE § 2, LOG 2026-09-25):
  whole-block ablations on the 3 classes (xDAWN, FB, broadband, log-var); the
  slow block on Dreyer only (+0.0 vs FB); CAR/Laplacian/8–30 Hz with EEGNet and
  Riemann on Dreyer 2-class (CAR best, 8–30 hurts); pooling, alignment,
  personalisation, pre-training, EEGNet variants. The thesis Riemann/FBCSP
  comparator is flat (AUC 0.53, `MODELS.md` § 5.5).
- **Recipe features today** (`riemann_steptype._feature_blocks`, copied into
  `riemann_sealed.py`): xDAWN covariances → TS; broadband OAS covariance → TS;
  broadband per-channel log-variance; 4-band filter bank (4–8, 8–13, 13–30,
  30–45 Hz) covariances → TS. Then shrinkage LDA with uniform priors. Every
  covariance is over the **whole 4 s window**: no temporal structure beyond
  xDAWN's class templates, and no band-power topography outside the TS.
- **Why "spatial filtering" alone cannot help this model** (checked
  2026-09-29): Riemannian tangent space at the class-agnostic mean is
  affine-invariant. A full-rank spatial transform A (CAR is rank-1 deficient;
  Laplacian/CSD is full rank) maps the TS vectors by an orthogonal transform. The
  LDA shrinkage target (μI) is rotation-invariant, so TS + shrinkage-LDA is
  invariant up to the OAS covariance shrinkage and the log-var block. Spatial
  gains therefore need **structural** changes, not re-referencing:
  - local (regional) covariances, which drop cross-region terms and shrink the
    dimension;
  - supervised low-rank spatial subspaces;
  - explicit per-band power topographies;
  - lagged connectivity (imaginary coherence) that TS cannot represent.

  The re-referencing runs are kept as a cheap **control** that tests this
  argument.

## 2. Data and assets

No downloads. All data is cached locally and pre-approved. Results go to
`~/codabench/logs/sealed_f0929/` (gitignored; commit only `RESULTS.md` /
`STATUS.md` with `git add -f`). The report and figures go to
`codabench/reports/features_0929/`.

## 3. Goal and priorities

Deliverables:
1. `analysis/xfeat.py` (+ `xfeat_temporal.py`, `xfeat_spatial.py`): opt-in
   feature blocks for the harness. The spec is `riemann:xd=1,fb=1,x=<b1>+<b2>`,
   optionally with `blocks=` to drop a base block. The defaults stay
   bit-identical.
2. `reports/features_0929/activation_eda.md` + figures: where and when the 3
   classes differ (band-power topographies, ERD/ERS time courses, band × time
   separability), from **training sessions only**.
3. The screen and confirmation tables with paired subject-bootstrap CIs, in
   `logs/sealed_f0929/RESULTS.md`. The LOG gets one entry per phase. SEALED_RECIPE
   § 2 gets a "Phase 7 — temporal/spatial blocks" section and § 5 is updated.
4. If a block is adopted or promising (§ 5 rules), it is ported into
   `riemann_sealed.py` as an opt-in. The defaults stay bit-identical; harness
   parity and 500 Hz sizing are checked, and the block is wired into
   `train_sealed.sh` / `release_ablations.sh` / `release_summarize.py` /
   RELEASE_DAY as a release-day ablation with a pre-registered rule.

Priority order:
1. Screen the candidates on the sealed-like proxy. This is the user's question,
   and new evidence.
2. The activation EDA. It explains the result and is the "investigation" half.
3. Replication on the other proxies, then confirmation under the deployed
   recipe (blend_calib).
4. Solver integration of anything that passes (or, if nothing passes, the
   chained fallback below).

**Chained fallback, if no candidate is adopted or promising after Phase 3:**
the strict per-fold references in the LOSO blend-weight search (SEALED_RECIPE
§ 5, "Before release day"). Harness `choose_w` and solver `_choose_blend_w` must
stay bit-identical to each other; report the weights before and after on Zhou,
`mock_sealed_s` and `mock_sealed_120`.

## 4. Candidates (pre-registered; names are the `x=` block ids)

All blocks are fitted on training windows only. Each block sees the same
preprocessed (and, under router alignment, whitened) windows as the base union.
The baseline is `riemann:xd=1,fb=1` (the recipe union).

| id | family | what it captures | spec |
|---|---|---|---|
| `tseg2`, `tseg3` | temporal | FB covariances of K non-overlapping time segments → TS each: how the spatial pattern evolves over the trial (early cue response vs sustained task) | `riemann:xd=1,fb=1,x=tseg2` |
| `acm3x2`, `acm2x4` | temporal | augmented (time-delay-embedded) broadband covariance, order p, lag in samples at 120 Hz (scaled with sfreq) → TS: temporal autocorrelation and lagged cross-channel coupling (Carrara & Papadopoulo) | `x=acm3x2` |
| `fb8` | temporal/spectral | 8-band FB TS (1–4, 4–8, 8–10, 10–13, 13–18, 18–25, 25–30, 30–45 Hz) replacing the 4-band FB | `riemann:xd=1,fb=1,blocks=xdawn+broad+logvar,x=fb8` |
| `fbd` | spectral | adds a 1–4 Hz band TS to the 4-band FB | `x=fbd` |
| `bpt4` | temporal | per band × channel log-power in 4 time bins (ERD/ERS time course) | `x=bpt4` |
| `slow` | temporal | the solver's existing < 4 Hz waveform bins (`slow_block`, never measured on the 3 classes) | `sl=1` |
| `fblv`, `fbrlv` | spatial | per band × channel log-power topography; relative (minus the channel's broadband log-power) variant | `x=fblv` |
| `reg` | spatial | regional covariances: channels grouped into ~6–9 scalp regions by 10-05 position, per FB band → TS | `x=reg` |
| `csp8` | spatial | per FB band, 8 multi-class CSP filters (pyriemann, AJD) → 8×8 covariances → TS: supervised low-rank spatial subspace | `x=csp8` |
| `icoh` | spatial | per FB band imaginary coherence (upper triangle): lagged connectivity, blind to volume conduction | `x=icoh` |
| `ref=laplacian`, `ref=car` | control | re-referencing; predicted ≈ 0 (§ 1) | `riemann:xd=1,fb=1,ref=laplacian` |

At most **2 EDA-motivated candidates** may be added after Phase 1b, under the
same rules (named and justified in the LOG before they run).

## 5. Rules

- Commit on `feat/codabench-track2` only. Push to `personal` after every commit;
  no `org`, tags, merges, PRs or uploads. Add paths explicitly: the working tree
  has another session's uncommitted `codabench/README.md`,
  `MODEL_ARCHITECTURE.md`, `analysis/arch_figures.py` and `figures/`. Never
  stage them.
- Frozen: `codabench/submissions/`, `eegnet_steptype_wu1.py`,
  `2026-competition/tracks/bci_decoding/outputs/`. `riemann_sealed.py` and
  `riemann_steptype.py` stay untouched until Phase 4. Every default path must
  then stay bit-identical (max |dP| = 0).
- Evidence:
  - The split is each subject's last session, fixed.
  - The EDA uses training sessions only.
  - The metric is cell-averaged balanced accuracy.
  - The noise floor is ~2 points on 9 subjects.
  - Deterministic fits, so the uncertainty is a paired bootstrap over subjects
    (10,000 resamples, `sealed_bootstrap.py` logic).
  - Report online-64 (rule-dependent) separately, never as the decision
    metric.
  - Check the data-shape line (subjects, windows, channels, classes, feature
    count) of every run.
- Winner's curse: ~15 candidates on 9 subjects. The best screen Δ of null
  candidates is ~+1.5–2 points by chance. That is why adoption needs a CI and
  replication, and why the release data has the final word.
- Resources: ≤ 2 heavy lanes, `XS_THREADS=10` each, < 7 GB RSS per job.
- HANDOFF is updated after every phase.

### Phase 0: brief, harness hook (estimate 20 min, 17:50–18:10)
Write `analysis/xfeat.py`: the block registry and a `RiemannXModel` that fits the
base `RiemannModel`, then the extra blocks, then refits the LDA on the union.
Hook the `x=` / `sl=` spec keys into `xsess_lib.make_model`. **Done when:** a spec
without `x=` still builds the old `RiemannModel` (same class, same code path), and
the brief is committed.

### Phase 1: build (Workflow, ≤ 5 agents, estimate 75 min)
- 1a (parallel): temporal blocks; spatial blocks; activation EDA script.
- 1b: verification. Smoke-test every block on zhou2016 and a 2-subject
  Scherer slice. Build a feature-count table at 30 ch and 43 ch. Check that the
  baseline spec reproduces the committed Phase 4 numbers **exactly**: Scherer
  3-class persubject/none **0.493**, pooled/none **0.443**, persubject/router
  0.499, pooled/router 0.452 (the 3-dp values in LOG 2026-09-25 Phase 4; compare
  at full precision with `logs/sealed_p4/results_scherer2015.jsonl`).

**Gate:** the baseline is reproduced to 1e-9, or stop and find out why before
screening.

### Phase 2: screen (estimate 2.5 h wall on 2 lanes)
- 2a: Scherer 3-class, every candidate + baseline, modes pooled + persubject ×
  aligns none + `router-psd:riemann`.
- 2b: Tangermann (4-cl.), Scherer 5-class and Zhou, every candidate + baseline,
  modes pooled + persubject × `router-psd:riemann` only.

The **screen score** is the mean of the pooled and persubject cells under the
router alignment.

**Decision (screen pass):**
- Δ screen score vs baseline ≥ **+1.0 point** on Scherer 3-class, **and**
- mean Δ over {Tangermann, Scherer 5-class, Zhou} ≥ **−0.5**.

Up to **3** passing candidates advance, ranked by the Scherer 3-class Δ; on a tie
the one with fewer features goes first. If the best temporal and the best spatial
candidates both pass, their union advances too, as a 4th entry. If none passes:
"no gain" for the family, and go to the chained fallback (§ 3) after Phase 3's
controls are written up.

### Phase 3: confirm under the deployed recipe (estimate 1.5 h)
Run `sealed_personal.py --family <spec> --align router-psd:riemann` on Scherer
3-class, Tangermann, Scherer 5-class and Zhou, for each advancing candidate and
the baseline. Run online-64 on Scherer 3-class for information only. Compute
paired bootstraps of blend_calib (router ids) vs the baseline's blend_calib.

**Decision:**
- **ADOPT** (recommended default for the release-day candidate, still subject to
  the release ablation): Scherer 3-class blend_calib Δ ≥ **+2.0** with a 95 % CI
  > 0, **and** mean Δ over the other three proxies ≥ 0 with none < −2.0.
- **PROMISING** (a release-day ablation, not default): Δ ≥ +1.0 on Scherer
  3-class and ≥ 0 on ≥ 2 of the 3 other proxies; the CI may cross 0.
- **NO GAIN** otherwise. Record it plainly in SEALED_RECIPE § 2; no
  integration.

On a tie between an adopted single block and a union, keep the single block.

**Added 18:10, before any candidate number existed:** a reverse-time replication
on Scherer 3-class (`--split first`: train on the later session, test on the
first; tag `f0929r`). It runs the screen configs for the advancing candidates and
the baseline. If a candidate's reverse-split screen Δ is < 0, an ADOPT verdict is
downgraded to PROMISING.

### Phase 4: integrate (estimate 2.5 h) — only for ADOPT / PROMISING
- Port the block(s) into `riemann_sealed.py` as an opt-in (`xblocks=()`), with
  a chunked path for 500 Hz.
- Gates:
  - defaults bit-identical (max |dP| = 0 on `mock_sealed_s` and zhou2016_xsess;
    zhou flow train = replay = 0.770000);
  - solver features == harness features (≤ 1e-10) on Scherer 3-class;
  - on `mock_sealed_500`, fit time ≤ 1.5× the current 30.4 min and peak RSS
    ≤ 16 GiB (else restrict the block, e.g. fewer bands, and re-gate, or leave
    it harness-only with a note).
- Wire `RECIPE_XBLOCKS` into `train_sealed.sh`, an `xb` step into
  `release_ablations.sh`, and a rule into `release_summarize.py` + RELEASE_DAY
  § 6: *adopt on the release replica (calib:3) only if ≥ +1.0 point and not
  worse on any context*. Verify on `mock_sealed_s` through benchopt.

### Phase 5: final review (Workflow, 2–3 lenses, ≤ 45 min), then wrap-up
Lenses: ML correctness/leakage (fit-on-train only, segment/lag math, TS
reference), bit-identity and deployment path, and doc accuracy.

## 6. Keeping jobs alive and never stale

- Lane scripts `scripts/sprint0929_f*.sh` are built on `sealed_lib.sh` (steps
  with `.done` markers, per-study result files, thread caps). Launch each in
  the foreground of a background Bash call:
  `wsl.exe bash -lc 'bash ~/codabench/scripts/sprint0929_f2.sh'`.
- Before launch, write an ETA row in STATUS.md. Arm the Monitor on
  `monitor_loop.sh ~/codabench/logs/sealed_f0929 300` and re-arm on expiry. The
  harness prints `[fit]`/`[done]` per config, which is the heartbeat.
- SLOW (> 1.5× ETA) / STALLED (10 min silent) / DEAD: read the step log tail,
  `ps` the PID, check `free -g`. Fix the root cause, kill by PID, re-run the lane
  (done configs are skipped by key).
- Nights run ~1.7× slower (E-cores); revise ETAs rather than calling a stall.
  Never edit a `.sh` a running lane is executing.

## 7. Time budget and order

| Target | Phase |
|---|---|
| 17:50–18:10 | 0 brief + hook (the f1 baseline/control lanes start 17:55) |
| 18:10–19:30 | 1 build (workflow); the baseline reproduction runs alongside |
| 19:30–22:00 | 2 screen (2 lanes); EDA write-up in the gaps |
| 22:00–23:30 | 3 confirm |
| 23:30–02:00 | 4 integrate, or the chained fallback |
| 02:00–02:45 | 5 review + fixes |
| 02:45–04:15 | slack (a second round of EDA-motivated candidates, if earned) |
| 04:15–05:45 | wrap-up reserve (90 min): SEALED_RECIPE, LOG, HANDOFF, memory, summary |

**Launch cutoff:** no job starts unless 1.5 × its ETA fits before **04:15**.

## 8. Contingencies

- **Graz + BrainHero released** (check the tracks page once, at ~04:15): stop
  screening at the next step boundary and switch to RELEASE_DAY.md (the
  download is pre-approved). Record the switch.
- **A block is too slow at 43 ch** (> 3× the baseline per config on the
  proxies): drop it from Phase 2b and note it; do not shrink others to make it
  fit.
- **The baseline does not reproduce:** stop and diagnose (library drift,
  cache change). Never screen against a baseline that moved.
- **Anything needing the user** (uploads, forum post, weights): skip it and
  list it under "needs you".
