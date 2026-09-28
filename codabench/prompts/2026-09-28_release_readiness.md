# Sprint 2026-09-28: sealed-release readiness (dress rehearsal on a mock)

Time frame: Mon 2026-09-28 16:55 → **Tue 2026-09-29 04:55** EDT (the user said
"12 hours"). The user is away and not expected to answer. The goal was chosen by
the value check (§ 3, no focus given). The user asked for a *workflow*: the build,
review and runbook-check phases run as multi-agent Workflow calls, and compute runs
as watched, resumable lanes. Work autonomously and decide each next step from the
previous result using the rules below. Never let a job go stale (§ 6).

## 1. Context (verified 2026-09-28 17:00)

- Workstream: Codabench Track 2 (`codabench/`), branch `feat/codabench-track2`,
  commit prefix `codabench:`. Read first: `SEALED_RECIPE.md` (§ 1, § 3, § 5),
  `HANDOFF.md`, the tail of `LOG.md`.
- Environment: WSL Ubuntu, `source ~/codabench/env.sh`, CPU only (20 threads,
  31 GB RAM). `~/codabench` → the repo's `codabench/`. Caches live in
  `~/neuralbench/xsess_cache/{zhou2016,tangermann2012,scherer2015,zyma2019,zhou2016_xsess}`.
- What counts: the **sealed phase** (Oct 28 – Nov 21; 1 submission/day; top 3
  replayed within ± 2σ). Warm-up does not rank. Graz + BrainHero was still "coming
  soon" at 17:00 today. It could land any day, and the sealed phase opens 30 days
  from now.
- Sealed data (tracks page): 20 participants × 6 sessions, 43 EEG + 2 EMG + 2 EOG
  at 500 Hz, 3 cued commands (MI / CALC / WORD). 10 training participants are
  fully labelled. 10 evaluation participants have sessions 1–3 labelled
  (calibration) and sessions 4–6 hidden (test). Metric: balanced accuracy averaged
  over **subject × session × context** cells (Graz vs BrainHero contexts).
- Upstream `2026-competition` has 2 new commits since our clone (docs; default
  study → dreyer2023). There is still no sealed-data loader. Not pulled: the clone
  has local overlay edits.
- **Already measured, do not repeat** (SEALED_RECIPE § 2; LOG 09-25): base family,
  pooling, alignment variants, blend_calib, 3-class blocks, cross-dataset
  pre-training (no gain), train-only alignment (hurts), batch statistics, EEGNet
  variants (6–12 pts below Riemann), more participants (no gain), 8–30 Hz band-pass
  (hurts).

**Claims checked in the code today (the reason for this sprint):**
1. SEALED_RECIPE § 3 says "the harness already scores subject × session ×
   context cells if the cache has a context column". **False.**
   `xsess_cache.py` saves `context.npy`, but `load()` never reads it,
   `load_study()` drops it, and every `L.score(...)` call passes no `ctx`
   (`sealed_run.py:195`, `sealed_personal.py:118,242`). On release day the cell
   metric would silently ignore contexts.
2. SEALED_RECIPE § 1/§ 3 say the blend weight is chosen by
   leave-one-calibration-session-out. **False.** `blend_cv_folds`
   (`sealed_personal.py:88-102`) holds out only each subject's last training
   session, and on the sealed structure that is session 6 for the training
   participants. The solver cannot choose w itself; `train_sealed.sh` bakes in
   the harness's w.
3. The harness split is "last session per subject = test" (`xsess_split`). The
   sealed structure (3 test sessions, only for evaluation participants, while
   fully labelled participants train on all 6) is **not expressible**.
   `train_sealed.sh` validates on the wrong split.
4. The solver holds every training window in float64 (`_prep`). At 500 Hz × 47
   ch × 4 s × ~14 k windows that is ~10 GB before features. Nothing has been
   timed or sized above 30 ch / 120 Hz.
5. Nothing tests 3 consecutive test sessions per subject, which is what
   `adapt="online"` will face. Its buffer crosses session boundaries.
6. The solver reads session ids from `train_loader.dataset.seg_ds.triggers`
   (`riemann_sealed.py:293-313`); `nb_task.py:236` supplies `subject_id`. Both
   are verified for the NeuralBench path.

Open questions the sprint must **not** silently decide: test-time use of
unlabelled windows (`adapt="online"` stays off by default; forum question is the
user's); any download (none needed).

## 2. Data and assets

| Name | Path | What | Approved |
|---|---|---|---|
| proxy caches | `~/neuralbench/xsess_cache/*` | real windows, 4 studies | local |
| mock sealed (new) | `~/neuralbench/xsess_cache/mock_sealed{_s,_120,_500}/` | synthetic, sealed structure | generated, no download |

The mock is for **plumbing, timing and ablation-machinery checks only**. Its
accuracies say nothing about the recipe. Only real-proxy numbers (Phase 3) may
change the recipe.

## 3. Goal and priorities

Value check (no focus given): candidates were (a) release readiness (this),
(b) the online-mode stream/order stress test, (c) Riemann + EEGNet ensemble
(SEALED_RECIPE § 5 #3: +1–2 pts expected, below the ~2-pt noise floor), and
(d) thesis doc-consistency and leakage checks (no deadline; public-repo edits
need a PR the user asks for). (a) ranks first. It has a hard external deadline,
three false claims sit on the release-day path, and a failure on release day
costs sealed submissions. (b) is folded in as Phase 3 because it is cheap and
real-data. (c) and (d) are runners-up.

Deliverables:
1. Harness that expresses the sealed structure: context cells, replica splits,
   and a leave-one-calibration-session-out blend weight. Defaults must reproduce
   the committed numbers bit-for-bit.
2. A mock sealed study generator plus a benchopt dataset for it, so the whole
   release pipeline runs end to end today.
3. A solver that fits and predicts at 43–47 ch × 500 Hz within time and memory,
   with an optional self-chosen blend weight and channel-type selection.
4. `scripts/release_ablations.sh`: every SEALED_RECIPE § 3 ablation, pre-scripted
   as lanes on the replica split, with a summarizer and bootstrap.
5. **`codabench/RELEASE_DAY.md`**: a runbook from download to zip, with timings
   from the dress rehearsal and pre-registered release-day decision rules. It is
   verified by an agent that follows it literally on the mock.
6. Real-proxy results for the multi-session test stream and order robustness of
   online re-centring (Phase 3).
7. Corrected docs: SEALED_RECIPE § 1/§ 3/§ 5, solver docstring, HANDOFF, LOG.

Priority if time runs short: 1 → 3 → 2 → 5 → 4 → 6 → 7 (7's false-claim fixes are
mandatory in wrap-up regardless).

## 4. Rules

- Branch `feat/codabench-track2`. Commit after each phase with prefix
  `codabench:` and push to `personal` straight away. No tags, merges or PRs, and
  no push to `org`. Only the main loop commits; workflow agents never commit.
- Frozen: `codabench/submissions/`, `solvers/bci_decoding/eegnet_steptype_wu1.py`,
  `2026-competition/tracks/bci_decoding/outputs/` (benchopt writes it; we never
  edit it by hand). No uploads. No downloads.
- Log dirs `codabench/logs/sprint0928_*` are gitignored; commit only
  `RESULTS.md`, `STATUS.md` and `HANDOFF.md` from them (`git add -f`).
- Evidence: no tuning on test sessions. Deterministic Riemann fits need one run;
  their uncertainty is across subjects, so use a paired subject-bootstrap
  (`analysis/sealed_bootstrap.py`). Noise floor ≈ 2 points on the proxies. Check
  the `[data]` shape line of every run. Say "no gain" plainly.
- **Backwards compatibility:** every harness/solver change is opt-in. With
  default arguments, outputs must equal the committed results (regression gate,
  Phase 1).
- Resources: ≤ 2 CPU-heavy jobs at once, `XS_THREADS`/OMP/OPENBLAS/MKL capped per
  job (10 each for 2 lanes; build agents' smoke tests ≤ 4 threads), ≤ 7 GB RSS per
  job, **except** the Phase 2 500 Hz sizing run, which runs alone.
- Rule-dependent methods (`adapt="online"`) are reported in separate columns and
  never become a default.

## 5. Phases and decision rules

Each phase: estimate first, then run, time-log from the log stamps, write
RESULTS.md, append a LOG entry, commit + push, apply the rule.

### Phase 0: brief + environment (estimate 20 min)
Brief committed and pushed; HANDOFF points here; the WSL env imports; caches
present. **Done when** pushed.

### Phase 1: build (Workflow "sealed-readiness-build", estimate 3 h)
Parallel builders on **disjoint files**, then integration, then an adversarial
review:
- **A: mock generator + benchopt dataset.** New files `analysis/mock_sealed.py`
  and `datasets/mock_sealed.py`.
- **B: harness.** `analysis/xsess_cache.py` (`load` only), `xsess_lib.py`,
  `sealed_run.py`, `sealed_personal.py`: context plumbing,
  `--split last|calib:K|replica:K`, `--wcv last|loso`, `--test_subjects`,
  `--chans`.
- **C: solver.** `solvers/bci_decoding/riemann_sealed.py`: float32 memory path,
  `blend_w="auto"` (LOSO over calibration sessions when session ids exist; else
  the fixed value), `chans` selection, timing prints.
- **E: stream/order experiment script.** New `analysis/sealed_stream.py` for
  Phase 3 (real proxies, standalone).
- **D: integration** (after A–C): `train_sealed.sh` (SPLIT/WCV env, replica
  validation), `scripts/release_ablations.sh` + `analysis/release_summarize.py`,
  and a small-scale end-to-end run on `mock_sealed_s`.
- **Review:** two skeptics, one on leakage/split correctness and one on
  regression/solver-harness agreement. Findings are fixed before Phase 2.

**Gate (must pass before Phase 2):**
- (i) With defaults, `sealed_personal.py` on `zhou2016` and on `scherer2015
  --classes 0,1,3` reproduces the committed blend_calib rows: same `blend_calib_w`,
  cell within 1e-4. `sealed_run.py` meanlr + riemann rows on zhou2016 reproduce
  within 1e-4. Failure = a bug; fix it first.
- (ii) Mock small: the `[data]` line shows 20 subjects, 6 sessions, 2 contexts,
  47 ch, 3 classes. Replica split: test = 10 eval subjects × sessions 4–6,
  n_cells = 60.
- (iii) The solver on `mock_sealed_s` through benchopt: train score = read-only
  replay score (exact), and within 2 points (pooled BA) of the harness
  blend_calib on the same split.
- (iv) `blend_w="auto"` in the solver equals the harness's `--wcv loso` weight on
  the same data.

### Phase 2: dress rehearsal at realistic size (estimate 2 h, runs alone)
Mock at 120 Hz (`mock_sealed_120`) and at 500 Hz (`mock_sealed_500`), full size
(~14 k windows). `train_sealed.sh` with the replica split, then
`release_ablations.sh`, time-logged, peak RSS recorded (`/usr/bin/time -v`).
- **Sizing rule:** at 500 Hz, fit ≤ 45 min, peak RSS ≤ 20 GB, and predict of the
  whole test set ≤ 10 min at 10 threads (and ≤ 30 min at 2 threads, as a stand-in
  for an unknown scoring CPU) → "ready at 500 Hz". Otherwise fix the bottleneck
  (chunking / float32 / decimation option) and re-run once. If a decimation option
  is needed, the release-day rule is pre-registered here: use the lowest rate
  whose replica cell score is within 1 point of the best rate.
- **Machinery rules** (mock ground truth, injected by construction):
  (a) n_cells = 60 on the replica, and the per-context columns exist;
  (b) the channel ablation flags the injected drifting EMG confound, i.e. the
  "keep a non-EEG channel only if it helps on the replica test sessions *and*
  the paired CI excludes 0" rule rejects EMG when the confound decays;
  (c) the subject×context alignment variant recovers the injected context shift
  (it must beat subject-only alignment on the mock).
  If (a)–(c) fail, the ablation code is wrong: fix it. None of these numbers go
  into the recipe.

### Phase 3: online re-centring on a realistic test stream (real proxies, estimate 1.5 h, can overlap Phase 2's lighter lane)
`analysis/sealed_stream.py` on the real caches:
- **3a multi-session stream:** Zhou (3 sessions): calibrate on session 1, test
  sessions 2 + 3 in recording order. Online-64 vs clean (blend_calib, router).
  Report per test session and the first-32-windows-after-boundary accuracy.
  Also buffer 32 / 64 / 128.
- **3b order robustness:** on Tangermann, Scherer 3-class and Zhou, the test
  stream in (i) recording order, (ii) subjects interleaved but sessions in
  order, (iii) fully shuffled (sessions mixed; Zhou only, since it has 2 test
  sessions in 3a).
- **Decision:** online's gain over clean is "order-robust" if under (ii) it keeps
  ≥ 75 % of its recording-order gain on every proxy. Otherwise the forum question
  must stress order and the runbook adds the guard below. If under 3a the second
  test session's gain is < 0 for ≥ 3 of 4 Zhou subjects, or the post-boundary
  accuracy drops > 5 points below that session's mean, record a
  "session-boundary lag" risk and add a buffer-reset guard option
  (`buffer_reset`) evaluated on the same data. Adopt the guard only if its gain
  is ≥ 2 points with a CI excluding 0 on the stream. Otherwise record "no gain".
- Recipe impact: only the online-mode section of SEALED_RECIPE and the forum
  question may change. The clean recipe is untouched.

### Phase 4: runbook + verification (Workflow "runbook-verify", estimate 1.5 h)
Write `RELEASE_DAY.md` (download → overlay → cache → EDA → replica validation →
ablations → decision rules → bake → train → replay → zip; never upload), with
dress-rehearsal timings. Then a fresh agent follows it **literally** on
`mock_sealed_s` and reports every step that fails or is ambiguous. A second agent
cross-checks every claim in RELEASE_DAY.md and SEALED_RECIPE against the code.
Fix, then re-verify the failed steps only.

### Phase 5 (optional, only if ≥ 2.5 h remain before the reserve): SEALED_RECIPE § 5 #3
The Riemann + pooled-EEGNet ensemble on Scherer 3-class, with the weight chosen by
chronological halves of the calibration session. Adopt only if it gains ≥ 2
points and the paired subject-bootstrap CI excludes 0. Otherwise record "no
gain".

## 6. Keeping jobs alive and never stale

- Compute lanes: phase scripts `scripts/sprint0928_pN.sh` built on
  `scripts/sealed_lib.sh` (`step name eta timeout cmd`, `.done` markers,
  STATUS.md). They are launched from the Bash tool in the foreground of
  `wsl.exe bash -lc '...'` with `run_in_background: true`, never `nohup … &`.
- ETA row in STATUS.md before each launch. Monitor on
  `scripts/monitor_loop.sh <logdir> 300`, re-armed until ALL DONE. Every long
  loop prints a heartbeat.
- SLOW (> 1.5× ETA) / STALLED (10 min without log growth) / DEAD → investigate
  with the long-runs checklist before anything else. Kill by PID only.
- Workflow agents run WSL commands in the foreground with `timeout` (≤ 10 min
  each) and ≤ 4 threads. Anything longer goes to a main-loop lane.
- Machine: sleep is disabled (AC/DC standby 0, checked 17:00). Expect ~1.7×
  slower nights (E-cores).

## 7. Time budget and order

| Phase | Target start | Estimate |
|---|---|---|
| 0 brief | 17:05 | 20 min |
| 1 build workflow | 17:30 | 3 h (→ ~20:30) |
| 2 dress rehearsal | 20:45 | 2 h |
| 3 stream/order (overlaps 2's light lane) | 21:00 | 1.5 h |
| 4 runbook + verify | 23:00 | 1.5 h |
| 5 optional ensemble | ≤ 00:55 | 2.5 h |
| wrap-up reserve | **03:25** | 90 min |

**Launch cutoff:** no job starts unless 1.5 × its ETA fits before 03:25.

## 8. Contingencies

- **Graz + BrainHero released mid-sprint** (tracks page, checked once more at
  ~23:00): downloading is pre-approved (to `Z:\Projects\codabench\neural_compet\`,
  then copied to WSL). Stop Phase 5, finish the current phase, then run the
  runbook on the real data. That is exactly what this sprint prepares for.
- A needed approval comes up (download, weights): skip it, note it under "needs
  you", and take the next item.
- A workflow agent fails or returns garbage: rerun that agent once with a
  sharper prompt, or do the item in the main loop.
- Upstream changes the BCI dataset/objective API: record it; do not pull over
  the local overlay edits without checking the diff.
