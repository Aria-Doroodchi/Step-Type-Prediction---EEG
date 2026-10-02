# Sprint 2026-10-01: dual shrinkage LDA, deployment-test zip, strict LOSO references

Time frame: Thu 2026-10-01 17:36 → **Fri 2026-10-02 05:36** EDT (the user said
"run a 12h sprint looking at a goal Claude sees fit"; no focus given). The user
is not expected to answer. Chosen by the value check (§ 3). Work autonomously;
decide each next step from the previous result with the rules below; never let
a job go stale (§ 6). Read the whole brief before starting.

## 1. Context (verified 2026-10-01 17:40–17:55)

- Workstream: Codabench Track 2 (`codabench/`), branch `feat/codabench-track2`,
  commit prefix `codabench:`. Read first: `DIRECTIONS.md`, `HANDOFF.md` top,
  `SEALED_RECIPE.md` § 5, `RELEASE_DAY.md` § 6 rules 3b and 4, § 8.
- Environment: WSL Ubuntu, `source ~/codabench/env.sh`, CPU only (20 threads,
  31 GB RAM). sklearn 1.9.1, numpy 2.5.2, scipy 1.18.1, pyriemann 0.12.
- What counts: the **sealed phase** (Oct 28 – Nov 21). The Graz + BrainHero
  data is still "coming soon" (tracks page read 17:40 today).
- Power: sleep is disabled on AC and DC (read 17:50; not changed). Desktop, no
  battery.
- **Already measured, do not repeat:** everything in DIRECTIONS § 4 and
  SEALED_RECIPE § 2. In particular the feature-block screen/confirm of
  2026-09-29 (bpt4 adopted, icoh PROMISING, tseg3 no gain) and the 500 Hz
  sizing re-measured alone (recipe 1,157 s, bpt4+icoh 2,062 s = 1.78×,
  `logs/sealed_f0929/RESULTS_sizing.md`).
- **Verified in the code before planning (17:45):**
  - Every shrinkage LDA is `fit_shrinkage_lda` (identical block in
    `solvers/bci_decoding/riemann_steptype.py` and `riemann_sealed.py`):
    sklearn's own fit up to `LDA_FAST_P = 4000` features, above it sklearn's
    fit with the lstsq step replaced by a Cholesky solve of the full p × p
    Ledoit-Wolf covariance (`_CholeskyLsqrLDA`).
  - The `auto` weight search (`RiemannSealedModel._choose_blend_w`) refits the
    pooled LDA and one LDA per subject in each of its 6 folds.
  - **Benchmark (17:40–17:45, random data, 10 threads, idle machine, the
    current solver):** pooled LDA n = 8,400: 9.7 s at p = 5,073 and 31.3 s at
    p = 9,373; **20 per-subject LDAs (n = 420): 39.5 s and 200.0 s**. The
    sizing log's folds took 130 s (recipe) and 260 s (bpt4+icoh). So the
    per-subject p × p solves are most of icoh's extra cost, and n ≪ p there.
- Open questions this sprint must not decide: online re-centring (organisers),
  the icoh absolute-vs-relative sizing policy (the user's decision 4, unless the
  gate passes outright under its own rule), uploads.

## 2. Data and assets

All local, nothing to download: proxy caches `~/neuralbench/xsess_cache/
{scherer2015,tangermann2012,zhou2016,zyma2019}`, mock caches `mock_sealed_s`
(gates) and `mock_sealed_500` (sizing), Dreyer 2023 in
`~/neuralbench/benchopt_data` (deployment test).

## 3. Goal and priorities

The value check (17:40–17:55, no user focus) ranked DIRECTIONS § 6 against the
evidence. Picked, in order:

1. **Dual (n < p) shrinkage LDA**: the same Ledoit-Wolf LDA solved through the
   n × n Woodbury form whenever the fit has fewer rows than features (and
   p > LDA_FAST_P). Expected: no accuracy change (equivalence ≤ 1e-8 in
   probabilities); per-subject LDA time down ≥ 5×; the 500 Hz fit down by
   roughly 15–25 % (recipe) and 50 %+ (bpt4+icoh). It may let icoh pass the
   1.5× sizing gate, and it shortens every release-day flow.
2. **Deployment-test candidate zip** (background compute during 1): Riemann-
   Sealed with the recipe's blocks (bpt4) trained on Dreyer, zipped, replayed
   locally. Turns the user's decision 2 into a single upload. No upload.
3. **Strict per-fold whitening references in the LOSO weight search**
   (harness `choose_w --wcv loso` and solver `_choose_blend_w`, kept in parity):
   removes the known ~2-point CV bias and the oracle-aligned fold when a
   (subject, context) pair lives in one calibration session (RELEASE_DAY rule 4
   "known bias").

Runners-up, not this sprint: restricted icoh (dominated by 1 if 1 works);
Riemann + EEGNet ensemble (+1–2, at the noise floor, half a day); the thesis
full-window leakage check (`MODELS.md:1018`, other workstream, no deadline).

Deliverables: the code + its gates; `logs/sprint1001/RESULTS_*.md`; LOG entries;
RELEASE_DAY § 6 rules 3b/4 and § 8 updated to the measured numbers;
SEALED_RECIPE § 5; HANDOFF; DIRECTIONS.

## 4. Rules

- Commit on `feat/codabench-track2` only, push to `personal` after every commit.
  No tags, merges, PRs, uploads. Commit only `RESULTS*.md` / `STATUS.md` from
  log dirs (`git add -f`).
- Frozen: `codabench/submissions/`, `eegnet_steptype_wu1.py`, benchopt
  `outputs/`. Leave the untracked `MODEL_ARCHITECTURE.md`, `analysis/
  arch_figures.py`, `figures/` and the README.md edit of another session alone
  (not ours to commit).
- **Defaults stay bit-identical where they are bit-identical today**: every
  fit with p ≤ LDA_FAST_P (all proxies without icoh) must not change at all.
  The dual path is a numerical re-route of an existing estimator, so it is
  judged by equivalence, never by a score.
- Never tune on test. Phase 3 is a correctness change: it is decided by its
  parity and bit-identity gates; test-session scores are reported for
  information only.
- Resources: ≤ 2 heavy jobs; thread caps per job (`step_lib.sh` /
  `sealed_lib.sh`); sizing runs **alone** (nothing else heavy running).
- Never edit a `.sh` a running job uses; never edit a solver `.py` while a
  benchopt job that imports it is starting (Python reads it at import; wait
  for the first log line).
- Update HANDOFF after every phase.

## 5. Phases and decision rules

### Phase 0: brief (17:55–18:15)
This file, committed and pushed; HANDOFF points at it. **Done when:** pushed.

### Phase 1: dual LDA (est. 2 h, 18:15–20:15)
- In the shared LDA block (both solver files, identical text): a subclass
  `_DualLsqrLDA._solve_lstsq` that keeps sklearn's `means_`, priors and
  2-class handling, and computes the coefficients from the low-rank form of
  sklearn's own estimate. Per class g (sklearn `_cov`, shrinkage "auto"):
  `StandardScaler` (its own `scale_`, zero-variance → 1), Z0 = centred scaled
  rows, Ledoit-Wolf δ_g from the n_g × n_g Gram (same formula as
  `ledoit_wolf_shrinkage`), μ_g = ‖Z0‖²/(n_g p). Then
  Σ = diag(λ) + Uᵀ diag(c) U with U = stacked (Z0_g · scale_g),
  c_g = π_g (1 − δ_g)/n_g, λ = Σ_g π_g δ_g μ_g scale_g², solved as
  Λ^{-1/2}(I − Vᵀ(I_n + VVᵀ)^{-1}V)Λ^{-1/2}B with V = C^{1/2}UΛ^{-1/2}.
- Used only when p > LDA_FAST_P **and** n < p; anything non-finite, λ ≤ 0, or a
  relative residual ‖Σ coef − B‖/‖B‖ > LDA_MAX_RESID (computed in low-rank form)
  falls back to the existing Cholesky path. `covariance_` is not formed
  (the solver already drops it above LDA_FAST_P).
- Checker `analysis/lda_dual_check.py`: random matrices (3- and 2-class, a
  constant feature, a 2-row class, p ∈ {4,500, 9,373}, n ∈ {60, 420, 3,000}),
  plus real feature matrices (mock `mock_sealed_s` 43 ch recipe and
  bpt4+icoh). Compares dual vs `fast=False` (sklearn) and vs Cholesky.
- **Decision D1:** enable the dual path by default iff, on every matrix,
  max |ΔP| ≤ 1e-8 vs both references, argmax identical on 100 % of rows,
  δ_g relative difference ≤ 1e-10 vs `ledoit_wolf_shrinkage`, and the
  20 × (n = 420, p = 9,373) benchmark is ≥ 5× faster than today's 200 s.
  Otherwise keep it opt-in (`LDA_DUAL = False`), record why, skip Phase 2's
  sizing and go to Phase 3.

### Phase 1b: deployment-test candidate (background, est. 45 min compute)
- Train Riemann-Sealed on `BCI[study=dreyer2023]` (SUBMISSIONS.md command) with
  the recipe's blocks (`xblocks=bpt4`), into
  `logs/wu_riemann_deploytest/submission`; zip at the zip root as
  `train_sealed.sh` does; replay inference-only from the zip and compare with
  the train score; list the zip and the joblib's object types.
- **Done when:** replay = train score, zip checked, SUBMISSIONS.md has a
  "ready, not uploaded" row and the upload checklist. Score irrelevant.

### Phase 2: gates and sizing (est. 2 h, 20:15–22:15)
- G1 `analysis/xblocks_gate.py` (defaults bit-identity + harness parity):
  ALL GATES PASS as on 2026-09-30.
- G2 zhou regression (`train_sealed.sh` default flow): 0.770000.
- G3 mock flows: default `mock_sealed_s` 0.483333 (5,073 features: the dual
  path is now used for the per-subject LDAs) and bpt4 0.5125, equal to 6 dp.
- G3b a real-data icoh config that already used the Cholesky path (Scherer
  3-class, bpt4+icoh, 4,395+ features at 30 ch): same blend_calib score as the
  committed row.
- G4 500 Hz sizing **alone**, sequential (`sprint0929_f4.sh`-style lane,
  `PREFIX=sz3`): none, bpt4, bpt4_icoh. Est. 60–75 min.
- **Decision D2** (on G4, same series, alone):
  - fit(bpt4+icoh) ≤ 1.5 × fit(recipe) → icoh **passes** rule 3b's 500 Hz
    gate: update RELEASE_DAY rule 3b and § 8, mark DIRECTIONS decision 4
    resolved.
  - > 1.5× → icoh stays out at 500 Hz; record the new absolute numbers;
    decision 4 stays with the user, with them.
  - Either way, § 8's release-day budget takes the new fit times.
  - Any of G1–G3b failing → the dual path goes back to opt-in, D2 is not
    applied, and the failure is the finding.

### Phase 3: strict LOSO references (est. 3 h, 22:15–01:15)
- Opt-in first: harness `sealed_personal.py --wref strict` and solver parameter
  `wcv_ref="strict"` (default `"all"` = today's behaviour). Under strict, each
  fold's whitening references (per subject, per (subject, context) pair, per
  (subject, session) under online) come from that fold's **fit rows only**; a
  group with no fit rows takes the reference the deployment would give it
  (pair → its subject's W from fit rows → W_global from fit rows).
- Gates: defaults bit-identical (G1-style, default `"all"`); solver vs harness
  under strict on `mock_sealed_s` (and its context variant if the mock has
  contexts): same chosen w and fold cell scores to 1e-6.
- Measurements (information): chosen w and LOSO CV under all vs strict on the
  proxies with ≥ 2 training sessions and the mock; the test-session score of
  each chosen w.
- **Decision D3:** if both gates pass, `strict` becomes the release-day
  setting (RELEASE_DAY rule 4: `WREF=strict` in the commands, the "known bias"
  paragraph replaced by the result). The solver/harness defaults stay `"all"`
  so every committed number reproduces. If a gate fails and cannot be fixed
  before 01:15, leave it opt-in and document.

### Phase 4: review and docs (est. 1.5 h, 01:15–02:45)
One independent review agent per change (dual LDA, strict references) reading
the diff against this brief; fix confirmed findings; rerun the affected gate.
Then LOG, RESULTS, RELEASE_DAY, SEALED_RECIPE § 5.

## 6. Keeping jobs alive and never stale
- Launch each lane in the foreground of a Bash tool call with
  `run_in_background: true` (WSL stays attached), with `sealed_lib.sh` steps
  (`.done` markers, thread caps, STATUS rows with ETA).
- Watch with `monitor_loop.sh <logdir> 300` in a Monitor
  (`stdbuf -oL tr -d '\r'`). The solver prints a heartbeat per blend_w fold.
- Past 1.5× an ETA: read the live log, check for a stall, fix the cause.
- Machine: E-cores at night (~1.7× slower); sizing only when nothing else
  heavy runs.

## 7. Time budget and order
Phase 0 17:55 → Phase 1 18:15 (Phase 1b compute in the background from
~18:20) → Phase 2 ~20:15 → Phase 3 ~22:15 → Phase 4 ~01:15 → slack until
04:06. **Launch cutoff:** no job starts unless 1.5 × its ETA fits before
**04:06** (= deadline − 90 min reserve). **Wrap-up reserve:** 04:06–05:36.

## 8. Contingencies
- **Data released** (check the tracks page once more around 01:00): stop at the
  next clean point, commit, and switch to RELEASE_DAY.md § 1 (download is
  pre-approved) with whatever this sprint has verified.
- Dual LDA not equivalent: keep it opt-in, skip D2, go to Phase 3.
- Dreyer training > 1.5× ETA: diagnose; the deployment test can use the
  default solver without bpt4 if bpt4 is the cause (record it).
- Anything needing approval: skip, list under "needs you".

## Addendum (2026-10-01 19:05, before any of its numbers)

Phases 1–3 finished early: D1 PASS, the gates PASS, D3 strict adopted. Two
additions, run after the 500 Hz sizing (which runs alone):

### Phase 5: full-size 120 Hz rehearsal of the new release-day commands (est. 1.5–2 h)
- On `mock_sealed_120` (full size, 47-ch cache): RELEASE_DAY § 5 (a) without
  STEPS (all 12 steps, `SPEC=…,x=bpt4 XB=icoh WREF=strict`, 2 lanes × 6
  threads), then `release_summarize.py`. Then § 7 step 1 (replica, `auto`) with
  the DEC the summarizer prints.
- **Done when:** every step rc = 0; the DECISIONS block's `train_sealed.sh:`
  line carries `WREF=strict`; step 1 has train = replay, MATCH and fold scores
  EQUAL (4 dp).
- **Use:** the wall times replace § 8's release-day budget rows (2026-09-28,
  pre-dual). No recipe decision: it is the mock. A failure is a release-day
  bug, to fix first.

### Phase 6: the deployment-test zip from the final solver (est. 15 min)
`sprint1001_deploytest.sh` once more after the last solver change.
SUBMISSIONS.md gets the ready (not uploaded) row and checklist.

### Phase 7 (added 2026-10-01 20:05, before any of its numbers): personal LDA with a shared covariance (screen)
- **Idea.** The per-subject ("calib") LDA has its own class means but a
  Ledoit-Wolf covariance from only ~120–700 windows against 3–9 k features.
  Its shrinkage is near 1, so it is close to a nearest-mean classifier. A
  personal LDA with the subject's class means and a covariance mixed with
  the pooled (all subjects', re-centred) within-class covariance,
  Σ_k = γ Σ_pooled + (1 − γ) Σ_k^LW, is regularised discriminant analysis with
  subject-to-subject transfer. It was not tested here before (grep: no
  "shared/pooled covariance" in LOG / SEALED_RECIPE).
- **Harness only, opt-in** (`sealed_personal.py`): variants `calibpc1`
  (γ = 1: pooled covariance, personal means) and `calibpc5` (γ = 0.5), each
  with its own `blend_*` weight chosen by the existing training CV
  (`--wcv last`, release settings, `WREF=all` as the committed rows). The
  defaults are unchanged; the committed rows are the reference.
- **Runs:** recipe x=bpt4, router-psd, on Scherer 3-class, Tangermann, Zhou
  and Scherer 5-class (the four confirmation proxies).
- **Rule** (blend variant vs `blend_calib`, router ids, cell metric; paired
  subject bootstrap as `sealed_bootstrap.py`; two variants tried, so
  stricter than a single screen):
  - **PROMISING** (a release-day ablation candidate, after a solver port in a
    later sprint): Scherer 3-class Δ ≥ +1.5 with the 95 % CI excluding 0, and
    the mean Δ over the other three ≥ 0.
  - **NO GAIN** otherwise; say so plainly. Never adopted into the recipe from
    this screen alone.
- **Launch only if** Phase 5 has finished, and only if its ETA × 1.5 fits
  before 04:06.
