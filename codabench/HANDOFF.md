# HANDOFF — sprint 2026-10-01: dual LDA, strict CV references, release-day rehearsal

**The short version for decisions is [DIRECTIONS.md](DIRECTIONS.md)**: status,
decisions pending, model, next steps. This file is the detailed session handoff.

Plan of record:
[prompts/2026-10-01_dual_lda_strict_wcv.md](prompts/2026-10-01_dual_lda_strict_wcv.md)
(17:36 → 05:36, two addenda). Branch `feat/codabench-track2`, pushed to
`personal` after every commit. Logs: `logs/sealed_s1001/` (STATUS.md,
`RESULTS_*.md`), `logs/sealed_s1001_rel120/` (the rehearsal's ablations),
`logs/s1001_*` (train_sealed flows). Record: LOG.md "2026-10-01".

## State of this sprint (updated 2026-10-01 22:45): sprint complete

| Phase | Status | Result |
|---|---|---|
| 0 orient + brief | ✅ db8b35c | per-subject LDAs were 200 s of a 260-s fold at 500 Hz (bpt4+icoh): picked the dual LDA |
| 1 dual (n < p) shrinkage LDA | ✅ f8e9175 | `_DualLsqrLDA` (both solver files): within 4e-12 of the Cholesky solve, 50–60× faster per subject. D1 PASS |
| 1b/6 deployment-test zip | ✅ 71607ea | Dreyer, bpt4 baked: train = replay 0.615873. 4.1 MiB (the first build was a 2.4 GB zip: `covariance_` is now dropped at every size). NOT uploaded |
| 2 gates | ✅ 07f4883 | regression 0.770000, mock flows 0.483333 / 0.512500, xblocks gate, Scherer icoh rows, ablations: all identical |
| 2 G4 500 Hz sizing | ✅ 26acd44 | recipe 15.7 min (was 19.3), bpt4 1.13×, bpt4+icoh **1.23× PASS** (was 1.78×), strict +12 min. D2: icoh passes rule 3b at 500 Hz |
| 3 strict CV references | ✅ 36902b3 | solver `wcv_ref` = harness `--wref` to 4 dp (pairs and subjects); defaults identical. D3: `WREF=strict` is the release-day setting |
| 4 reviews | ✅ bb280ac | dual LDA (3 minors fixed), strict code (6 minors fixed), integration review (1 pre-existing major fixed: step 2 now reads W1 / HARNESS_FROM from `R1`, and train_sealed.sh stops on a baked weight ≠ the seeded harness weight; checked) |
| 5 full-size 120 Hz rehearsal | ✅ 71607ea | § 5 (a) 12 steps 69 min; § 7 step 1 28 min, MATCH, EQUAL, train = replay 0.676944. Found and fixed: summarizer online-row bug |
| 7 shared-covariance personal LDA | ✅ 05ac612 | **NO GAIN** (Scherer 3-class −2.66 / −0.86) |
| 8 harness ctx_min for router-psdctx | ✅ 71607ea | check PASS; regression identical |
| extra checks | ✅ 4403f5a | the deployment joblib loads identically under sklearn 1.6–1.8, numpy 1.26–2.5, pyriemann 0.7–0.11; strict parity with eeg+eog; final zhou regression identical (0.770000, \|dP\| = 0) |

## What is running

Nothing. No uploads were made. The zips under `logs/` are
local only. The deployment-test zip is for the user to upload (SUBMISSIONS.md).

## Facts a resumer needs (this sprint)

- **`LDA_DUAL`** (both solver files) is on by default.
  - It applies above `LDA_FAST_P` = 4,000 features, with n < p and float64
    input, and only after a once-per-process self-check against sklearn's
    `_class_cov`.
  - Every proxy without icoh is unaffected (bit-identical).
  - Checker: `analysis/lda_dual_check.py`.
- **`WREF=strict`** is in RELEASE_DAY's DEC and § 5 commands. The defaults of
  `train_sealed.sh`, `release_ablations.sh`, the solver and the harness stay
  `all`, so every committed number reproduces.
  - Under strict the harness rows carry `wref=strict` in their config key;
    online-alignment rows stay `all`.
  - `train_sealed.sh` refuses `WREF=strict` for non-router alignments and
    switches it to `all` under `ADAPT=online`.
- **`train_sealed.sh` now prints a fold-score line** ("fold scores per w …
  EQUAL (4 dp) | DIFFER"). It is information; the weight gate stays MATCH.
  On zhou2016_xsess, DIFFER is expected: benchopt trains on 1,176 windows,
  the harness on 1,200.
- **`xsess_lib.read_results(path)`** returns the rows of every
  `results_*.jsonl` in that folder. Filter on `study` (now done in
  `compare_results.py` and `pc_screen.py`).
- **The harness's router-psdctx alignment** now gives a pair with < 16
  training windows its subject's reference, as the solver does
  (`info["ctx_small_pairs"]`).
- **New tools:**
  - `analysis/compare_results.py` (row-by-row equivalence of two runs);
  - `analysis/lda_dual_check.py`;
  - `analysis/ctxmin_check.py`;
  - `analysis/pc_screen.py` (Phase 7, no gain);
  - lane scripts `scripts/sprint1001_*.sh`.

## What the user needs to do

1. Ask the organisers on **Discord** (https://discord.gg/yZv8KqKMpH) whether
   `predict()` may use statistics of unlabelled test windows. The listed email
   `neurips2026-eeg-emg-competition@googlegroups.com` bounced (2026-10-02), and
   GitHub issues get no organiser replies. Order and persistence are answered
   by the public code (LOG 2026-10-02).
   A draft (2026-10-02) also asks whether running statistics count as
   "training" (the guide says not to train during `predict`) and whether the
   model stays loaded between calls. It is still the largest lever.
2. ~~Upload the deployment-test zip~~ **done 2026-10-02**: submission 956971,
   Finished, 0.62, 58.68 s (SUBMISSIONS.md).
3. Review rule 4 (`WREF=strict`) and rule 3b (icoh now passes at 500 Hz) in
   RELEASE_DAY § 6.
4. Optional: merge `docs/sprint0928-consistency`, which fixes the stale
   perf-loop numbers on `main`; REVE/LaBraM weights are still parked.
5. Optional (thesis, for a later sprint): say what the epoch's t = 0 is and
   whether that cue already shows the step direction. The full-CNV window
   (0–2 s, the 0.714 headline) beats the late window (1–2 s) by +0.09 AUC.
   If the t = 0 cue differs between straight and diagonal, the 0–1 s gain may
   be a visual-cue effect. The repo does not document it, so the check could
   not be set up this sprint.

The sections below are the previous sprints' handoffs; their facts still hold
except where this section supersedes them (icoh's 500 Hz verdict, rule 4's
guard, the release-day budget).

---

# HANDOFF — sprint 2026-09-29/30: temporal and spatial feature blocks

Resume from this file alone. Plan of record:
[prompts/2026-09-29_temporal_spatial_features.md](prompts/2026-09-29_temporal_spatial_features.md)
(17:46 → 05:45, with an erratum at the end). Branch `feat/codabench-track2`,
pushed to `personal` after every commit. Logs: `~/codabench/logs/sealed_f0929/`
(STATUS.md, `RESULTS_*.md`, per-study `results_<study>.jsonl`). The lanes
`scripts/sprint0929_f*.sh` are resumable.

## State of this sprint (updated 2026-09-30 02:10): sprint complete

**Outcome:**
- **The band-power time course `bpt4` is adopted** into the release-day recipe
  (`RECIPE_SPEC=riemann:xd=1,fb=1,x=bpt4`).
  - Blend_calib on the sealed-like proxy (Scherer WORD / SUB / HAND): +4.35
    (95 % CI +2.05, +6.67).
  - Positive point estimates on all 4 proxies, with the CI above 0 on 3.
  - It passes the 500 Hz sizing gate (1.06×).
  - Caveat: the confirmation reuses the screen's test sessions.
- **Imaginary coherence `icoh` (lagged connectivity) is PROMISING.** It becomes
  the release-day ablation `xa`. Post-hoc, bpt4+icoh gave +6.51. But it
  **fails the 500 Hz sizing gate**: 1.78× the recipe's fit measured alone
  (34 vs 19 min), so rule 3b keeps it out on a 500 Hz cache for now.
- **No gain under the deployed recipe:**
  - time-segment covariances (`tseg3`);
  - topography-type spatial blocks: CSP, regional covariances, band-power maps,
    re-referencing.
- Tables: SEALED_RECIPE § 2 "Phase 7"; LOG 2026-09-29/30.

| Phase | Status | Result |
|---|---|---|
| 0 brief + harness hook | ✅ 8c50541, 7e0d008 | `analysis/xfeat.py` (block registry, `RiemannXModel`), spec keys `x=` / `sl=`, `--split first` |
| 1 build (workflow, 5 agents) | ✅ 9b3719b | 12 verified blocks + 2 EDA-motivated (`tcut1000`, `fbfrom1000`); activation EDA in `reports/features_0929/`; baseline bit-exact; slow block / CAR / Laplacian: no gain |
| 2 screen | ✅ fe0efe4 | tseg3 +3.33, bpt4 +2.59, icoh +2.59 on Scherer 3-cl.; `RESULTS_screen.md` |
| 3 confirm | ✅ 4562ac4 | bpt4 ADOPT, icoh PROMISING, tseg3 NO GAIN (+1.96), tseg3+icoh ADOPT by the letter but not deployed; reverse split: no downgrade; `RESULTS_confirm.md`, `RESULTS_reverse.md` |
| 4 integrate + verify | ✅ e1b7402 | solver opt-in `xblocks` (tseg<K>, bpt<K>, icoh): defaults \|dP\| = 0, harness parity exact. `train_sealed.sh` bakes `RECIPE_SPEC … x=`. `release_ablations.sh` steps `xb` / `xa` (env `XB`), summarizer rule 3b. zhou regression 0.770000 = replay. Mock bpt4 flow: train = replay = harness 0.5125. Mock default flow 0.483333 = committed. 500 Hz sizing: bpt4 1.06× PASS; bpt4+icoh **FAIL** (1.56× contended, **1.78× alone**: 2,062 vs 1,157 s) |
| 4 extras (post-hoc, information) | ✅ e6af308 | bpt4+icoh: +6.51 / +1.81 / +2.16 / +3.33; online-64 Scherer 3-cl.: recipe 0.561, bpt4 0.592, bpt4+icoh 0.599 |
| 5 review (workflow, 6 agents) | ✅ eb39e70, 80d4da3 | 36 confirmed findings fixed. Two majors: the release-day summarizer lost `x=bpt4` (fixed and hardened: reads `config.txt`), and rule 3b's reading of the brief (documented; the icoh rule is now literal). LOG "Corrections" block, brief erratum |

## What is running

Nothing. The last job, the 500 Hz sizing re-measured alone, ended at 02:03
(`logs/sealed_f0929/RESULTS_sizing.md`). No uploads were made. The zips under
`logs/` (mock, regression) are local only; never upload them.

## Facts a resumer needs (this sprint)

- **The recipe is `x=bpt4` only through the release-day commands.**
  - Every script default and the solver default stay `xblocks=""`, so the
    regression gate still reproduces 0.770000.
  - RELEASE_DAY § 5 passes `SPEC=…,x=bpt4 XB=icoh`.
  - § 7 takes `RECIPE_SPEC` from the DECISIONS; the `DEC` default carries
    `x=bpt4`.
- **benchopt:** blocks are joined with `_` (`xblocks="bpt4_icoh"`), because `+`
  may be parsed as arithmetic. `train_sealed.sh` converts `x=bpt4+icoh`.
- **`release_summarize.py`** now reads `spec=` / `xb=` from the run's
  `config.txt` and exits on a mismatching `--spec` / `--xb`.
- **Harness `x=` blocks** run per 512 MiB chunk. Stateless blocks skip the
  full-array fit, and chunked = one-shot exactly.

## What the user needs to do (the previous sprint's items, plus 3 and 4)

1. Post the organiser question (SEALED_RECIPE § 5 step 1).
2. A warm-up deployment-test upload before 2026-10-25 (SUBMISSIONS.md). The
   candidate would now carry bpt4 (`xblocks="bpt4"`); the joblib gains only
   plain numpy arrays for it.
3. Review rule 3b (RELEASE_DAY § 6), in particular the ADOPT reading: bpt4
   stays unless the replica is ≥ +1.0 without it.
4. Decide the icoh sizing policy for a 500 Hz release: it fails the brief's
   relative gate (1.78×) but meets the older absolute rule (fit 34 ≤ 45 min).
   Either accept the absolute rule, or keep icoh out at 500 Hz. The other way
   to lift the block is a restricted icoh (2 bands) or a dual LDA solve, which
   is sprint work.
5. Optional: review / merge `docs/sprint0928-consistency`; approve REVE /
   LaBraM weights (still parked).

The sections below are the previous sprints' handoffs (2026-09-28/29,
sealed-release readiness); every fact in them still holds.

## State of the previous sprint (updated 2026-09-29 02:30): sprint complete

| Phase | Status | Result (details: LOG.md 2026-09-28/29 entries) |
|---|---|---|
| 0 brief | ✅ a44526a | three false claims found in the code before planning |
| 1 build (workflow, 9 agents) | ✅ bd931df | sealed-structure harness, mock sealed study, fast LDA, release scripts; defaults bit-identical |
| 3 online stream/order (real proxies) | ✅ bd931df | online gain not order-robust (+3.0/+4.4 interleaved vs +3.5/+7.5 recording order); no session lag; reset guard no gain |
| 2 dress rehearsal (full-size mock, 120 + 500 Hz) | ✅ ee5377d | every gate exact; 500 Hz sizing passes (fit 30 min, 12 GB); release-split ablations catch the EMG trap |
| 4a solver `align="subject_context"` | ✅ ee5377d, e64c58a | deploys the harness's router-psdctx; adversarially reviewed |
| 4b claims check + `release_eda.py` | ✅ ee5377d | 30 doc mismatches found |
| 4c runbook fix + literal verification | ✅ ceb3827, 5d39463 | RELEASE_DAY.md followed step by step on the mock; § 9 is the log |
| 5 final 3-lens review + fixes | ✅ (last commit) | channel-name guard in the solver; silent-failure paths closed; LOSO bias documented |

## What is running (previous sprint)

Nothing. No uploads were made. The zips under `logs/` are mock or regression
candidates only; never upload them.

## Where things are (previous sprint)

- **[RELEASE_DAY.md](RELEASE_DAY.md)**: the release-day runbook, from download to
  a checked zip. It holds the pre-registered rules, the dress-rehearsal timings
  and the verification log.
- **[SEALED_RECIPE.md](SEALED_RECIPE.md)**: the recipe and the evidence. § 5 lists
  the ranked next steps, including the two added by the final review.
- **Mock study:** `analysis/mock_sealed.py` (caches `mock_sealed_s`, `_120`,
  `_500`) and `datasets/mock_sealed.py` (benchopt, path form only;
  `split=organisers|replica_full`).
- **Harness** (`analysis/xsess_lib.py`, `sealed_run.py`, `sealed_personal.py`):
  splits `last` | `calib:K` | `replica:K` (mock only), context cells,
  `--wcv loso`, `--chans`, `--pool`, `router-psdctx`, `--router_cap`,
  `--wvariant`, `--mmap`.
- **Cache builder** (`xsess_cache.py`): `--picks`, `--sealed`,
  `--eval_subjects`, `--hidden_labelled`.
- **Release tools:**
  - `scripts/train_sealed.sh`. Its env is documented in the header: SPLIT,
    TEST_SUBJECTS, BLEND_W, WCV, CHANS, RECIPE_*, DATASET, GATE, FORCE_ZIP,
    HARNESS_FROM, RUN_NAME, STOP_AFTER, SUBMISSION_DIR.
  - `scripts/release_ablations.sh`.
  - `analysis/release_summarize.py` (the pre-registered rules).
  - `analysis/release_eda.py`.
  - `scripts/monitor_loop.sh`.
- **Solver** (`solvers/bci_decoding/riemann_sealed.py`), new opt-ins:
  - `blend_w="auto"`;
  - `chans`;
  - `align="subject_context"`;
  - channel-name check at load;
  - fast shrinkage LDA above 4,000 features;
  - float32 memory path.
- **Online-order study:** `analysis/sealed_stream.py`; results in
  `logs/sprint0928_p3/`.
- **Thesis docs fixes:** branch `docs/sprint0928-consistency` (853f9c5).

## Facts a resumer needs (previous sprint)

- **The release-day replica** is `SPLIT=calib:3 TEST_SUBJECTS=<the 10 fully
  labelled participants>`. `replica:3` is refused on non-mock caches.
- **Decide on that split only.** On `replica:3` the drifting EMG trap was
  invisible (+0.8); on `calib:3` it costs −3.7 to −4.2 points.
- **Expect 43 EEG channels at 120 Hz.** NeuralBench's default extractor keeps
  EEG channels only and resamples to 120 Hz. EMG/EOG reach `predict()` only if
  the organisers override `neuro.picks`.
- **The fast LDA (Cholesky)** is used only above 4,000 features, so every
  committed proxy number is reproduced bit for bit.
- **LOSO weight search: known bias.** Each held-out session's whitening
  reference includes its own windows: ~2 points of CV bias, never changing a
  weight so far, but it becomes oracle alignment when a (subject, context) pair
  lives in one calibration session. See RELEASE_DAY rule 4 for the interim
  guard. The final all-data run bakes the replica's weight; it does not use
  `auto`.
- **benchopt caches on parameters, not data.** Use `--no-cache` after a cache
  rebuild; train_sealed.sh passes it.
- **Line endings in `2026-competition`.** It is a Windows checkout: 81 of 93
  files are CRLF in the working tree, so plain WSL `git` can list CRLF-only
  diffs. Use `git -c core.autocrlf=true`.
- **WSL calls from the Bash tool:** start the heredoc with `exec 2>&1`.
  Otherwise stderr can overwrite the start of stdout in the captured output.

## What the user needs to do (previous sprint)

1. **Post the organiser question** (SEALED_RECIPE § 5 step 1). It now also asks
   about test-window **order and batching**: the online gain drops from
   +3.5 to +7.5 in recording order to +3.0 to +4.4 when windows arrive
   interleaved.
2. **A deployment-test upload during warm-up** (before 2026-10-25). No
   sklearn/pyriemann joblib has ever run on the scoring image. The command is in
   `SUBMISSIONS.md`.
3. **Review the new pre-registered rule 6 precedence** in RELEASE_DAY § 6
   (router-psdctx is kept over online unless the organisers confirm recording
   order). It was written tonight, before any release data.
4. Optional: review/merge the thesis docs branch `docs/sprint0928-consistency`
   (pushed to `personal`, no PR).
5. Optional: approve the REVE/LaBraM weights (still parked).
