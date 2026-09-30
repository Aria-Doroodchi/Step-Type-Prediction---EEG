# HANDOFF — sprint 2026-09-29/30: temporal and spatial feature blocks

Resume from this file alone. Plan of record:
[prompts/2026-09-29_temporal_spatial_features.md](prompts/2026-09-29_temporal_spatial_features.md)
(17:46 → 05:45). Branch `feat/codabench-track2`, pushed to `personal` after
every commit. Logs: `~/codabench/logs/sealed_f0929/` (STATUS.md, per-study
`results_<study>.jsonl`; lanes `scripts/sprint0929_f*.sh` are resumable).

## State of this sprint (updated 2026-09-30 00:15)

**Outcome so far:** the band-power time course **`bpt4` is adopted** into the
recipe for release day. It gains +4.4 points blend_calib on the sealed-like
proxy and is positive on all 4 proxies. Imaginary coherence **`icoh`** is
PROMISING and becomes the release-day ablation `xa`. Time-segment covariances
(`tseg3`) showed no gain under the deployed recipe. Topography-type spatial
blocks showed no gain. Tables: SEALED_RECIPE § 2 "Phase 7"; LOG 2026-09-29.

| Phase | Status | Result |
|---|---|---|
| 0 brief + harness hook | ✅ 8c50541, 7e0d008 | `analysis/xfeat.py` (block registry, `RiemannXModel`), spec keys `x=` / `sl=`, `--split first` |
| 1 build (workflow, 5 agents) | ✅ 9b3719b | 12 verified blocks + 2 EDA-motivated (`tcut1000`, `fbfrom1000`); activation EDA in `reports/features_0929/`; baseline bit-exact; slow block / CAR / Laplacian: no gain |
| 2 screen | ✅ fe0efe4 | tseg3 +3.3, bpt4 +2.6, icoh +2.6 on Scherer 3-cl.; `logs/sealed_f0929/RESULTS_screen.md` |
| 3 confirm | ✅ 4562ac4 | bpt4 ADOPT, icoh PROMISING, tseg3 NO GAIN, tseg3+icoh ADOPT but dominated by bpt4; reverse split: no downgrade; `RESULTS_confirm.md`, `RESULTS_reverse.md` |
| 4 integrate | ✅ | solver opt-in `xblocks` (tseg<K>, bpt<K>, icoh), defaults \|dP\| = 0, harness parity exact; `train_sealed.sh` bakes `RECIPE_SPEC=…,x=bpt4`; `release_ablations.sh` steps `xb` / `xa` (env `XB`) + summarizer rule 3b; 500 Hz sizing: bpt4 **1.06×** PASS, bpt4+icoh 1.56× (borderline, measured under contention); zhou regression 0.770000 = replay; mock bpt4 flow train = replay = harness 0.5125 |
| 4 extras (post-hoc, info) | running | bpt4+icoh on the proxies; online-64 for the baseline / bpt4 / bpt4+icoh (`CANDS="riemann:xd=1,fb=1,x=bpt4 riemann:xd=1,fb=1,x=bpt4+icoh" sprint0929_f3.sh`) |
| 5 review | next | workflow, 3 lenses + skeptics |

The sections below are the previous sprint's handoff (2026-09-28/29, sealed-release
readiness); every fact in them still holds.

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

## What is running

Nothing. No uploads were made. The zips under `logs/` are mock or regression
candidates only; never upload them.

## Where things are

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

## Facts a resumer needs

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

## What the user needs to do

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
