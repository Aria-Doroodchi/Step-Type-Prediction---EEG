# HANDOFF — sprint 2026-09-28: sealed-release readiness

Resume from this file alone. Plan of record:
[prompts/2026-09-28_release_readiness.md](prompts/2026-09-28_release_readiness.md).
Branch `feat/codabench-track2`, pushed to `personal` after every commit. The
weekend handoff (2026-09-25) it replaces is in git history (a44526a).

## State (updated 2026-09-29 00:00; final update at the sprint's end)

| Phase | Status | Result (details: LOG.md 2026-09-28 entries) |
|---|---|---|
| 0 brief | ✅ a44526a | three false claims found in the code before planning |
| 1 build (workflow, 9 agents) | ✅ bd931df | sealed-structure harness, mock sealed study, fast LDA, release scripts; defaults bit-identical |
| 3 online stream/order (real proxies) | ✅ bd931df | online gain not order-robust (+3.0/+4.4 interleaved vs +3.5/+7.5 rec. order); no session lag; reset guard no gain |
| 2 dress rehearsal (full-size mock, 120 + 500 Hz) | ✅ ee5377d | all gates exact; 500 Hz sizing passes (fit 30 min, 12 GB); release-split ablations catch the EMG trap |
| 4a solver `align="subject_context"` | ✅ ee5377d (review R3 pending) | deploys the harness's router-psdctx; baked by train_sealed.sh |
| 4b claims check + `release_eda.py` | ✅ ee5377d | 30 doc mismatches found → fixed in 4c |
| 4c runbook fix + literal verification | running | RELEASE_DAY.md rewrite, script gate enforcement, V1 literal run |

## What is running

TBD at the end of the sprint (normally nothing).

## Where things are

- **[RELEASE_DAY.md](RELEASE_DAY.md)**: the release-day runbook, from download to
  zip, with pre-registered rules and dress-rehearsal timings.
- **[SEALED_RECIPE.md](SEALED_RECIPE.md)**: the recipe and the evidence; § 5
  lists the ranked next steps.
- **Mock study:** `analysis/mock_sealed.py` (caches `mock_sealed_s`, `_120`,
  `_500`) and `datasets/mock_sealed.py` (benchopt; path form only:
  `-d "../datasets/mock_sealed.py[study=...]"`).
- **Harness:**
  - `analysis/xsess_lib.py`, `sealed_run.py`, `sealed_personal.py`: splits
    `last` | `calib:K` | `replica:K`, context cells, `--wcv loso`, `--chans`,
    `--pool`, `router-psdctx`, `--router_cap`, `--wvariant`, `--mmap`;
  - `xsess_cache.py`: `--picks`, `--sealed`, `--eval_subjects`,
    `--hidden_labelled`.
- **Release tools:**
  - `scripts/train_sealed.sh` (env documented in its header);
  - `scripts/release_ablations.sh`;
  - `analysis/release_summarize.py` (the pre-registered rules);
  - `analysis/release_eda.py`.
- **Solver:** `solvers/bci_decoding/riemann_sealed.py`, with new opt-in options:
  - `blend_w="auto"`;
  - `chans`;
  - `align="subject_context"`;
  - fast shrinkage LDA above 4,000 features;
  - float32 memory path.
- **Online-order study:** `analysis/sealed_stream.py`, results in
  `logs/sprint0928_p3/`.

## Facts a resumer needs

- On the released data the internal replica is **`SPLIT=calib:3
  TEST_SUBJECTS=<the 10 fully labelled participants>`**. `replica:3` is for the
  mock only.
- Release-day DECISIONS must be read on that split: on `replica:3`, the drifting
  EMG trap is invisible (+0.8), while on `calib:3` it costs −3.7 to −4.2 points.
- NeuralBench's default EEG extractor keeps EEG channels only and resamples to
  120 Hz. Expect 43 ch at 120 Hz; EMG/EOG reach `predict()` only if the organisers
  override `neuro.picks`.
- The fast LDA (a Cholesky solve) is used only above 4,000 features, so every
  committed proxy number is reproduced bit for bit.
- benchopt caches on parameters, not data: pass `--no-cache` after rebuilding
  any cache under the same study name.
- WSL `git` in `2026-competition` shows ~82 CRLF-only diffs. Use
  `git -c core.autocrlf=true`.

## What the user needs to do

1. **Post the organiser question** (SEALED_RECIPE § 5 step 1). It now also asks
   about test-window **order and batching**: the online gain halves when windows
   arrive interleaved.
2. Optional: review/merge the thesis docs branch `docs/sprint0928-consistency`
   (853f9c5; pushed to `personal`, no PR).
3. Optional: approve the REVE/LaBraM weights (still parked).
