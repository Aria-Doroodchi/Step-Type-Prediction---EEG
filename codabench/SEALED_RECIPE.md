# Sealed-phase recipe (Track 2, cross-session, 3-class)

Evidence-ranked training recipe for the Track 2 sealed phase (Oct 28 – Nov 21),
built over the weekend of 2026-09-25/27 on within-subject, cross-session proxies,
because the Graz + BrainHero training data is not released yet. All numbers are
**cell-averaged balanced accuracy** on each subject's *last* session, the
sealed metric without the context dimension. Logs, timings and every decision
are in [LOG.md](LOG.md) (2026-09-25 entries); raw tables in
`logs/sealed_p*/RESULTS.md`; code in `analysis/` and `scripts/sealed_*.sh`.

## 1. The recipe

**Riemann-Sealed** (`solvers/bci_decoding/riemann_sealed.py`): the
Riemann-StepType feature union (xDAWN covariances, broadband tangent space,
log-variance **and** the 4-band filter-bank tangent space) with shrinkage LDA.
**Since 2026-09-29 it also includes the band-power time course `bpt4`:** per
filter-bank band, per channel, the log power in 4 one-second bins. That is
688 features at 43 ch, set with `xblocks="bpt4"`, which `train_sealed.sh` bakes
from `RECIPE_SPEC=riemann:xd=1,fb=1,x=bpt4`.
- **Evidence (§ 2, Phase 7):** it gained +4.35 points on the sealed-like proxy
  under the deployed blend_calib (95 % CI +2.05, +6.67), with a positive
  point estimate on all four proxies and a CI above 0 on three.
- **Caveat:** that confirmation reuses the test sessions of the screen that
  selected bpt4 from 17 candidates, so its CI is optimistic. The independent
  checks are positive but not significant: the reverse split +1.4 and Zhou
  +2.2.
- **Defaults:** the solver's own default stays `xblocks=""`, and so does
  every script default, so the regression gate still reproduces the committed
  flow.
- **Release-day commands:** RELEASE_DAY § 5 passes `SPEC=…,x=bpt4` to the
  ablations; § 7 takes `RECIPE_SPEC` from § 6's DECISIONS via `DEC`, whose
  default carries `x=bpt4`. The replica decides by rule 3b.

The model is made cross-session-aware in three steps that need **no subject or
session id at prediction time**:

1. A **subject router** (log-PSD 1–45 Hz per channel, shrinkage LDA over the
   calibration subjects) names the subject of each test window. It identifies
   the subject of a later-session window 87–100 % of the time on the proxies.
   A window it is unsure about gets the global whitening and the pooled model.
   "Unsure" means its max posterior is below the 1st percentile of the router's
   out-of-fold training confidence, capped at 0.5 (`riemann_sealed.py`:
   `min(percentile 1, 0.5)`). The cap matters: with one calibration session the
   uncapped threshold saturates at 1.0, and 49 % of Zhou's later windows fell
   back (LOG 2026-09-28 Phase 3).
2. **Per-subject whitening** of every window with its (routed) subject's
   Riemannian-mean reference from the calibration data. If the release-day
   context rule adopts it, routing and whitening are instead per (subject,
   context) pair (`align="subject_context"`, § 3).
3. **blend_calib personalisation**: the pooled LDA's probabilities are blended
   with those of the routed subject's own LDA (fitted on the shared features),
   P = w·P_pooled + (1−w)·P_subject. w is chosen on the calibration data only.
   On the proxies it came out 0, 0.5 or 0.75, from the harness's held-out last
   calibration session or chronological halves. For the release, the solver
   chooses w itself (`blend_w="auto"`, since 2026-09-28) by
   leave-one-calibration-session-out and stores it in the joblib. It uses the
   same LOSO rule as the harness's `--wcv loso`:
   - on `mock_sealed_s` and zhou2016 the weights were identical, with
     bit-identical fold scores with the filter bank (CV equal to 4 dp without
     it);
   - on the full-size 120 Hz mock both chose 0.75 (MATCH).

   Small differences are expected on NeuralBench overlays with a validation
   slice: the loader withholds that slice from the solver, while the harness
   trains on it. The mock has no such slice. Until 2026-09-28 this document
   claimed LOSO while the code held out only the last session.

**Conditional upgrade, pending the organisers' answer:** *online per-subject
re-centring* (`adapt="online"`). Each routed subject's whitening reference is the
mean covariance of the last 64 test windows routed to it. This is the single
largest lever measured. In recording order it adds **+3.5 to +7.5 points** to the
clean recipe on the three proxies where the gain is significant (Tangermann,
Scherer, Scherer 3-class). On Zhou the gain is +1.8 and not significant (CI −3.0
to +4.8). That is at or above what the (unattainable) oracle session re-centring
gives. **It depends on the test order** (sprint 2026-09-28,
Phase 3, `logs/sprint0928_p3/`): in recording order a 64-window batch is mostly one
subject, so the buffer is essentially the current batch. With subjects interleaved
window by window, the gain shrinks to **+3.0 (Tangermann) and +4.4 (Scherer
3-class)**. Both are still significant, but that is only 58–85 % of the
recording-order gain. N = 32 kept the most under interleaving. A session change
inside the test stream causes no measurable lag. It uses unlabelled test windows,
which the rules neither allow nor forbid explicitly (§ 5, step 1).

Not in the recipe: EEGNet (6–12 points below Riemann in every comparison, with
or without last-layer calibration or alignment); cross-dataset pre-training
(§ 2, Phase 5); train-only alignment (hurts); batch-level statistics (fragile,
rule-dependent, superseded by online re-centring).

## 2. Evidence

Riemann-StepType, xDAWN + filter bank unless stated; single deterministic fits
(Riemann) or mean ± SD over 3 seeds (neural). Differences under ~2 points are
noise on these 4–9-subject datasets.

**Phase 1 — base family and pooling** (no alignment):

| Model | Tangermann pooled / per-subj. (4 cl.) | Scherer pooled / per-subj. (5 cl.) | Zhou pooled / per-subj. (3 cl.) |
|---|---|---|---|
| MeanLogReg | 0.258 / 0.270 | 0.218 / 0.193 | 0.435 / 0.457 |
| Riemann xDAWN + FB | 0.639 / **0.775** | 0.259 / **0.320** | **0.772** / 0.708 |
| Riemann FB, no xDAWN | 0.544 / 0.726 | 0.277 / 0.301 | 0.728 / 0.667 |
| braindecode EEGNet | 0.622 ± 0.026 / 0.395 ± 0.008 | 0.295 ± 0.014 / 0.213 ± 0.016 | 0.592 ± 0.073 / 0.477 ± 0.017 |
| EEGNet-StepType | 0.644 ± 0.037 / 0.546 ± 0.065 | 0.281 ± 0.010 / 0.261 ± 0.015 | 0.754 ± 0.063 / 0.561 ± 0.003 |

**Phase 2 — alignment without ids** (per-subject / pooled):

| Proxy | none | oracle (test session's stats) | train-only | router (clean) | online-64 (rule-dep.) |
|---|---|---|---|---|---|
| Tangermann | 0.775 / 0.639 | 0.824 / 0.726 | 0.727 / 0.588 | 0.780 / 0.687 | **0.831** / 0.729 |
| Scherer | 0.320 / 0.259 | 0.383 / 0.293 | 0.303 / 0.283 | 0.322 / 0.277 | **0.385** / 0.301 |
| Zhou | 0.708 / 0.772 | 0.763 / 0.742 | 0.732 / 0.770 | 0.688 / 0.767 | **0.783** / 0.747 |

Router accuracy (per window, cross-session): Tangermann 0.965, Scherer 0.866,
Zhou 1.000; Scherer 3-class 0.897. Why the clean router barely helps per-subject
models: whitening a subject's train and test windows with the same reference is
nearly invisible to a tangent space at that subject's own mean (affine
invariance). Only test-session statistics fix *session* drift.

**Phase 3 — pooling vs personalisation** (router ids; clean router alignment /
online):

| Proxy | pooled | calib | blend | **blend_calib** | per-subject |
|---|---|---|---|---|---|
| Tangermann | 0.687 / 0.729 | 0.792 / 0.824 | 0.792 / 0.836 | **0.802 / 0.837** | 0.777 / 0.820 |
| Scherer | 0.277 / 0.301 | 0.322 / 0.375 | 0.304 / 0.363 | **0.322 / 0.375** | 0.302 / 0.362 |
| Zhou | 0.767 / 0.747 | 0.708 / 0.778 | 0.757 / 0.802 | **0.778** / 0.797 | 0.688 / 0.783 |
| Scherer 3-class (Phase 4) | 0.452 / 0.485 | 0.473 / 0.558 | 0.499 / 0.544 | 0.486 / **0.561** | 0.486 / 0.546 |

EEGNet-StepType, router alignment, pooled / last-layer calibration (router ids):
Tangermann 0.653 / 0.678, Scherer 0.295 / 0.303, Zhou 0.704 / 0.729.

**Phase 4 — the sealed-like classes (WORD, SUB, HAND) and feature blocks**
(Scherer 3-class, per-subject; none / router / online): all blocks 0.493 / 0.499 /
0.573; no xDAWN 0.491 / 0.476 / 0.556; filter bank only 0.484 / 0.463 / 0.553;
xDAWN only 0.419 / 0.472 / 0.483; MeanLogReg 0.336 (chance 0.333). Zyma
arithmetic vs rest, cross-subject: FB only 0.734, all 0.723, MeanLogReg 0.512;
0.76–0.78 with per-person re-centring.

**Phase 5 — cross-dataset pre-training** (EEGNet-StepType pre-trained on Dreyer +
Tangermann + Zhou, 27,776 windows, 11 shared channels; fine-tuned on Scherer
3-class; 3 seeds):

| EEGNet-StepType, 11 ch | per-subject: none / router / online | pooled: none / router / online |
|---|---|---|
| from scratch | 0.349 / **0.466** / 0.438 | 0.438 / 0.414 / 0.430 |
| pre-trained (raw, then EA-aligned for router/online) | 0.410 / 0.420 / 0.429 | 0.364 / 0.342 / 0.346 |
| Riemann FB TS, 11 ch: reference Scherer vs MI + Scherer | 0.436 vs 0.431 | 0.436 vs 0.436 |
| *reference: Riemann-Sealed recipe, 30 ch* | *0.486 clean, 0.561 online (blend_calib)* | *0.452* |

Pre-training helps only unaligned per-subject training (+6.1). Everywhere else it
is level or worse (pooled −7 to −8; pooled fine-tunes early-stop at epoch 1–3),
so: **no gain from cross-dataset pre-training at this scale.** The 11-channel set
shared by the MI datasets would itself cost ~6 points against Scherer's 30
channels. REVE / LaBraM remain the documented, not-run next step (2026-09-25
feasibility on this CPU: 20–24 min frozen embedding pass for Dreyer, 45 min per
epoch full fine-tune; weights not downloaded, needs the user's approval).

**Phase 7 — temporal and spatial feature blocks** (sprint 2026-09-29, brief
`prompts/2026-09-29_temporal_spatial_features.md`, LOG 2026-09-29; harness
blocks in `analysis/xfeat_temporal.py` / `xfeat_spatial.py`, spec
`riemann:xd=1,fb=1,x=<id>`). Every recipe covariance spans the whole 4 s
window, so the question was whether time structure or spatial structure adds
class information.
- **Screen:** 14 blocks + 3 controls. The metric is the mean of pooled and
  persubject under the clean router, Δ vs the recipe union, paired 95 % CI.
- **Confirmation:** the passing blocks under the deployed blend_calib on the 4
  proxies, plus a reverse-time replication (train on the later session, test
  on the first).

| block | what it adds | screen Δ, Scherer 3-cl. | blend_calib Δ: Scherer 3-cl. / Tangermann / Scherer 5-cl. / Zhou | verdict |
|---|---|---|---|---|
| **bpt4** | log band power, 4 bands × 4 one-second bins × channel (ERD/ERS time course) | +2.59 (+0.42, +4.95) 8/9 | **+4.35 (+2.05, +6.67)** / +1.50 (+0.12, +3.05) / +2.70 (+1.13, +4.68) / +2.17 (−1.50, +6.33) | **ADOPT** (recipe default for release day) |
| icoh | imaginary coherence per band: lagged connectivity, blind to volume conduction | +2.59 (+0.96, +4.27) 7/9 | +2.17 (+0.29, +4.43) / +1.27 / +1.97 (+0.56, +3.61) / **−3.50** | PROMISING → release-day ablation `xa` |
| tseg3 | FB covariances of 3 time segments → TS | +3.33 (+1.52, +5.76) 9/9 | +1.96 (+0.00, +4.19) / −0.23 / +1.02 / −1.50 | no gain under the deployed recipe (and +11 k features at 43 ch) |
| tseg3+icoh (pre-registered union) | both | — | +3.20 (+1.65, +5.09) / +0.50 / +1.47 / −1.83 | ADOPT by the letter; bpt4 has the higher point estimate on every proxy (paired CI crosses 0 on 3 of 4); ~20 k features at 43 ch: not deployed (post-hoc, on cost and parsimony) |
| *bpt4+icoh (post-hoc, not pre-registered)* | *both* | — | *+6.51 (+4.34, +9.11) 9/9 / +1.81 (+0.77, +3.12) / +2.16 / +3.33* | *information: supports the release-day `xa` step; 1.56× fit at 500 Hz* |
| tcut1000, tseg2, acm3x2, fbd | cue-second split; 2 segments; time-delay-embedded covariance; 1–4 Hz TS | +1.04 to +2.82 | (not advanced: same family as tseg3, or lower) | — |
| fbfrom1000 (control) | the FB without the cue second | +1.55 | — | the FB does **not** depend on the cue second |
| csp8, fblv, fbrlv, reg, fb8, acm2x4, slow block | CSP subspace, band-power topography, regional covariances, 8 bands, … | −0.35 to +0.56 | — | no gain |
| CAR, Laplacian (controls) | fixed spatial filters | −0.05, −1.84 (Tangermann −3.43) | — | no gain (Laplacian hurts) |

Read:
- **Temporal blocks gave the largest and most consistent gains under the
  deployed recipe.** bpt4, where in the trial the band power changes, holds
  across sessions.
  - The segment covariances (tseg3) gain +6.66 without alignment (pooled and
    per-subject; pooled alone +9.2), +3.33 under the router, and +1.96 under
    blend_calib.
  - That pattern is *consistent with* router whitening and personalisation
    absorbing most of what they add. It was not tested directly.
- **The one spatial gain is lagged connectivity (icoh).** The structural
  spatial blocks gained nothing in the screen: CSP subspace, regional
  covariances and band-power topography. Neither did re-referencing (the
  Laplacian hurts).
  - These are empirical findings. The invariance argument once given for them
    was wrong: sklearn's shrinkage LDA standardises features, so the pipeline
    is not rotation-invariant (brief erratum).
- **Selection caveat:** the Scherer 3-class confirmation reuses the test
  sessions of the screen that picked these blocks from 17, so its CIs are
  optimistic. The reverse-time replication (train on the later session, test
  on the first) kept every sign: bpt4 +1.44 (−0.42, +3.33), icoh +2.22
  (+0.56, +4.72), tseg3 +3.43 (+1.53, +5.51).
- **Under online-64** (rule-dependent, information only): bpt4 keeps its gain,
  Scherer 3-cl. blend_calib 0.592 vs 0.561 for the recipe.
- A training-sessions-only activation EDA
  (`reports/features_0929/activation_eda.md`) located the information:
  - alpha, posterior-right, after ~1.5 s;
  - WORD beta decrease at F3;
  - early frontal delta/theta, eye-movement suspect; the FB and bpt4 start at
    4 Hz;
  - no EMG signature at 30–45 Hz.

**How sure are we? Paired bootstrap over subjects** (95 % CI of the mean
per-subject difference, 10,000 resamples; `analysis/sealed_bootstrap.py`):

| Proxy | Claim | A − B | Subjects A > B |
|---|---|---|---|
| Tangermann | per-subject vs pooled (no alignment) | +0.135 (+0.110, +0.160) | 9/9 |
| Scherer | per-subject vs pooled (no alignment) | +0.060 (+0.017, +0.106) | 7/9 |
| Zhou | per-subject vs pooled (no alignment) | −0.063 (−0.130, −0.000) | 1/4 |
| Tangermann | blend_calib vs pooled (clean) | +0.115 (+0.096, +0.134) | 9/9 |
| Scherer | blend_calib vs pooled (clean) | +0.045 (+0.012, +0.081) | 6/9 |
| Zhou | blend_calib vs pooled (clean) | +0.012 (+0.000, +0.023) | 2/4 |
| Scherer 3-cl. | blend_calib vs pooled (clean) | +0.034 (+0.001, +0.065) | 6/9 |
| Tangermann | blend_calib (clean) vs per-subject, no alignment | +0.027 (−0.018, +0.073) | 5/9 |
| Scherer | blend_calib (clean) vs per-subject, no alignment | +0.003 (−0.011, +0.015) | 6/9 |
| Zhou | blend_calib (clean) vs per-subject, no alignment | +0.070 (+0.015, +0.118) | 3/4 |
| Tangermann | online-64 vs clean (blend_calib) | +0.035 (+0.016, +0.054) | 8/9 |
| Scherer | online-64 vs clean (blend_calib) | +0.052 (+0.025, +0.082) | 8/9 |
| Zhou | online-64 vs clean (blend_calib) | +0.018 (−0.030, +0.048) | 3/4 |
| Scherer 3-cl. | online-64 vs clean (blend_calib) | +0.075 (+0.044, +0.110) | 9/9 |
| Scherer 3-cl. | xDAWN: all blocks vs no xDAWN (per-subject) | +0.002 (−0.024, +0.031) | 5/9 |

Read: personalisation beats pooling where each subject has one calibration
session, and pooling beats it with few subjects and two sessions (Zhou).
blend_calib is never significantly worse than either, and its training-chosen
weight is what makes it safe in both regimes. Online re-centring is a
significant gain on 3 of 4 proxies. **xDAWN's contribution on the 3 sealed-like
classes is not measurable** (+0.2, CI −2.4 to +3.1); it stays in the recipe for
its +4–5 points on the motor-imagery proxies (Phase 1) and costs ~nothing, but it
is the first block to drop if the release data disagrees.

## 3. Untested until the Graz + BrainHero data arrives

What the proxies cannot tell us, and the ablation that settles each point once
the data is released. The first run on the new data should use the **ten fully
labelled training participants as an internal replica of the sealed split**.
Every participant's sessions 1–3 train, and the fully labelled participants'
sessions 4–6 are scored with the cell metric (`SPLIT=calib:3
TEST_SUBJECTS=<their indices>`). The organisers' split has the same structure for
the ten evaluation participants; as `replica:3` it runs on the mock only, and the
scripts refuse it on any other cache.

| Open point | Why the proxies can't answer it | Ablation on the released data |
|---|---|---|
| **EMG (2) and EOG (2) channels** | no proxy has them; the default EEG pick drops them anyway | EEG only vs EEG + EOG vs EEG + EMG vs all 47, cross-session on the replica split. This runs on a 47-channel cache (`<study>_x`, `xsess_cache.py --picks eeg,emg,eog`); on the default 43-channel cache these ablations equal the recipe row. Keep a non-EEG channel only if it helps *and* the gain holds on sessions 4–6, and only if the official loader delivers it. The thesis trap: an "easy" class separated by gross muscle or eye activity that drifts across days. Watch word association (speech-like muscle activity) and calculation (eye movements) |
| **500 Hz, 43 EEG channels** | proxies are 14–30 ch at 120 Hz | Check what the sealed loader delivers (NeuralBench resamples EEG tasks to 120 Hz so far). The solver's sizing passes at 500 Hz on the full-size mock (2026-09-28): fit with `blend_w="auto"` 30.4 min, peak 11.8 GiB, read-only replay of 3,600 windows 77 s at 10 threads and 66 s at 2 threads (RELEASE_DAY § 8). A 1–45 Hz vs 1–100 Hz router comparison has no option yet: the band is hard-coded in `xsess_lib.SubjectRouter._feats` and `riemann_sealed._psd_features` |
| **"Context" cells (Graz vs BrainHero)** | proxies have one context | Since 2026-09-28 the harness scores subject × session × context cells: it reads `context.npy`, which `xsess_cache.py` saves from a `context`/`condition`/`paradigm` column. Before that date the column was saved but never read, although this row said otherwise. Ablation `al_ctx` compares `router-psd` with `router-psdctx` (router and whitening per (subject, context) pair). The solver's matching option `align="subject_context"` exists since 2026-09-28, and `train_sealed.sh` bakes it for `RECIPE_ALIGN=router-psdctx:<kind>`. On `mock_sealed_s` it gave train = replay = harness = 0.547222. It cannot be combined with `ADAPT=online` (the solver raises NotImplementedError and `train_sealed.sh` refuses). A pair with fewer than `ctx_min` = 16 training windows keeps its subject's whitening. The solver reads the context from the same column names as `xsess_cache.py` (`_CTX_COLUMNS`); a column with another name must be added to both (RELEASE_DAY § 2) |
| **Three calibration sessions per evaluation participant** | Tangermann and Scherer have one; only Zhou has two training sessions | Blend weight by leave-one-calibration-session-out (`--wcv loso` / `blend_w="auto"`, implemented 2026-09-28). With one calibration session the router degrades fast on later sessions (0.72 → 0.35 on Zhou): check router accuracy per test session 4, 5, 6 (`analysis/release_eda.py` section 4; 0.997 on the full-size mock with three calibration sessions). Offline training with session ids also allows per-(subject, session) training alignment: on Zhou it added +2.4 per-subject (train-only, Riemannian) |
| **Ten training participants with all six sessions** | no proxy has extra fully labelled people | Pooled model and router trained on everyone vs on the test subjects only (`pool_test`, harness `--pool test`). On the replica the ten fully labelled participants stand in for the evaluation participants. The one ablation restricts the pooled model and the router together; there is no separate router-only ablation. The solver has no option to train on a subset yet, so a `pool = test` decision cannot be deployed (`release_summarize.py` prints a NOTE) |
| **xDAWN and class-specific cues** | Scherer's window starts at the visual cue | EDA: evoked response to the cue per class; ablate xDAWN on the replica split |
| **Extra feature blocks: bpt4 (default), icoh (candidate)** (sprint 2026-09-29) | 4–9 subjects per proxy; bpt4's first bin holds the cue response; icoh was −3.5 on Zhou (4 subjects) | `release_ablations.sh` with `SPEC=riemann:xd=1,fb=1,x=bpt4 XB=icoh`: step `xb` drops bpt4 only if the replica is ≥ +1.0 without it; step `xa` adds icoh only if ≥ +1.0 and no context worse, and on a 500 Hz cache only after a fit ≤ 1.5× the recipe (bpt4 + icoh measured 1.56× under contention). RELEASE_DAY rule 3b |
| **Test window order and batch composition** | proxies are in recording order by construction | The clean recipe routes per window and does not care. The online upgrade needs windows of one subject to arrive near each other; the Dreyer test loader delivers recording order (81 % of batches single-subject). Measured 2026-09-28 (Phase 3): with subjects interleaved the online gain drops to 58–85 % of its recording-order value (still significant on Tangermann and Scherer), and no lag at a session change |
| **Class balance per cell** | proxies are balanced | Report per-cell class counts in the EDA; uniform LDA priors already match the balanced-accuracy metric |

## 4. Reproduce end to end

```bash
bash ~/codabench/scripts/train_sealed.sh <data_home> <study> <modality/task> [overlay]
```

**For the released data, follow [RELEASE_DAY.md](RELEASE_DAY.md).** It covers
the replica split (`SPLIT=calib:3 TEST_SUBJECTS=<the fully labelled
participants>`), the § 3 ablations (`scripts/release_ablations.sh`), the
pre-registered release-day rules (`analysis/release_summarize.py`) and timings
from a full-size dress rehearsal on a mock study with the sealed structure
(`analysis/mock_sealed.py`).
- **Rehearsed on the full-size mock on 2026-09-28:** the `train_sealed.sh` flow
  on the organisers' split, the 500 Hz sizing, and the ablation machinery (6 of
  10 steps timed at full size).
- **Rehearsable on the mock since 2026-09-29:** the release-day replica through
  benchopt (`calib:3` with `split=replica_full`, the stand-in for the
  `<study>_xsess` overlay). See RELEASE_DAY § 9.
- **Only the release can exercise:** the organisers' loader, the real NeuralBench
  overlays (`<study>_xsess`, `<study>_all`) and the cache build at release size.

Steps (resumable, logs in `logs/train_sealed_<study>/`): cache every window with
subject/session/run(/context) → validate the recipe in the harness (MeanLogReg
floor, pooled vs per-subject, none vs router alignment) → Phase 3 personalisation
(chooses the blend weight on calibration data) → bake the chosen settings into a
candidate copy of the solver as its defaults (Codabench runs defaults) → benchopt
training on the study overlay → read-only inference replay (must reproduce the
score) → zip into the log folder. It never uploads and never writes
`codabench/submissions/`. `ADAPT=online` switches on the conditional upgrade.

**Verified end to end** (2026-09-25 23:17–23:29, `zhou2016_xsess`): clean recipe
w = 0.75, train 0.770 = read-only replay 0.770 (harness 0.778); `ADAPT=online`
w = 0.5, train 0.792 = replay 0.792 (harness 0.797). Tangermann through benchopt:
0.805 clean, 0.841 online (harness 0.802 / 0.837). Each run takes under 3 minutes.
The clean `zhou2016_xsess` flow was re-checked on 2026-09-28 at 20:06:00–20:08:01
(`logs/sprint0928_D_zhou_regress/STATUS.md`): train 0.770000 = replay 0.770000.
It is also the release-day regression gate (RELEASE_DAY § 0).

## 5. Next steps (ranked)

| # | Step | Expected gain | Effort | Risk |
|---|---|---|---|---|
| 1 | **Ask the organisers** whether `predict()` may use statistics of the unlabelled test windows it has already received (online per-subject re-centring; no labels, no training on the sealed split). Suggested wording below. If yes: add `ADAPT=online RECIPE_ALIGN=online-<N>:riemann` to the release-day settings, e.g. `ADAPT=online RECIPE_ALIGN=online-32:riemann SPLIT=calib:3 TEST_SUBJECTS=$FULL bash ~/codabench/scripts/train_sealed.sh $DH <study>`. N comes from RELEASE_DAY § 6 rule 6. It cannot be combined with `router-psdctx` | **+3.5 to +7.5 points** in recording order on the three proxies where the gain is significant; +1.8, not significant, on Zhou. **+3.0 to +4.4** (Tangermann, Scherer 3-class, N = 64) if test windows arrive interleaved across subjects. The single largest lever found | one forum post (the user's action); code ready and verified | rule only. If the answer is no, the clean recipe stands unchanged |
| 2 | **Release day** ([RELEASE_DAY.md](RELEASE_DAY.md)): use the ten fully labelled participants as a replica split (everyone's sessions 1–3 train, their sessions 4–6 are scored). Then run the EMG/EOG ablation, context cells, blend weight by leave-one-calibration-session-out and the xDAWN ablation (§ 3), and train the candidate | correctness: the proxies are small (4–9 people), so the ranking of close variants can change | ~1 day. At the sealed size one `train_sealed.sh` flow takes 41 min (fixed weight) to 61 min (`auto`) at 120 Hz on the mock, and the ablations 1.5–2 h. RELEASE_DAY § 8 budgets 5.5–6.5 h from download to a checked zip, reading included | loader surprises (47 ch at 500 Hz, contexts, how sessions are labelled) |
| 3 | **Riemann + pooled-EEGNet probability ensemble** on the 3 classes. Pooled EEGNet reaches 0.476 there, close to the Riemann recipe (0.486–0.499), and its errors may differ. Weight on held-out calibration sessions, never equal weights (the Riemann LDA is overconfident: an equal blend lost 0.75 points in the warm-up) | +1–2 points (the warm-up ensemble gave +0.8) | half a day: both models into one `submission.py`, weight search in the harness | small; adds inference time and a second model to audit |

**Added by the sprint's final review (2026-09-29), ranked with the table above:**

- **Between steps 1 and 2 — a deployment test during warm-up.** This is the
  user's upload. No sklearn/pyriemann joblib has ever run on the scoring image.
  Riemann-Sealed ships 22 sklearn LDAs pickled with sklearn 1.9.1 and pyriemann
  0.12 objects, while the image's scikit-learn is unpinned. The sealed phase
  hides logs and allows one submission per day. The command is in
  `SUBMISSIONS.md`. It costs 1 warm-up upload and takes ~30 min of training
  locally.
- **Before release day — strict per-fold references in the LOSO weight search**
  (harness `choose_w` and solver `_choose_blend_w`, kept bit-identical). Today
  each held-out session is whitened with a reference that includes its own
  unlabelled windows: ~2 points of CV bias. When a (subject, context) pair lives
  in one calibration session, that fold is oracle-aligned (10–17 points). It has
  not changed a chosen weight so far. RELEASE_DAY rule 4 carries the interim
  guard. Effort: ~2–3 h with the bit-identity gate.

**Added by the feature sprint (2026-09-29):**

- **Release day:** run the ablations with `SPEC=riemann:xd=1,fb=1,x=bpt4
  XB=icoh` (steps `xb`, `xa`; RELEASE_DAY § 5, rule 3b). bpt4 is the
  default. The replica decides whether it stays and whether icoh joins. The
  post-hoc union bpt4+icoh (+6.51 on Scherer 3-class, positive on all four
  proxies) makes `xa` the ablation most likely to change the recipe.
- **Before release day: re-measure the bpt4+icoh fit at 500 Hz alone.** It
  measured 1.56× the recipe under contention, a FAIL of the 1.5× gate. A
  re-measure on an idle machine closed the sprint (`logs/sealed_f0929/
  RESULTS_sizing.md`, `sz2_*` rows; LOG 2026-09-30). If it still fails, rule
  3b on a 500 Hz cache needs either a restricted icoh (e.g. 8–13 and 13–30 Hz
  only) or a recorded budget deviation.
- **If online re-centring is allowed (step 1):** bpt4 kept its gain under
  `online-64` on Scherer 3-class (information run: 0.592 vs 0.561).
- **Not now: time-segment covariances (tseg3).** They showed no gain under the
  deployed recipe and add 11 k features at 43 ch. At that size, 20 per-subject
  LDAs would solve 16 k × 16 k systems from ~500 windows each; a dual (n < d)
  shrinkage-LDA solve would be needed first.

Not recommended now: more participants (warm-up: no gain), REVE/LaBraM on this
CPU (hours per epoch; revisit on a GPU or with a frozen probe on the release),
train-only alignment (hurts), batch-level statistics (superseded by online).

**Suggested question to the organisers** (for the Codabench forum, the user to post):
> In the sealed phase, may a submission's `predict()` use statistics of the
> unlabelled windows it has received so far, for example keeping a running mean
> covariance per inferred user to re-centre later windows? It uses no labels
> and no sealed data for training, but it is test-time adaptation. Also, in what
> order and in what batches will the sealed test windows reach `predict()`:
> grouped by subject and session in recording order (as in the Dreyer warm-up),
> or interleaved/shuffled across participants? And is the model object kept
> between `predict()` calls?

The order question matters on its own. In recording order the online gain is
+3.5 to +7.5 points on the three proxies where it is significant. With subjects
interleaved it is +3.0 (Tangermann) and +4.4 (Scherer 3-class), the two
significant proxies measured that way (§ 1).
