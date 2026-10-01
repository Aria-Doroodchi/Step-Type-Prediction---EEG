# Log

Dated record of setup steps, runs and timings (estimate vs actual), newest
last. Times are local (WSL `date`), from the run logs in `logs/`.

## 2026-09-23 — environment + Track 2 setup

| Time | Step | Estimate | Actual | Result |
|---|---|---|---|---|
| 17:52 | `uv` + Python 3.12 venv (WSL) | ~40 s | 1 s | ✅ |
| 17:52 | CPU torch 2.14.0+cpu | 1–2 min | 5 s | ✅ `cuda_available: False` |
| 17:52 | neuralbench 0.3.1 + deps | 2–4 min | ~35 s | ✅ matches competition pins |
| 17:53–17:54 | NeuralBench sample data (1.65 GB) | 5–15 min (with debug) | 1 min 36 s | ✅ |
| 17:54 | `neuralbench eeg audiovisual_stimulus --debug` | (above) | 9 s | ✅ trained on CPU, exit 0 |
| 17:59 | `codabench/` folder built; competition repo cloned @ `1ff8ce3` | — | — | config moved here, WSL links made |
| 18:00 | benchopt 1.10.0 + moabb 1.7.2 | ~1 min | <1 s | ✅ torch still +cpu |
| 18:00 | Track 2 smoke test, Simulated | 1–3 min | fails, then 8 s | ❌ `No module named 'benchmark_utils'` (Windows git symlinks) → fixed → ✅ all 4 baselines |
| 18:01–18:04 | prepare `tangermann2012` | minutes | 3 min 23 s | ✅ 1.2 GB |
| 18:03 | our 2 solvers on Simulated | ~1 min | 8 s | ✅ after fixes (empty-val fallback; z-score made optional) |
| 18:05 | tangermann2012: Riemann-StepType + MeanLogReg | few min | 31 s | ✅ Riemann **0.536** (4-class, chance 0.25) |
| 18:05 | tangermann2012: both EEGNets | — | — | ❌ stale Simulated weights in `outputs/` → fixed (skip weights when training) |
| 18:06–18:08 | tangermann2012: EEGNet-StepType + upstream EEGNet | 2–6 min | 1 min 48 s | ✅ ours **0.454** (early stop @ 15), upstream **0.580** |
| 18:09 | `test_track2_solvers.sh tangermann2012 20` | <1 min | 19 s | ✅ script + path-solver params work |
| 18:04:57–18:18:58 | prepare `dreyer2023` (download 22 GB) | 1–3 h (all studies) | 14 min | ❌ OSF `500` on subject 59 → `corrupted, expected 520 timelines but found 516`; retries 2–3 fail in seconds (no re-download) |
| 18:11 | **Stieger 2021 blocked**: 399 GB vs 118 GB free on C: | — | — | guard killed all 3 attempts in ≤1 s; nothing written. Script now skips it below 450 GB free |
| 18:20 | subject 59 fetched by hand (44 MB, 3 s), extracted | — | — | OSF answered 200 on retry |
| 18:22 | prepare `dreyer2023` again | — | 9 s | ❌ still 516: `timelines.csv` index (86 subjects) is never rebuilt once written |
| 18:24–18:34 | stale index set aside (`timelines.csv.bak_missing_sub59`), prepare re-run | 5–15 min + extraction | 10 min 14 s | ✅ index rebuilt: 520 timelines / 87 subjects; `1/1 datasets ready`; data 25 GB total |
| 18:34–18:35 | **test batch** `test_track2_solvers.sh dreyer2023 40` | 5–10 min | 47 s | ✅ Riemann-StepType **0.716**, EEGNet-StepType **0.651** (3 epochs), MeanLogReg 0.681 (full train); chance 0.50 |
| 18:35–18:36 | platform replay: both exported submissions, read-only, inference-only | ~1 min each | 17 s / 12 s | ✅ scores reproduced exactly |

## 2026-09-24 — preprocessing

| Time | Step | Estimate | Actual | Result |
|---|---|---|---|---|
| 17:11 | inspect Dreyer windows (what the model receives) | <1 min | 13 s | 27 ch × 480 (120 Hz), robust-scaled, 78 % of power < 4 Hz; train/val/test = 12,392 / 3,360 / 5,040 windows |
| 17:25 | unit check of `WindowPreproc` (CAR, CSD Laplacian, 8–30 Hz) on real windows | <1 min | 13 s | ✅ drift removed (0–4 Hz 84 % → 0 %); Laplacian C3 row sensible |
| 17:26 | first grid attempt | — | 9 s | ❌ benchopt parsed `8to30` as `8'to30'` → quote values |
| 17:26–17:29 | 40-batch sweep: 2 solvers × {none, car, laplacian} × {none, 8to30} | ~3 min | 3 min 38 s | CAR best (EEGNet 0.711, Riemann 0.718); band-pass 8–30 hurts (EEGNet chance, Riemann ~0.60) |
| 17:42–17:53 | Dreyer EDA `analysis/dreyer_eda.py` (inventory, PSD, effect sizes, ERP/TF, cross-subject checks, within-subject CV) | ~10 min | **11 min 23 s** (683 s), on estimate; figs 00–09 + `summary.json` | cross-subject (train+val → test) LDA: slow < 4 Hz waveform, all 27 ch **0.769**; mu power 0.645; beta 0.609; delta/theta power 0.55–0.59. **Within-subject** 5-fold (median over 87 subjects): slow LDA 0.725, mu/beta Riemann 0.696; 92 % / 85 % of subjects above chance (0.563); the two are uncorrelated across subjects (r = −0.02) |
| 17:55–18:01 | EDA figure fixes + re-render (cached windows) | ~6 min | 5 min 41 s | figure 03 panel replaced (artifact % per channel), label/legend overlaps fixed; `reports/dreyer_eda/README.md` written |
| 18:03 | smoke test: EEGNet `seed` parameter, `--output`, `summarize_runs.py` | <1 min | 20 s | ✅ seeds differ (SD 0.027 at 5 batches) |
| 18:03:55–20:23:24 | **overnight phase 1** `scripts/overnight_2026-09-24.sh`: full Dreyer (train 12,392 → test 5,040 windows) | 2–2.5 h (ETA 20:30) | **2 h 20 min**, on estimate. EEGNet ran ~10 min/run (never early-stopped before epoch 40; revised ETA 21:15 mid-run, beat it) | `logs/overnight_2026-09-24/RESULTS.md`. **EEGNet-StepType 0.779–0.790** (car + patience 20: 0.790 ± 0.002, 3 seeds); upstream braindecode EEGNet 0.778 ± 0.013; Riemann-StepType with xDAWN 0.754 (car 0.746), **without xDAWN 0.585–0.592**; Torch-Linear 0.689; MeanLogReg 0.681 |
| 20:23:48–23:31:58 | **overnight phase 2** `scripts/overnight_2026-09-24_phase2.sh`: Riemann xDAWN nfilter 2/4/8 + 1–40 Hz; EEGNet 100 epochs / patience 20; EEGNet z-score; both refs × 3 seeds | ~3.5 h (ETA ~00:15; revised to ~01:30 at 21:05) | **3 h 8 min**, under estimate. Epochs slowed from ~12 s to ~20 s after ~20:30 with no competing load (WSL the only CPU user); most likely Windows moving background work to E-cores once the desk went idle. Not changed: that's a system setting | `logs/overnight_2026-09-24_p2/RESULTS_all.md`. **EEGNet-StepType 100 epochs / patience 20: 0.806 ± 0.016 (ref none), 0.803 ± 0.010 (car)**, +0.013 to +0.027 over 50 epochs; z-scoring no gain (0.777–0.780); Riemann nfilter 8 **0.760**, nfilter 2 0.66–0.70, 1–40 Hz band-pass 0.71 (the sub-1 Hz content matters). One benign joblib cache `EOFError` (a truncated cache entry, recomputed automatically) |
| 23:33:13–01:46:59 | **overnight phase 3** `scripts/overnight_2026-09-24_phase3.sh`: train frozen candidate `EEGNet-StepType-WU1` (defaults = 100 ep, patience 20, ref none, seed 33) → platform replay → ZIP in `submissions/` (not uploaded); EEGNet 200 epochs / patience 30 × 3 seeds | ~3.5–4 h (ETA ~03:30) | **2 h 14 min**, under estimate (WU1 train 21 min, replay 11 s, 200-epoch grid 1 h 53 min) | **WU1 = 0.82004**; the read-only, inference-only replay reproduces **0.82004 exactly**. ZIP `submissions/eegnet_steptype_wu1_2026-09-24.zip` (submission.py 17 KB + weights.pt 15 KB, defaults checked). This seed is at the top of its config's 3-seed range (0.806 ± 0.016), so expect ~0.81 ± 0.02 on the leaderboard. **200 epochs: 0.810 ± 0.008**, +0.004 over 100 (two runs early-stopped at 154/179): plateau. Stopped scheduling here; further Dreyer-only tuning mostly overfits a warm-up test set that doesn't count |

**State at end of the 2026-09-24 overnight session:** best Dreyer model
EEGNet-StepType, 100 epochs, patience 20 (0.806 ± 0.016; frozen candidate WU1
= 0.820), zipped and replay-verified, **not uploaded** (the user's call). The
5 uploads per day are unused. Full table: `logs/overnight_2026-09-24_p3/RESULTS_all.md`.

**State at end of 2026-09-23 session:** environment, competition code, Track 2 data
(tangermann2012 + dreyer2023) and two ported thesis solvers are ready and
tested. Stieger 2021 is deliberately not downloaded (disk). Next: full
Dreyer training runs, then a first warm-up upload; see
[TRACK2_BCI.md § Next steps](TRACK2_BCI.md#next-steps-suggested).

## 2026-09-25 — Riemann-StepType: slow-waveform and filter-bank blocks

Added two optional blocks to `solvers/bci_decoding/riemann_steptype.py`, both
off by default: `slow_block` (0.1–4 Hz band-pass, mean in 0.25 s bins over
0.5–4.0 s = 14 bins × 27 ch) and `filterbank` (4–8, 8–13, 13–30, 30–45 Hz →
OAS covariance → one tangent space per band). They filter on top of the
solver's own preprocessing, so blocks 1–3 are unchanged; the saved `parts`
hold only builtins + pyriemann/sklearn objects (checked by unpickling without
the solver module). The fit log line now carries the per-block feature count.
`scripts/test_track2_solvers.sh` fixed: a grid key unknown to one solver made
benchopt abort the whole run; that solver is now skipped.

| Time | Step | Estimate | Actual | Result |
|---|---|---|---|---|
| 16:00:29–16:01:36 | smoke test `test_track2_solvers.sh dreyer2023 10 "slow_block=[True],filterbank=[True]"` | 1–2 min | 67 s, on estimate | ✅ rc 0, 2,431 features; 0.662 on 640 training windows (pipeline check only) |
| 16:02:24–16:15:19 | full-Dreyer grid `scripts/riemann_blocks_2026-09-25.sh` (5 configs, one step each) | 7–8 min (ETA 16:13; revised to 16:16 at 16:08) | **12 min 55 s, ~1.7× over**. Configs without the filter bank ran on estimate (1:12–1:15); each filter-bank config took 3:17–3:36, not ~1.5–2 min: four Riemannian-mean tangent-space fits over 12,392 matrices. Progress steady, no stall | every fit line `X=(12392, 27, 480)`; no tracebacks; `logs/riemann_blocks_2026-09-25/RESULTS.md` |

| Config (ref none, nfilter 4) | Features | Bal. acc. | Δ vs 0.754 | Step wall time |
|---|---|---|---|---|
| xDAWN, blocks off | 541 (xdawn 136 + broad 378 + logvar 27) | 0.75377 | 0 (identical to phase 1) | 1:15 |
| xDAWN + slow | 919 (+ slow 378) | 0.75635 | +0.003 | 1:12 |
| **xDAWN + filter bank** | 2,053 (+ fb 1,512) | **0.76964** | **+0.016** | 3:34 |
| xDAWN + slow + filter bank | 2,431 | 0.76865 | +0.015 | 3:36 |
| no xDAWN + slow + filter bank | 2,295 | 0.71825 | −0.036 (xDAWN off alone: 0.592) | 3:17 |

Sealed-phase caveat: the slow lateralised cue is a left/right asymmetry and
is unlikely to transfer to MI vs calculation vs word association, so any
warm-up gain from it is provisional. The one block that did help (the filter
bank, mu/beta power) is the one more likely to transfer.

## Next step toward 0.93

**a. Scores.**

| Best Riemann (xDAWN + filter bank) | EEGNet-StepType-WU1 | Target | Remaining gap |
|---|---|---|---|
| 0.770 | 0.820 | 0.930 | **11.0 points** (16.0 from the best Riemann) |

**b. Hypotheses.**
- *Slow block on top of xDAWN:* no gain (+0.3 points, 0.754 → 0.756, inside
  the ±1.2-point binomial 95 % interval on 5,040 windows); xDAWN's prototype
  block already carries the class-average slow waveform.
- *Filter bank:* a small gain, +1.6 points (0.754 → 0.770), just outside
  that interval (windows cluster in 21 people, so treat it as likely, not
  certain); adding the slow block on top of it: no gain (0.769).
- *Neither replaces xDAWN:* both blocks without it reach 0.718, 3.6 points
  below xDAWN alone.

**c. Can hand-feature blocks close 11 points? No.** The simple slow-waveform
LDA floor is 0.77; the best union reaches 0.770 with 2,053 features, i.e. it
only matches that floor. The union without xDAWN (0.718) doesn't even
reach the slow block's standalone 0.77 (the EDA's LDA, trained on
train + val), so piling blocks into one shrinkage LDA dilutes rather than
adds. EEGNet is at 0.82 and no Riemann config has beaten it (best: 5 points
below). **Feature engineering on this path is exhausted for the warm-up.**
Beyond this path, none of our evidence reaches 0.93: the optimistic sum of
the steps below lands near 0.885. Dreyer's test participants (61–81) are
public, so before spending sessions on a leaderboard 0.93, check what the top
scores rest on. A warm-up score says nothing about the sealed phase.

**d. Candidate next steps, ranked.**
1. **Pretrained REVE encoder** (braindecode `REVE`, `brain-bzh/reve-base`).
   *Gain:* +3 to +5 points (≈0.85–0.87). *Evidence:* the organisers' Stieger
   table, EEGNet 58.6 % → REVE 68.0 %, is a 23 % relative cut in error
   (41.4 → 32.0 %); the same cut on WU1's 18.0 % error gives ≈0.86. That is
   an extrapolation across datasets and class counts, and assumes fine-tuning.
   *Effort:* 2–3 sessions: a frozen-encoder probe, then an overnight CPU
   fine-tune, then packaging. Blockers: the weights are gated (the user must
   accept the terms on HuggingFace; no HF token on this machine yet); ~290 MB
   of fp32 weights plus the position-bank JSON must ship in the ZIP (Codabench
   size limit not checked); input must be resampled 120 → 200 Hz. *Transfer
   risk:* lowest of the three. The encoder is pretrained across paradigms
   rather than tuned to Dreyer's lateralised cue, it takes arbitrary montages
   (4-D position embedding), and the evidence comes from the organisers' own
   track. Open question: the sealed EMG/EOG channels have no scalp position.
2. **Probability-average ensemble, WU1 + Riemann (xDAWN + filter bank).**
   *Gain:* 0 to +1.5 points. *Evidence:* weaker than it looks. r = −0.02 is
   between the two EDA *cues* across people, not between these two models'
   errors. EEGNet sees the < 4 Hz band (78 % of the power) and likely uses the
   same slow cue xDAWN captures, and the Riemann model is 5 points weaker, so
   an equal-weight average can also lose. *Effort:* < 1 session (both models
   exist: LDA `predict_proba` + WU1 softmax in one `submission.py`; measure
   error overlap on the test split first). *Transfer risk:* low for the
   mechanism; the gain is provisional to the extent it rides on the slow cue.
3. **EEGNeX / ATCNet from scratch, WU1 recipe.** *Gain:* 0 to +2 points.
   *Evidence:* the from-scratch line has plateaued (100 → 200 epochs +0.004;
   stock braindecode EEGNet 0.778 ≈ our port). *Effort:* 1 session + ~1–2 h
   CPU per 3-seed config. *Transfer risk:* medium: a from-scratch net on
   Dreyer learns Dreyer's cues. I would not run it now.

**e. Recommendation.** Build and score a frozen-encoder REVE probe on full
Dreyer next session, as the gate for an overnight CPU fine-tune.

> *Precondition (user):* accept the data-usage terms for `brain-bzh/reve-base`
> on HuggingFace, read its licence for competition use, and log in inside WSL
> (`hf auth login`). *Prompt:* On `feat/codabench-track2`, add a frozen-encoder
> REVE probe for Track 2. Read the docstring of braindecode's `models/reve.py`
> (200 Hz input, position bank, `REVE.from_pretrained`) and copy the structure
> of `2026-competition/tracks/bci_decoding/solvers/eegnet.py` into a new
> `codabench/solvers/bci_decoding/reve_probe.py`: inside the model, resample
> each window 120 → 200 Hz, map `meta["ch_names"]` to the position bank, run
> the frozen `reve-base` encoder (no grad, CPU, batch 64) to pooled embeddings,
> fit a shrinkage-LDA head, and save head + encoder weights + position-bank
> JSON so `load_model` never touches the network. Add
> `codabench/scripts/reve_probe_2026-09-26.sh` with the `step` pattern of
> `scripts/riemann_blocks_2026-09-25.sh`: first `max_batches=10` (time it,
> extrapolate, and stop if the full embedding pass extrapolates past 45 min),
> then the full run `bash ~/codabench/scripts/reve_probe_2026-09-26.sh`;
> verify `X=(12392, 27, 480)` in the fit log line. Record the score, embedding
> time per 1k windows and saved size in `LOG.md` and `SUBMISSIONS.md`, then
> decide: probe ≥ 0.80 → queue an overnight fine-tune; probe < 0.75 → drop
> REVE. Do not upload. Expected wall time ≈1.5–2 h: code ≈1 h, 10-batch test
> ≈3 min, full run ≈15–30 min (17.4k windows × ~20 GFLOPs, to be replaced by
> the measured rate).

## 2026-09-25 — exploration: narrowing the next step (4 parallel probes)

| Time | Step | Estimate | Actual | Result |
|---|---|---|---|---|
| ~16:37–~16:47 | workflow `track2-next-step-exploration`: 4 probes in parallel, CPU split 6/8/6 threads, no downloads | 12–15 min | **9 min 45 s**, under estimate (ensemble 16:38:05–16:44:31, reve 16:38:05–16:44:01, desk 16:38:02–~16:42:30, data 16:38:03–16:47:05) | outputs in `logs/explore_2026-09-25/{ensemble,reve,desk,data}/` |
| 16:48:54–16:50 | follow-up checks: REVE gating (HF API), test-loader order | <1 min | ~1 min | see below |

**Sanity:** from the EDA cache, WU1 reproduces 0.82004, Riemann with the
filter bank 0.76964 and the Riemann baseline 0.75377, all exactly.

**Findings.**
- **Leaderboard (public Codabench API, ~16:41):** 36 entries. The #1 score
  is 0.93 (fact sheet: "zero test-pool labels"), then 0.91 ×3 and 0.90 ×4.
  **0.93 is exactly first place**; 8 entries are at or above 0.90. Our 0.82
  (adoroodchi, 14:51 UTC) has 19 entries above it. The best published
  cross-subject Dreyer result found is ~0.74 (different split), and the
  original within-subject online accuracy was 0.63. The #2 entry is a
  masked-autoencoder model.
- **Ensemble WU1 + Riemann (filter bank), measured:** with the weight chosen
  on val only (w_WU1 = 0.85) it scores **0.828 (+0.8 points**; subject
  bootstrap 95 % CI +0.2 to +1.4). The test-tuned oracle weight gives
  0.832. An equal-weight average *loses* 0.75 points because the Riemann
  LDA is overconfident (mean max-prob 0.88 vs accuracy 0.77). The errors
  overlap: error φ 0.34, per-subject r 0.74, and both fail on the same hard
  subjects. The Riemann baseline adds nothing.
- **More participants (Riemann learning curve), measured:** +0.007 per 10
  participants up to 52. Training on train+val scores **−0.010** (baseline)
  and **−0.001** (filter bank), both inside the draw-to-draw noise (SD
  0.008 across different 52-participant draws). More data from the same
  distribution is not a lever. Treat differences under ~0.01 between
  candidates as noise.
- **REVE, offline feasibility (random weights, 8 shared threads):**
  69.4 M params, 265 MB fp32 / 132 MB fp16 (storage quota 15 GB, no
  per-ZIP cap). Costs: a frozen embedding pass ≈20–24 min; full fine-tune
  **45.5 min/epoch** (impractical); last 2 blocks + head on cached layer-20
  tokens 4.4 min/epoch after a ≈22 min caching pass. bf16 is unusable (no
  native support on the i7-12700K). **Not gated** (HF API `gated=False`;
  the braindecode docstring is out of date). It still needs your approval to
  download the weights plus `positions.json`.
- **REVE, desk evidence:** in the REVE paper, a frozen or linear-probed
  REVE-Base trails full fine-tuning by 11–21 points on motor imagery.
  **Dreyer 2023 is in REVE's pretraining corpus**, including the test
  participants' unlabelled EEG, so any warm-up gain is inflated. Our 120 Hz
  input leaves 60–100 Hz empty relative to pretraining. The organisers'
  REVE 68.0 % may be a probe or a fine-tune (their sources disagree).
- **Test windows reach `predict` in order.** `benchmark_utils/nb_task.py`
  shuffles only the train split. In the cache (assumed to be in loader
  order), the 5,040 test windows form 21 contiguous runs, one per
  participant, and 81 % of 64-window batches come from one participant.
  Per-subject re-centring from unlabelled test windows is therefore
  mechanically possible without a subject id.

**Revised ranking.**
1. **Unsupervised per-subject alignment** (Euclidean alignment of each
   subject's windows, or Riemannian re-centring). *Gain:* unmeasured. It is
   the most plausible explanation for the 0.90–0.93 cluster: those entries
   beat anything published by far, the #1 fact sheet stresses "zero
   test-pool labels", and subject shift is large here (±1–2 points just from
   which participants you train on). *Effort:* 1 session to measure.
   *Transfer:* the best of any option: the sealed phase is cross-session
   within subject, which is the textbook case for re-centring (REUSING § 6).
   *Risk:* it depends on the scorer keeping windows in recording order,
   19 % of batches straddle two subjects, and whether it's allowed must be
   checked against the rules.
2. **Ensemble** as a last-mile add-on: +0.8 points measured. Choose the
   weight on held-out subjects; never use equal weights.
3. **REVE**: demoted for the warm-up (probe likely below WU1, fine-tuning
   costs hours on this CPU, Dreyer is in its pretraining corpus). Revisit
   when the sealed data arrive (47 ch, 500 Hz), where a montage-agnostic
   encoder matters.
Dropped: more participants / train+val (measured, no gain); EEGNeX/ATCNet
from scratch (the plateau evidence stands; no new evidence for it).

## 2026-09-25 — data folder on Z: (`\192.168.50.157\Beast_PC\Projects\codabench`)

User decision: `Z:\Projects\codabench` is the data folder. Downloaded the
organisers' Track 2 datasets except Stieger 2021 (held pending questions);
Graz + BrainHero 2026 is not released yet. Downloads went through
`neuralset.Study(...).download()` (same call as `benchopt prepare`) into a WSL
staging folder, then robocopy to `Z:\Projects\codabench\neural_compet\`
(benchopt's layout). WSL cannot see Z: until it is mounted (needs sudo).

| Time | Step | Estimate | Actual | Result |
|---|---|---|---|---|
| 17:23:34–17:27:53 | download to `~/neuralbench/stage_z/`: Zyma 2019 (NEMAR), Scherer 2015 = BNCI2015_004, Zhou 2016 (Zenodo) | 5–15 min | 4 min 19 s, under | ✅ Zyma 72 EDF = 36 people × (rest + arithmetic), 176 MB; Scherer 18 recordings = 9 people × 2 sessions; Zhou 24 = 4 people × 3 sessions × 2 runs. MOABB's Zenodo fetch warns `InsecureRequestWarning` (no TLS certificate check: library behaviour) |
| 17:24:17 | first copy of existing Dreyer + Tangermann data | — | 1 s | ❌ Git Bash collapsed `\wsl.localhost` to `\wsl.localhost`, nothing copied → rerun from PowerShell |
| 17:24:32–17:31:08 | robocopy existing `benchopt_data/neural_compet` (Dreyer, Tangermann, window cache) → Z: | 5–15 min | 6 min 36 s, on estimate | ✅ 4,453 files, 24.7 GB, 0 failed (≈76 MB/s) |
| 17:28:16–17:29:41 | robocopy staging → Z: | ~2 min | 1 min 25 s | ✅ 316 files, 3.76 GB, 0 failed (larger than the 2.2 GB staged: links stored as full copies) |

To check before using Scherer 2015: NeuralFetch/MOABB label its classes
math / letter / rotation / count / baseline, while the paper describes word
association, mental subtraction, spatial navigation, hand and feet imagery.

## 2026-09-25 (evening) — sealed-phase prep, Phase 0: cross-session harness

Weekend plan: `prompts/` weekend prompt (commit d3019b7). Goal: an evidence-ranked
sealed-phase recipe (`SEALED_RECIPE.md`) from within-subject cross-session proxies.

**Machine prep.** `powercfg` read-only check at 18:06: sleep-after and
hibernate-after are already **0 (never) on AC and DC**, so nothing was changed
(previous values = current values = 0).

| Time | Step | Estimate | Actual | Result |
|---|---|---|---|---|
| 18:06:53–18:06:57 | rsync `stage_z/neural_compet/*` → `benchopt_data/neural_compet/` | 1–2 min | 4 s | ✅ Scherer 3.4 GB, Zhou 273 MB, Zyma 176 MB now under one `BENCHOPT_DATA_HOME` |
| 18:08:38–18:09:26 | first cache build, Zhou 2016 (smoke) | 1–3 min | 48 s | ✅ but 23 windows had MOABB's `-100` end-of-run code (one per run) counted as a 4th class → dropped, labels re-indexed |
| 18:09:57–18:12:13 | caches: Zhou, Tangermann, Scherer, Zyma (`scripts/sealed_p0_caches.sh`) | ≈8 min | **2 min 16 s**, under | ✅ see table below |
| 18:13:42–18:14:56 | harness smoke test on Zhou: 4 model families × 2 modes, 6 alignment conditions | ≈2 min | 1 min 14 s | ✅ all paths run |
| ~18:18–18:22 | benchopt overlays + load test | — | ~4 min | ✅ after one fix (an empty val split is rejected → 2 % index val slice) |

**Caches** (`~/neuralbench/xsess_cache/<study>/`, rows in recording order):

| Study | X | Subjects × sessions | Classes | Notes |
|---|---|---|---|---|
| zhou2016 | (1800, 14, 480) | 4 × 3 | 3: left hand, right hand, feet (600 each) | `-100` markers dropped |
| tangermann2012 | (5184, 22, 480) | 9 × 2 (`0train`, `1test`, different days) | 4 (1296 each) | 288 / subject / session |
| scherer2015 | (3550, 30, 480) | 9 × 2 | 5 (710 each) | window 3–7 s (task default); two sessions have 175 trials, the rest 200 |
| zyma2019 | (1659, 20, 600) | 35 × 1 | arithmetic 420 / rest 1239 | 5-s windows; one channel is `A2-A1` (dropped by the harness); 35 of 36 people |

**Scherer 2015 label mapping, resolved.** The BNCI `.mat` files carry their own
`classes` field: `['WORD' 'SUB' 'NAV' 'HAND' 'FEET']` for codes 1–5 (checked in
`download/MNE-bnci-data/~bci/database/004-2015/A.mat`, both sessions). The NEMAR
README and MOABB's event names (`math=1, letter=2, rotation=3, count=4,
baseline=5`) come from a docstring for a different dataset and are wrong:
**"math" is word association, "letter" is mental subtraction, "count" is right-hand
MI.** In the cache, label 0 = code 1 WORD, 1 = SUB, 2 = NAV, 3 = HAND, 4 = FEET.
Sealed-like 3-class subset: labels 0, 1, 3 (WORD, SUB, HAND).

**Harness** (`analysis/xsess_lib.py`, `analysis/sealed_run.py`,
`analysis/sealed_summarize.py`): test = each subject's last session; cell-averaged
balanced accuracy over (subject, session) cells (the proxies have no "context")
plus pooled balanced accuracy and a per-subject table. Models reuse the solver code
(Riemann-StepType feature union, EEGNet-StepType network) plus MeanLogReg and the
upstream braindecode EEGNet (20 epochs). EEGNet-StepType early-stops on the val
session where a subject has 3 sessions (Zhou), else on 2 held-out subjects
(pooled) or a stratified 20 % (per-subject), then refits on all training windows
for the best epoch count. Alignment = per-group whitening X ← R^-1/2 X (Euclidean
mean = EA, He & Wu 2020; Riemannian mean = re-centring), applied at the signal
level so every covariance block and EEGNet see it. Conditions: none, oracle,
train-only, router (per window, training statistics only), routerb (64-window
batch vote), batch (64-window batch statistics).

**benchopt overlays** (`config/nb_overlays/motor_imagery/`, installed into the
venv + registered in the untracked `bci_studies.py` by
`scripts/install_xsess_overlays.sh`): `tangermann2012_xsess` (2539 / 53 / 2592
train/val/test), `zhou2016_xsess` (1176 / 24 / 600, 3 classes, `-100` filtered),
`scherer2015_xsess` (1763 / 37 / 1750, 5 classes), `scherer2015_xsess3` (1055 / 25 /
1050: WORD, SUB, HAND). Val is a 2 % index slice because neuralset rejects an empty
split, and a solver only ever sees the train loader. Session-based validation lives
in the harness instead.

**Rules check (batch-level alignment).** `codabench/pages/terms.md` does not mention
test-time adaptation; Rules 4 and 6 forbid training on or accessing the sealed test
split and modifying the scoring harness. The track page says the split isolates
drift "without additional calibration". Verdict: using the statistics of the batch
that `predict` receives is **not explicitly prohibited but rule-dependent**
(transductive, and it depends on the scorer's batching and ordering). Report it, do
not adopt it without asking the organisers. The per-window router uses training
statistics only and is clean.

**Incident, Phase 1 first launch (18:16–18:22).** Both lanes stalled on their first
Riemann fit (> 150 s silent where a smoke fit took 9 s): 49 threads per process,
OpenBLAS defaulting to 20 threads in each of two processes on 20 cores (spinning
BLAS threads). Cause: `XS_THREADS` only capped torch. Fix: `sealed_lib.sh` exports
`OMP/OPENBLAS/MKL_NUM_THREADS=$XS_THREADS`. Relaunched 18:22; the same fit then
took 14 s. Also: two lanes appending to one file on `/mnt/c` lost a STATUS row,
so results now go to one `results_<study>.jsonl` per study and the reader skips
malformed lines. On the Windows side, `python3` is the Store alias and hangs, so
edit with the Edit tool, not Python.

**Data-release check (18:53, Fri):** tracks page still says Graz + BrainHero is
"coming soon". Recorded from the page (not in our docs before): **20 participants
× 6 sessions, 80 h, 47 ch at 500 Hz. Ten *training* participants have all six
sessions labelled; the ten *evaluation* participants have sessions 1–3 labelled
(calibration) and 4–6 hidden.** The contexts are "varied Graz and BrainHero
contexts"; the drift sources named are time of day, electrode replacement,
physiology and mental strategy. Consequences: (1) the ten training participants
give an exact internal replica of the sealed split (calibrate on 1–3, test on 4–6)
for validating the recipe; (2) hidden windows come only from the ten evaluation
participants, each with three calibration sessions: the Zhou-like regime (3
sessions), where the subject router is at 100 %, not the one-calibration-session
regime of Tangermann/Scherer.

## 2026-09-25 (evening) — Phase 2 prep: subject routing without ids

Router-only study (`analysis/router_eval.py`, no decoder): train on each subject's
training sessions, route every last-session window to a training subject.
2 threads, alongside the Phase 1 lanes. Tables: `logs/sealed_p2/router_eval*.md`.

| Time | Step | Estimate | Actual |
|---|---|---|---|
| 18:27:47–18:31:24 | iteration 1: distance to subject mean (Euclid / Riemann), TS, per-band log-var, filter-bank TS | 2–4 min | 3 min 37 s |
| 18:32:01–18:36:47 | iteration 2: log-PSD, combinations; fallback threshold changed to outlier semantics | 3–5 min | 4 min 46 s |
| 18:37–18:41:40 | iteration 3: 0.5 Hz PSD; accuracy on fallback windows | 3–5 min | ~4 min |

Per-window cross-session subject accuracy (window / 64-window batch vote):

| Fingerprint | Zhou (4 subj, 2 train sessions) | Tangermann (9, 1) | Scherer (9, 1) |
|---|---|---|---|
| distance to mean, Riemannian | 0.577 / 0.603 | 0.652 / 0.667 | 0.676 / 0.737 |
| tangent space (LDA) | 0.710 | 0.754 | 0.577 |
| filter-bank TS (LDA) | 0.995 | 0.878 | 0.809 |
| per-band log-variance (LDA) | 0.980 | 0.945 | 0.831 |
| **log-PSD 1–45 Hz, 1 Hz bins (LDA)** | **1.000** / 0.927 | **0.965** / 0.951 | **0.866** / 0.926 |
| log-PSD 0.5 Hz bins | 0.990 | 0.942 | 0.865 |

- **Chosen router: log-PSD + shrinkage LDA** (`SubjectRouter("psd")`). ≥ 90 % on
  2 of 3 proxies; Scherer (patients, 1 calibration session, 5 very different
  mental tasks) stays at 0.87 after three iterations. The sealed regime (3
  calibration sessions per evaluation participant) resembles Zhou, where two
  training sessions already give 100 %.
- Batch voting over contiguous 64-window batches is often *worse* than per-window
  routing (batches straddle subjects: only 70 % of Zhou test batches are
  single-subject) and is rule-dependent anyway. Not adopted.
- Fallback: a window whose max posterior is below the 1st percentile of the
  training out-of-fold posteriors gets the global reference. The first rule
  (accuracy-based) fell back on 96 % of Zhou windows and was replaced. Routing
  accuracy on the fallback windows is only 43–57 %, so the fallback catches real
  failures.

## 2026-09-25 (evening) — Phase 1: cross-session baselines

`scripts/sealed_p1.sh`, two lanes × 10 threads. Results: `logs/sealed_p1/RESULTS.md`.

| Time | Step | Estimate | Actual |
|---|---|---|---|
| 18:16:46–18:22:17 | first launch | — | ❌ thrashed (OpenBLAS 20 threads × 2 processes), killed; see Phase 0 incident |
| 18:22:19–19:03:29 | lane B: Scherer fast / braindecode EEGNet / EEGNet-StepType | 53 min | **41 min**, under |
| 18:22:19–19:36:56 | lane A: Tangermann + Zhou | 69 min (revised to ~100 at 18:46) | **75 min**, 1.1× the first estimate |

The Tangermann EEGNet-StepType step took 48 min against 40: pooled runs early-stop
at 70–96 epochs and refit, ~9–11 min each. The watchdog's "idle" column is
misleading for these runs (the runner logs only per config), so a heartbeat line
every 10 epochs was added for later phases.

Cell-averaged balanced accuracy on each subject's last session (seed-mean ± SD,
3 seeds for the neural models; the Riemann and MeanLogReg fits are deterministic):

| Model | Tangermann pooled / per-subj. (4-cl., chance .25) | Scherer pooled / per-subj. (5-cl., .20) | Zhou pooled / per-subj. (3-cl., .33) |
|---|---|---|---|
| MeanLogReg | 0.258 / 0.270 | 0.218 / 0.193 | 0.435 / 0.457 |
| Riemann xDAWN, no FB | 0.663 / 0.679 | 0.252 / 0.315 | 0.635 / 0.622 |
| Riemann xDAWN + FB | 0.639 / **0.775** | 0.259 / **0.320** | **0.772** / 0.708 |
| Riemann no xDAWN, FB | 0.544 / 0.726 | 0.277 / 0.301 | 0.728 / 0.667 |
| Riemann no xDAWN, no FB | 0.501 / 0.599 | 0.264 / 0.294 | 0.555 / 0.478 |
| braindecode EEGNet (20 ep) | 0.622 ± 0.026 / 0.395 ± 0.008 | 0.295 ± 0.014 / 0.213 ± 0.016 | 0.592 ± 0.073 / 0.477 ± 0.017 |
| EEGNet-StepType (ES + refit) | 0.644 ± 0.037 / 0.546 ± 0.065 | 0.281 ± 0.010 / 0.261 ± 0.015 | 0.754 ± 0.063 / 0.561 ± 0.003 |

**Decisions.**
- **Base family = Riemann-StepType with xDAWN + filter bank**: best on all three
  (Tangermann +13 points over the best EEGNet, Scherer +2.5 over braindecode
  EEGNet, Zhou +1.8 over EEGNet-StepType, which is within its seed SD of 0.063).
  EEGNet stays in Phases 2–3 as the deep control.
- **Pooled vs per-subject: carry both.** Riemann per-subject beats pooled on 2 of
  3 (Tangermann +13.6, Scherer +6.1; Zhou pooled +6.4). This is the opposite of
  the thesis pooling result: with a same-subject calibration session, a
  subject's own covariance model beats a pooled one unless the data is very
  small (Zhou: 4 subjects, 300 training windows each). EEGNet prefers pooled on
  all three (too few windows per subject to train a network).
- The filter bank helps most per-subject (Tangermann +9.6 points with xDAWN).
  xDAWN, an ERP method, still adds on MI cross-session (Tangermann per-subject
  +4.9, Zhou pooled +4.4); dropping it is not supported by Phase 1.
- Scherer 5-class is weak everywhere (best 0.32; no subject above 0.43). The
  sealed-like 3-class subset is Phase 4.

## 2026-09-25 (evening) — Phase 2: alignment without ids

`scripts/sealed_p2.sh` (lanes) + `scripts/sealed_p2b.sh` (online re-centring).
Decision table: `python analysis/sealed_decide.py --tag p1 --tag p2`.

| Time | Step | Estimate | Actual |
|---|---|---|---|
| 19:04:01–19:47:44 | s_riem: Scherer, 2 Riemann specs × 2 modes × 9 conditions | 55 min | 44 min, under |
| 19:47:44–20:06:28 | s_eeg: Scherer EEGNet-StepType pooled, 3 conditions × 3 seeds | 40 min | 19 min, under |
| 19:37:04–20:00:47 | t_riem: Tangermann, 36 configs | 40 min | 24 min, under |
| 20:00:47–20:06:44 | z_riem: Zhou, 36 configs | 8 min | 6 min |
| 20:06:44–20:24:07 | z_eeg: Zhou EEGNet-StepType, 9 runs | 15 min | 17 min |
| 20:07–20:25:05 | p2b Riemann online (Zhou, Scherer, Tangermann) | 28 min | 18 min, under |

Cell-averaged balanced accuracy, Riemann-StepType with xDAWN + filter bank
(riemannian-mean kind unless noted; `E` = Euclidean):

| Proxy / mode | none | oracle | train-only | **router (clean)** | batch-64 | **online-64** |
|---|---|---|---|---|---|---|
| Tangermann pooled | 0.639 | 0.726 | 0.588 | **0.687** | 0.723 | 0.729 |
| Tangermann per-subject | 0.775 | 0.824 | 0.727 | 0.780 | 0.818 | **0.831** |
| Scherer pooled | 0.259 | 0.293 | 0.283 | 0.277 | 0.300 | 0.301 |
| Scherer per-subject | 0.320 | 0.383 | 0.303 | 0.322 | 0.382 | **0.385** |
| Zhou pooled | **0.772** | 0.742 | 0.770 | 0.767 | 0.745 | 0.747 |
| Zhou per-subject | 0.708 | 0.763 | 0.732 | 0.688 | 0.763 | **0.783** |

EEGNet-StepType pooled (Euclidean, 3 seeds): Scherer none 0.281 ± 0.010, oracle
0.305, router 0.295, train-only 0.292; Zhou none 0.754 ± 0.063, oracle **0.819 ±
0.019**, router 0.704 ± 0.117 (hurts, unstable), train-only 0.756 ± 0.031.

**Decision (weekend rules), stated plainly.**
1. **Alignment is in the recipe**: the oracle gains ≥ 2 points on all three
   proxies (per-subject +4.9 / +6.3 / +5.5).
2. **The clean router (d) fails the adoption bar**: accuracy passes on 2 of 3
   (0.965, 0.866, 1.000), but it recovers only ~55 % of the oracle gain for pooled
   models and ~0–10 % for per-subject models. This is expected: whitening a
   subject's train and test windows with the *same* training reference is
   (nearly) invisible to a tangent space at that subject's own mean (affine
   invariance). Without test-session statistics, alignment fixes *between-subject*
   shift (it helps pooled models) but not *between-session* drift.
3. **The prescribed fallback (c) train-only is rejected on evidence**: it
   *lowers* the score on Tangermann (−5 to −6) and in every per-subject case.
4. **What recovers the gain is test-time statistics.** Batch-64 recovers
   88–100 %, and **online per-subject re-centring (route each window, whiten with
   the mean of the last 64 test windows routed to the same subject) matches or
   beats the oracle on all three proxies**: per-subject +5.6 / +6.5 / +7.5 points.
   It tracks within-session drift and does not care where batches start. It is
   rule-dependent (transductive), so it becomes the recipe's *conditional
   upgrade*, pending the organisers' answer.
5. Clean default for Phase 3: per-subject Riemann (model chosen by the router),
   pooled + router alignment as the pooled component.

**Phase 2, remaining steps.** 20:24:07–21:06:45 t_eeg (Tangermann EEGNet-StepType,
6 runs): 42 min, under the 70-min estimate. p2b EEGNet online steps 20:25–20:39.
Phase 2 ALL DONE 21:06:49.

EEGNet-StepType pooled, 3 seeds (Euclidean kind):

| Proxy | none | oracle | router | train-only | online-128 (rule-dep.) |
|---|---|---|---|---|---|
| Tangermann | 0.644 ± 0.037 | 0.690 | 0.653 (19 % of the gain) | — | — |
| Scherer | 0.281 ± 0.010 | 0.305 | 0.295 (58 %) | 0.292 | 0.294 |
| Zhou | 0.754 ± 0.063 | 0.819 ± 0.019 | 0.704 ± 0.117 (hurts) | 0.756 | **0.824** |

The same pattern as Riemann: the clean router recovers little, test-time
statistics recover all of it. EEGNet stays below per-subject Riemann everywhere.

## 2026-09-25 (night) — Phase 3: pooling vs personalisation

`scripts/sealed_p3.sh` (lanes) + `scripts/sealed_p3b.sh` (rerun with the new
`blend_calib` variant). Tables: `logs/sealed_p3/RESULTS.md`
(`analysis/sealed_decide_p3.py`).

| Time | Step | Estimate | Actual |
|---|---|---|---|
| 20:39:40–21:27:54 | lane B: Scherer, 2 Riemann families (clean), 1 online, EEGNet 3 seeds | 49 min | 48 min, on |
| 21:06:58–21:56:05 | lane A: Tangermann + Zhou, same + EEGNet | 74–90 min | 49 min, under |
| 21:28–22:14:19 | p3b: blend_calib rerun, 3 proxies × {router, online} | 40 min | 46 min, 1.15× |

Variants (all on the same aligned data): **pooled**; **calib** = pooled feature
extractor + per-subject LDA; **blend** = w·pooled + (1−w)·per-subject model;
**blend_calib** = w·pooled + (1−w)·calib; **per-subject**. w is chosen on
training data only (Zhou: val session; others: two chronological halves of the
training session). Personal variants are scored with the router's subject ids
(a wrong route applies the wrong subject's model; outliers get pooled).

Riemann xDAWN + FB, router ids (oracle ids within ~1 point, except Scherer ~1–2):

| Proxy | alignment | pooled | calib | blend | **blend_calib** (w) | per-subject |
|---|---|---|---|---|---|---|
| Tangermann | router (clean) | 0.687 | 0.792 | 0.792 | **0.802** (0.5) | 0.777 |
| Scherer | router (clean) | 0.277 | **0.322** | 0.304 | **0.322** (0.0) | 0.302 |
| Zhou | router (clean) | 0.767 | 0.708 | 0.757 | **0.778** (0.75) | 0.688 |
| Tangermann | online-64 (rule-dep.) | 0.729 | 0.824 | 0.836 | **0.837** | 0.820 |
| Scherer | online-64 (rule-dep.) | 0.301 | **0.375** | 0.363 | **0.375** | 0.362 |
| Zhou | online-64 (rule-dep.) | 0.747 | 0.778 | **0.802** | 0.797 | 0.783 |

EEGNet-StepType (router alignment, 3 seeds): pooled / calib (last-layer
fine-tune, router ids): Tangermann 0.653 / 0.678, Scherer 0.295 / 0.303, Zhou
0.704 / 0.729. Calibration helps EEGNet by about 1–3 points, but it stays 6–12
points below Riemann blend_calib.

**Decision.** **blend_calib is best or tied-best on all three proxies**, clean and
rule-dependent alike, so it goes into the recipe. Read literally, the rule's
"ties go to the simpler variant" would pick calib on Tangermann (−1.0) and
Scherer (identical), but calib loses 7 points on Zhou, the proxy with two
training sessions per subject, closest to the sealed three. blend_calib contains
calib (w = 0) and pooled (w = 1) as special cases and picks w on training data
(0.0 / 0.5 / 0.75 here), so it degrades gracefully in both regimes.
Router ids cost < 1 point where the router is ≥ 96 % accurate (Tangermann,
Zhou) and ~1–2 points on Scherer (87 %). Differences between blend,
blend_calib and calib are mostly under 2 points on single deterministic fits:
the robust finding is personal > pooled on one-calibration-session data, and
that a training-chosen blend recovers pooled's advantage when pooling wins.

## 2026-09-25 (night) — Phase 4: the sealed-like 3 classes (Scherer) + Zyma

Scherer 2015 restricted to **WORD (word association), SUB (mental subtraction),
HAND (right-hand kinesthetic MI)** = cache labels 0, 1, 3; X = (2130, 30, 480),
1080 train / 1050 test, chance 0.333. `scripts/sealed_p4.sh`.

| Time | Step | Estimate | Actual |
|---|---|---|---|
| 21:10:33–~21:33 | Zyma, 21 configs, 3 threads (run early as a light job) | 10–20 min | ~23 min |
| 21:56:30–22:34:01 | s3_ablate, 48 configs (8 block sets × 2 modes × 3 alignments) | 30 min (revised 50 at 22:26) | 38 min, 1.25× |
| 22:34–22:54:33 | three personalisation runs (clean, online, no-xDAWN) | 45 min | 20 min, under |

**Block ablation**, Riemann-StepType on the 3 classes, cell score (none / clean
router / online-64, rule-dependent):

| Blocks | per-subject | pooled |
|---|---|---|
| all (xDAWN + FB + broadband + log-var) | **0.493** / 0.499 / **0.573** | **0.443** / 0.452 / 0.485 |
| no xDAWN (FB + broadband + log-var) | 0.491 / 0.476 / 0.556 | 0.431 / 0.431 / 0.476 |
| filter bank only | 0.484 / 0.463 / 0.553 | 0.435 / 0.433 / 0.487 |
| no FB (xDAWN + broadband + log-var) | 0.479 / 0.501 / 0.555 | 0.423 / 0.448 / 0.473 |
| broadband + log-var | 0.446 / 0.443 / 0.509 | 0.423 / 0.406 / 0.450 |
| log-var only | 0.432 / — / — | 0.381 / 0.361 / 0.413 |
| xDAWN only | 0.419 / 0.472 / 0.483 | 0.386 / 0.413 / 0.437 |
| MeanLogReg | 0.336 | 0.353 |

**Personalisation on the 3 classes** (all blocks, router ids): clean pooled 0.452,
calib 0.473, per-subject 0.486, blend **0.499**, blend_calib 0.486 (router 0.897
accurate); online pooled 0.485, calib 0.558, per-subject 0.546, blend 0.544,
blend_calib **0.561**.

**Zyma 2019** (arithmetic vs rest, 35 people, cross-subject 5-fold, chance 0.5):
filter bank only 0.734, all blocks 0.723, xDAWN only 0.622, broadband + log-var
0.627, MeanLogReg 0.512; with each held-out person re-centred on their own
unlabelled windows: FB only 0.779, all blocks 0.762, xDAWN only 0.778.

**Findings and decision.**
- **The sealed classes are separable across sessions** with these features:
  ~0.49–0.50 clean and ~0.56–0.57 with online re-centring against 0.333 chance
  (per-subject, one calibration session). Mental calculation is detectable at all
  (Zyma, cross-subject 0.73–0.78).
- **The filter bank carries most of the 3-class signal** (FB alone 0.484 vs all
  blocks 0.493). **xDAWN still adds a small, consistent amount**: removing it
  costs 0.2–2.3 points in 6 of 6 per-subject/pooled × alignment comparisons, and
  0.9–2.0 in the personalised variants. On the MI proxies it added 4–5 points
  (Phase 1). By the rule's second branch ("if xDAWN still matters, keep both"):
  **keep xDAWN + filter bank**. Caveat for the sealed data: xDAWN is an ERP
  method, and a window that starts at a class-specific visual cue gives it
  cue-evoked potentials to learn. That is legitimate for the score but not
  "mental-task" signal. Check it with the EDA on the release.
- Personal variants are within ~2.5 points of each other here (blend 0.499 vs
  blend_calib 0.486 clean; blend_calib best with online): no reason to change
  the Phase 3 choice.

**Phase 4, lane B** (22:54:40–23:17:12, 23 min vs 50 estimated): EEGNet-StepType
on the 3 classes, 30 ch, 3 seeds: pooled **0.476 ± 0.010** (above pooled Riemann,
0.443, below the Riemann recipe, 0.486–0.499), per-subject 0.363 ± 0.012;
pooled with oracle Euclidean alignment 0.469 ± 0.036, online-64 0.459 ± 0.033
(alignment does not help pooled EEGNet here). Phase 4 ALL DONE 23:17:15.

## 2026-09-25 (night) — Phase 6 checks: the release-day pipeline end to end

`scripts/train_sealed.sh <data_home> <study> <task> [overlay]` on the
`zhou2016_xsess` overlay, exactly as it will run on the Graz + BrainHero release:
cache → harness validation → blend weight on calibration data → candidate solver
with the chosen settings baked in as **defaults** (Codabench runs defaults) →
benchopt training → read-only inference replay → zip in the log folder.

| Time | Run | Result |
|---|---|---|
| 23:17:22–23:20:09 | clean recipe (router alignment, blend_calib) | ✅ 2 min 47 s; w = 0.75; train **0.770** = replay 0.770 (harness 0.778; the loader withholds the 2 % val slice) |
| ~23:21 | solver `adapt=online` on Zhou via benchopt | ❌ 0.743 < 0.770 without it (harness 0.797) |
| ~23:23 | same on Tangermann | ✅ 0.841 vs 0.805 (harness 0.837 vs 0.802): faithful with one training session |
| 23:24 | fix + Zhou re-check | ✅ 0.792 |
| 23:26:32–23:29:19 | `ADAPT=online` end to end | ✅ w = 0.5; train **0.792** = replay 0.792 |

**Incident (solver online mode on multi-session data).** Symptom: the online
variant scored below the clean one on Zhou. Cause: the harness centres *training*
windows per (subject, session), matching the session-local centring of online
test windows. The solver's `fit` saw only `subject_id` and centred each subject's
sessions pooled, a train/test mismatch that costs nothing with one training
session (Tangermann) and 2.7 points with two (Zhou). Fix: `fit` reads session ids
from the NeuralBench trigger table under the loader when present (logged as
`session_ids=yes`) and centres per (subject, session) in online mode.
Also found and fixed in `train_sealed.sh`: the blend weight was read from the
wrong folder, must be chosen under the same alignment the solver uses (w = 0.75
under the router vs 0.5 under online on Zhou; the wrong one cost 3 points), and
online runs now get their own tag and output folder.

## 2026-09-25 (night) — Phase 5: cross-dataset pre-training

`scripts/sealed_p5.sh`, `analysis/sealed_pretrain.py`. Channels: the 11 shared by
Dreyer, Tangermann and Scherer (Fz FC3 FCz FC4 C3 Cz C4 CP3 CPz CP4 Pz; all four
datasets share only 9, and Zhou's missing Fz/Pz are spline-interpolated, which
the weights show is sensible: Fz ≈ 1.08·FCz + 0.3·FC3/FC4 − 0.3·C-row). All
data already at 120 Hz. Pre-training pool: Dreyer 20,792 + Tangermann 5,184 +
Zhou 1,800 = 27,776 windows, one EEGNet-StepType trunk with a head per dataset,
10 % of subjects per dataset held out for early stopping.

| Time | Step | Estimate | Actual |
|---|---|---|---|
| 22:14:30–22:40:29 | raw pre-training × 3 seeds | 75 min | 26 min (4.4–15.7 min each; ES at epoch 18–28) |
| 22:40:29–22:53:18 | fine-tune (raw) + Riemann reference analogue | 30 min | 13 min |
| 22:53:27–23:21:24 | Euclidean-aligned pre-training × 3 | 75 min | 28 min |
| 23:21:24–23:37:19 | fine-tune with online-64 and with router alignment | 60 min | 16 min |

Total 83 min against the 3–4 h estimate (this CPU with 10 threads per lane is
faster on 11-channel windows than the 30-channel estimate assumed).

Held-out-subject accuracy of the pre-trained heads: raw Dreyer 0.77, Tangermann
0.43, Zhou 0.57; aligned pre-training raises Tangermann's held-out subject to 0.57.

Scherer 3-class (WORD, SUB, HAND) on the 11 channels, cell score, 3 seeds:

| EEGNet-StepType | none | router (EA, clean) | online-64 (EA, rule-dep.) |
|---|---|---|---|
| scratch, per-subject | 0.349 ± 0.016 | **0.466 ± 0.032** | 0.438 ± 0.014 |
| pre-trained, per-subject | 0.410 ± 0.005 | 0.420 ± 0.042 | 0.429 ± 0.046 |
| scratch, pooled | 0.438 ± 0.028 | 0.414 ± 0.021 | 0.430 ± 0.024 |
| pre-trained, pooled | 0.364 ± 0.034 | 0.342 ± 0.003 | 0.346 ± 0.016 |

Riemann analogue (filter-bank + broadband tangent space, 11 ch): reference point
from Scherer training covariances vs from MI + Scherer: pooled 0.436 vs 0.436,
per-subject 0.436 vs 0.431.

**Decision: no gain from cross-dataset pre-training at this scale.** Pre-training
helps only the weakest setup (per-subject EEGNet without alignment, +6.1, where
training from scratch often early-stops at epoch 1). In every other same-condition
comparison it is level or worse: per-subject with EA −4.6, pooled −7 to −8. Pooled
fine-tuning early-stops at epoch 1–3 (the new head overfits two held-out subjects'
validation loss), so a gentler protocol (frozen trunk first, lower LR) might narrow
the pooled gap. The best pre-trained number (0.429) is below the best from-scratch
EEGNet (0.466) and far below the 30-channel Riemann recipe (0.486–0.499 clean,
0.561 online). Adopting the 11-channel harmonised set would itself cost ~6 points.
The Riemann reference point makes no difference (tangent-space LDA is nearly
invariant to it). REVE / LaBraM stay the documented, not-run next step: 2026-09-25
feasibility was a 20–24 min frozen embedding pass for Dreyer and 45 min per epoch
full fine-tune on this CPU; weights not downloaded (needs the user's approval).
Side finding: per-subject Euclidean alignment helps EEGNet itself (0.349 → 0.466),
unlike the affine-invariant Riemann pipeline.

## 2026-09-25 (night) — Phase 6: deliverable

- `SEALED_RECIPE.md` complete: recipe (§ 1), evidence tables for every phase plus a
  paired subject-level bootstrap of the key claims (§ 2,
  `analysis/sealed_bootstrap.py`), what stays untested until Graz + BrainHero
  with an ablation plan (§ 3), the one-command path (§ 4), ranked next steps
  and a drafted organiser question (§ 5).
- `Riemann-Sealed` defaults now match the recipe (`personal="blend"`,
  `adapt="none"`); `train_sealed.sh` sets `blend_w` from calibration data and
  checks the baked-in defaults before training.
- Bootstrap honesty note: xDAWN's contribution on the 3 sealed-like classes is
  +0.2 points (95 % CI −2.4 to +3.1), not measurable. It stays for its MI gains,
  and it is the first block to drop if the release data disagrees.
- Whole weekend plan (Phases 0–6) done by ~23:55 Friday, well ahead of the
  8–12 h compute budget (~5.5 h wall, two lanes). Nothing uploaded.

## 2026-09-28 (evening) — Sprint: sealed-release readiness (brief `prompts/2026-09-28_release_readiness.md`)

A 12-hour sprint (16:55 → 04:55). The user asked for a workflow and gave no focus;
the value check picked release readiness (brief § 3). Claims checked in the code
before planning: the harness ignored context cells (the cache saved `context.npy`,
nothing read it), the blend weight was not leave-one-calibration-session-out (only
the last training session was held out), the sealed split (3 test sessions for
the evaluation participants only) was not expressible, and nothing above
30 ch / 120 Hz had been sized.

| Time | Step | Estimate | Actual | Result |
|---|---|---|---|---|
| 16:55–17:03 | Phase 0: orient, value check, brief | 50 min | 8 min | brief a44526a pushed; tracks page still "coming soon" (17:00) |
| 17:06–18:43 | Phase 1 builders (workflow): A mock study, B harness, C solver, E stream script | 3 h (whole Phase 1) | 1 h 37 min for the builders | ✅ all four done; gates below |
| 18:45– | Phase 1 resumed with inserted step L (fast LDA) | — | — | see the Phase 1 entry |

Also, in idle main-loop time: a thesis doc-consistency pass on its own branch
`docs/sprint0928-consistency` (853f9c5, pushed to personal, no PR). It fixes the
perf-loop confirm CV design (the +0.028 compared 5×2 vs 4×1 CV; like-for-like is
+0.031 paired), stale pooling/TF/window notes and CLAUDE.md branch names, and adds
stim-ledger errata.

## 2026-09-28 (evening) — Sprint Phase 3: online re-centring on a realistic test stream

Script `analysis/sealed_stream.py` (agent E); tables `logs/sprint0928_p3/RESULTS.md`
and `BOOTSTRAP.md`. Sanity: it reproduces the committed Phase 3 rows exactly (clean
/ online-64: Tangermann 0.802/0.837, Scherer 3-class 0.486/0.561, Zhou 0.778/0.797;
diff 0.0000). Its whitening is bit-identical to `sealed_run.aligned_data`.

| Time | Step | Estimate | Actual | Result |
|---|---|---|---|---|
| 17:15–17:19 | Zhou sanity + last/stream rows | ~5 min | 3.5 min | on |
| 17:19–17:31 | Tangermann | 6 min | 12 min (hit the 600 s cap once, resumed) | over 2×: choose_w CV + CPU shared with 3 agents |
| 17:31–17:58 | Scherer 3-class | 3.5 min | 27 min incl. one timeout | over: fixed with a calib-only weight CV and per-fold checkpoints |
| 18:00–18:09 | buffer-reset guard runs | 8.5 min | 9 min | on |

Online-64 minus clean (rule-dependent), cell score, paired subject bootstrap:

| proxy | recording order | subjects interleaved | fully shuffled |
|---|---|---|---|
| Tangermann (9 subj.) | +3.5 (+1.6, +5.4) | +3.0 (+1.2, +5.0) | +3.1 (+1.3, +4.9) |
| Scherer 3-class (9) | +7.5 (+4.3, +10.8) | +4.4 (+2.4, +6.4) | +4.6 (+1.9, +7.4) |
| Zhou last session (4) | +1.8 (−3.0, +4.8) | +0.5 (−2.7, +3.7) | +1.6 (−2.0, +5.1) |
| Zhou stream: calib s0, test s1+s2 (4) | +3.5 (−3.3, +10.3) | +2.2 (−2.3, +6.8) | +1.7 (−4.0, +7.4) |

**Decisions (pre-registered, brief § 5 Phase 3).**
- **Order robustness: NOT order-robust.** With subjects interleaved, online-64 keeps
  85 / 58 / 27 / 65 % of its recording-order gain (rule: ≥ 75 % on every proxy).
  The gain shrinks but stays positive, and it stays significant on Tangermann and
  Scherer. Mechanism: in recording order a 64-window batch is mostly one subject,
  so the buffer is essentially the current batch, including look-ahead within it.
  Interleaved, each subject gets ~7 windows per batch. Consequence: the organiser
  question must also ask how test windows are ordered and batched, and the recipe
  quotes the interleaved gain (+3 to +4.4) as the conservative figure.
- **Session-boundary lag: none.** The second-session gain is < 0 for 2/4 Zhou
  subjects (trigger ≥ 3). The first 32 post-boundary windows score 3.3 points
  *above* that session's mean (trigger: a drop > 5).
- **Buffer-reset guard: no gain.** −0.1 (−0.5, +0.3). Not adopted. It never fired
  at a session boundary.
- Descriptive only (choosing N on test sessions would be test tuning): N = 32 kept
  the most gain under interleaving on all 4 proxies. On release day, choose N on
  the replica split under the announced order.

**New risk found.** With one calibration session, the log-PSD router's accuracy
falls to 0.718 on the next session and 0.353 on the one after (Zhou stream;
chance 0.25). The harness's fallback threshold saturates at 1.0 (0.49 of windows
fall back); the solver caps it at 0.5. The sealed data gives three calibration
sessions, but the runbook must report router accuracy per test session on the
replica split.

## 2026-09-28 (evening) — Sprint Phase 1: build (workflow `sealed-readiness-build`)

The workflow ran 9 agents. Builders worked in parallel on disjoint files: A (mock
sealed study + benchopt dataset), B (harness), C (solver), E (Phase 3 stream
script). The main loop then inserted L (fast LDA) when B and C found a blocking
bottleneck. After that came D (integration), two adversarial reviewers (R1:
leakage/splits; R2: regression/agreement) and F (fixed all 9 review findings).

| Time | Step | Estimate | Actual | Result |
|---|---|---|---|---|
| 17:06–18:43 | builders A, B, C, E | (3 h for all of Phase 1) | 1 h 37 min | ✅ |
| 18:43–18:45 | workflow stopped and resumed with step L inserted (A–E cached) | — | 2 min | — |
| 18:45–19:37 | L: fast LDA, router cap, --wvariant calib, 500 Hz harness memory | ~45 min | 52 min | ✅ |
| 19:37–20:35 | D: train_sealed.sh, release_ablations.sh, release_summarize.py, mock e2e | ~1 h | 58 min | ✅ (1 ablation step left to Phase 2: over the 600 s agent cap) |
| 20:35–21:00 | R1 + R2 | ~30 min | 25 min | pass_with_fixes: 4 major, 5 minor findings |
| 21:00–21:55 | F: fixes + re-gate | ~30 min | 55 min | ✅ all 9 fixed, all 4 gates pass |
| **Phase 1** | | **3 h** | **4 h 49 min (1.6×)** | the overrun is the inserted L step plus the review fixes |

**What exists now (all opt-in; defaults reproduce every committed number
bit-for-bit, re-checked independently by B, L, R2 and F):**
- **Harness** (`xsess_lib`, `sealed_run`, `sealed_personal`, `xsess_cache`):
  - context cells everywhere;
  - `--split last | calib:K | replica:K`, `--test_subjects`;
  - hidden (split 2) rows of a release cache never enter training;
  - `--wcv loso` (leave-one-calibration-session-out), `--chans`, `--pool test`;
  - `router-psdctx` alignment, `--router_cap 0.5` (matches the solver);
  - `--wvariant calib` (1.6× faster), `--mmap`;
  - `xsess_cache.py --picks`, and `ch_types` / release-structure fields in meta.
- **Solver** (`riemann_sealed.py`):
  - float32 storage with chunked float64 maths;
  - `blend_w="auto"` (the same LOSO as the harness; the weight is stored in the
    joblib);
  - `chans`; stage timings and RSS prints;
  - a trigger-table sanity check.
- **Fast shrinkage LDA** (a Cholesky solve instead of sklearn's SVD lstsq, above
  4,000 features; always a plain sklearn object; predict_proba within 1e-9):
  - one 5,073-feature fit: 38.7 s → 3.8 s;
  - it makes the 43-channel recipe feasible, where 21 LDAs × 7 LOSO folds would
    otherwise take hours.
- **Mock sealed study** (`analysis/mock_sealed.py`, `datasets/mock_sealed.py`):
  - 20 subjects × 6 sessions × 2 contexts, 43 EEG + 2 EMG + 2 EOG, 3 classes;
  - `_s` / `_120` / `_500` sizes;
  - injected session drift, a context shift, a drifting WORD↔EMG confound and a
    stable CALC↔EOG cue.
- **Scripts:**
  - `train_sealed.sh`: SPLIT / TEST_SUBJECTS / WCV / CHANS / DATASET /
    BLEND_W=auto / RUN_NAME / STOP_AFTER; guards against stale rows and changed
    settings; refuses a sealed-looking cache without SPLIT; kills its step on a
    signal;
  - `release_ablations.sh` (two lanes, every SEALED_RECIPE § 3 ablation);
  - `release_summarize.py` (applies the pre-registered release-day rules with a
    paired bootstrap).

**Gates (brief § 5, Phase 1).**
- (i) Regression: zhou2016 15/15 and scherer 3-class 9/9 committed rows match
  with |d| = 0. The zhou2016_xsess default flow gives train 0.770 = replay 0.770.
- (ii) Mock `[data]` line: 20 subjects × 6 sessions × 2 contexts, 47 → 43 ch,
  60 test cells.
- (iii) Mock: benchopt train = read-only replay exactly (0.483333); gap to the
  harness 0.0000.
- (iv) Solver `blend_w="auto"` = harness `--wcv loso`:
  - fb=0: w 0.25, CV equal to 4 dp;
  - fb=1: w 0.75, folds bit-identical (fold slices).

**Machinery rules (mock, verified by construction, not recipe evidence).**
- (a) 60 cells with per-context columns.
- (b) Decided on the **release-day split** (calib:3, test = the 10 fully labelled
  participants, hidden rows excluded): EEG+EMG −4.2 points (CI −8.3, −0.4), so
  the rule rejects EMG. On replica:3 it had shown +0.8, because training then held
  the decoupled later sessions. That was review finding R1-3, and the reason the
  release-day DECISIONS must use calib:3.
- (c) `router-psdctx` recovers the injected context shift: +6.4 (CI +2.8,
  +10.0).

**Found and fixed, each a release-day failure that would have gone unnoticed:**
- contexts were ignored;
- the blend weight was not LOSO;
- the sealed split could not be expressed;
- LDA fit time at 43 ch: hours;
- the mock's hidden rows leaked into training on calib:3;
- channel typing: harness and solver disagreed on real caches;
- train_sealed.sh silently validated on the proxy split for a real cache;
- a misaligned trigger table was accepted silently;
- stale harness rows were reused after a cache rebuild.

**Open (goes to Phase 4):** the solver has no (subject, context) alignment yet.
On the mock the pre-registered rule adopts `router-psdctx`, and train_sealed.sh
now refuses to bake it rather than silently train without it.

C noted that on zhou2016 LOSO picks w = 0.5, where the committed last-session
rule picks 0.75 (benchopt test 0.742 vs 0.770; 4 subjects, within noise).
`auto` is not the default.

## 2026-09-28 (night) — Sprint Phase 2: dress rehearsal at realistic size (mock)

Scripts `scripts/sprint0928_p2.sh` (two lanes, full-size 120 Hz mock) and
`scripts/sprint0928_p2b.sh` (solver sizing at 500 Hz). Logs:
`logs/sealed_sprint0928_p2*/`, `logs/train_sealed_mock_sealed_120*/` and
`logs/sealed_sprint0928_abl120/RESULTS.md`. Every run shared the CPU with a
Phase 4a agent, and 2b also shared it with lane A, so all times are upper bounds.

| Time | Step | Estimate | Actual | Result |
|---|---|---|---|---|
| 21:55–22:11 | mock_sealed_s `wcv_loso` ablation (left over from Phase 1) | 8 min | 16 min (2×: 3 jobs + agent on 20 threads; progressing, not stalled) | ✅ w 0.75 = `last` |
| 21:55–22:54 | lane B: 6 ablations on mock_sealed_120, **release-day split** (calib:3, test = 10 fully labelled participants, 3,600 hidden rows excluded) | 90 min | 58 min | ✅ table below |
| 22:11–22:52 | lane A: `train_sealed.sh` default flow, mock_sealed_120 | 35 min | 41 min | ✅ train = replay = harness = 0.614722 |
| 22:52–23:53 | lane A: `train_sealed.sh` BLEND_W=auto, mock_sealed_120 (harness `--wcv loso` 38 min, solver fit with its own LOSO ~22 min) | 70 min | 61 min | ✅ solver w 0.75 = harness LOSO w 0.75 (MATCH); train = replay = harness = 0.614722 |
| 22:54–23:28 | 2b: solver fit with `blend_w="auto"` at **500 Hz** (14,400 × 47 × 2000; 43 ch kept), 2 replays | 60 min | 34 min | ✅ sizing rule passes |

**Sizing (500 Hz, the worst case).** Fit 30.4 min: collect 8 s, router 82 s,
alignment 83 s, features 265 s, LDA 75 s, 6-fold weight search 1,313 s. Peak RSS
12.4 GB. Read-only replay of all 3,600 test windows took 77 s at 10 threads and
66 s at 2 threads, with 2.3 GB RSS. Train = replay (10 threads) = replay
(2 threads) = 0.612778 exactly. The solver chose w = 0.75 by LOSO. The rule (fit
≤ 45 min, RSS ≤ 20 GB, predict ≤ 10 / 30 min) **passes** with a wide margin. The
sealed data can arrive at 500 Hz and the recipe still fits the budget.

**Machinery rules at full size on the release-day split** (mock ground truth,
not recipe evidence). `release_summarize.py` DECISIONS, cell score vs the recipe
(0.604), paired bootstrap over the 10 test subjects:

| ablation | Δ points (95 % CI) | subjects up | pre-registered rule → decision |
|---|---|---|---|
| EEG + EOG | +1.0 (+0.1, +1.9) | 7/10 | below the +2 bar → EEG only |
| EEG + EMG (drifting confound) | **−3.7 (−5.2, −2.4)** | 0/10 | rejected: the trap is caught |
| all 47 channels | −2.8 (−4.7, −0.6) | 2/10 | rejected |
| router-psdctx (injected context shift) | **+5.9 (+3.8, +8.1)** | 9/10 | adopted (by construction) |
| no xDAWN | +4.1 (+2.6, +6.0) | 10/10 | drop xDAWN (a mock property) |

The summarizer prints the combined `train_sealed.sh` command for the adopted
settings, plus a NOTE that the solver cannot yet deploy router-psdctx (Phase 4a
fixes this). Rules (a) and (b) of the brief hold at full size, and (c) holds.

**Incident (22:10).** The Phase 2 watchdog reported both lanes STALLED at
15 min. Cause: each lane step wraps an inner script in `/usr/bin/time -v`, so
the outer step log only grows when the step ends. The inner logs showed
healthy progress (LOSO fold 5/6, ~1.4 min per fold). Fix: watch the inner
log dirs (`sealed_sprint0928_abl120`) instead.

**Note.** `scripts/summarize_runs.py` prints a hard-coded "Dreyer 2023 test
split" header on every RESULTS.md, including the mock runs. This is cosmetic;
the header is to be fixed in Phase 4.

## 2026-09-28/29 (night) — Sprint Phase 4a/4b: deployable context alignment, claims check, EDA script

| Time | Step | Estimate | Actual | Result |
|---|---|---|---|---|
| 21:57–23:33 | 4a G: `align="subject_context"` in Riemann-Sealed, developed on a scratch copy and swapped in atomically after the running job had frozen its candidate | ~1 h | 1 h 36 min (≈37 min of it waiting for the swap window) | ✅ 4 gates |
| 23:33–00:10 | 4a R3: adversarial review | ~30 min | 37 min | pass_with_fixes (1 minor) |
| 00:10–00:43 | 4a F2: fix | ~20 min | 33 min | ✅ bit-identical defaults |
| 23:40–00:02 | 4b V2 claims check + V3 `analysis/release_eda.py` | ~1 h | 22 min | 96 claims checked, 30 mismatches, 16 gaps; EDA script ✅ |

**4a: context alignment deploys.** The solver option `align="subject_context"` is
a log-PSD router over (subject, context) pairs, with whitening per pair from
training data. The personal LDAs stay per subject. Pairs with fewer than
`ctx_min=16` windows keep the subject's W, and with no context column it falls
back to `align="subject"`. `adapt="online"` together with it raises
NotImplementedError. `train_sealed.sh` bakes it for `RECIPE_ALIGN=router-psdctx:<kind>`.

Gates:
- defaults bit-identical to before (zhou2016, mock);
- on mock_sealed_s the benchopt train equals the read-only replay, and both
  equal the harness `router-psdctx` row (0.547222; pair accuracy 0.9944 on both
  sides);
- `blend_w="auto"` fold scores are bit-identical to the harness's `choose_w`
  under psdctx;
- the joblib holds only numpy/sklearn/pyriemann objects.

R3's one minor finding: the trigger-table row-order check caught only misorder
*across* subjects. F2 added a record_id/onset consistency check. It is used when
the loader supplies those fields; NeuralBench's supplies only subject_id, and
that limitation is documented. The tell is the router OOF pair accuracy
falling well below `release_eda.py`'s.

**4b: the claims check paid for itself.** RELEASE_DAY.md was drafted before most
of the code existed, and 30 of its 96 checked claims were wrong or misleading.
Among them:
- the loader-inspection snippet failed on today's benchopt API;
- `replica:3` was not refused on real data;
- `train_sealed.sh` zipped whatever the gate said;
- the final run recomputed the harness steps under a new RUN_NAME;
- in WSL `git stash` would have touched 82 CRLF-only files;
- the `--eval_subjects` default is inverted under the replica overlay;
- a timing figure was misread.

All of these went to Phase 4c.

`analysis/release_eda.py` produces:
- cells and class balance;
- the context layout;
- router accuracy **per test session** and per context (it reproduces E's 0.718 → 0.353 on Zhou calib:1 exactly);
- session drift (it detects the mock's injected drift: 3–5× the calibration baseline);
- an evoked-response check (positive on Zhou, negative on the mock).

It runs in 2.6 min on the full-size 120 Hz mock.

## 2026-09-29 (night) — Sprint Phase 4c: runbook fixed and verified literally on the mock

The workflow `runbook-fix-and-verify` ran S (scripts) and W (runbook rewrite) in
parallel. Then V1 followed RELEASE_DAY.md **literally** on the mock, and F4 fixed
what V1 found and filled RELEASE_DAY § 9.

| Time | Step | Estimate | Actual | Result |
|---|---|---|---|---|
| 23:54–00:38 | S: scripts; W: RELEASE_DAY.md + SEALED_RECIPE.md from the 30 mismatches / 16 gaps | ~45 min | 44 min | ✅ S 5/5 checks; W all items mapped |
| 00:38–01:20 | V1: literal run on mock_sealed_s (§§ 0, 2, 3.3 substitute, 4, 5, 6, 7 steps 1–3, 8) | ~75 min | 42 min | 4 FAIL, 7 AMBIGUOUS, rest PASS; every gate number exact |
| 01:20–01:38 | F4: fixes + § 9 | ~30 min | 18 min | ✅ all FAIL/AMBIGUOUS resolved, each affected step re-run |

**What S changed.**
- `train_sealed.sh` enforces rule 7:
  - no zip on a failed replay, on train ≠ replay, or (with `GATE=replica`) on a
    harness gap or solver/harness weight mismatch;
  - `FORCE_ZIP=1` overrides only the weight;
  - the gate reads benchopt's own parquet names.
- `HARNESS_FROM` reuses harness rows under a new RUN_NAME: 0 fits, 2–16 s
  instead of minutes.
- `replica:K` is refused on any non-mock cache, by the harness and both scripts.
- `datasets/mock_sealed.py split=replica_full` lets the release-day replica run
  through benchopt on the mock.
- `summarize_runs.py` no longer prints the Dreyer header on other studies.

**What V1 found** (each would have bitten on release day):
- § 0's regression would have overwritten the frozen
  `outputs/Riemann-Sealed-Cand/`. `train_sealed.sh` now writes to
  `<logdir>/submission` whenever RUN_NAME is set.
- `pip freeze` silently produced empty files, because the uv venv has no pip:
  now `uv pip freeze`.
- Through `wsl.exe`, stderr overwrote the start of stdout in the captured
  output, which could hide a gate line: the call template now starts with
  `exec 2>&1`.
- `monitor_loop.sh` never surfaced ERROR / STOPPED / not-done rows: it now
  prints them and exits.
- The summarizer's wcv default contradicted rule 4, and the § 7 block carried
  another cache's decision.

**Verified end to end on the mock** (RELEASE_DAY § 9): replica step 1
(calib:3, `split=replica_full`, `align="subject_context"`, `blend_w="auto"`)
gives train = replay = harness = 0.602778, weight MATCH. The final step 2
(`HARNESS_FROM`, `GATE=final`) gives train = replay = 0.547222. The zip check
confirms the baked defaults. The zhou2016_xsess regression still gives
0.770000 = replay.

**Could not be rehearsed** (stated in RELEASE_DAY): the organisers' loader, real
NeuralBench overlays (`<name>_xsess` / `<name>_all`) and their registration, and
a cache build at release size.

## 2026-09-29 (night) — Sprint Phase 5: final adversarial review + fixes

The optional Phase 5 of the brief (the Riemann + EEGNet ensemble) needed 2.5 h
and did not fit before the 03:25 reserve. The time went instead to a final
review of every line changed tonight (a44526a..HEAD, ~5,000 lines), through three
independent lenses, followed by fixes.

| Time | Step | Estimate | Actual | Result |
|---|---|---|---|---|
| 01:40–02:26 | review, 3 lenses (ML correctness/leakage, script robustness, Codabench deployment path) | ~45 min | 46 min | 2 + 8 + 3 verified findings (no blocker) |
| 02:11–02:26 | fixes X1 (solver) + X2 (scripts) | ≤ 45 min | 15 min | ✅ all gates |

**Fixed:**
- **Solver (major): the channel pick was positional and never checked against
  channel names.** A different channel order at scoring time would have run
  silently near chance (0.335 vs 0.475 on the mock). The joblib now stores the
  training channel names. `load_model` reorders by name, drops extras, and raises
  on a missing channel. Defaults are bit-identical (max |dP| = 0), and benchopt
  train = replay = 0.483333 as before.
- **Solver (minor):** `ch_names=None` no longer crashes.
- **Scripts (major):**
  - `release_ablations.sh` exited 0 when a requested step crashed or was
    mistyped, and the summarizer then silently kept the default. That could
    have flipped a release-day decision, e.g. a crashed `al_ctx`. It now
    validates STEPS/LANES and exits 1 on any requested step without `.done`;
    the summarizer prints `MISSING`.
  - `train_sealed.sh` could zip a different model than the one that passed the
    gates. It now hashes the submission folder at the end of training and
    re-checks the hash before the replay and before zipping.
- **Scripts (minor):** STOP_AFTER validation, a HARNESS_FROM source-gate check,
  `monitor_loop.sh` (stale ALL DONE, paths with spaces, the lanes pattern) and
  two-decimal deltas in the summarizer.
- zhou2016_xsess regression: train = replay = 0.770000, gap 0.0083 OK.

**Not fixed; documented for the next session:**
- **LOSO weight-search bias (major).** Each held-out session is whitened with a
  reference that includes its own unlabelled windows: ~2 points of CV bias on
  Zhou, < 1 on the mock, never flipping a complete run's weight. If a
  (subject, context) pair lives in one calibration session, that fold is
  oracle-aligned (10–17 points).
  - Interim guard in RELEASE_DAY rule 4.
  - The strict per-fold fix (harness + solver, bit-identical) is SEALED_RECIPE
    § 5's first code step.
- **The final all-data run re-chose w over folds the replica never validated**
  (minor): RELEASE_DAY § 7 step 2 now bakes step 1's weight.
- **Deployment risk (major, needs the user):** no sklearn/pyriemann joblib has
  ever run on the scoring image, and scikit-learn is unpinned. A warm-up upload
  is the test (SUBMISSIONS.md).

Verified fine: see the reviewers' checked_ok lists in the workflow journal. Among
them:
- the fast-LDA degenerate cases, with the correct lstsq fallback;
- the router-threshold rules, moot under the 0.5 cap;
- the organisers' own ingestion + scoring programs on tonight's zips, scores
  exact;
- CPU-only and 1–4 thread inference, identical;
- batch size 1;
- training determinism across thread counts.

## 2026-09-29 (evening) — Sprint: temporal and spatial feature blocks (brief `prompts/2026-09-29_temporal_spatial_features.md`)

The user asked for a 12-hour workflow sprint (17:46 → 05:45) with the focus
"investigate temporal features and improve spatial features so the differences
in brain activation can be captured". The value check kept that focus and aimed
it at the Track 2 sealed recipe:
- only the sealed phase ranks;
- the filter bank carries most of the 3-class signal (Phase 4, 2026-09-25);
- no temporal-structure or spatial-structure variant had been measured, only
  whole-block ablations;
- nothing blocks it: the data is not released and no approvals are needed.

Runners-up were the Riemann + EEGNet ensemble and the strict LOSO fix (the
chained fallback). Every covariance in the recipe spans the whole 4 s window.

### Phase 0/1 — harness hook, build workflow, baseline and controls

| Time | Step | Estimate | Actual | Result |
|---|---|---|---|---|
| 17:46–17:52 | orient + value check | ≤ 15 min | 6 min | proceed on the user's focus; the tracks page still said "coming soon"; sleep disabled on AC/DC |
| 17:52–18:05 | brief, `analysis/xfeat.py` hook (`x=` / `sl=` spec keys), `xfeat_selftest.py`, lane scripts, summarizer, `--split first` | 20 min | 13 min | ✅ 8c50541, 7e0d008; spec without `x=` still builds the plain `RiemannModel` |
| 17:55:10–18:22:02 | f1: baseline reproduction + no-new-code candidates (slow block, CAR, Laplacian), 4 proxies | ~25 min | 27 min, on estimate (s5 1.45× over: the build agents shared the CPU) | ✅ baseline **bit-exact** vs `logs/sealed_p4` (4 configs, per-subject max \|Δ\| = 0.0) |
| 17:59– | workflow `sprint0929-feature-build` (5 agents: temporal / spatial builders + adversarial verifiers, activation EDA) | 75 min | temporal verified 18:22 | 7 temporal blocks: no code bugs; independent re-implementations match exactly; leakage checks pass |

**Controls (Phase 2 screen metric; Δ in points vs the recipe union, paired over
subjects, 95 % bootstrap CI):**

| spec | Scherer 3-cl. Δ screen (router) | Scherer 3-cl. Δ none | Tangermann | Scherer 5-cl. | Zhou |
|---|---|---|---|---|---|
| slow block (`sl=1`) | −0.4 (−2.5, +1.7) | +1.2 (−0.7, +2.7) | +0.8 (−0.1, +1.6) | +0.2 | +0.1 |
| CAR | −0.1 (−2.0, +1.9) | −0.7 | **−1.8 (−2.5, −0.9)** | +0.2 | +2.3 (−0.5, +5.0) |
| Laplacian (CSD) | −1.8 (−4.6, +1.2) | −3.0 (−5.9, +0.2) | **−3.4 (−5.4, −1.6)** | +0.5 | +1.8 |

- **No gain from the slow block or re-referencing.** The brief predicted ≈ 0
  for re-referencing: TS + shrinkage LDA is affine-invariant in exact
  arithmetic. The Laplacian in fact *hurts* on the MI proxy. The OAS covariance
  shrinkage toward μI is not affine-invariant, and the CSD amplifies
  high-spatial-frequency noise, so the invariance holds only approximately. Fixed
  spatial filters stay out of the recipe.

**Phase 1 results (workflow `sprint0929-feature-build`, 17:59–18:40, 41 min vs 75
estimated; 5 agents, 0 errors).**
- **12 blocks built and adversarially verified:**
  - temporal (`analysis/xfeat_temporal.py`): tseg2/3, acm3x2, acm2x4, fb8,
    fbd, bpt4;
  - spatial (`analysis/xfeat_spatial.py`): fblv, fbrlv, reg, csp8, icoh.
- **Verification:** no code bugs. Independent re-implementations match
  exactly, and features do not change when the test batch is split, permuted,
  transformed one window at a time or mixed with junk windows. Only docstring
  corrections were made.
- **Notes from the verifiers:**
  - fbrlv is an exact linear re-parametrisation of fblv given the base
    log-var block; count them as one family.
  - reg's regions are lopsided on Scherer (the merge rule is not
    left/right symmetric). Kept as built.
  - Every `x=` config costs ≥ 2× the baseline, because the wrapper refits the
    base model and the union LDA.
  - tseg3 at 43 ch makes a 16.4 k-feature union (about 2.2 GB per d × d
    matrix), which needs the Phase 4 sizing gate.
- **Activation EDA** (`reports/features_0929/activation_eda.md`, training
  sessions only, 530 s):
  - The classes differ strongly **within** subjects but idiosyncratically:
    ANOVA F up to ~6, while the group t is ≤ 3.6.
  - Alpha information is posterior-right and emerges after ~1.5 s. Beta is a
    WORD decrease at F3. 30–45 Hz is near chance, and there is no EMG
    signature.
  - Delta/theta information is early (0–2 s) and frontal (F7/F8/AFz): likely
    eye movements, with indirect evidence only (no EOG on the proxy).
  - TS beats band power by 5–14 points (4-band TS 0.684 vs power 0.546,
    within-session CV).
  - Time-resolved TS (4 × 1 s) beats whole-window TS in delta/theta (+6 to
    +7) and in alpha (+4.8, n.s.).
  - A class-specific evoked response exists only in the first second.

**EDA-motivated candidates, registered here at 18:45 before they run (brief
§ 4 allows 2; same rules as the others):**
1. `tcut1000`: the FB4 TS over two segments, [0, 1 s) and [1 s, end). The
   first second is a different regime (cue-evoked burst, early ERD) from the
   sustained task period where the alpha/beta class information lives. It
   separates them at 2 blocks' cost instead of tseg's equal cuts.
2. `fbfrom1000` (spec `blocks=xdawn+broad+logvar,x=fbfrom1000`): a control.
   The recipe's FB TS computed on 1 s → end only, replacing the full-window FB.
   It measures how much of the FB's accuracy depends on the cue second. It is
   information for release-day decisions (the sealed windows' cue content is
   unknown) and will not be adopted on the screen score alone.

### Phase 2 — screen (18:23–~20:35; `scripts/sprint0929_f2.sh` FAM=T, `sprint0929_f2b.sh` FAM=SE)

| Time | Step | Estimate | Actual | Result |
|---|---|---|---|---|
| 18:23:00–19:27:51 | f2 FAM=T: 7 temporal blocks × 4 proxies | 65 min | 65 min, on estimate | tseg3, bpt4, tseg2, acm3x2, fbd pass |
| 19:13–19:23 | lane A idle 10 min | — | — | **incident:** the watchdog pipe `… \| tr -d '\r' \| grep` block-buffered in `tr`, so END rows never reached the Monitor. Fixed with `stdbuf -oL tr`. The f2b guard was narrowed to its own lane so lane A could start before lane B's s5_T ended |
| 19:23:43– | f2b FAM=SE lane A; lane B from 19:27:56 | 55 / 60 min | (see Phase 2 results) | |
| 19:31–19:36 | solver port gates (`analysis/xblocks_gate.py`) | ~8 min | 4 min 54 s, under | defaults max \|dP\| = 0.0; harness parity for tseg3_bpt4: max \|dF\| = 0, max \|dP\| = 0 |

**Decision on the Phase 3 set (recorded 20:16, before Phase 3 and before the
Scherer 5-class rows of the S/E blocks).**
- By Scherer 3-class Δ the ranking is: tseg3 +3.3, tcut1000 +2.8, bpt4 +2.6,
  icoh +2.6 (the tie goes to fewer features: bpt4).
- The literal top-3 would be tseg3, tcut1000 and bpt4, with icoh only inside
  the union.
- **Deviation:** tcut1000 is the same block family as tseg3: filter-bank
  covariances over time segments, differing only in where the cuts fall.
  Advancing both spends a Phase 3 slot, about an hour of compute, on a
  near-duplicate, against the premise of the top-3 rule (test the strongest
  *distinct* candidates). So the family advances once, with its better member
  (tseg3), and tcut1000's screen result is reported as corroboration.
- **Phase 3 set:**
  - `tseg3`, `bpt4` and `icoh` (icoh subject to its replication check once the
    Scherer 5-class rows land);
  - the pre-registered union of the best temporal and best spatial blocks,
    `tseg3+icoh`.

**Phase 2 results** (f2b ended 20:40:02; lane A 51 min vs 55 estimated, lane B 72
vs 60, 1.2× over; full tables in `logs/sealed_f0929/RESULTS_screen.md`). The
screen metric is the mean of pooled and persubject under the clean router
alignment. Δ is in points vs the recipe union, paired over subjects, with a
95 % bootstrap CI.

| block | what | Scherer 3-cl. Δ | Tangermann | Scherer 5-cl. | Zhou | verdict |
|---|---|---|---|---|---|---|
| **tseg3** | FB covariances of 3 time segments → TS | **+3.3 (+1.5, +5.7) 9/9** | −0.1 | +2.4 (+1.0, +4.0) | −1.2 | pass |
| tcut1000 | FB covariances of [0, 1 s) and [1 s, end) | +2.8 (+1.7, +4.5) 9/9 | +0.2 | +2.1 (+0.9, +3.5) | −1.5 | pass (same family as tseg3) |
| **bpt4** | band-power time course (4 bands × 4 bins × C) | **+2.6 (+0.4, +5.0) 8/9** | +1.1 | +2.3 (+1.5, +3.1) | +2.9 (+0.5, +5.5) 4/4 | pass; the most consistent |
| **icoh** | imaginary coherence per band (lagged connectivity) | **+2.6 (+0.9, +4.2) 7/9** | +1.2 | +1.7 (+0.6, +2.8) | +0.3 | pass; the only spatial block that helps |
| tseg2 | 2 segments | +1.7 | −0.4 | +0.5 | −1.2 | pass |
| acm3x2 | augmented (time-delay) covariance | +1.5 | −1.7 (−3.8, −0.0) | +1.6 | +0.4 | pass (hurts MI) |
| fbd | 1–4 Hz band TS | +1.0 | −0.4 | +0.8 | +0.3 | pass (marginal) |
| fbfrom1000 | control: FB without the cue second | +1.6 | +0.5 | +0.8 | +1.0 | the FB does **not** depend on the cue second |
| fb8, acm2x4, csp8, fblv, fbrlv, reg, slow, CAR, Laplacian | | −1.8 to +0.6 | | | | no gain |

- **Temporal structure is where the missing information was.** Splitting the
  4 s window in time, as covariance structure (tseg / tcut) or as band power
  (bpt4), gains 2.6–3.3 points on the sealed-like classes.
  - The biggest gains are without alignment (tseg3 +6.7 pooled + persubject).
    Router whitening absorbs part of it.
- **The one spatial gain is connectivity, not topography.** Imaginary
  coherence, the lagged coupling that zero-lag covariances cannot represent,
  helps (+2.6, and on all 3 other proxies ≥ 0).
- **Spatial topography blocks add nothing.** CSP, band-power topography,
  regional covariances and re-referencing gain nothing over the tangent space,
  consistent with the affine-invariance argument.
- The EDA's prediction for bpt4 ("≈ 0 on top of the FB TS") was wrong
  cross-session; within-session CV rankings did not transfer.

**Added 21:30, before any number for this union exists:** if `bpt4` and
`icoh` are both ADOPT in Phase 3, their union `bpt4+icoh` runs through the same
Phase 3 configs. It is post-hoc and labelled so. The recommended default is
the union only if **both** of these hold:
- on Scherer 3-class blend_calib it beats the better single block by ≥ +1.0;
- on the mean of the other three proxies it is not worse than that block.

Otherwise the default is the better single block. Either way the release-day
ablation tests the chosen default's blocks against the recipe on the replica.

Sizing smoke test (21:15–21:25, `analysis/xblocks_sizing.py`, mock_sealed_s,
bpt4_icoh, fixed w, 2 threads while two lanes ran): 9,373 features. The LDA
stage took 463 s of 589: 21 LDAs, about 92 % of the cost in the 20
per-subject fits, which solve a d × d system from ~100 windows each. So:
- bpt4 (+688 features at 43 ch) should cost ~nothing at the sealed size;
- icoh (+3,612) is expected to add ~5–10 min to the 30.4 min 500 Hz auto fit;
- tseg3 (+11,352) is expected to fail the 1.5× gate without a dual (n < d)
  LDA solve.

### Phase 3 — confirmation under the deployed recipe (20:14–~23:45; `scripts/sprint0929_f3.sh`)

`sealed_personal.py` blend_calib, router ids, clean router alignment, on the 4
proxies (tags `f0929p`, `f0929p5`), plus the reverse-time replication
(`sealed_run --split first`, tag `f0929r`).

| Time | Step | Estimate | Actual | Result |
|---|---|---|---|---|
| 20:14:46–20:26:53 | baseline, Scherer 3-cl. | 20 min | 12 min | blend_calib **0.486** = SEALED_RECIPE (pooled 0.452, calib 0.473, per-subject 0.486, blend 0.499: all reproduced) |
| 20:26–21:35 | lane A: tseg3 / bpt4 / icoh / union on Scherer 3-cl., then Zhou ×5 | ~70 min | 69 min | |
| 21:35:57–22:10:36 | reverse split, 5 specs × 4 configs | 35 min | 35 min, on estimate | every Δ ≥ 0: no downgrades |
| 22:10:41 | lane A stopped by hand before the online-64 info steps | — | — | frees the slot for the 500 Hz sizing; the online steps are resumable |
| 20:40:06– | lane B: Tangermann ×5, then Scherer 5-cl. ×5 | ~180 min | (running) | the steps slowed by ~1.3× while the sizing ran |

**Blend_calib Δ vs the recipe** (points, paired over subjects, 95 % CI;
`logs/sealed_f0929/RESULTS_confirm.md`):

| block | Scherer 3-cl. | Tangermann | Scherer 5-cl. | Zhou | reverse split (screen) | verdict |
|---|---|---|---|---|---|---|
| **bpt4** | **+4.4 (+2.0, +6.7) 8/9** | +1.5 (+0.1, +3.1) 6/9 | +2.7 (+1.2, +4.6) 8/9 | +2.2 (−1.5, +6.3) 3/4 | +1.4 (−0.4, +3.3) | **ADOPT** |
| icoh | +2.2 (+0.3, +4.4) 5/9 | +1.3 (−0.2, +2.8) | +2.0 (+0.6, +3.6) 8/9 | **−3.5** (−8.0, +1.2) 1/4 | +2.2 (+0.6, +4.7) | **PROMISING** (the Zhou guard) |
| tseg3 | +2.0 (−0.0, +4.1) 4/9 | −0.2 | +1.0 | −1.5 | +3.4 (+1.5, +5.5) | **NO GAIN** (the CI touches 0; 2 of 3 others < 0) |
| tseg3+icoh | +3.2 (+1.6, +5.1) 8/9 | +0.5 | (running) | −1.8 | +4.2 (+1.8, +6.9) | (pending) |

- **bpt4 is the new recipe default:** `RECIPE_SPEC=riemann:xd=1,fb=1,x=bpt4`.
  It is the band-power time course, 4 bands × 4 one-second bins × channels:
  688 extra features at 43 ch.
  - It gains on every proxy, and its gain *grows* under the deployed
    blend_calib (+2.6 in the screen → +4.4).
  - Release day still decides it: step `xb` of `release_ablations.sh` drops
    the blocks if the replica shows ≥ +1.0 without them (RELEASE_DAY rule 3b).
- **icoh becomes the release-day ablation `xa` (`XB=icoh`),** added on top of
  bpt4 only if the replica shows ≥ +1.0 and no context < −1.0.
- **tseg3: no gain under the deployed recipe.** Its large pooled/none gains
  (+6.7) and the reverse split (+3.4) do not survive personalisation and the
  MI proxies. Router whitening plus the per-subject LDA already capture most of
  what the segment covariances add. It is also the most expensive
  (+11.4 k features at 43 ch) and would need a dual LDA solve to pass sizing.
- **The 21:30 union rule is moot.** icoh is not ADOPT, so bpt4+icoh is not
  run as a pre-registered candidate. It runs below as post-hoc information
  only.

**Phase 3 final** (lane B ended 23:39:32; `logs/sealed_f0929/RESULTS_confirm.md`).
The pre-registered union **tseg3+icoh** also meets ADOPT by the letter:
- Scherer 3-cl. +3.2 (+1.6, +5.1) 8/9;
- Tangermann +0.5, Scherer 5-cl. +1.5, Zhou −1.8: mean +0.07, none < −2.0.

It is **not deployed**:
- bpt4 is better on every proxy: +4.4 vs +3.2, +1.5 vs +0.5, +2.7 vs +1.5,
  +2.2 vs −1.8.
- At 43 ch the union is ~20 k features. That is the sizing problem of tseg3,
  which the brief's Phase 4 gate would fail without a dual LDA solve.
- The brief's tie rule ("keep the single block") covers only a tie between a
  union and one of its own blocks; this is a clear loss to another adopted
  block.
- It is recorded as "ADOPT, dominated by bpt4".

The CIs above come from one bootstrap RNG run over all rows in table order, so
an interval can move in the last digit as rows are added (tseg3's lower bound
read −0.0 at 21:33 and −0.1 at 23:40); the verdicts did not change.

**Incident, 22:10:** lane A was stopped by hand before its online-64 info
steps, to free the CPU slot for the 500 Hz sizing. The killed first online
step (`p3_s3on_1d2f9d19`) left no END row; its `.start`/`.pid` were removed so
the watchdog would not report DEAD. The step is resumable with `LANES=A`.

### Phase 4 — sizing at the sealed size (`scripts/sprint0929_f4.sh`, `analysis/xblocks_sizing.py`)

Riemann-Sealed on `mock_sealed_500` through the mock's own dataset class, so
it gets session and context ids: 10,800 training windows, 43 EEG ch × 2000
samples, `blend_w="auto"` with 6 LOSO folds, 10 threads, with one other lane
running.

| Time | Variant | Estimate | Fit | Peak RSS | Features | Predict | Gate (≤ 1.5× fit, ≤ 16 GiB) |
|---|---|---|---|---|---|---|---|
| 22:10:43–22:44:03 | recipe (none) | 35 min | 1,961 s (dress rehearsal alone: 1,826 s) | 11.32 GB | 5,073 | 27.7 ms/window | reference |
| 22:44:03–23:19:14 | bpt4 | 45 min | 2,073 s = **1.06×** | 11.97 GB | 5,761 | 24.4 ms/window | **PASS** |
| 23:19:28–00:10:55 | bpt4+icoh | 45 min | 3,061 s = **1.56×** | 12.99 GB | 9,373 | 24.2 ms/window | borderline FAIL: measured while 2 other jobs ran (the reference had 1); the extra time is in the LDAs (the 6 LOSO folds 2,235 s vs 1,460). icoh is only the release-day ablation `xa`, so this does not block anything. If the replica adds it, budget ~+20 min per 500 Hz flow, or re-measure alone |

**Phase 4 verification** (`scripts/sprint0929_f5.sh`, 23:39:59–00:12:02, 32 min
vs ~40 estimated; 0 ERROR rows):
- **Regression gate with defaults:** `train_sealed.sh` on zhou2016_xsess gives
  train 0.770000 = replay 0.770000, harness 0.778333 (w = 0.75), gap 0.0083 OK.
  This is the committed flow exactly, and the candidate bakes `xblocks=none`
  (2.6 min).
- **The flow with `RECIPE_SPEC=riemann:xd=1,fb=1,x=bpt4`** on mock_sealed_s
  (organisers' split): the candidate bakes `xblocks=bpt4` (5,761 features).
  Benchopt train 0.512500 = replay 0.512500 = harness 0.512500, gap 0.0000
  (16 min). The zip is local and was **not uploaded**. The wrapper step showed
  STALLED at 12 min; that is the known wrapper trap, and the inner log was
  growing.
- **`release_ablations.sh` rehearsal** on mock_sealed_s: the release replica
  (calib:3, full participants), `SPEC=…,x=bpt4 XB=icoh`, steps base / xb / xa.
  - The summarizer prints "keep the blocks" (xb +0.56 < +1.0) and "ADD icoh"
    (xa +1.94, contexts +2.22 / +1.67).
  - The final line is `RECIPE_SPEC=riemann:xd=1,fb=1,x=bpt4+icoh` for
    `train_sealed.sh`, which turns the `+` into `xblocks="bpt4_icoh"`.
  - The mock's accuracies mean nothing; this checks the machinery.

### Phase 5 — final review (workflow `sprint0929-final-review`, 00:10–00:40, 6 agents) and fixes

- **Review:** three lenses (ML correctness/evidence, deployment path, doc
  claims), each followed by a skeptic who tried to refute every finding.
  45 findings, 36 confirmed (ml 11/13, deploy 5/8, docs 20/24); the rest
  were refuted.
- **Two majors,** each found by two lenses:
  - **The release-day summarizer commands in RELEASE_DAY § 5 did not pass
    `--spec …,x=bpt4 --xb icoh`.** The documented flow ends on those manual
    commands, so the final DECISIONS would have silently scored the xb row as
    "base" and dropped bpt4 from `RECIPE_SPEC`.
    Fixed:
    - the commands now pass `--spec` / `--xb`;
    - `release_summarize.py` also reads `spec=` / `xb=` from the run's
      `config.txt` and exits on a mismatch;
    - tested: an old run is unchanged vs HEAD except the blocks line, a new run
      reads its config, and a mismatched `--spec` exits.
  - **Rule 3b departed from the brief's Phase 4 rule without saying so.** The
    brief reads: adopt on the replica only if ≥ +1.0 and not worse on any
    context.
    Resolved (00:45, before any release number exists):
    - **bpt4 (ADOPT):** it starts in the recipe and is dropped only if the
      replica is ≥ +1.0 better without it. This is an interpretation: the
      brief itself separates ADOPT ("recommended default, still subject to the
      release ablation") from PROMISING ("a release-day ablation"), and it
      mirrors the xDAWN rule.
    - **icoh (PROMISING):** the brief's rule literally, ≥ +1.0 **and no
      context worse**. It was "no context < −1.0".
    - **Both fire:** only the larger single change is applied (the previous
      code always kept the blocks).
    - **On a 500 Hz cache,** icoh additionally needs a fit ≤ 1.5× the recipe.
- **Minors fixed:**
  - bpt4+icoh sizing is labelled **FAIL** (1.56×; 1.68× vs the rehearsal).
    It is being re-measured alone (below).
  - RELEASE_DAY:
    - the `DEC` default and the rule 6 sweep now carry `x=bpt4`;
    - the rule 3 example keeps `x=`;
    - the § 7 zip check greps and lists `xblocks`;
    - the § 8 budget covers 12 steps (plan 6–7 h).
  - **Harness memory at 500 Hz:** `RiemannXModel` computed the extra blocks on
    a full float64 copy of the training set. The extra blocks are now computed
    per 512 MiB chunk, and stateless blocks (bpt, icoh) skip the full-array
    fit. Chunked = one-shot: max |dF| = 0.0 over 15 forced chunks. Parity
    re-gated: bpt4 and icoh |dF| = |dP| = 0; defaults |dP| = 0 vs the
    pre-sprint solver, now logged after the icoh port
    (`logs/sealed_f0929/RESULTS_gates_0930.md`).
  - **`f0929_summarize.py`:**
    - per-comparison bootstrap seeds, so a CI no longer moves when rows are
      added;
    - float-safe thresholds and 2-dp output;
    - a `--reverse` header for the reverse-split table.
    - All verdicts unchanged.
  - **The invariance explanation of the spatial nulls was wrong.** sklearn's
    `shrinkage="auto"` LDA standardises the features before Ledoit-Wolf, so
    its target is diag(var), not μI. The skeptic measured up to 0.90 change in
    predict_proba under an orthogonal rotation. The spatial nulls are
    **empirical findings**; the invariance argument at most motivates why
    re-referencing was expected to do little. SEALED_RECIPE and this LOG are
    corrected (next block); the brief carries an erratum.

**Corrections to earlier statements in this sprint's entries** (review
2026-09-30; the entries above are left as written):
- *Phase 0/1 controls table:* the CIs are from the 18:22 f1 summary. The final
  values are in `RESULTS_screen.md` and differ in the last digit.
- *Phase 1:* "every `x=` config costs ≥ 2× the baseline" should read
  "~1–3× per config in the contended screen (up to ~4× for tseg2/tcut1000 on
  Tangermann)".
- *Controls read:* "TS + shrinkage LDA is affine-invariant in exact arithmetic"
  is wrong for sklearn's standardised Ledoit-Wolf LDA (see above). The
  Laplacian's loss is measured; the explanation is not.
- *Phase 2 read:* "consistent with the affine-invariance argument" is
  withdrawn. The CSP / regional / topography nulls are empirical; the argument
  never covered them (they are not fixed full-rank filters).
- *Phase 3 set decision:* the set was fixed when the Phase 3 lane launched
  (20:14:46 per STATUS) and written up at 20:16, before icoh's Scherer 5-class
  rows (20:22–20:27). A consequence of the tcut1000 dedupe: icoh got a
  single-block confirmation, and with it the release-day ablation xa.
- *Phase 3:*
  - tseg3 on Scherer 3-cl. is **+1.96** (+0.00, +4.19): below +2.0 *and* the
    CI touches 0. Its others' mean is −0.24; only 1 of 3 ≥ 0.
  - The union's others' mean is **+0.05**.
  - "bpt4's gain grows under blend_calib" should read "holds under blend_calib
    (a different metric on the same test sessions; that confirmation reuses
    the screen's test sessions, so its CI is optimistic)".
  - "Dominated by bpt4 on every proxy" should read "bpt4 has the higher point
    estimate on every proxy; the paired CI crosses 0 on 3 of 4; the union was
    not sized, projected from bpt4+icoh at 1.56×; chosen on cost and
    parsimony, post-hoc".
  - "tseg3: router whitening plus the per-subject LDA already capture most of
    what the segment covariances add" is a hypothesis. It is *consistent with*
    +6.7 (none) → +3.3 (router) → +1.96 (blend_calib); it was not tested.
- *Phase 3 incident:* the killed online step left no `.done` and no lane END
  row; a manual END row was added at 22:10:57.
- *Phase 4 sizing:*
  - the reference is "1,961 s re-measured with 1 other job (the 2026-09-28
    rehearsal also ran contended: 1,826 s)";
  - bpt4 is 1.06× vs the re-measure (1.14× vs the rehearsal);
  - the RSS columns are GiB (the solver's maxrss);
  - the 21:30 prediction for icoh (+5–10 min) came out at +16.5 min
    (under heavier contention);
  - the smoke-test output (21:15–21:25) was not kept.
- *Process deviation, not disclosed at the time:* the solver port (18534b2,
  d245478) was started during Phase 2, before the Phase 3 verdicts, to overlap
  with compute. Its defaults were verified bit-identical, and no Phase 2/3
  number uses the solver.
- *Brief § 5 (reverse split):* "added 18:10" should read "added ~18:03
  (7e0d008), before any reverse-split number and before any x= block number".

### 2026-09-30 (night) — closing runs after the review

| Time | Step | Estimate | Actual | Result |
|---|---|---|---|---|
| 00:12–01:04 | post-hoc bpt4+icoh (4 proxies) + online-64 info (baseline, bpt4, bpt4+icoh) | ~65 min | 52 min | bpt4+icoh blend_calib Δ +6.51 (+4.34, +9.11) 9/9 / +1.81 (+0.77, +3.12) / +2.16 (−1.09, +5.43) / +3.33 (−0.67, +8.67). Online-64, Scherer 3-cl. blend_calib: recipe **0.561** (= SEALED_RECIPE), bpt4 **0.592**, bpt4+icoh **0.599** (rule-dependent; information only) |
| 00:58:36–01:08:46 | defaults gate, mock arm: `train_sealed.sh` default flow on mock_sealed_s | 8 min | 10 min | train 0.483333 = replay = harness 0.483333 (w 0.75), gap 0.0000. That equals the committed flow (`logs/train_sealed_mock_sealed_s`), so the solver defaults stay bit-identical on the context / 43-ch path as well |
| 01:08– | 500 Hz sizing re-measured **alone** (`PREFIX=sz2`, recipe then bpt4+icoh) | ~85 min | (running) | |

- **bpt4+icoh is post-hoc and not a candidate.** It would meet the ADOPT rule,
  but icoh was selected after seeing results.
- It strengthens the prior for release-day step `xa`. It is not a reason to
  change the pre-registered default.
| 01:08:46–02:03:06 | 500 Hz sizing re-measured alone | ~85 min | 54 min, under (the idle machine fits 1.7× faster) | recipe **1,157 s** (19.3 min, 11.2 GiB, 11.8 ms/window); bpt4+icoh **2,062 s** (34.4 min, 13.3 GiB, 17.3 ms/window) = **1.78×: FAIL** of the brief's ≤ 1.5× gate. It is within the older absolute rule (fit ≤ 45 min). The extra time is the LDAs: LOSO folds 1,532 vs 829 s, LDA stage 199 vs 51 s at 9,373 vs 5,073 features |

- **icoh stays out on a 500 Hz cache under rule 3b** unless the user accepts
  the absolute rule, or a restricted icoh (2 bands) or a dual (n < d) LDA
  solve passes the gate.
- bpt4 alone was not re-measured. Its 1.06× compared two runs under the same
  one-other-job condition.
- Contention inflated the recipe's fit 1.7× (1,961 vs 1,157 s), so sizing
  ratios are only comparable within one series.

**Sprint wrap-up (02:10).**
- **Deliverables:** SEALED_RECIPE § 1 / § 2 Phase 7 / § 3 / § 5, RELEASE_DAY
  (rule 3b, § 5 commands, § 7 zip check, § 8 sizing), HANDOFF, the brief's
  erratum, and the activation EDA report.
- **Run bookkeeping:**
  - every run ended with rc = 0, except the lane-A online step killed by hand
    at 22:10 (resumed and completed at 00:35);
  - nothing is running;
  - no uploads were made.
- **The data release** was still "coming soon" at 01:10.

## 2026-10-01 (evening) — Sprint: dual shrinkage LDA, deployment-test zip, strict LOSO references (brief `prompts/2026-10-01_dual_lda_strict_wcv.md`)

17:36 → 05:36, no user focus; picked by the value check (DIRECTIONS § 6 items
2 and 4, plus decision 2's preparation). Logs: `logs/sealed_s1001/`.

### Phase 0 — orient, benchmark, brief (17:36–17:49)

- Tracks page 17:40: Graz + BrainHero still "coming soon". Sleep disabled on
  AC/DC (read, not changed).
- **Where the 500 Hz fit time goes** (17:40–17:45, `scratchpad/bench_lda.py`,
  random data, 10 threads, the committed solver): pooled LDA n = 8,400 took
  9.7 s at p = 5,073 and 31.3 s at p = 9,373; **20 per-subject LDAs
  (n = 420) took 39.5 s and 200.0 s**. Each `auto` fold refits all of them,
  so at bpt4+icoh's size the per-subject p × p solves are most of the fold
  (260 s in `sz2_bpt4_icoh`).
- Brief committed db8b35c.

### Phase 1 — dual (n < p) shrinkage LDA (17:49–18:06)

- `_DualLsqrLDA` in the shared LDA block (`riemann_steptype.py` and
  `riemann_sealed.py`, identical text). sklearn's estimate is per class
  StandardScaler → Ledoit-Wolf towards μ·I → rescale, i.e.
  Σ = diag(λ) + Uᵀ diag(c) U with U the n × p centred class rows. It is
  solved through the n × n Woodbury form; the shrinkage comes from the
  n_g × n_g Gram matrix with `ledoit_wolf_shrinkage`'s formula. It is used
  above `LDA_FAST_P` when n < p (`LDA_DUAL = True`). Anything non-finite, a
  non-positive diagonal or a full-system residual above `LDA_MAX_RESID`
  falls back to the Cholesky path. `covariance_` is not formed.
- **Gate D1** (`analysis/lda_dual_check.py`, 17:57:46–18:05:42, 8 min, est.
  15; `logs/sealed_s1001/RESULTS_d1.md`): **PASS.** The cases were 6 random
  matrices (iid; factor + 10^±2 scales at p = 9,373; n = 3,000; binary; a
  2-row class with constant and zero features; empirical priors) and 9 real
  ones (Scherer 3-class bpt4+icoh harness features, p = 4,875: pooled and 4
  subjects; mock_sealed_s, p = 5,987: 4 subjects).
  - **vs the Cholesky solve:** max |dP| ≤ 3.7e-12 in every case, argmax
    identical everywhere, coefficients within 7e-14 relative.
  - **Shrinkage** vs `ledoit_wolf_shrinkage`: ≤ 1.2e-15 relative.
  - The solver and harness copies give identical bits. The result pickles as
    a plain `LinearDiscriminantAnalysis` without `covariance_`.
  - **Speed:** 20 × (n = 420, p = 9,373) took 2.9 s with the dual solve and
    178 s with Cholesky, **60.7×**. Real per-subject fits ran at 0.02 s vs
    1.7–3.0 s, the Scherer pooled fit at 0.19 s vs 2.77 s.
  - **Deviation from the rule's letter (vs sklearn).** In the 2-row-class /
    constant-feature case, dual vs sklearn's own lstsq is 4.7e-8, above the
    1e-8 threshold. But Cholesky vs sklearn is the same 4.7e-8, and sklearn's
    solution has the largest full-system residual (1.4e-11 vs dual 2.4e-14 vs
    Cholesky 2.0e-14). sklearn's SVD lstsq is the less exact reference there,
    and the dual is no further from it than the committed Cholesky path. The
    checker accepts this case under that explicit condition (residual and
    ≤ 2 × Cholesky's deviation); every real case is within 1.3e-9 of sklearn.
- **Decision D1:** the dual path is on by default (`LDA_DUAL = True`).
  Phase 2's gates decide whether that stands.

### Phase 1b — deployment-test candidate (first pass 17:49–18:03)

`scripts/sprint1001_deploytest.sh`: Riemann-Sealed with `xblocks="bpt4"`
baked, trained on Dreyer (12,392 windows, 52 subjects, 27 ch, 2,485
features), replayed inference-only from a read-only copy. Training took
7.5 min (est. 30), the replay 32 s.
- Train **0.615873 = replay 0.615873** (Dreyer is cross-subject, so the score
  means nothing). The joblib holds only numpy arrays, sklearn
  `LinearDiscriminantAnalysis` and pyriemann `XdawnCovariances` /
  `TangentSpace` (table in `deploytest_v1/RESULTS.md`).
- **Finding: the joblib was 2.6 GB (zip 2.4 GB).** Below `LDA_FAST_P` the
  solver kept each LDA's unused p × p `covariance_`: 53 × 49 MB at
  p = 2,485. The sealed data (p ≈ 5,800 > 4,000) would not have hit this,
  but a warm-up upload would have spent 2.4 GB of the profile's 15 GB, and the
  worker would have had to load a 2.6 GB joblib.
- **Fix:** `_fit_lda` now drops `covariance_` at every size (prediction uses
  only `coef_` / `intercept_` / `classes_`; the predictions are unchanged).
  Re-run launched 18:07.
- Deployment re-run (18:05:49–18:13:31): train **0.615873 = replay
  0.615873**, the same score as before the fix. The joblib is **4.6 MB**
  (was 2.6 GB) and the zip 4.1 MiB
  (`logs/sealed_s1001/deploytest/riemann_sealed_deploytest_dreyer2023_2026-10-01.zip`,
  NOT uploaded). It carries the dual LDA but not yet Phase 3's `wcv_ref`
  parameter. Rebuild it from the final solver at the end of the sprint.

### Phase 2 — gates for the dual LDA (18:07:46–18:25:27, 18 min; est. 45)

`scripts/sprint1001_g.sh`, two lanes at 5 threads, contended by the
deployment re-run for its first 6 min. Results:
`logs/sealed_s1001/RESULTS_gates.md`. **All PASS**: each run reproduces
its committed run.

| Gate | Result | vs committed |
|---|---|---|
| zhou regression flow (`train_sealed.sh`, defaults) | train 0.770000 = replay; harness 0.778333 (w 0.75) | identical; 17 harness rows, worst \|dP\| 1.0e-12 |
| mock_sealed_s default flow (p = 5,073: dual per-subject and pooled LDAs) | train 0.483333 = replay = harness (w 0.75) | identical; 13 rows, 3.6e-13. Solver LDA stage **34.3 s → 1.1 s** |
| mock_sealed_s x=bpt4 flow | train 0.512500 = replay = harness (w 0.5) | identical; 13 rows, 1.5e-11 |
| `xblocks_gate.py --xblocks icoh` | gate 1 defaults bit-identical vs the pre-sprint solver (\|dP\| = 0); gate 2 harness = solver with icoh (p = 4,395, dual in both), acc 0.4352 | as 2026-09-30 |
| Scherer 3-class bpt4+icoh, `sealed_personal` (p = 4,875) | blend_calib router-id 0.551058 | 9 rows identical scores and weights, 1.3e-12. Run time 351 s vs 613 s (both contended) |
| `release_ablations.sh` base/xb/xa on mock_sealed_s | 15 rows | identical, 3.7e-12. base 2.3 / xb 1.8 / xa 1.9 min vs 5.6 / 4.3 / 9.3 on 2026-09-29 |

**Decision.** The gates pass, so the dual path stays on by default (D1, D2's
precondition).

### Phase 4 (pulled forward) — review of the dual LDA (18:14–18:21, 1 agent)

The review found no blocker or major issue. It re-derived equivalence with
sklearn 1.9.1's source (1-sample class, constant features, shrinkage 0 and 1,
non-contiguous and string labels, unnormalised priors: estimate ≤ 2e-16,
coef ≤ 6e-16). Fallback state equals a Cholesky fit. Nothing reads
`covariance_`. The pickle holds no reference to the private classes. No
p × p allocation; I + VVᵀ has eigenvalues ≥ 1. Fixed (`_dual_coef`,
`_dual_selfcheck`, applied 18:25 after the gate lane, both files identical):
- **minor:** a failed dual attempt kept V (n × p) and M (n × n) alive during
  the Cholesky fallback (~1.2 GB at the sealed size). They are now freed on
  return;
- **minor:** a caller forcing `fast=True` below 4,000 features
  (`activation_eda.py`) took the dual path. The dual now also requires
  p > `LDA_FAST_P`;
- **minor:** the dual re-implements sklearn's `_cov`, while the scoring
  image's sklearn is unpinned (training is local, 1.9.1). A once-per-process
  self-check against `_class_cov` now switches the dual off, with a warning,
  on any disagreement;
- **nits:** dual only for float64 X; numerical errors (ValueError,
  ArithmeticError, MemoryError) fall back too; a LDA_STATS counting note. The
  probability gate saturates on the mock (\|dP\| ~1e-91); there the
  coefficient differences (≤ 2e-14) are the evidence.

Re-check `lda_dual_check.py --bench 5` after the fixes:
`RESULTS_d1_recheck.md` (running).

### Phase 3 — strict LOSO references: code (18:00–18:25)

- **Solver:** `wcv_ref="all" | "strict"` (default `all`; a Solver parameter,
  baked by `train_sealed.sh`). Under strict, `_fold_refs` takes each fold's
  whitening references from its fit rows only. With align `subject` that is
  per subject. With `subject_context` it is per (subject, context) pair with
  ≥ `ctx_min` fit rows, else the subject's; a subject without fit rows gets
  the global reference. X is not overwritten, and every block is recomputed
  per fold. Strict equals `all` under `adapt="online"` and `align="none"`.
- **Harness:** `sealed_personal.py --wref all|strict` (`strict_fold_X`, the
  same references and float32 whitening). `wref` enters the config key only
  when strict (KEY_DEFAULTS); non-router alignments fall back to `all` with a
  log line.
- **Scripts:** `train_sealed.sh WREF=` (passed to the harness, baked, used in
  the row filter; CONF and config.txt change only when strict).
  `release_ablations.sh WREF=` reaches every `pers` step;
  `release_summarize.py` carries `WREF=` into its `train_sealed.sh` line.
- Gates and information runs: `scripts/sprint1001_p3.sh`, launched 18:26
  (`RESULTS_p3.md`).
