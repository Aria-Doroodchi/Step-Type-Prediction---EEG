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
