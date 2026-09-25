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
