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
| 18:04:57 → | prepare `dreyer2023` | 1–3 h (all studies) | *running* | |
