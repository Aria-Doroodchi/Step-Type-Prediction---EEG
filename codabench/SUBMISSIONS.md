# Submissions ledger

One row per Codabench upload (and per notable local run worth comparing).
Warm-up scores are indicative only: public test data, leakage possible.

## Codabench uploads

| Date | Track / phase | Solver + key params | ZIP | Local bal. acc. | Codabench score | Notes |
|---|---|---|---|---|---|---|
| *ready* | Track 2 / warm-up | **EEGNet-StepType-WU1** (`solvers/bci_decoding/eegnet_steptype_wu1.py`): 100 epochs, patience 20, ref none, seed 33 | `submissions/eegnet_steptype_wu1_2026-09-24.zip` | 0.820 (config mean 0.806 ± 0.016, 3 seeds) | *not uploaded yet* | replay-verified (read-only, inference-only, identical score) |

## Local reference runs

| Date | Dataset | Solver | Setting | Bal. acc. | Log |
|---|---|---|---|---|---|
| 2026-09-23 | tangermann2012 (4-class, 22 ch) | Riemann-StepType | full train, defaults | 0.536 | `logs/test_tangermann_1805.log` |
| 2026-09-23 | tangermann2012 | EEGNet-StepType | full train, early stop @ epoch 15 | 0.454 | `logs/test_tangermann_eegnets_1806.log` |
| 2026-09-23 | tangermann2012 | EEGNet (upstream braindecode baseline) | 20 epochs | 0.580 | `logs/test_tangermann_eegnets_1806.log` |
| 2026-09-23 | tangermann2012 | MeanLogReg (upstream floor) | full train | 0.266 | `logs/test_tangermann_1805.log` |
| 2026-09-23 | dreyer2023 (2-class, 27 ch) | Riemann-StepType | test batch: 40 batches (2,560 windows) | 0.716 | `logs/test_track2_dreyer2023_2026-09-23_1834.log` |
| 2026-09-23 | dreyer2023 | EEGNet-StepType | test batch: 40 batches, 3 epochs | 0.651 | same |
| 2026-09-23 | dreyer2023 | MeanLogReg (upstream floor) | full train split | 0.681 | same |
| 2026-09-24 | **dreyer2023, full train (12,392 windows)** | **EEGNet-StepType** | reference=car, patience=20, 50 epochs; 3 seeds | **0.790 ± 0.002** | `logs/overnight_2026-09-24/` |
| 2026-09-24 | dreyer2023 full | EEGNet-StepType | reference=car, patience=10; 3 seeds | 0.786 ± 0.008 | same |
| 2026-09-24 | dreyer2023 full | EEGNet-StepType | reference=none, patience=10 / 20; 3 seeds | 0.785 / 0.779 | same |
| 2026-09-24 | dreyer2023 full | EEGNet (upstream braindecode) | 20 epochs; 3 seeds | 0.778 ± 0.013 | same |
| 2026-09-24 | dreyer2023 full | Riemann-StepType | xDAWN on, reference none / car | 0.754 / 0.746 | same |
| 2026-09-24 | dreyer2023 full | Riemann-StepType | xDAWN off, reference none / car | 0.592 / 0.585 | same |
| 2026-09-24 | dreyer2023 full | Torch-Linear / MeanLogReg (upstream floors) | — | 0.689 / 0.681 | same |
| 2026-09-24 | dreyer2023 full | **EEGNet-StepType** | **100 epochs, patience 20**, ref none / car; 3 seeds | **0.806 ± 0.016 / 0.803 ± 0.010** | `logs/overnight_2026-09-24_p2/` |
| 2026-09-24 | dreyer2023 full | EEGNet-StepType | standardize=True, 50 epochs, ref car / none; 3 seeds | 0.780 / 0.777 | same |
| 2026-09-24 | dreyer2023 full | Riemann-StepType | xDAWN nfilter 8, ref car / none | 0.760 / 0.759 | same |
| 2026-09-24 | dreyer2023 full | Riemann-StepType | xDAWN nfilter 4 + 1–40 Hz band-pass / nfilter 2 | 0.71–0.72 / 0.66–0.70 | same |
| 2026-09-25 | dreyer2023 full | EEGNet-StepType | 200 epochs, patience 30, ref none; 3 seeds | 0.810 ± 0.008 | `logs/overnight_2026-09-24_p3/` |
| 2026-09-25 | dreyer2023 full | EEGNet-StepType-WU1 (candidate) | defaults (100 ep, pat 20, none, seed 33); train run / platform replay | 0.82004 / 0.82004 | same |
| 2026-09-25 | dreyer2023 full | Riemann-StepType | xDAWN on, ref none, blocks off (541 features): baseline re-run | 0.75377 (= phase 1 exactly) | `logs/riemann_blocks_2026-09-25/` |
| 2026-09-25 | dreyer2023 full | Riemann-StepType | xDAWN on + `slow_block` (919 features) | 0.75635 (+0.003) | same |
| 2026-09-25 | dreyer2023 full | **Riemann-StepType** | **xDAWN on + `filterbank`** (2,053 features) | **0.76964 (+0.016)**, best Riemann | same |
| 2026-09-25 | dreyer2023 full | Riemann-StepType | xDAWN on + `slow_block` + `filterbank` (2,431 features) | 0.76865 (+0.015) | same |
| 2026-09-25 | dreyer2023 full | Riemann-StepType | xDAWN **off** + `slow_block` + `filterbank` (2,295 features) | 0.71825 (−0.036; xDAWN off alone was 0.592) | same |

Chance: tangermann2012 = 0.25, dreyer2023 = 0.50. Dreyer test = participants
61–81 (= the Codabench warm-up evaluation subset).
