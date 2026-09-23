# Submissions ledger

One row per Codabench upload (and per notable local run worth comparing).
Warm-up scores are indicative only: public test data, leakage possible.

## Codabench uploads

| Date | Track / phase | Solver + key params | ZIP | Local bal. acc. | Codabench score | Notes |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | nothing submitted yet |

## Local reference runs

| Date | Dataset | Solver | Setting | Bal. acc. | Log |
|---|---|---|---|---|---|
| 2026-09-23 | tangermann2012 (4-class, 22 ch) | Riemann-StepType | full train, defaults | 0.536 | `logs/test_tangermann_1805.log` |
| 2026-09-23 | tangermann2012 | EEGNet-StepType | full train, early stop @ epoch 15 | 0.454 | `logs/test_tangermann_eegnets_1806.log` |
| 2026-09-23 | tangermann2012 | EEGNet (upstream braindecode baseline) | 20 epochs | 0.580 | `logs/test_tangermann_eegnets_1806.log` |
| 2026-09-23 | tangermann2012 | MeanLogReg (upstream floor) | full train | 0.266 | `logs/test_tangermann_1805.log` |

Chance on tangermann2012 = 0.25.
