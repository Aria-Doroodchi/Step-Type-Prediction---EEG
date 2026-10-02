# Submissions ledger

One row per Codabench upload (and per notable local run worth comparing).
Warm-up scores are indicative only: public test data, leakage possible.

## Codabench uploads

| Date | Track / phase | Solver + key params | ZIP | Local bal. acc. | Codabench score | Notes |
|---|---|---|---|---|---|---|
| *ready* | Track 2 / warm-up | **EEGNet-StepType-WU1** (`solvers/bci_decoding/eegnet_steptype_wu1.py`): 100 epochs, patience 20, ref none, seed 33 | `submissions/eegnet_steptype_wu1_2026-09-24.zip` | 0.820 (config mean 0.806 ± 0.016, 3 seeds) | 0.82 (public leaderboard, submitted 2026-09-25 14:51 UTC as adoroodchi; presumably WU1, confirm) | replay-verified (read-only, inference-only, identical score) |

### Recommended next upload: a deployment test (not a leaderboard attempt)

Recommended by the final sprint review, 2026-09-29. No sklearn/pyriemann joblib
has ever run on the Codabench scoring image (`tommoral/neural-compet:v2`). The only
upload so far is the torch EEGNet above. The image's scikit-learn is unpinned, and
pyriemann arrives only transitively. Riemann-Sealed ships 22 sklearn LDAs
(pickled with sklearn 1.9.1) plus pyriemann 0.12 objects. The sealed phase allows 1
submission per day and hides the logs, so find out **during warm-up** (open until
2026-10-25, 5/day):
- train Riemann-Sealed with its **default** parameters (what Codabench runs) on
  the warm-up study. Dreyer is cross-subject, so the score does not matter and the
  cross-session harness does not apply; train with benchopt directly:
  `cd ~/codabench/2026-competition && COMPET_SUBMISSION_DIR=$HOME/codabench/logs/wu_riemann_deploytest/submission benchopt run tracks/bci_decoding -d "BCI[study=dreyer2023]" -s ../solvers/bci_decoding/riemann_sealed.py -o "BCI-decoding[training=True]" --no-plot --no-html --no-cache --output wu_riemann_deploytest`,
  then zip the two files in that submission folder at the zip root (as
  `train_sealed.sh` does);
- upload the zip (the user's action);
- check that it reaches *Finished*, and read the ingestion log and its duration.

**Prepared 2026-10-01 (sprint 1001): ready to upload, NOT uploaded.**

| | |
|---|---|
| zip | `codabench/logs/sealed_s1001/deploytest/riemann_sealed_deploytest_dreyer2023_2026-10-01.zip` (4.1 MiB, at the zip root: `submission.py` + `riemann_sealed.joblib` 4.6 MB) |
| sha256 | `0cf3c91b57e9cfc0035316a05b83e9bb6c3a9509d0cafce694186873272e1c17` |
| model | Riemann-Sealed (the solver as of commit 71607ea: dual LDA, `wcv_ref`), `xblocks="bpt4"` baked as the sealed candidate will be; other parameters at their defaults (blend_w 0.5, align subject); trained on `BCI[study=dreyer2023]` by `scripts/sprint1001_deploytest.sh` |
| local check | train 0.615873 = read-only replay 0.615873 (inference only, from a read-only copy). Dreyer is cross-subject, so the score means nothing. The public warm-up leaderboard shows 0.82 for WU1: if Codabench displays the latest submission, keep WU1 as the one shown (the warm-up does not rank) |
| joblib holds | numpy arrays, sklearn `LinearDiscriminantAnalysis` (pooled, 52 per-subject, router), pyriemann `XdawnCovariances` / `TangentSpace`. No `covariance_` matrices: the first build was a 2.6 GB joblib (2.4 GB zip) before the solver dropped them at every size |

Checked locally (2026-10-01, `analysis/joblib_compat_check.py`): the joblib
loads and gives identical outputs (≤ 2e-14) under sklearn 1.6.1–1.8.0, numpy
1.26–2.5 and pyriemann 0.7–0.11, with only sklearn's InconsistentVersionWarning.
So the upload mainly tests the ingestion path (benchopt, the solver's imports,
time), not the pickles.

What to check after the upload:
1. It reaches *Finished*, not *Failed*.
2. The ingestion log shows no unpickling warning (sklearn
   `InconsistentVersionWarning`) and no pyriemann import error. A warning
   alone is information. A failure means the sealed candidate needs
   plain-array parameters instead of pickled objects: do this before
   Oct 28.
3. The duration. Locally the replay took 32–37 s for Dreyer's test set.
4. Record the row in the upload table above.

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
