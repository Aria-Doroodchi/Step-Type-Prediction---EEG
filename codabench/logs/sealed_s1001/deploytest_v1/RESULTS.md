# Deployment-test candidate (sprint 2026-10-01 Phase 1b; NOT uploaded)

| check | value |
|---|---|
| benchopt train (s1001_deploytest_train.parquet) | 0.615873 |
| read-only replay (s1001_deploytest_replay.parquet) | 0.615873 (EQUAL) |

## Object types in the joblib

| type | keys |
|---|---|
| `builtins.NoneType` | chan_idx |
| `builtins.float` | router.thr, sfreq |
| `builtins.int` | n_classes |
| `builtins.str` | ch_names[], estimator, fb_bands[], xblocks[] |
| `numpy.ndarray` | W, W_global, subjects |
| `pyriemann.estimation.XdawnCovariances` | xdawn |
| `pyriemann.tangentspace.TangentSpace` | broad_ts, fb_ts[], xdawn_ts |
| `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` | lda, lda_subject[], router.lda |

## Zip `riemann_sealed_deploytest_dreyer2023_2026-10-01.zip` (2409.0 MiB)

- `riemann_sealed.joblib` 2512.67 MiB
- `submission.py` 0.07 MiB
