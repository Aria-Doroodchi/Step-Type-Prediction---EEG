# Sprint 0928 Phase 3: paired subject bootstrap

Per-subject score = mean of the subject's cells (cell-averaged balanced accuracy); for shuf, the mean over seeds 0-2 first. A − B = mean per-subject difference with a 95 % percentile CI (10,000 resamples over subjects, `np.random.default_rng(0)` per comparison), and the number of subjects with A > B. Online rows are rule-dependent.

| proxy | A | B | order | A cell | B cell | A − B (95 % CI) | subjects A > B |
|---|---|---|---|---|---|---|---|
| Tangermann (4 cl., last session) | online-32 | clean | rec | 0.836 | 0.802 | +0.034 (+0.014, +0.056) | 8/9 |
| Tangermann (4 cl., last session) | online-32 | clean | inter | 0.835 | 0.802 | +0.033 (+0.017, +0.050) | 8/9 |
| Tangermann (4 cl., last session) | online-32 | clean | shuf | 0.835 | 0.802 | +0.033 (+0.014, +0.054) | 7/9 |
| Tangermann (4 cl., last session) | online-64 | clean | rec | 0.837 | 0.802 | +0.035 (+0.016, +0.054) | 8/9 |
| Tangermann (4 cl., last session) | online-64 | clean | inter | 0.832 | 0.802 | +0.030 (+0.012, +0.050) | 7/9 |
| Tangermann (4 cl., last session) | online-64 | clean | shuf | 0.833 | 0.802 | +0.031 (+0.013, +0.049) | 7/9 |
| Tangermann (4 cl., last session) | online-128 | clean | rec | 0.831 | 0.802 | +0.029 (+0.007, +0.049) | 7/9 |
| Tangermann (4 cl., last session) | online-128 | clean | inter | 0.829 | 0.802 | +0.027 (+0.008, +0.046) | 7/9 |
| Tangermann (4 cl., last session) | online-128 | clean | shuf | 0.833 | 0.802 | +0.031 (+0.014, +0.051) | 8/9 |
| Tangermann (4 cl., last session) | online-64+guard | online-64 | rec | 0.838 | 0.837 | +0.001 (+0.000, +0.002) | 2/9 |
| Tangermann (4 cl., last session) | online-64+guard | online-64 | inter | 0.831 | 0.832 | -0.001 (-0.003, +0.001) | 1/9 |
| Tangermann (4 cl., last session) | online-64+guard | online-64 | shuf | 0.833 | 0.833 | +0.000 (+0.000, +0.000) | 1/9 |
| Scherer 3-class (WORD/SUB/HAND, last session) | online-32 | clean | rec | 0.556 | 0.486 | +0.070 (+0.046, +0.101) | 9/9 |
| Scherer 3-class (WORD/SUB/HAND, last session) | online-32 | clean | inter | 0.546 | 0.486 | +0.060 (+0.036, +0.084) | 8/9 |
| Scherer 3-class (WORD/SUB/HAND, last session) | online-32 | clean | shuf | 0.537 | 0.486 | +0.051 (+0.023, +0.082) | 8/9 |
| Scherer 3-class (WORD/SUB/HAND, last session) | online-64 | clean | rec | 0.561 | 0.486 | +0.075 (+0.043, +0.108) | 9/9 |
| Scherer 3-class (WORD/SUB/HAND, last session) | online-64 | clean | inter | 0.530 | 0.486 | +0.044 (+0.024, +0.064) | 8/9 |
| Scherer 3-class (WORD/SUB/HAND, last session) | online-64 | clean | shuf | 0.532 | 0.486 | +0.046 (+0.019, +0.074) | 8/9 |
| Scherer 3-class (WORD/SUB/HAND, last session) | online-128 | clean | rec | 0.549 | 0.486 | +0.063 (+0.046, +0.081) | 9/9 |
| Scherer 3-class (WORD/SUB/HAND, last session) | online-128 | clean | inter | 0.517 | 0.486 | +0.031 (+0.011, +0.050) | 7/9 |
| Scherer 3-class (WORD/SUB/HAND, last session) | online-128 | clean | shuf | 0.525 | 0.486 | +0.039 (+0.018, +0.060) | 8/9 |
| Scherer 3-class (WORD/SUB/HAND, last session) | online-64+guard | online-64 | rec | 0.561 | 0.561 | -0.000 (-0.003, +0.003) | 1/9 |
| Scherer 3-class (WORD/SUB/HAND, last session) | online-64+guard | online-64 | inter | 0.530 | 0.530 | +0.000 (+0.000, +0.000) | 0/9 |
| Scherer 3-class (WORD/SUB/HAND, last session) | online-64+guard | online-64 | shuf | 0.532 | 0.532 | +0.000 (+0.000, +0.000) | 0/9 |
| Zhou (3 cl., last session) | online-32 | clean | rec | 0.800 | 0.778 | +0.022 (-0.012, +0.050) | 3/4 |
| Zhou (3 cl., last session) | online-32 | clean | inter | 0.795 | 0.778 | +0.017 (-0.005, +0.042) | 3/4 |
| Zhou (3 cl., last session) | online-32 | clean | shuf | 0.788 | 0.778 | +0.010 (-0.030, +0.050) | 2/4 |
| Zhou (3 cl., last session) | online-64 | clean | rec | 0.797 | 0.778 | +0.018 (-0.030, +0.048) | 3/4 |
| Zhou (3 cl., last session) | online-64 | clean | inter | 0.783 | 0.778 | +0.005 (-0.027, +0.037) | 2/4 |
| Zhou (3 cl., last session) | online-64 | clean | shuf | 0.794 | 0.778 | +0.016 (-0.020, +0.051) | 2/4 |
| Zhou (3 cl., last session) | online-128 | clean | rec | 0.787 | 0.778 | +0.008 (-0.038, +0.037) | 3/4 |
| Zhou (3 cl., last session) | online-128 | clean | inter | 0.758 | 0.778 | -0.020 (-0.065, +0.015) | 1/4 |
| Zhou (3 cl., last session) | online-128 | clean | shuf | 0.792 | 0.778 | +0.013 (-0.016, +0.042) | 2/4 |
| Zhou (3 cl., last session) | online-64+guard | online-64 | rec | 0.797 | 0.797 | +0.000 (+0.000, +0.000) | 0/4 |
| Zhou (3 cl., last session) | online-64+guard | online-64 | inter | 0.783 | 0.783 | +0.000 (+0.000, +0.000) | 0/4 |
| Zhou (3 cl., last session) | online-64+guard | online-64 | shuf | 0.794 | 0.794 | +0.000 (+0.000, +0.000) | 0/4 |
| Zhou stream (calib session 0; test sessions 1+2) | online-32 | clean | rec | 0.660 | 0.616 | +0.044 (-0.026, +0.115) | 2/4 |
| Zhou stream (calib session 0; test sessions 1+2) | online-32 | clean | inter | 0.641 | 0.616 | +0.025 (-0.033, +0.084) | 2/4 |
| Zhou stream (calib session 0; test sessions 1+2) | online-32 | clean | shuf | 0.634 | 0.616 | +0.018 (-0.042, +0.077) | 2/4 |
| Zhou stream (calib session 0; test sessions 1+2) | online-64 | clean | rec | 0.651 | 0.616 | +0.035 (-0.033, +0.103) | 2/4 |
| Zhou stream (calib session 0; test sessions 1+2) | online-64 | clean | inter | 0.638 | 0.616 | +0.022 (-0.023, +0.068) | 2/4 |
| Zhou stream (calib session 0; test sessions 1+2) | online-64 | clean | shuf | 0.633 | 0.616 | +0.017 (-0.040, +0.074) | 2/4 |
| Zhou stream (calib session 0; test sessions 1+2) | online-128 | clean | rec | 0.667 | 0.616 | +0.051 (+0.007, +0.096) | 3/4 |
| Zhou stream (calib session 0; test sessions 1+2) | online-128 | clean | inter | 0.630 | 0.616 | +0.014 (-0.033, +0.062) | 2/4 |
| Zhou stream (calib session 0; test sessions 1+2) | online-128 | clean | shuf | 0.627 | 0.616 | +0.011 (-0.043, +0.064) | 2/4 |
| Zhou stream (calib session 0; test sessions 1+2) | online-32+guard | online-32 | rec | 0.660 | 0.660 | -0.001 (-0.005, +0.003) | 1/4 |
| Zhou stream (calib session 0; test sessions 1+2) | online-32+guard | online-32 | inter | 0.641 | 0.641 | +0.000 (+0.000, +0.000) | 0/4 |
| Zhou stream (calib session 0; test sessions 1+2) | online-32+guard | online-32 | shuf | 0.634 | 0.634 | +0.000 (+0.000, +0.000) | 0/4 |
| Zhou stream (calib session 0; test sessions 1+2) | online-64+guard | online-64 | rec | 0.650 | 0.651 | -0.001 (-0.005, +0.003) | 1/4 |
| Zhou stream (calib session 0; test sessions 1+2) | online-64+guard | online-64 | inter | 0.638 | 0.638 | +0.000 (+0.000, +0.000) | 0/4 |
| Zhou stream (calib session 0; test sessions 1+2) | online-64+guard | online-64 | shuf | 0.633 | 0.633 | +0.000 (+0.000, +0.000) | 0/4 |
| Zhou stream (calib session 0; test sessions 1+2) | online-128+guard | online-128 | rec | 0.658 | 0.667 | -0.009 (-0.027, +0.002) | 1/4 |
| Zhou stream (calib session 0; test sessions 1+2) | online-128+guard | online-128 | inter | 0.630 | 0.630 | +0.000 (+0.000, +0.000) | 0/4 |
| Zhou stream (calib session 0; test sessions 1+2) | online-128+guard | online-128 | shuf | 0.627 | 0.627 | +0.000 (+0.000, +0.000) | 0/4 |

Zhou stream, second test session (session 2) only, recording order:

| A | B | A − B (95 % CI) | subjects A > B |
|---|---|---|---|
| online-32 | clean | +0.043 (-0.037, +0.117) | 3/4 |
| online-64 | clean | +0.023 (-0.060, +0.107) | 2/4 |
| online-128 | clean | +0.032 (-0.027, +0.090) | 2/4 |

