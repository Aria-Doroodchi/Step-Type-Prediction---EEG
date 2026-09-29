# Results (2 runs from 2 files)

Dreyer 2023 test split (participants 61-81), balanced accuracy, chance 0.50.

| solver | config | bal. acc. mean | SD | n | time (s) |
|---|---|---|---|---|---|
| Riemann-Sealed-Cand_auto | adapt=none,align=subject,bandpass=none,blend_w=auto,buffer=64,chans=eeg,estimator=oas,filterbank=True,kind=riemann,max_batches=None,nfilter=4,personal=blend,reference=none,slow_block=False,use_xdawn=True | 0.615 | 0.000 | 2 | 645 |

## Gate check

| check | value | verdict |
|---|---|---|
| benchopt train (train_sealed_mock_sealed_120_auto_train.parquet) | 0.614722 | |
| read-only replay (train_sealed_mock_sealed_120_auto_replay.parquet) | 0.614722 | EQUAL (exact) |
| harness blend_calib router-id pooled BA (cell 0.6147, w=0.75) | 0.614722 | abs(train - harness) = 0.0000: OK (<= 0.02) |

Candidate blend_w = "auto".
