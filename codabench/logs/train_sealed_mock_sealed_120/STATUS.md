| time | event |
|---|---|
| 22:11:13 | cache exists: /home/ali_d/neuralbench/xsess_cache/mock_sealed_120 (not rebuilt) |
| 22:11:13 | skip cache |
| 22:11:13 | config split=replica:3 test_subjects=default wcv=last chans=eeg spec=riemann:xd=1,fb=1 align=router-psd:riemann adapt=none blend_w=harness router_cap=0.5 wvariant=calib dataset=../datasets/mock_sealed.py[study=mock_sealed_120] (harness rows: logs/sealed_train_sealed_mock_sealed_120; cache built=2026-09-28 17:51:37 X.npy=1299456128) |
| 22:11:13 | START preflight |
| 22:11:19 | END preflight rc=0 |
| 22:11:19 | preflight: X=(14400, 43, 480) subjects=20 sessions/subject=6 train=10800 test=3600 val_session_windows=0 K=3 class_counts=[4800, 4800, 4800] dropped_ch=['EMG1', 'EMG2', 'EOG1', 'EOG2'] threads=10 split=replica:3 pool=all chans=eeg(43/47 ch) ch_types=eeg:43,emg:2,eog:2(meta) contexts=2 test_subjects=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19] test_sessions=[3, 4, 5] test_cells=60 hidden_in_train=0 router_cap=0.5 |
| 22:11:19 | START validate |
| 22:28:33 | END validate rc=0 |
| 22:28:33 | START personal_none |
| 22:44:05 | END personal_none rc=0 |
| 22:44:05 | candidate /home/ali_d/codabench/logs/train_sealed_mock_sealed_120/riemann_sealed_cand.py: personal=blend blend_w=0.75 adapt=none use_xdawn=True filterbank=True kind=riemann buffer=64 chans=eeg |
| 22:44:05 | START train |
| 22:51:36 | END train rc=0 |
| 22:51:36 | solver: fitting on X=(10800, 43, 480), subjects=20, classes=[0, 1, 2], features=5073, align=subject/riemann, personal=blend, router_thr=0.500, session_ids=yes, chans=eeg (43 kept) |
| 22:51:36 | blend weight: harness w=0.75 (split=replica:3 wcv=last); solver trained with the baked w=0.75 |
| 22:51:37 | START replay |
| 22:52:25 | END replay rc=0 |
| 22:52:26 | gate: train 0.614722 replay 0.614722 EQUAL; harness blend_calib pooled 0.614722 (w=0.75), gap 0.0000 OK |
| 22:52:29 | zipped /home/ali_d/codabench/logs/train_sealed_mock_sealed_120/riemann_sealed_mock_sealed_120_2026-09-28.zip (NOT uploaded) |
| 22:52:29 | ALL DONE |
