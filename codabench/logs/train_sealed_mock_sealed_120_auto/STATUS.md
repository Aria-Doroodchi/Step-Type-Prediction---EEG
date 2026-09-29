| time | event |
|---|---|
| 22:52:31 | cache exists: /home/ali_d/neuralbench/xsess_cache/mock_sealed_120 (not rebuilt) |
| 22:52:31 | skip cache |
| 22:52:31 | config split=replica:3 test_subjects=default wcv=loso chans=eeg spec=riemann:xd=1,fb=1 align=router-psd:riemann adapt=none blend_w=auto router_cap=0.5 wvariant=calib dataset=../datasets/mock_sealed.py[study=mock_sealed_120] (harness rows: logs/sealed_train_sealed_mock_sealed_120; cache built=2026-09-28 17:51:37 X.npy=1299456128) |
| 22:52:31 | START preflight |
| 22:52:36 | END preflight rc=0 |
| 22:52:36 | preflight: X=(14400, 43, 480) subjects=20 sessions/subject=6 train=10800 test=3600 val_session_windows=0 K=3 class_counts=[4800, 4800, 4800] dropped_ch=['EMG1', 'EMG2', 'EOG1', 'EOG2'] threads=10 split=replica:3 pool=all chans=eeg(43/47 ch) ch_types=eeg:43,emg:2,eog:2(meta) contexts=2 test_subjects=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19] test_sessions=[3, 4, 5] test_cells=60 hidden_in_train=0 router_cap=0.5 |
| 22:52:36 | START validate |
| 22:52:41 | END validate rc=0 |
| 22:52:41 | START personal_none |
| 23:30:25 | END personal_none rc=0 |
| 23:30:25 | candidate /home/ali_d/codabench/logs/train_sealed_mock_sealed_120_auto/riemann_sealed_cand.py: personal=blend blend_w="auto" adapt=none use_xdawn=True filterbank=True kind=riemann buffer=64 chans=eeg |
| 23:30:25 | START train |
| 23:52:38 | END train rc=0 |
| 23:52:38 | solver: fitting on X=(10800, 43, 480), subjects=20, classes=[0, 1, 2], features=5073, align=subject/riemann, personal=blend, router_thr=0.500, session_ids=yes, chans=eeg (43 kept) |
| 23:52:38 | blend weight: harness w=0.75 (split=replica:3 wcv=loso) vs solver auto w=0.75: MATCH |
| 23:52:39 | START replay |
| 23:53:20 | END replay rc=0 |
| 23:53:21 | gate: train 0.614722 replay 0.614722 EQUAL; harness blend_calib pooled 0.614722 (w=0.75), gap 0.0000 OK |
| 23:53:24 | zipped /home/ali_d/codabench/logs/train_sealed_mock_sealed_120_auto/riemann_sealed_mock_sealed_120_2026-09-28.zip (NOT uploaded) |
| 23:53:24 | ALL DONE |
