| time | event |
|---|---|
| 23:58:20 | config STUDY=mock_sealed_s SPLIT=calib:3 TEST_SUBJECTS=0,1,2,3,4,5,6,7,8,9 SPEC=riemann:xd=1,fb=1,x=bpt4 ALIGN=router-psd:riemann WCV=last CAP=0.5 context=1 LANES=AB STEPS=base xb xa threads=5 |
| 23:58:23 | preflight: X=(2880, 43, 480) subjects=20 sessions/subject=6 train=1440 test=720 val_session_windows=0 K=3 class_counts=[960, 960, 960] dropped_ch=['EMG1', 'EMG2', 'EOG1', 'EOG2'] threads=5 split=calib:3 pool=all chans=eeg(43/47 ch) ch_types=eeg:43,emg:2,eog:2(meta) contexts=2 test_subjects=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] test_sessions=[3, 4, 5] test_cells=60 hidden_in_train=0 hidden_excluded=720 router_cap=0.5 |
| 23:58:23 | START base (ETA 20 min) |
| 00:02:40 | END xb rc=0 |
| 00:02:40 | START xa (ETA 25 min) |
| 00:04:02 | END base rc=0 |
| 00:11:59 | END xa rc=0 |
| 00:11:59 | lanes AB finished; ch_eeg_eog not done |
