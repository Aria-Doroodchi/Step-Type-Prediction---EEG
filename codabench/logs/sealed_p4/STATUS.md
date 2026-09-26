| time | event |
|---|---|
| 21:56:23 | ETA p4 lane A (Scherer 3-class ablation + personalisation): ~75 min, done by ~23:15; lane B after p3b. zyma configs already done early (logs/sealed_p4/zyma_early.log, 21:10-21:33) |
| 21:56:25 | config ALIGN_C=router-psd:riemann ALIGN_RD=online-64:riemann |
| 21:56:26 | START s3_ablate (ETA 30 min) |
| 22:26:36 | ETA revised: s3_ablate is 48 configs (8 specs x 2 modes x 3 aligns), 28 done at 30 min -> ~50 min (1.7x), steady progress; lane A ~23:20 |
| 22:34:01 | END s3_ablate rc=0 |
| 22:34:01 | START s3_personal (ETA 15 min) |
| 22:41:41 | END s3_personal rc=0 |
| 22:41:41 | START s3_personal_rd (ETA 15 min) |
| 22:49:15 | END s3_personal_rd rc=0 |
| 22:49:15 | START s3_personal_nx (ETA 15 min) |
| 22:54:33 | END s3_personal_nx rc=0 |
| 22:54:33 | lanes A finished; zyma not done |
| 22:54:37 | ETA p4 lane B: zyma (configs already done) + s3_eeg ~30 + s3_eeg_al ~20 = ~50 min, done by ~23:50 |
| 22:54:40 | config ALIGN_C=router-psd:riemann ALIGN_RD=online-64:riemann |
| 22:54:40 | START zyma (ETA 20 min) |
| 22:54:47 | END zyma rc=0 |
| 22:54:47 | START s3_eeg (ETA 30 min) |
| 23:06:03 | END s3_eeg rc=0 |
| 23:06:03 | START s3_eeg_al (ETA 20 min) |
| 23:17:12 | END s3_eeg_al rc=0 |
| 23:17:15 | ALL DONE (RESULTS.md written) |
