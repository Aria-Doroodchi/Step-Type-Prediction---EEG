| time | event |
|---|---|
| 18:16:41 | ETA p1 ≈ 70 min (2 lanes: Tangermann+Zhou / Scherer), done by ~19:35 |
| 18:16:46 | START t_fast (ETA 4 min) |
| 18:22:17 | INCIDENT: first launch thrashed (OpenBLAS 20 thr x 2 lanes), killed; relaunch with BLAS/OMP capped at 10 |
| 18:22:19 | START s_fast (ETA 5 min) |
| 18:26:05 | END t_fast rc=0 |
| 18:26:05 | START t_bd (ETA 8 min) |
| 18:27:52 | END s_fast rc=0 |
| 18:27:52 | START s_bd (ETA 8 min) |
| 18:37:06 | END t_bd rc=0 |
| 18:37:06 | START t_st (ETA 40 min) |
| 18:38:48 | END s_bd rc=0 |
| 18:38:48 | START s_st (ETA 40 min) |
| 18:46:45 | ETA revised: EEGNet-ST pooled Tangermann ~11 min/run (ES up to 100 ep + refit, ~3 s/epoch), not ~7; lane B (Scherer) ~19:10, lane A (Tangermann+Zhou) ~20:05. Log is silent between configs (no per-epoch lines in this run's code) - idle >600 s here is not a stall while CPU ~930 % |
| 19:03:29 | END s_st rc=0 |
| 19:24:50 | END t_st rc=0 |
| 19:24:50 | START z_fast (ETA 2 min) |
| 19:25:54 | END z_fast rc=0 |
| 19:25:54 | START z_bd (ETA 3 min) |
| 19:28:04 | END z_bd rc=0 |
| 19:28:04 | START z_st (ETA 12 min) |
| 19:36:54 | END z_st rc=0 |
| 19:36:56 | ALL DONE (RESULTS.md written) |
