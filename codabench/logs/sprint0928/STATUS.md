| time | event |
|---|---|
| 16:55:02 | sprint start (deadline 04:55, reserve from 03:25) |
| 17:02:57 | Phase 0 done: brief a44526a pushed |
| 17:06:40 | START Phase 1 workflow sealed-readiness-build (wf_fe9ec9cf-f0b), ETA 3 h -> ~20:10 |
| 18:43:27 | Phase 1: builders A/B/C/E done (A 17:06-~17:57, B -17:56, E -18:10, C -18:43) |
| 18:45:00 | Phase 1 resumed with inserted step L (fast LDA; bottleneck found by B and C); new ETA ~21:45 (1.6x the 3 h estimate) |
| 21:55:36 | Phase 1 DONE (workflow complete 21:55; 1.8x estimate). START Phase 2 lanes sprint0928_p2.sh: ETA lane A ~115 min, lane B ~90 min -> ~23:55 |
| 21:57:14 | START Phase 4a workflow solver-context-alignment (wf_4c175d8d-d8d): G + R3 + F2, ETA ~1.5 h, overlaps Phase 2 (agents at <= 4 threads; Phase 2 timings inflated accordingly) |
| 22:41:11 | tracks page check #2: Graz + BrainHero still "coming soon" (no contingency triggered) |
| 22:54:06 | Phase 2 lane B (abl120) done 21:55-22:54 (58 min vs 90 est, under). t120 done 22:11-22:52 (41 min vs 35, on). START Phase 2b (500 Hz solver sizing) concurrently with t120_auto (one other 10-thread job + agent G at 4 threads): timings are upper bounds. ETA ~60 min |
| 23:28:12 | Phase 2b DONE 22:54-23:28 (34 min vs 60 est, under): 500 Hz fit 30.4 min, RSS 12.4 GB, replay 77 s (t10) / 66 s (t2), train = replay = 0.612778 -> sizing rule PASSES |
| 23:29:44 | START Phase 4b workflow runbook-claims-and-eda (wf_d9155f1c-cbd): V2 claims checker + V3 release_eda.py, ETA ~1 h |
| 23:53:24 | Phase 2 ALL DONE (21:55-23:53, 1 h 58 min vs ~2 h est, on). t120_auto: MATCH w=0.75, train=replay=harness 0.614722 |
| 23:55:00 | G's patches applied (train_sealed.sh bakes align=subject_context; summarizer NOTE); R3 review of G still running |
| 23:55:09 | START Phase 4c workflow runbook-fix-and-verify (wf_7738f1de-e8f): S scripts + W runbook rewrite -> V1 literal run on mock -> F4, ETA ~2.5 h (-> ~02:30) |
| 00:43:59 | Phase 4a DONE (workflow 21:57-00:43): G subject_context, R3 pass_with_fixes (1 minor), F2 fix; committed |
