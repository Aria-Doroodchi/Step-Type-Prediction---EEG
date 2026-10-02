# step-2 guard check, 2026-10-01 22:25:45 (W1=0.75)

## s1001_final120_neg
-  22:19:15  HARNESS_FROM: seeded logs/sealed_s1001_final120_neg from logs/sealed_s1001_replica120 (13 rows, 13 probs files) 
-  22:19:15  START preflight 
-  22:19:19  END preflight rc=0 
-  22:19:19  START validate 
-  22:19:23  END validate rc=0 
-  22:19:23  START personal_none 
-  22:20:10  END personal_none rc=0 
-  22:20:10  ERROR: BLEND_W=0.5 differs from the harness weight 0.75 seeded from HARNESS_FROM=s1001_replica120: was W1 read from that run's STATUS.md? (FORCE_ZIP=1 overrides) 

## s1001_final120
-  22:20:11  HARNESS_FROM: seeded logs/sealed_s1001_final120 from logs/sealed_s1001_replica120 (13 rows, 13 probs files) 
-  22:20:11  START preflight 
-  22:20:15  END preflight rc=0 
-  22:20:15  START validate 
-  22:20:19  END validate rc=0 
-  22:20:19  START personal_none 
-  22:21:03  END personal_none rc=0 
-  22:21:03  START train 
-  22:25:15  END train rc=0 
-  22:25:15  blend weight: harness w=0.75 (split=calib:3 wcv=loso wref=strict); solver trained with the baked w=0.75 
-  22:25:15  START replay 
-  22:25:44  END replay rc=0 
-  22:25:45  gate: train 0.638889 replay 0.638889 EQUAL; harness blend_calib pooled 0.676944 (w=0.75), gap 0.0381 FLAG 
-  22:25:45  zipped /home/ali_d/codabench/logs/s1001_final120/riemann_sealed_mock_sealed_120_2026-10-01.zip (NOT uploaded) 
