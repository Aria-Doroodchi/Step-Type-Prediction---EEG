# Sprint 2026-10-01 Phase 4 re-verification after the strict review fixes, 2026-10-01 18:58:48

## s1001_rep2_all
-  18:58:20  blend weight: harness w=0.75 (split=calib:3 wcv=loso wref=all) vs solver auto w=0.75: MATCH 
-  18:58:20  fold scores per w: solver [0.4819, 0.484, 0.5201, 0.5889, 0.5861] vs harness [0.4819,0.484,0.5201,0.5889,0.5861]: EQUAL (4 dp) (information; the weight gate is the line above) 
-  18:58:33  gate: train 0.602778 replay 0.602778 EQUAL; harness blend_calib pooled 0.602778 (w=0.75), gap 0.0000 OK 

## s1001_rep2_strict
-  18:58:37  blend weight: harness w=0.75 (split=calib:3 wcv=loso wref=strict) vs solver auto w=0.75: MATCH 
-  18:58:37  fold scores per w: solver [0.4854, 0.4861, 0.5181, 0.5799, 0.5785] vs harness [0.4854,0.4861,0.5181,0.5799,0.5785]: EQUAL (4 dp) (information; the weight gate is the line above) 
-  18:58:47  gate: train 0.602778 replay 0.602778 EQUAL; harness blend_calib pooled 0.602778 (w=0.75), gap 0.0000 OK 

## rows: s1001_rep2_all vs committed replica_mock_sealed_s
13 new rows, 13 matched; worst |dP| 6.79e-13
COMPARE mock_sealed_s: PASS

## rows: s1001_rep2_strict vs s1001_rep_strict (before the fixes)
13 new rows, 13 matched; worst |dP| 0.00e+00
COMPARE mock_sealed_s: PASS
