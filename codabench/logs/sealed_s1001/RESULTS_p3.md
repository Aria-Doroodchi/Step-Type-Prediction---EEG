# Sprint 2026-10-01 Phase 3: strict LOSO references, 2026-10-01 18:51:50

## replica_mock_sealed_s
-  01:08:48  blend weight: harness w=0.75 (split=calib:3 wcv=loso) vs solver auto w=0.75: MATCH 
-  01:09:00  gate: train 0.602778 replay 0.602778 EQUAL; harness blend_calib pooled 0.602778 (w=0.75), gap 0.0000 OK 
-  01:34:43  blend weight: harness w=0.75 (split=calib:3 wcv=loso) vs solver auto w=0.75: MATCH 
-  01:34:44  gate: train 0.602778 replay 0.602778 EQUAL; harness blend_calib pooled 0.602778 (w=0.75), gap 0.0000 OK 
- solver: [Riemann-Sealed] blend_w=auto -> 0.75 (loso, 3 folds, context=yes, pair-whitened; cell scores [0.4819, 0.484, 0.5201, 0.5889, 0.5861] for w=[0.0, 0.25, 0.5, 0.75, 1.0])
- harness: [01:04:35]   blend_calib: w_pooled=0.75 (training CV [0.482, 0.484, 0.52, 0.589, 0.586])
- harness fold: [01:02:29]   wcv=loso (loso) fold 1/3: val sessions [0] of 20 subjects (n_fit=960 n_val=480) cell for w=[0.0, 0.25, 0.5, 0.75, 1.0]: blend_calib [0.496, 0.5, 0.538, 0.581, 0.583]
- harness fold: [01:03:33]   wcv=loso (loso) fold 2/3: val sessions [1] of 20 subjects (n_fit=960 n_val=480) cell for w=[0.0, 0.25, 0.5, 0.75, 1.0]: blend_calib [0.469, 0.469, 0.502, 0.585, 0.588]
- harness fold: [01:04:35]   wcv=loso (loso) fold 3/3: val sessions [2] of 20 subjects (n_fit=960 n_val=480) cell for w=[0.0, 0.25, 0.5, 0.75, 1.0]: blend_calib [0.481, 0.483, 0.521, 0.6, 0.588]

## s1001_rep_all
-  18:33:30  blend weight: harness w=0.75 (split=calib:3 wcv=loso wref=all) vs solver auto w=0.75: MATCH 
-  18:33:44  gate: train 0.602778 replay 0.602778 EQUAL; harness blend_calib pooled 0.602778 (w=0.75), gap 0.0000 OK 
- solver: [Riemann-Sealed] blend_w=auto -> 0.75 (loso, 3 folds, context=yes, pair-whitened; cell scores [0.4819, 0.484, 0.5201, 0.5889, 0.5861] for w=[0.0, 0.25, 0.5, 0.75, 1.0])
- harness: [18:31:15]   blend_calib: w_pooled=0.75 (training CV [0.482, 0.484, 0.52, 0.589, 0.586])
- harness fold: [18:30:14]   wcv=loso (loso) fold 1/3: val sessions [0] of 20 subjects (n_fit=960 n_val=480) cell for w=[0.0, 0.25, 0.5, 0.75, 1.0]: blend_calib [0.496, 0.5, 0.538, 0.581, 0.583]
- harness fold: [18:30:44]   wcv=loso (loso) fold 2/3: val sessions [1] of 20 subjects (n_fit=960 n_val=480) cell for w=[0.0, 0.25, 0.5, 0.75, 1.0]: blend_calib [0.469, 0.469, 0.502, 0.585, 0.588]
- harness fold: [18:31:15]   wcv=loso (loso) fold 3/3: val sessions [2] of 20 subjects (n_fit=960 n_val=480) cell for w=[0.0, 0.25, 0.5, 0.75, 1.0]: blend_calib [0.481, 0.483, 0.521, 0.6, 0.588]

## s1001_rep_strict
-  18:41:44  blend weight: harness w=0.75 (split=calib:3 wcv=loso wref=strict) vs solver auto w=0.75: MATCH 
-  18:42:05  gate: train 0.602778 replay 0.602778 EQUAL; harness blend_calib pooled 0.602778 (w=0.75), gap 0.0000 OK 
- solver: [Riemann-Sealed] blend_w=auto -> 0.75 (loso, 3 folds, context=yes, pair-whitened; cell scores [0.4854, 0.4861, 0.5181, 0.5799, 0.5785] for w=[0.0, 0.25, 0.5, 0.75, 1.0])
- harness: [18:39:26]   blend_calib: w_pooled=0.75 (training CV [0.485, 0.486, 0.518, 0.58, 0.578])
- harness fold: [18:38:22]   wcv=loso wref=strict (loso) fold 1/3: val sessions [0] of 20 subjects (n_fit=960 n_val=480) cell for w=[0.0, 0.25, 0.5, 0.75, 1.0]: blend_calib [0.508, 0.51, 0.533, 0.59, 0.588]
- harness fold: [18:38:54]   wcv=loso wref=strict (loso) fold 2/3: val sessions [1] of 20 subjects (n_fit=960 n_val=480) cell for w=[0.0, 0.25, 0.5, 0.75, 1.0]: blend_calib [0.469, 0.467, 0.5, 0.577, 0.569]
- harness fold: [18:39:26]   wcv=loso wref=strict (loso) fold 3/3: val sessions [2] of 20 subjects (n_fit=960 n_val=480) cell for w=[0.0, 0.25, 0.5, 0.75, 1.0]: blend_calib [0.479, 0.481, 0.521, 0.573, 0.579]

## s1001_rep_spsd
-  18:48:54  blend weight: harness w=0.75 (split=calib:3 wcv=loso wref=strict) vs solver auto w=0.75: MATCH 
-  18:49:10  gate: train 0.548611 replay 0.548611 EQUAL; harness blend_calib pooled 0.548611 (w=0.75), gap 0.0000 OK 
- solver: [Riemann-Sealed] blend_w=auto -> 0.75 (loso, 3 folds, context=yes; cell scores [0.4632, 0.4611, 0.4889, 0.509, 0.4986] for w=[0.0, 0.25, 0.5, 0.75, 1.0])
- harness: [18:47:06]   blend_calib: w_pooled=0.75 (training CV [0.463, 0.461, 0.489, 0.509, 0.499])
- harness fold: [18:46:14]   wcv=loso wref=strict (loso) fold 1/3: val sessions [0] of 20 subjects (n_fit=960 n_val=480) cell for w=[0.0, 0.25, 0.5, 0.75, 1.0]: blend_calib [0.477, 0.475, 0.512, 0.527, 0.515]
- harness fold: [18:46:40]   wcv=loso wref=strict (loso) fold 2/3: val sessions [1] of 20 subjects (n_fit=960 n_val=480) cell for w=[0.0, 0.25, 0.5, 0.75, 1.0]: blend_calib [0.442, 0.444, 0.465, 0.5, 0.5]
- harness fold: [18:47:06]   wcv=loso wref=strict (loso) fold 3/3: val sessions [2] of 20 subjects (n_fit=960 n_val=480) cell for w=[0.0, 0.25, 0.5, 0.75, 1.0]: blend_calib [0.471, 0.465, 0.49, 0.5, 0.481]

## s1001_zh_all
-  18:28:37  blend weight: harness w=0.5 (split=last wcv=loso wref=all) vs solver auto w=0.5: MATCH 
-  18:28:58  gate: train 0.741667 replay 0.741667 EQUAL; harness blend_calib pooled 0.760000 (w=0.5), gap 0.0183 OK 
- solver: [Riemann-Sealed] blend_w=auto -> 0.5 (loso, 2 folds, context=no; cell scores [0.6549, 0.6532, 0.6659, 0.6508, 0.6419] for w=[0.0, 0.25, 0.5, 0.75, 1.0])
- harness: [18:28:01]   blend_calib: w_pooled=0.5 (training CV [0.657, 0.658, 0.674, 0.642, 0.636])
- harness fold: [18:27:43]   wcv=loso (loso) fold 1/2: val sessions [0] of 4 subjects (n_fit=586 n_val=614) cell for w=[0.0, 0.25, 0.5, 0.75, 1.0]: blend [0.671, 0.671, 0.682, 0.638, 0.622] blend_calib [0.692, 0.697, 0.699, 0.633, 0.622]
- harness fold: [18:28:01]   wcv=loso (loso) fold 2/2: val sessions [1] of 4 subjects (n_fit=614 n_val=586) cell for w=[0.0, 0.25, 0.5, 0.75, 1.0]: blend [0.619, 0.622, 0.665, 0.658, 0.649] blend_calib [0.621, 0.618, 0.648, 0.651, 0.649]

## s1001_zh_strict
-  18:31:52  blend weight: harness w=0.5 (split=last wcv=loso wref=strict) vs solver auto w=0.5: MATCH 
-  18:32:13  gate: train 0.741667 replay 0.741667 EQUAL; harness blend_calib pooled 0.760000 (w=0.5), gap 0.0183 OK 
- solver: [Riemann-Sealed] blend_w=auto -> 0.5 (loso, 2 folds, context=no; cell scores [0.6502, 0.6564, 0.6698, 0.6614, 0.6577] for w=[0.0, 0.25, 0.5, 0.75, 1.0])
- harness: [18:31:10]   blend_calib: w_pooled=0.5 (training CV [0.65, 0.656, 0.669, 0.662, 0.656])
- harness fold: [18:30:52]   wcv=loso wref=strict (loso) fold 1/2: val sessions [0] of 4 subjects (n_fit=586 n_val=614) cell for w=[0.0, 0.25, 0.5, 0.75, 1.0]: blend [0.673, 0.675, 0.695, 0.649, 0.644] blend_calib [0.697, 0.701, 0.705, 0.653, 0.644]
- harness fold: [18:31:10]   wcv=loso wref=strict (loso) fold 2/2: val sessions [1] of 4 subjects (n_fit=614 n_val=586) cell for w=[0.0, 0.25, 0.5, 0.75, 1.0]: blend [0.6, 0.602, 0.645, 0.674, 0.667] blend_calib [0.604, 0.611, 0.633, 0.671, 0.667]

## harness rows: s1001_rep_all vs replica_mock_sealed_s
| key | old cell | new cell | d cell | d pooled | w old/new | max abs dP |
|---|---|---|---|---|---|---|
| `mock_sealed_s|all|meanlr|pooled|none|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|router_cap=0.5` | 0.434722 | 0.434722 | 0.0e+00 | 0.0e+00 | - | 0.0e+00 |
| `mock_sealed_s|all|meanlr|pooled|router-psdctx:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|router_cap=0.5` | 0.436111 | 0.436111 | 0.0e+00 | 0.0e+00 | - | 0.0e+00 |
| `mock_sealed_s|all|meanlr|persubject|none|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|router_cap=0.5` | 0.375000 | 0.375000 | 0.0e+00 | 0.0e+00 | - | 0.0e+00 |
| `mock_sealed_s|all|meanlr|persubject|router-psdctx:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|router_cap=0.5` | 0.376389 | 0.376389 | 0.0e+00 | 0.0e+00 | - | 0.0e+00 |
| `mock_sealed_s|all|riemann:xd=1,fb=1|pooled|none|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|router_cap=0.5` | 0.526389 | 0.526389 | 0.0e+00 | 0.0e+00 | - | 2.9e-13 |
| `mock_sealed_s|all|riemann:xd=1,fb=1|pooled|router-psdctx:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|router_cap=0.5` | 0.595833 | 0.595833 | 0.0e+00 | 0.0e+00 | - | 1.4e-13 |
| `mock_sealed_s|all|riemann:xd=1,fb=1|persubject|none|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|router_cap=0.5` | 0.493056 | 0.493056 | 0.0e+00 | 0.0e+00 | - | 2.7e-13 |
| `mock_sealed_s|all|riemann:xd=1,fb=1|persubject|router-psdctx:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|router_cap=0.5` | 0.472222 | 0.472222 | 0.0e+00 | 0.0e+00 | - | 5.3e-14 |
| `mock_sealed_s|all|riemann:xd=1,fb=1/pooled|none|router-psdctx:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|wcv=loso|router_cap=0.5|wvariant=calib` | 0.595833 | 0.595833 | 0.0e+00 | 0.0e+00 | 0.75/0.75 | 1.4e-13 |
| `mock_sealed_s|all|riemann:xd=1,fb=1/calib|oracle-id|router-psdctx:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|wcv=loso|router_cap=0.5|wvariant=calib` | 0.501389 | 0.501389 | 0.0e+00 | 0.0e+00 | 0.75/0.75 | 6.8e-13 |
| `mock_sealed_s|all|riemann:xd=1,fb=1/calib|router-id|router-psdctx:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|wcv=loso|router_cap=0.5|wvariant=calib` | 0.500000 | 0.500000 | 0.0e+00 | 0.0e+00 | 0.75/0.75 | 6.8e-13 |
| `mock_sealed_s|all|riemann:xd=1,fb=1/blend_calib|oracle-id|router-psdctx:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|wcv=loso|router_cap=0.5|wvariant=calib` | 0.602778 | 0.602778 | 0.0e+00 | 0.0e+00 | 0.75/0.75 | 1.7e-13 |
| `mock_sealed_s|all|riemann:xd=1,fb=1/blend_calib|router-id|router-psdctx:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|wcv=loso|router_cap=0.5|wvariant=calib` | 0.602778 | 0.602778 | 0.0e+00 | 0.0e+00 | 0.75/0.75 | 1.7e-13 |

13 new rows, 13 matched; worst |dP| 6.79e-13
COMPARE mock_sealed_s: PASS

## information: all vs strict (sealed_personal, riemann:xd=1,fb=1,x=bpt4, router-psd, release settings)

| study | wcv | wref | blend_calib w | CV per w | test cell (router-id) |
|---|---|---|---|---|---|
| scherer2015 | last | all | 0.5 | [0.5898, 0.587, 0.6019, 0.5657, 0.5574] | 0.5399 |
| scherer2015 | last | strict | 0.5 | [0.5889, 0.5898, 0.6019, 0.5574, 0.5537] | 0.5399 |
| tangermann2012 | last | all | 0.5 | [0.7951, 0.7963, 0.8079, 0.7272, 0.7149] | 0.8206 |
| tangermann2012 | last | strict | 0.5 | [0.7975, 0.7986, 0.8071, 0.7269, 0.713] | 0.8206 |
| zhou2016 | loso | all | 0.5 | [0.6472, 0.6513, 0.6837, 0.6747, 0.671] | 0.7983 |
| zhou2016 | loso | strict | 0.75 | [0.6413, 0.6446, 0.6654, 0.6751, 0.6687] | 0.8000 |
