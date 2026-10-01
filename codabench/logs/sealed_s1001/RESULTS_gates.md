# Sprint 2026-10-01 Phase 2 gates (dual LDA): 2026-10-01 18:25:17

- s1001_regress:  18:11:42  gate: train 0.770000 replay 0.770000 EQUAL; harness blend_calib pooled 0.778333 (w=0.75), gap 0.0083 OK 
- s1001_mock_default:  18:19:07  gate: train 0.483333 replay 0.483333 EQUAL; harness blend_calib pooled 0.483333 (w=0.75), gap 0.0000 OK 
- s1001_mock_bpt4:  18:25:16  gate: train 0.512500 replay 0.512500 EQUAL; harness blend_calib pooled 0.512500 (w=0.5), gap 0.0000 OK 
- xblocks_gate: GATE 1 (defaults bit-identical): PASS;GATE 2 (harness parity, icoh): PASS;ALL GATES PASS;

## sealed_s1001_regress vs sealed_regress_f0929 (zhou2016_xsess)

| key | old cell | new cell | d cell | d pooled | w old/new | max abs dP |
|---|---|---|---|---|---|---|
| `zhou2016_xsess|all|meanlr|pooled|none|33` | 0.435000 | 0.435000 | 0.0e+00 | 0.0e+00 | - | 0.0e+00 |
| `zhou2016_xsess|all|meanlr|pooled|router-psd:riemann|33` | 0.435000 | 0.435000 | 0.0e+00 | 0.0e+00 | - | 0.0e+00 |
| `zhou2016_xsess|all|meanlr|persubject|none|33` | 0.456667 | 0.456667 | 0.0e+00 | 0.0e+00 | - | 0.0e+00 |
| `zhou2016_xsess|all|meanlr|persubject|router-psd:riemann|33` | 0.460000 | 0.460000 | 0.0e+00 | 0.0e+00 | - | 0.0e+00 |
| `zhou2016_xsess|all|riemann:xd=1,fb=1|pooled|none|33` | 0.771667 | 0.771667 | 0.0e+00 | 0.0e+00 | - | 1.0e-12 |
| `zhou2016_xsess|all|riemann:xd=1,fb=1|pooled|router-psd:riemann|33` | 0.766667 | 0.766667 | 0.0e+00 | 0.0e+00 | - | 2.1e-13 |
| `zhou2016_xsess|all|riemann:xd=1,fb=1|persubject|none|33` | 0.708333 | 0.708333 | 0.0e+00 | 0.0e+00 | - | 5.9e-13 |
| `zhou2016_xsess|all|riemann:xd=1,fb=1|persubject|router-psd:riemann|33` | 0.688333 | 0.688333 | 0.0e+00 | 0.0e+00 | - | 2.0e-13 |
| `zhou2016_xsess|all|riemann:xd=1,fb=1/pooled|none|router-psd:riemann|33` | 0.766667 | 0.766667 | 0.0e+00 | 0.0e+00 | 0.5/0.5; 0.75/0.75 | 2.1e-13 |
| `zhou2016_xsess|all|riemann:xd=1,fb=1/calib|oracle-id|router-psd:riemann|33` | 0.708333 | 0.708333 | 0.0e+00 | 0.0e+00 | 0.5/0.5; 0.75/0.75 | 6.4e-13 |
| `zhou2016_xsess|all|riemann:xd=1,fb=1/calib|router-id|router-psd:riemann|33` | 0.708333 | 0.708333 | 0.0e+00 | 0.0e+00 | 0.5/0.5; 0.75/0.75 | 6.4e-13 |
| `zhou2016_xsess|all|riemann:xd=1,fb=1/persubject|oracle-id|router-psd:riemann|33` | 0.688333 | 0.688333 | 0.0e+00 | 0.0e+00 | 0.5/0.5; 0.75/0.75 | 2.0e-13 |
| `zhou2016_xsess|all|riemann:xd=1,fb=1/persubject|router-id|router-psd:riemann|33` | 0.688333 | 0.688333 | 0.0e+00 | 0.0e+00 | 0.5/0.5; 0.75/0.75 | 2.0e-13 |
| `zhou2016_xsess|all|riemann:xd=1,fb=1/blend|oracle-id|router-psd:riemann|33` | 0.756667 | 0.756667 | 0.0e+00 | 0.0e+00 | 0.5/0.5; 0.75/0.75 | 1.1e-13 |
| `zhou2016_xsess|all|riemann:xd=1,fb=1/blend|router-id|router-psd:riemann|33` | 0.756667 | 0.756667 | 0.0e+00 | 0.0e+00 | 0.5/0.5; 0.75/0.75 | 1.1e-13 |
| `zhou2016_xsess|all|riemann:xd=1,fb=1/blend_calib|oracle-id|router-psd:riemann|33` | 0.778333 | 0.778333 | 0.0e+00 | 0.0e+00 | 0.5/0.5; 0.75/0.75 | 2.7e-13 |
| `zhou2016_xsess|all|riemann:xd=1,fb=1/blend_calib|router-id|router-psd:riemann|33` | 0.778333 | 0.778333 | 0.0e+00 | 0.0e+00 | 0.5/0.5; 0.75/0.75 | 2.7e-13 |

17 new rows, 17 matched; worst |dP| 1.01e-12
COMPARE zhou2016_xsess: PASS

## sealed_s1001_mock_default vs sealed_f0929_mock_default (mock_sealed_s)

| key | old cell | new cell | d cell | d pooled | w old/new | max abs dP |
|---|---|---|---|---|---|---|
| `mock_sealed_s|all|meanlr|pooled|none|33|split=replica:3|router_cap=0.5` | 0.450000 | 0.450000 | 0.0e+00 | 0.0e+00 | - | 0.0e+00 |
| `mock_sealed_s|all|meanlr|pooled|router-psd:riemann|33|split=replica:3|router_cap=0.5` | 0.472222 | 0.472222 | 0.0e+00 | 0.0e+00 | - | 0.0e+00 |
| `mock_sealed_s|all|meanlr|persubject|none|33|split=replica:3|router_cap=0.5` | 0.394444 | 0.394444 | 0.0e+00 | 0.0e+00 | - | 0.0e+00 |
| `mock_sealed_s|all|meanlr|persubject|router-psd:riemann|33|split=replica:3|router_cap=0.5` | 0.415278 | 0.415278 | 0.0e+00 | 0.0e+00 | - | 0.0e+00 |
| `mock_sealed_s|all|riemann:xd=1,fb=1|pooled|none|33|split=replica:3|router_cap=0.5` | 0.472222 | 0.472222 | 0.0e+00 | 0.0e+00 | - | 3.6e-13 |
| `mock_sealed_s|all|riemann:xd=1,fb=1|pooled|router-psd:riemann|33|split=replica:3|router_cap=0.5` | 0.480556 | 0.480556 | 0.0e+00 | 0.0e+00 | - | 1.7e-13 |
| `mock_sealed_s|all|riemann:xd=1,fb=1|persubject|none|33|split=replica:3|router_cap=0.5` | 0.459722 | 0.459722 | 0.0e+00 | 0.0e+00 | - | 3.1e-13 |
| `mock_sealed_s|all|riemann:xd=1,fb=1|persubject|router-psd:riemann|33|split=replica:3|router_cap=0.5` | 0.452778 | 0.452778 | 0.0e+00 | 0.0e+00 | - | 5.6e-14 |
| `mock_sealed_s|all|riemann:xd=1,fb=1/pooled|none|router-psd:riemann|33|split=replica:3|router_cap=0.5|wvariant=calib` | 0.480556 | 0.480556 | 0.0e+00 | 0.0e+00 | 0.75/0.75 | 1.7e-13 |
| `mock_sealed_s|all|riemann:xd=1,fb=1/calib|oracle-id|router-psd:riemann|33|split=replica:3|router_cap=0.5|wvariant=calib` | 0.472222 | 0.472222 | 0.0e+00 | 0.0e+00 | 0.75/0.75 | 3.2e-13 |
| `mock_sealed_s|all|riemann:xd=1,fb=1/calib|router-id|router-psd:riemann|33|split=replica:3|router_cap=0.5|wvariant=calib` | 0.468056 | 0.468056 | 0.0e+00 | 0.0e+00 | 0.75/0.75 | 3.5e-13 |
| `mock_sealed_s|all|riemann:xd=1,fb=1/blend_calib|oracle-id|router-psd:riemann|33|split=replica:3|router_cap=0.5|wvariant=calib` | 0.483333 | 0.483333 | 0.0e+00 | 0.0e+00 | 0.75/0.75 | 1.3e-13 |
| `mock_sealed_s|all|riemann:xd=1,fb=1/blend_calib|router-id|router-psd:riemann|33|split=replica:3|router_cap=0.5|wvariant=calib` | 0.483333 | 0.483333 | 0.0e+00 | 0.0e+00 | 0.75/0.75 | 1.3e-13 |

13 new rows, 13 matched; worst |dP| 3.63e-13
COMPARE mock_sealed_s: PASS

## sealed_s1001_mock_bpt4 vs sealed_f0929_mock_bpt4 (mock_sealed_s)

| key | old cell | new cell | d cell | d pooled | w old/new | max abs dP |
|---|---|---|---|---|---|---|
| `mock_sealed_s|all|meanlr|pooled|none|33|split=replica:3|router_cap=0.5` | 0.450000 | 0.450000 | 0.0e+00 | 0.0e+00 | - | 0.0e+00 |
| `mock_sealed_s|all|meanlr|pooled|router-psd:riemann|33|split=replica:3|router_cap=0.5` | 0.472222 | 0.472222 | 0.0e+00 | 0.0e+00 | - | 0.0e+00 |
| `mock_sealed_s|all|meanlr|persubject|none|33|split=replica:3|router_cap=0.5` | 0.394444 | 0.394444 | 0.0e+00 | 0.0e+00 | - | 0.0e+00 |
| `mock_sealed_s|all|meanlr|persubject|router-psd:riemann|33|split=replica:3|router_cap=0.5` | 0.415278 | 0.415278 | 0.0e+00 | 0.0e+00 | - | 0.0e+00 |
| `mock_sealed_s|all|riemann:xd=1,fb=1,x=bpt4|pooled|none|33|split=replica:3|router_cap=0.5` | 0.523611 | 0.523611 | 0.0e+00 | 0.0e+00 | - | 7.3e-12 |
| `mock_sealed_s|all|riemann:xd=1,fb=1,x=bpt4|pooled|router-psd:riemann|33|split=replica:3|router_cap=0.5` | 0.504167 | 0.504167 | 0.0e+00 | 0.0e+00 | - | 3.2e-12 |
| `mock_sealed_s|all|riemann:xd=1,fb=1,x=bpt4|persubject|none|33|split=replica:3|router_cap=0.5` | 0.550000 | 0.550000 | 0.0e+00 | 0.0e+00 | - | 1.5e-11 |
| `mock_sealed_s|all|riemann:xd=1,fb=1,x=bpt4|persubject|router-psd:riemann|33|split=replica:3|router_cap=0.5` | 0.480556 | 0.480556 | 0.0e+00 | 0.0e+00 | - | 1.8e-12 |
| `mock_sealed_s|all|riemann:xd=1,fb=1,x=bpt4/pooled|none|router-psd:riemann|33|split=replica:3|router_cap=0.5|wvariant=calib` | 0.504167 | 0.504167 | 0.0e+00 | 0.0e+00 | 0.5/0.5 | 3.2e-12 |
| `mock_sealed_s|all|riemann:xd=1,fb=1,x=bpt4/calib|oracle-id|router-psd:riemann|33|split=replica:3|router_cap=0.5|wvariant=calib` | 0.493056 | 0.493056 | 0.0e+00 | 0.0e+00 | 0.5/0.5 | 4.9e-12 |
| `mock_sealed_s|all|riemann:xd=1,fb=1,x=bpt4/calib|router-id|router-psd:riemann|33|split=replica:3|router_cap=0.5|wvariant=calib` | 0.488889 | 0.488889 | 0.0e+00 | 0.0e+00 | 0.5/0.5 | 4.9e-12 |
| `mock_sealed_s|all|riemann:xd=1,fb=1,x=bpt4/blend_calib|oracle-id|router-psd:riemann|33|split=replica:3|router_cap=0.5|wvariant=calib` | 0.515278 | 0.515278 | 0.0e+00 | 0.0e+00 | 0.5/0.5 | 2.4e-12 |
| `mock_sealed_s|all|riemann:xd=1,fb=1,x=bpt4/blend_calib|router-id|router-psd:riemann|33|split=replica:3|router_cap=0.5|wvariant=calib` | 0.512500 | 0.512500 | 0.0e+00 | 0.0e+00 | 0.5/0.5 | 2.4e-12 |

13 new rows, 13 matched; worst |dP| 1.50e-11
COMPARE mock_sealed_s: PASS

## sealed_s1001p vs sealed_f0929p (scherer2015)

| key | old cell | new cell | d cell | d pooled | w old/new | max abs dP |
|---|---|---|---|---|---|---|
| `scherer2015|0-1-3|riemann:xd=1,fb=1,x=bpt4+icoh/pooled|none|router-psd:riemann|33` | 0.472487 | 0.472487 | 0.0e+00 | 0.0e+00 | 0.5/0.5; 0.5/0.5 | 7.0e-13 |
| `scherer2015|0-1-3|riemann:xd=1,fb=1,x=bpt4+icoh/calib|oracle-id|router-psd:riemann|33` | 0.545370 | 0.545370 | 0.0e+00 | 0.0e+00 | 0.5/0.5; 0.5/0.5 | 1.3e-12 |
| `scherer2015|0-1-3|riemann:xd=1,fb=1,x=bpt4+icoh/calib|router-id|router-psd:riemann|33` | 0.544048 | 0.544048 | 0.0e+00 | 0.0e+00 | 0.5/0.5; 0.5/0.5 | 1.3e-12 |
| `scherer2015|0-1-3|riemann:xd=1,fb=1,x=bpt4+icoh/persubject|oracle-id|router-psd:riemann|33` | 0.542328 | 0.542328 | 0.0e+00 | 0.0e+00 | 0.5/0.5; 0.5/0.5 | 1.1e-12 |
| `scherer2015|0-1-3|riemann:xd=1,fb=1,x=bpt4+icoh/persubject|router-id|router-psd:riemann|33` | 0.537302 | 0.537302 | 0.0e+00 | 0.0e+00 | 0.5/0.5; 0.5/0.5 | 1.1e-12 |
| `scherer2015|0-1-3|riemann:xd=1,fb=1,x=bpt4+icoh/blend|oracle-id|router-psd:riemann|33` | 0.553968 | 0.553968 | 0.0e+00 | 0.0e+00 | 0.5/0.5; 0.5/0.5 | 6.3e-13 |
| `scherer2015|0-1-3|riemann:xd=1,fb=1,x=bpt4+icoh/blend|router-id|router-psd:riemann|33` | 0.550132 | 0.550132 | 0.0e+00 | 0.0e+00 | 0.5/0.5; 0.5/0.5 | 6.4e-13 |
| `scherer2015|0-1-3|riemann:xd=1,fb=1,x=bpt4+icoh/blend_calib|oracle-id|router-psd:riemann|33` | 0.553307 | 0.553307 | 0.0e+00 | 0.0e+00 | 0.5/0.5; 0.5/0.5 | 6.6e-13 |
| `scherer2015|0-1-3|riemann:xd=1,fb=1,x=bpt4+icoh/blend_calib|router-id|router-psd:riemann|33` | 0.551058 | 0.551058 | 0.0e+00 | 0.0e+00 | 0.5/0.5; 0.5/0.5 | 6.6e-13 |

9 new rows, 9 matched; worst |dP| 1.33e-12
COMPARE scherer2015: PASS

## sealed_s1001_abl_s_A,sealed_s1001_abl_s_B vs sealed_f0929_abl_s_A,sealed_f0929_abl_s_B (mock_sealed_s)

| key | old cell | new cell | d cell | d pooled | w old/new | max abs dP |
|---|---|---|---|---|---|---|
| `mock_sealed_s|all|riemann:xd=1,fb=1,x=bpt4/pooled|none|router-psd:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|router_cap=0.5|wvariant=calib` | 0.545833 | 0.545833 | 0.0e+00 | 0.0e+00 | 0.5/0.5 | 3.7e-12 |
| `mock_sealed_s|all|riemann:xd=1,fb=1,x=bpt4/calib|oracle-id|router-psd:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|router_cap=0.5|wvariant=calib` | 0.518056 | 0.518056 | 0.0e+00 | 0.0e+00 | 0.5/0.5 | 3.6e-12 |
| `mock_sealed_s|all|riemann:xd=1,fb=1,x=bpt4/calib|router-id|router-psd:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|router_cap=0.5|wvariant=calib` | 0.516667 | 0.516667 | 0.0e+00 | 0.0e+00 | 0.5/0.5 | 3.6e-12 |
| `mock_sealed_s|all|riemann:xd=1,fb=1,x=bpt4/blend_calib|oracle-id|router-psd:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|router_cap=0.5|wvariant=calib` | 0.544444 | 0.544444 | 0.0e+00 | 0.0e+00 | 0.5/0.5 | 2.7e-12 |
| `mock_sealed_s|all|riemann:xd=1,fb=1,x=bpt4/blend_calib|router-id|router-psd:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|router_cap=0.5|wvariant=calib` | 0.543056 | 0.543056 | 0.0e+00 | 0.0e+00 | 0.5/0.5 | 2.7e-12 |
| `mock_sealed_s|all|riemann:xd=1,fb=1/pooled|none|router-psd:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|router_cap=0.5|wvariant=calib` | 0.544444 | 0.544444 | 0.0e+00 | 0.0e+00 | 0.75/0.75 | 1.4e-14 |
| `mock_sealed_s|all|riemann:xd=1,fb=1/calib|oracle-id|router-psd:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|router_cap=0.5|wvariant=calib` | 0.484722 | 0.484722 | 0.0e+00 | 0.0e+00 | 0.75/0.75 | 7.1e-14 |
| `mock_sealed_s|all|riemann:xd=1,fb=1/calib|router-id|router-psd:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|router_cap=0.5|wvariant=calib` | 0.486111 | 0.486111 | 0.0e+00 | 0.0e+00 | 0.75/0.75 | 7.1e-14 |
| `mock_sealed_s|all|riemann:xd=1,fb=1/blend_calib|oracle-id|router-psd:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|router_cap=0.5|wvariant=calib` | 0.548611 | 0.548611 | 0.0e+00 | 0.0e+00 | 0.75/0.75 | 1.8e-14 |
| `mock_sealed_s|all|riemann:xd=1,fb=1/blend_calib|router-id|router-psd:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|router_cap=0.5|wvariant=calib` | 0.548611 | 0.548611 | 0.0e+00 | 0.0e+00 | 0.75/0.75 | 1.8e-14 |
| `mock_sealed_s|all|riemann:xd=1,fb=1,x=bpt4+icoh/pooled|none|router-psd:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|router_cap=0.5|wvariant=calib` | 0.562500 | 0.562500 | 0.0e+00 | 0.0e+00 | 1.0/1.0 | 2.1e-12 |
| `mock_sealed_s|all|riemann:xd=1,fb=1,x=bpt4+icoh/calib|oracle-id|router-psd:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|router_cap=0.5|wvariant=calib` | 0.504167 | 0.504167 | 0.0e+00 | 0.0e+00 | 1.0/1.0 | 2.6e-12 |
| `mock_sealed_s|all|riemann:xd=1,fb=1,x=bpt4+icoh/calib|router-id|router-psd:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|router_cap=0.5|wvariant=calib` | 0.506944 | 0.506944 | 0.0e+00 | 0.0e+00 | 1.0/1.0 | 2.6e-12 |
| `mock_sealed_s|all|riemann:xd=1,fb=1,x=bpt4+icoh/blend_calib|oracle-id|router-psd:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|router_cap=0.5|wvariant=calib` | 0.562500 | 0.562500 | 0.0e+00 | 0.0e+00 | 1.0/1.0 | 2.1e-12 |
| `mock_sealed_s|all|riemann:xd=1,fb=1,x=bpt4+icoh/blend_calib|router-id|router-psd:riemann|33|split=calib:3|test_subjects=0,1,2,3,4,5,6,7,8,9|router_cap=0.5|wvariant=calib` | 0.562500 | 0.562500 | 0.0e+00 | 0.0e+00 | 1.0/1.0 | 2.1e-12 |

15 new rows, 15 matched; worst |dP| 3.69e-12
COMPARE mock_sealed_s: PASS
