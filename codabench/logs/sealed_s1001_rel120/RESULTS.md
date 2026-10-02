# Release ablations: s1001_rel120 / mock_sealed_120

Split calib:3 (test subjects 0,1,2,3,4,5,6,7,8,9), router cap 0.5, X=(14400, 43, 480) (after the channel pick), train=7200 test=3600 windows, test cells per row [60] (OK: one value), contexts ['brainhero', 'graz']. Cell = balanced accuracy averaged over subject x session x context cells of the test sessions; pooled = over all test windows. Recipe: riemann:xd=1,fb=1,x=bpt4, router-psd:riemann, chans eeg, pool all, wcv last. Rows read from sealed_s1001_rel120, sealed_s1001_rel120_A, sealed_s1001_rel120_B.

## Recipe variants (blend_calib, router ids)

| variant | cell | vs recipe | brainhero | graz | pooled | w | router acc | s |
|---|---|---|---|---|---|---|---|---|
| base (recipe) | 0.6278 |  | 0.629 | 0.627 | 0.6278 | 0.5 | 0.997 | 573 |
| chans eeg+eog | 0.6342 | +0.64 | 0.633 | 0.636 | 0.6342 | 0.5 | 0.998 | 603 |
| chans eeg+emg | 0.5847 | -4.31 | 0.590 | 0.579 | 0.5847 | 0.5 | 0.997 | 597 |
| chans all | 0.5953 | -3.25 | 0.596 | 0.594 | 0.5953 | 0.5 | 0.998 | 457 |
| align router-psdctx:riemann | 0.6769 | +4.92 | 0.669 | 0.684 | 0.6769 | 0.75 | 0.999 | 397 |
| no xDAWN (riemann:xd=0,fb=1,x=bpt4) | 0.6367 | +0.89 | 0.636 | 0.637 | 0.6367 | 0.5 | 0.997 | 560 |
| without blocks (riemann:xd=1,fb=1) | 0.6042 | -2.36 | 0.608 | 0.600 | 0.6042 | 0.5 | 0.997 | 402 |
| with blocks (riemann:xd=1,fb=1,x=bpt4+icoh) | 0.6208 | -0.69 | 0.616 | 0.626 | 0.6208 | 0.5 | 0.997 | 573 |
| wcv loso | 0.6278 | +0.00 | 0.629 | 0.627 | 0.6278 | 0.5 | 0.997 | 744 |
| pool test | 0.6353 | +0.75 | 0.641 | 0.630 | 0.6353 | 0.5 | 0.997 | 227 |
| online-64 (RULE-DEPENDENT) | 0.6497 | +2.19 | 0.646 | 0.653 | 0.6497 | 0.5 | 0.997 | 339 |

## All rows

| spec | mode | align | chans | pool | wcv | cell | brainhero | graz | pooled | n_cells | router acc | fallback | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| meanlr | persubject | none | eeg | all |  | 0.3914 | 0.391 | 0.392 | 0.3914 | 60 |  |  | 0 |
| meanlr | persubject | router-psd:riemann | eeg | all |  | 0.3908 | 0.387 | 0.395 | 0.3908 | 60 | 0.997 | 0.000 | 18 |
| meanlr | persubject | router-psdctx:riemann | eeg | all |  | 0.3972 | 0.392 | 0.403 | 0.3972 | 60 | 0.999 | 0.000 | 16 |
| meanlr | pooled | none | eeg | all |  | 0.4219 | 0.420 | 0.424 | 0.4219 | 60 |  |  | 0 |
| meanlr | pooled | router-psd:riemann | eeg | all |  | 0.4136 | 0.407 | 0.420 | 0.4136 | 60 | 0.997 | 0.000 | 36 |
| meanlr | pooled | router-psdctx:riemann | eeg | all |  | 0.4164 | 0.406 | 0.427 | 0.4164 | 60 | 0.999 | 0.000 | 34 |
| riemann:xd=0,fb=1,x=bpt4/blend_calib | oracle-id | router-psd:riemann | eeg | all | last | 0.6364 | 0.636 | 0.637 | 0.6364 | 60 | 0.997 | 0.000 | 560 |
| riemann:xd=0,fb=1,x=bpt4/blend_calib | router-id | router-psd:riemann | eeg | all | last | 0.6367 | 0.636 | 0.637 | 0.6367 | 60 | 0.997 | 0.000 | 560 |
| riemann:xd=0,fb=1,x=bpt4/calib | oracle-id | router-psd:riemann | eeg | all | last | 0.6228 | 0.625 | 0.621 | 0.6228 | 60 | 0.997 | 0.000 | 560 |
| riemann:xd=0,fb=1,x=bpt4/calib | router-id | router-psd:riemann | eeg | all | last | 0.6233 | 0.625 | 0.622 | 0.6233 | 60 | 0.997 | 0.000 | 560 |
| riemann:xd=0,fb=1,x=bpt4/pooled | none | router-psd:riemann | eeg | all | last | 0.6222 | 0.607 | 0.638 | 0.6222 | 60 | 0.997 | 0.000 | 560 |
| riemann:xd=1,fb=1,x=bpt4 | persubject | none | eeg | all |  | 0.6719 | 0.658 | 0.686 | 0.6719 | 60 |  |  | 88 |
| riemann:xd=1,fb=1,x=bpt4 | persubject | router-psd:riemann | eeg | all |  | 0.6017 | 0.602 | 0.602 | 0.6017 | 60 | 0.997 | 0.000 | 104 |
| riemann:xd=1,fb=1,x=bpt4 | persubject | router-psdctx:riemann | eeg | all |  | 0.5950 | 0.594 | 0.596 | 0.5950 | 60 | 0.999 | 0.000 | 101 |
| riemann:xd=1,fb=1,x=bpt4 | pooled | none | eeg | all |  | 0.6136 | 0.600 | 0.627 | 0.6136 | 60 |  |  | 145 |
| riemann:xd=1,fb=1,x=bpt4 | pooled | router-psd:riemann | eeg | all |  | 0.5964 | 0.578 | 0.615 | 0.5964 | 60 | 0.997 | 0.000 | 165 |
| riemann:xd=1,fb=1,x=bpt4 | pooled | router-psdctx:riemann | eeg | all |  | 0.6683 | 0.659 | 0.678 | 0.6683 | 60 | 0.999 | 0.000 | 163 |
| riemann:xd=1,fb=1,x=bpt4+icoh/blend_calib | oracle-id | router-psd:riemann | eeg | all | last | 0.6208 | 0.616 | 0.626 | 0.6208 | 60 | 0.997 | 0.000 | 573 |
| riemann:xd=1,fb=1,x=bpt4+icoh/blend_calib | router-id | router-psd:riemann | eeg | all | last | 0.6208 | 0.616 | 0.626 | 0.6208 | 60 | 0.997 | 0.000 | 573 |
| riemann:xd=1,fb=1,x=bpt4+icoh/calib | oracle-id | router-psd:riemann | eeg | all | last | 0.5944 | 0.592 | 0.597 | 0.5944 | 60 | 0.997 | 0.000 | 573 |
| riemann:xd=1,fb=1,x=bpt4+icoh/calib | router-id | router-psd:riemann | eeg | all | last | 0.5944 | 0.592 | 0.597 | 0.5944 | 60 | 0.997 | 0.000 | 573 |
| riemann:xd=1,fb=1,x=bpt4+icoh/pooled | none | router-psd:riemann | eeg | all | last | 0.6022 | 0.589 | 0.616 | 0.6022 | 60 | 0.997 | 0.000 | 573 |
| riemann:xd=1,fb=1,x=bpt4/blend_calib | oracle-id | online-64:riemann | eeg | all | last | 0.6494 | 0.646 | 0.653 | 0.6494 | 60 | 0.997 | 0.000 | 339 |
| riemann:xd=1,fb=1,x=bpt4/blend_calib | oracle-id | router-psd:riemann | all | all | last | 0.5947 | 0.596 | 0.593 | 0.5947 | 60 | 0.998 | 0.000 | 457 |
| riemann:xd=1,fb=1,x=bpt4/blend_calib | oracle-id | router-psd:riemann | eeg | all | last | 0.6278 | 0.629 | 0.627 | 0.6278 | 60 | 0.997 | 0.000 | 573 |
| riemann:xd=1,fb=1,x=bpt4/blend_calib | oracle-id | router-psd:riemann | eeg | all | loso | 0.6278 | 0.629 | 0.627 | 0.6278 | 60 | 0.997 | 0.000 | 744 |
| riemann:xd=1,fb=1,x=bpt4/blend_calib | oracle-id | router-psd:riemann | eeg | test | last | 0.6339 | 0.641 | 0.627 | 0.6339 | 60 | 0.997 | 0.000 | 227 |
| riemann:xd=1,fb=1,x=bpt4/blend_calib | oracle-id | router-psd:riemann | eeg+emg | all | last | 0.5847 | 0.590 | 0.579 | 0.5847 | 60 | 0.997 | 0.000 | 597 |
| riemann:xd=1,fb=1,x=bpt4/blend_calib | oracle-id | router-psd:riemann | eeg+eog | all | last | 0.6342 | 0.633 | 0.636 | 0.6342 | 60 | 0.998 | 0.000 | 603 |
| riemann:xd=1,fb=1,x=bpt4/blend_calib | oracle-id | router-psdctx:riemann | eeg | all | last | 0.6769 | 0.669 | 0.684 | 0.6769 | 60 | 0.999 | 0.000 | 397 |
| riemann:xd=1,fb=1,x=bpt4/blend_calib | router-id | online-64:riemann | eeg | all | last | 0.6497 | 0.646 | 0.653 | 0.6497 | 60 | 0.997 | 0.000 | 339 |
| riemann:xd=1,fb=1,x=bpt4/blend_calib | router-id | router-psd:riemann | all | all | last | 0.5953 | 0.596 | 0.594 | 0.5953 | 60 | 0.998 | 0.000 | 457 |
| riemann:xd=1,fb=1,x=bpt4/blend_calib | router-id | router-psd:riemann | eeg | all | last | 0.6278 | 0.629 | 0.627 | 0.6278 | 60 | 0.997 | 0.000 | 573 |
| riemann:xd=1,fb=1,x=bpt4/blend_calib | router-id | router-psd:riemann | eeg | all | loso | 0.6278 | 0.629 | 0.627 | 0.6278 | 60 | 0.997 | 0.000 | 744 |
| riemann:xd=1,fb=1,x=bpt4/blend_calib | router-id | router-psd:riemann | eeg | test | last | 0.6353 | 0.641 | 0.630 | 0.6353 | 60 | 0.997 | 0.000 | 227 |
| riemann:xd=1,fb=1,x=bpt4/blend_calib | router-id | router-psd:riemann | eeg+emg | all | last | 0.5847 | 0.590 | 0.579 | 0.5847 | 60 | 0.997 | 0.000 | 597 |
| riemann:xd=1,fb=1,x=bpt4/blend_calib | router-id | router-psd:riemann | eeg+eog | all | last | 0.6342 | 0.633 | 0.636 | 0.6342 | 60 | 0.998 | 0.000 | 603 |
| riemann:xd=1,fb=1,x=bpt4/blend_calib | router-id | router-psdctx:riemann | eeg | all | last | 0.6769 | 0.669 | 0.684 | 0.6769 | 60 | 0.999 | 0.000 | 397 |
| riemann:xd=1,fb=1,x=bpt4/calib | oracle-id | online-64:riemann | eeg | all | last | 0.6369 | 0.633 | 0.641 | 0.6369 | 60 | 0.997 | 0.000 | 339 |
| riemann:xd=1,fb=1,x=bpt4/calib | oracle-id | router-psd:riemann | all | all | last | 0.5842 | 0.589 | 0.579 | 0.5842 | 60 | 0.998 | 0.000 | 457 |
| riemann:xd=1,fb=1,x=bpt4/calib | oracle-id | router-psd:riemann | eeg | all | last | 0.6114 | 0.615 | 0.608 | 0.6114 | 60 | 0.997 | 0.000 | 573 |
| riemann:xd=1,fb=1,x=bpt4/calib | oracle-id | router-psd:riemann | eeg | all | loso | 0.6114 | 0.615 | 0.608 | 0.6114 | 60 | 0.997 | 0.000 | 744 |
| riemann:xd=1,fb=1,x=bpt4/calib | oracle-id | router-psd:riemann | eeg | test | last | 0.6119 | 0.620 | 0.604 | 0.6119 | 60 | 0.997 | 0.000 | 227 |
| riemann:xd=1,fb=1,x=bpt4/calib | oracle-id | router-psd:riemann | eeg+emg | all | last | 0.5739 | 0.581 | 0.567 | 0.5739 | 60 | 0.997 | 0.000 | 597 |
| riemann:xd=1,fb=1,x=bpt4/calib | oracle-id | router-psd:riemann | eeg+eog | all | last | 0.6158 | 0.617 | 0.614 | 0.6158 | 60 | 0.998 | 0.000 | 603 |
| riemann:xd=1,fb=1,x=bpt4/calib | oracle-id | router-psdctx:riemann | eeg | all | last | 0.6181 | 0.612 | 0.624 | 0.6181 | 60 | 0.999 | 0.000 | 397 |
| riemann:xd=1,fb=1,x=bpt4/calib | router-id | online-64:riemann | eeg | all | last | 0.6372 | 0.633 | 0.642 | 0.6372 | 60 | 0.997 | 0.000 | 339 |
| riemann:xd=1,fb=1,x=bpt4/calib | router-id | router-psd:riemann | all | all | last | 0.5847 | 0.589 | 0.580 | 0.5847 | 60 | 0.998 | 0.000 | 457 |
| riemann:xd=1,fb=1,x=bpt4/calib | router-id | router-psd:riemann | eeg | all | last | 0.6114 | 0.615 | 0.608 | 0.6114 | 60 | 0.997 | 0.000 | 573 |
| riemann:xd=1,fb=1,x=bpt4/calib | router-id | router-psd:riemann | eeg | all | loso | 0.6114 | 0.615 | 0.608 | 0.6114 | 60 | 0.997 | 0.000 | 744 |
| riemann:xd=1,fb=1,x=bpt4/calib | router-id | router-psd:riemann | eeg | test | last | 0.6133 | 0.620 | 0.607 | 0.6133 | 60 | 0.997 | 0.000 | 227 |
| riemann:xd=1,fb=1,x=bpt4/calib | router-id | router-psd:riemann | eeg+emg | all | last | 0.5739 | 0.581 | 0.567 | 0.5739 | 60 | 0.997 | 0.000 | 597 |
| riemann:xd=1,fb=1,x=bpt4/calib | router-id | router-psd:riemann | eeg+eog | all | last | 0.6161 | 0.617 | 0.615 | 0.6161 | 60 | 0.998 | 0.000 | 603 |
| riemann:xd=1,fb=1,x=bpt4/calib | router-id | router-psdctx:riemann | eeg | all | last | 0.6178 | 0.612 | 0.623 | 0.6178 | 60 | 0.999 | 0.000 | 397 |
| riemann:xd=1,fb=1,x=bpt4/pooled | none | online-64:riemann | eeg | all | last | 0.6033 | 0.581 | 0.626 | 0.6033 | 60 | 0.997 | 0.000 | 339 |
| riemann:xd=1,fb=1,x=bpt4/pooled | none | router-psd:riemann | all | all | last | 0.5631 | 0.557 | 0.569 | 0.5631 | 60 | 0.998 | 0.000 | 457 |
| riemann:xd=1,fb=1,x=bpt4/pooled | none | router-psd:riemann | eeg | all | last | 0.5964 | 0.578 | 0.615 | 0.5964 | 60 | 0.997 | 0.000 | 573 |
| riemann:xd=1,fb=1,x=bpt4/pooled | none | router-psd:riemann | eeg | all | loso | 0.5964 | 0.578 | 0.615 | 0.5964 | 60 | 0.997 | 0.000 | 744 |
| riemann:xd=1,fb=1,x=bpt4/pooled | none | router-psd:riemann | eeg | test | last | 0.5683 | 0.563 | 0.573 | 0.5683 | 60 | 0.997 | 0.000 | 227 |
| riemann:xd=1,fb=1,x=bpt4/pooled | none | router-psd:riemann | eeg+emg | all | last | 0.5536 | 0.548 | 0.559 | 0.5536 | 60 | 0.997 | 0.000 | 597 |
| riemann:xd=1,fb=1,x=bpt4/pooled | none | router-psd:riemann | eeg+eog | all | last | 0.5975 | 0.592 | 0.603 | 0.5975 | 60 | 0.998 | 0.000 | 603 |
| riemann:xd=1,fb=1,x=bpt4/pooled | none | router-psdctx:riemann | eeg | all | last | 0.6683 | 0.659 | 0.678 | 0.6683 | 60 | 0.999 | 0.000 | 397 |
| riemann:xd=1,fb=1/blend_calib | oracle-id | router-psd:riemann | eeg | all | last | 0.6044 | 0.608 | 0.601 | 0.6044 | 60 | 0.997 | 0.000 | 402 |
| riemann:xd=1,fb=1/blend_calib | router-id | router-psd:riemann | eeg | all | last | 0.6042 | 0.608 | 0.600 | 0.6042 | 60 | 0.997 | 0.000 | 402 |
| riemann:xd=1,fb=1/calib | oracle-id | router-psd:riemann | eeg | all | last | 0.5892 | 0.590 | 0.588 | 0.5892 | 60 | 0.997 | 0.000 | 402 |
| riemann:xd=1,fb=1/calib | router-id | router-psd:riemann | eeg | all | last | 0.5886 | 0.590 | 0.587 | 0.5886 | 60 | 0.997 | 0.000 | 402 |
| riemann:xd=1,fb=1/pooled | none | router-psd:riemann | eeg | all | last | 0.6014 | 0.589 | 0.614 | 0.6014 | 60 | 0.997 | 0.000 | 402 |

## DECISIONS: release-day rules (pre-registered 2026-09-28)

```
recipe: riemann:xd=1,fb=1,x=bpt4 / router-psd:riemann / chans eeg / pool all / wcv last: cell 0.6278 (60 cells)
channels  eeg+eog vs eeg: +0.64 pts (95 % CI -0.58, +2.14; 6/10 subjects up): fails (>= +2.0 and CI > 0)
channels  eeg+emg vs eeg: -4.31 pts (95 % CI -6.22, -2.05; 2/10 subjects up): fails (>= +2.0 and CI > 0)
channels  all vs eeg: -3.25 pts (95 % CI -4.94, -1.36; 2/10 subjects up): fails (>= +2.0 and CI > 0)
  -> chans = eeg
context   router-psdctx vs router-psd: +4.92 pts (95 % CI +3.33, +6.33; 9/10 subjects up) -> ADOPT router-psdctx (>= +2.0 and CI > 0)
xDAWN     xd=0 vs xd=1: +0.89 pts (95 % CI +0.00, +1.64; 8/10 subjects up) -> keep xDAWN (drop if xd=0 >= +1.0)
blocks    without vs with the recipe's x=: -2.36 pts (95 % CI -3.64, -1.19; 1/10 subjects up) (per context -2.06, -2.67) -> keep the blocks (an adopted block stays unless the replica is >= +1.0 without it)
blocks    adding icoh vs the recipe: -0.69 pts (95 % CI -2.53, +1.11; 4/10 subjects up) (per context -1.28, -0.11) -> do not add icoh (add only if >= +1.0 and no context worse)
wcv       loso vs last: +0.00 pts (95 % CI +0.00, +0.00; 0/10 subjects up) (w: loso 0.5, last 0.5) -> wcv = loso (loso unless <= -2.0)
pool      test-only vs all: +0.75 pts (95 % CI +0.03, +1.42; 8/10 subjects up) -> pool = all (test only if >= +2.0 and CI > 0)
online-64 (RULE-DEPENDENT, report only) vs clean: +2.19 pts (95 % CI +0.69, +3.78; 7/10 subjects up)

result: spec riemann:xd=1,fb=1,x=bpt4 / align router-psdctx:riemann / chans eeg / pool all / wcv loso
train_sealed.sh: RECIPE_SPEC=riemann:xd=1,fb=1,x=bpt4 RECIPE_ALIGN=router-psdctx:riemann CHANS=eeg WCV=loso BLEND_W=auto SPLIT=calib:3 TEST_SUBJECTS=0,1,2,3,4,5,6,7,8,9 WREF=strict bash ~/codabench/scripts/train_sealed.sh ...
(each rule compared one change with the recipe; train_sealed.sh's harness steps score the combined settings)
NOTE: router-psdctx deploys as Riemann-Sealed align="subject_context" (train_sealed.sh bakes it from RECIPE_ALIGN); it has no online mode, so it cannot be combined with ADAPT=online
```
