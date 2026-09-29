# Release ablations: sprint0928_abl120 / mock_sealed_120

Split calib:3 (test subjects 0,1,2,3,4,5,6,7,8,9), router cap 0.5, X=(14400, 43, 480) (after the channel pick), train=7200 test=3600 windows, test cells per row [60] (OK: one value), contexts ['brainhero', 'graz']. Cell = balanced accuracy averaged over subject x session x context cells of the test sessions; pooled = over all test windows. Recipe: riemann:xd=1,fb=1, router-psd:riemann, chans eeg, pool all, wcv last. Rows read from sealed_sprint0928_abl120, sealed_sprint0928_abl120_A, sealed_sprint0928_abl120_B.

## Recipe variants (blend_calib, router ids)

| variant | cell | vs recipe | brainhero | graz | pooled | w | router acc | s |
|---|---|---|---|---|---|---|---|---|
| base (recipe) | 0.6042 |  | 0.608 | 0.600 | 0.6042 | 0.5 | 0.997 | 603 |
| chans eeg+eog | 0.6144 | +1.0 | 0.624 | 0.604 | 0.6144 | 0.5 | 0.998 | 608 |
| chans eeg+emg | 0.5672 | -3.7 | 0.569 | 0.565 | 0.5672 | 0.5 | 0.997 | 614 |
| chans all | 0.5764 | -2.8 | 0.582 | 0.571 | 0.5764 | 0.5 | 0.998 | 735 |
| align router-psdctx:riemann | 0.6631 | +5.9 | 0.656 | 0.671 | 0.6631 | 1.0 | 0.999 | 520 |
| no xDAWN (riemann:xd=0,fb=1) | 0.6450 | +4.1 | 0.641 | 0.649 | 0.6450 | 0.75 | 0.997 | 552 |
| wcv loso | not run |  | | |  |  |  |  |
| pool test | not run |  | | |  |  |  |  |
| online-64 (RULE-DEPENDENT) | not run |  | | |  |  |  |  |

## All rows

| spec | mode | align | chans | pool | wcv | cell | brainhero | graz | pooled | n_cells | router acc | fallback | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemann:xd=0,fb=1/blend_calib | oracle-id | router-psd:riemann | eeg | all | last | 0.6453 | 0.641 | 0.650 | 0.6453 | 60 | 0.997 | 0.000 | 552 |
| riemann:xd=0,fb=1/blend_calib | router-id | router-psd:riemann | eeg | all | last | 0.6450 | 0.641 | 0.649 | 0.6450 | 60 | 0.997 | 0.000 | 552 |
| riemann:xd=0,fb=1/calib | oracle-id | router-psd:riemann | eeg | all | last | 0.5894 | 0.589 | 0.589 | 0.5894 | 60 | 0.997 | 0.000 | 552 |
| riemann:xd=0,fb=1/calib | router-id | router-psd:riemann | eeg | all | last | 0.5889 | 0.589 | 0.588 | 0.5889 | 60 | 0.997 | 0.000 | 552 |
| riemann:xd=0,fb=1/pooled | none | router-psd:riemann | eeg | all | last | 0.6311 | 0.624 | 0.638 | 0.6311 | 60 | 0.997 | 0.000 | 552 |
| riemann:xd=1,fb=1/blend_calib | oracle-id | router-psd:riemann | all | all | last | 0.5767 | 0.582 | 0.572 | 0.5767 | 60 | 0.998 | 0.000 | 735 |
| riemann:xd=1,fb=1/blend_calib | oracle-id | router-psd:riemann | eeg | all | last | 0.6044 | 0.608 | 0.601 | 0.6044 | 60 | 0.997 | 0.000 | 603 |
| riemann:xd=1,fb=1/blend_calib | oracle-id | router-psd:riemann | eeg+emg | all | last | 0.5678 | 0.569 | 0.566 | 0.5678 | 60 | 0.997 | 0.000 | 614 |
| riemann:xd=1,fb=1/blend_calib | oracle-id | router-psd:riemann | eeg+eog | all | last | 0.6147 | 0.624 | 0.605 | 0.6147 | 60 | 0.998 | 0.000 | 608 |
| riemann:xd=1,fb=1/blend_calib | oracle-id | router-psdctx:riemann | eeg | all | last | 0.6631 | 0.656 | 0.671 | 0.6631 | 60 | 0.999 | 0.000 | 520 |
| riemann:xd=1,fb=1/blend_calib | router-id | router-psd:riemann | all | all | last | 0.5764 | 0.582 | 0.571 | 0.5764 | 60 | 0.998 | 0.000 | 735 |
| riemann:xd=1,fb=1/blend_calib | router-id | router-psd:riemann | eeg | all | last | 0.6042 | 0.608 | 0.600 | 0.6042 | 60 | 0.997 | 0.000 | 603 |
| riemann:xd=1,fb=1/blend_calib | router-id | router-psd:riemann | eeg+emg | all | last | 0.5672 | 0.569 | 0.565 | 0.5672 | 60 | 0.997 | 0.000 | 614 |
| riemann:xd=1,fb=1/blend_calib | router-id | router-psd:riemann | eeg+eog | all | last | 0.6144 | 0.624 | 0.604 | 0.6144 | 60 | 0.998 | 0.000 | 608 |
| riemann:xd=1,fb=1/blend_calib | router-id | router-psdctx:riemann | eeg | all | last | 0.6631 | 0.656 | 0.671 | 0.6631 | 60 | 0.999 | 0.000 | 520 |
| riemann:xd=1,fb=1/calib | oracle-id | router-psd:riemann | all | all | last | 0.5636 | 0.569 | 0.558 | 0.5636 | 60 | 0.998 | 0.000 | 735 |
| riemann:xd=1,fb=1/calib | oracle-id | router-psd:riemann | eeg | all | last | 0.5892 | 0.590 | 0.588 | 0.5892 | 60 | 0.997 | 0.000 | 603 |
| riemann:xd=1,fb=1/calib | oracle-id | router-psd:riemann | eeg+emg | all | last | 0.5525 | 0.559 | 0.546 | 0.5525 | 60 | 0.997 | 0.000 | 614 |
| riemann:xd=1,fb=1/calib | oracle-id | router-psd:riemann | eeg+eog | all | last | 0.5939 | 0.599 | 0.589 | 0.5939 | 60 | 0.998 | 0.000 | 608 |
| riemann:xd=1,fb=1/calib | oracle-id | router-psdctx:riemann | eeg | all | last | 0.6078 | 0.608 | 0.608 | 0.6078 | 60 | 0.999 | 0.000 | 520 |
| riemann:xd=1,fb=1/calib | router-id | router-psd:riemann | all | all | last | 0.5636 | 0.569 | 0.558 | 0.5636 | 60 | 0.998 | 0.000 | 735 |
| riemann:xd=1,fb=1/calib | router-id | router-psd:riemann | eeg | all | last | 0.5886 | 0.590 | 0.587 | 0.5886 | 60 | 0.997 | 0.000 | 603 |
| riemann:xd=1,fb=1/calib | router-id | router-psd:riemann | eeg+emg | all | last | 0.5522 | 0.559 | 0.545 | 0.5522 | 60 | 0.997 | 0.000 | 614 |
| riemann:xd=1,fb=1/calib | router-id | router-psd:riemann | eeg+eog | all | last | 0.5939 | 0.599 | 0.589 | 0.5939 | 60 | 0.998 | 0.000 | 608 |
| riemann:xd=1,fb=1/calib | router-id | router-psdctx:riemann | eeg | all | last | 0.6075 | 0.608 | 0.607 | 0.6075 | 60 | 0.999 | 0.000 | 520 |
| riemann:xd=1,fb=1/pooled | none | router-psd:riemann | all | all | last | 0.5706 | 0.571 | 0.570 | 0.5706 | 60 | 0.998 | 0.000 | 735 |
| riemann:xd=1,fb=1/pooled | none | router-psd:riemann | eeg | all | last | 0.6014 | 0.589 | 0.614 | 0.6014 | 60 | 0.997 | 0.000 | 603 |
| riemann:xd=1,fb=1/pooled | none | router-psd:riemann | eeg+emg | all | last | 0.5636 | 0.561 | 0.566 | 0.5636 | 60 | 0.997 | 0.000 | 614 |
| riemann:xd=1,fb=1/pooled | none | router-psd:riemann | eeg+eog | all | last | 0.6017 | 0.605 | 0.598 | 0.6017 | 60 | 0.998 | 0.000 | 608 |
| riemann:xd=1,fb=1/pooled | none | router-psdctx:riemann | eeg | all | last | 0.6631 | 0.656 | 0.671 | 0.6631 | 60 | 0.999 | 0.000 | 520 |

## DECISIONS: release-day rules (pre-registered 2026-09-28)

```
recipe: riemann:xd=1,fb=1 / router-psd:riemann / chans eeg / pool all / wcv last: cell 0.6042 (60 cells)
channels  eeg+eog vs eeg: +1.0 pts (95 % CI +0.1, +1.9; 7/10 subjects up): fails (>= +2.0 and CI > 0)
channels  eeg+emg vs eeg: -3.7 pts (95 % CI -5.2, -2.4; 0/10 subjects up): fails (>= +2.0 and CI > 0)
channels  all vs eeg: -2.8 pts (95 % CI -4.7, -0.6; 2/10 subjects up): fails (>= +2.0 and CI > 0)
  -> chans = eeg
context   router-psdctx vs router-psd: +5.9 pts (95 % CI +3.8, +8.1; 9/10 subjects up) -> ADOPT router-psdctx (>= +2.0 and CI > 0)
xDAWN     xd=0 vs xd=1: +4.1 pts (95 % CI +2.6, +6.0; 10/10 subjects up) -> DROP xDAWN (drop if xd=0 >= +1.0)
wcv       wcv=loso row missing -> keep wcv=last
pool      pool=test row missing -> keep all
online-64 (rule-dependent) not run

result: spec riemann:xd=0,fb=1 / align router-psdctx:riemann / chans eeg / pool all / wcv last
train_sealed.sh: RECIPE_SPEC=riemann:xd=0,fb=1 RECIPE_ALIGN=router-psdctx:riemann CHANS=eeg WCV=last SPLIT=calib:3 TEST_SUBJECTS=0,1,2,3,4,5,6,7,8,9 bash ~/codabench/scripts/train_sealed.sh ...
(each rule compared one change with the recipe; train_sealed.sh's harness steps score the combined settings)
NOTE: Riemann-Sealed has no (subject, context) router yet: the adopted alignment needs a solver option before it can be trained
```
