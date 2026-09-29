# Release ablations: sprint0928_abl_s / mock_sealed_s

Split replica:3, router cap 0.5, X=(2880, 43, 480) (after the channel pick), train=2160 test=720 windows, test cells per row [60] (OK: one value), contexts ['brainhero', 'graz']. Cell = balanced accuracy averaged over subject x session x context cells of the test sessions; pooled = over all test windows. Recipe: riemann:xd=1,fb=1, router-psd:riemann, chans eeg, pool all, wcv last. Rows read from sealed_sprint0928_abl_s, sealed_sprint0928_abl_s_A, sealed_sprint0928_abl_s_B.

## Recipe variants (blend_calib, router ids)

| variant | cell | vs recipe | brainhero | graz | pooled | w | router acc | s |
|---|---|---|---|---|---|---|---|---|
| base (recipe) | 0.4833 |  | 0.461 | 0.506 | 0.4833 | 0.75 | 0.979 | 195 |
| chans eeg+eog | 0.5111 | +2.8 | 0.489 | 0.533 | 0.5111 | 0.75 | 0.978 | 205 |
| chans eeg+emg | 0.4917 | +0.8 | 0.461 | 0.522 | 0.4917 | 0.75 | 0.978 | 222 |
| chans all | 0.5069 | +2.4 | 0.508 | 0.506 | 0.5069 | 0.75 | 0.978 | 267 |
| align router-psdctx:riemann | 0.5472 | +6.4 | 0.519 | 0.575 | 0.5472 | 0.75 | 0.994 | 195 |
| no xDAWN (riemann:xd=0,fb=1) | 0.4986 | +1.5 | 0.481 | 0.517 | 0.4986 | 0.75 | 0.979 | 191 |
| wcv loso | 0.4833 | +0.0 | 0.461 | 0.506 | 0.4833 | 0.75 | 0.979 | 853 |
| pool test | 0.4458 | -3.7 | 0.417 | 0.475 | 0.4458 | 0.75 | 0.996 | 87 |
| online-64 (RULE-DEPENDENT) | 0.4917 | +0.8 | 0.453 | 0.531 | 0.4917 | 0.75 | 0.979 | 201 |

## All rows

| spec | mode | align | chans | pool | wcv | cell | brainhero | graz | pooled | n_cells | router acc | fallback | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| meanlr | persubject | none | eeg | all |  | 0.3944 | 0.356 | 0.433 | 0.3944 | 60 |  |  | 0 |
| meanlr | persubject | router-psd:riemann | eeg | all |  | 0.4153 | 0.392 | 0.439 | 0.4153 | 60 | 0.979 | 0.000 | 7 |
| meanlr | persubject | router-psdctx:riemann | eeg | all |  | 0.4083 | 0.392 | 0.425 | 0.4083 | 60 | 0.994 | 0.000 | 6 |
| meanlr | pooled | none | eeg | all |  | 0.4500 | 0.425 | 0.475 | 0.4500 | 60 |  |  | 0 |
| meanlr | pooled | router-psd:riemann | eeg | all |  | 0.4722 | 0.453 | 0.492 | 0.4722 | 60 | 0.979 | 0.000 | 33 |
| meanlr | pooled | router-psdctx:riemann | eeg | all |  | 0.4556 | 0.442 | 0.469 | 0.4556 | 60 | 0.994 | 0.000 | 31 |
| riemann:xd=0,fb=1/blend_calib | oracle-id | router-psd:riemann | eeg | all | last | 0.4986 | 0.481 | 0.517 | 0.4986 | 60 | 0.979 | 0.000 | 191 |
| riemann:xd=0,fb=1/blend_calib | router-id | router-psd:riemann | eeg | all | last | 0.4986 | 0.481 | 0.517 | 0.4986 | 60 | 0.979 | 0.000 | 191 |
| riemann:xd=0,fb=1/calib | oracle-id | router-psd:riemann | eeg | all | last | 0.4542 | 0.436 | 0.472 | 0.4542 | 60 | 0.979 | 0.000 | 191 |
| riemann:xd=0,fb=1/calib | router-id | router-psd:riemann | eeg | all | last | 0.4500 | 0.436 | 0.464 | 0.4500 | 60 | 0.979 | 0.000 | 191 |
| riemann:xd=0,fb=1/pooled | none | router-psd:riemann | eeg | all | last | 0.4931 | 0.467 | 0.519 | 0.4931 | 60 | 0.979 | 0.000 | 191 |
| riemann:xd=1,fb=1 | persubject | none | eeg | all |  | 0.4597 | 0.439 | 0.481 | 0.4597 | 60 |  |  | 30 |
| riemann:xd=1,fb=1 | persubject | router-psd:riemann | eeg | all |  | 0.4528 | 0.436 | 0.469 | 0.4528 | 60 | 0.979 | 0.000 | 36 |
| riemann:xd=1,fb=1 | persubject | router-psdctx:riemann | eeg | all |  | 0.4903 | 0.461 | 0.519 | 0.4903 | 60 | 0.994 | 0.000 | 34 |
| riemann:xd=1,fb=1 | pooled | none | eeg | all |  | 0.4722 | 0.458 | 0.486 | 0.4722 | 60 |  |  | 46 |
| riemann:xd=1,fb=1 | pooled | router-psd:riemann | eeg | all |  | 0.4806 | 0.453 | 0.508 | 0.4806 | 60 | 0.979 | 0.000 | 42 |
| riemann:xd=1,fb=1 | pooled | router-psdctx:riemann | eeg | all |  | 0.5403 | 0.508 | 0.572 | 0.5403 | 60 | 0.994 | 0.000 | 40 |
| riemann:xd=1,fb=1/blend_calib | oracle-id | online-64:riemann | eeg | all | last | 0.4917 | 0.453 | 0.531 | 0.4917 | 60 | 0.979 | 0.000 | 201 |
| riemann:xd=1,fb=1/blend_calib | oracle-id | router-psd:riemann | all | all | last | 0.5056 | 0.508 | 0.503 | 0.5056 | 60 | 0.978 | 0.000 | 267 |
| riemann:xd=1,fb=1/blend_calib | oracle-id | router-psd:riemann | eeg | all | last | 0.4833 | 0.461 | 0.506 | 0.4833 | 60 | 0.979 | 0.000 | 195 |
| riemann:xd=1,fb=1/blend_calib | oracle-id | router-psd:riemann | eeg | all | loso | 0.4833 | 0.461 | 0.506 | 0.4833 | 60 | 0.979 | 0.000 | 853 |
| riemann:xd=1,fb=1/blend_calib | oracle-id | router-psd:riemann | eeg | test | last | 0.4458 | 0.417 | 0.475 | 0.4458 | 60 | 0.996 | 0.000 | 87 |
| riemann:xd=1,fb=1/blend_calib | oracle-id | router-psd:riemann | eeg+emg | all | last | 0.4917 | 0.461 | 0.522 | 0.4917 | 60 | 0.978 | 0.000 | 222 |
| riemann:xd=1,fb=1/blend_calib | oracle-id | router-psd:riemann | eeg+eog | all | last | 0.5125 | 0.489 | 0.536 | 0.5125 | 60 | 0.978 | 0.000 | 205 |
| riemann:xd=1,fb=1/blend_calib | oracle-id | router-psdctx:riemann | eeg | all | last | 0.5472 | 0.519 | 0.575 | 0.5472 | 60 | 0.994 | 0.000 | 195 |
| riemann:xd=1,fb=1/blend_calib | router-id | online-64:riemann | eeg | all | last | 0.4917 | 0.453 | 0.531 | 0.4917 | 60 | 0.979 | 0.000 | 201 |
| riemann:xd=1,fb=1/blend_calib | router-id | router-psd:riemann | all | all | last | 0.5069 | 0.508 | 0.506 | 0.5069 | 60 | 0.978 | 0.000 | 267 |
| riemann:xd=1,fb=1/blend_calib | router-id | router-psd:riemann | eeg | all | last | 0.4833 | 0.461 | 0.506 | 0.4833 | 60 | 0.979 | 0.000 | 195 |
| riemann:xd=1,fb=1/blend_calib | router-id | router-psd:riemann | eeg | all | loso | 0.4833 | 0.461 | 0.506 | 0.4833 | 60 | 0.979 | 0.000 | 853 |
| riemann:xd=1,fb=1/blend_calib | router-id | router-psd:riemann | eeg | test | last | 0.4458 | 0.417 | 0.475 | 0.4458 | 60 | 0.996 | 0.000 | 87 |
| riemann:xd=1,fb=1/blend_calib | router-id | router-psd:riemann | eeg+emg | all | last | 0.4917 | 0.461 | 0.522 | 0.4917 | 60 | 0.978 | 0.000 | 222 |
| riemann:xd=1,fb=1/blend_calib | router-id | router-psd:riemann | eeg+eog | all | last | 0.5111 | 0.489 | 0.533 | 0.5111 | 60 | 0.978 | 0.000 | 205 |
| riemann:xd=1,fb=1/blend_calib | router-id | router-psdctx:riemann | eeg | all | last | 0.5472 | 0.519 | 0.575 | 0.5472 | 60 | 0.994 | 0.000 | 195 |
| riemann:xd=1,fb=1/calib | oracle-id | online-64:riemann | eeg | all | last | 0.4750 | 0.450 | 0.500 | 0.4750 | 60 | 0.979 | 0.000 | 201 |
| riemann:xd=1,fb=1/calib | oracle-id | router-psd:riemann | all | all | last | 0.4778 | 0.464 | 0.492 | 0.4778 | 60 | 0.978 | 0.000 | 267 |
| riemann:xd=1,fb=1/calib | oracle-id | router-psd:riemann | eeg | all | last | 0.4722 | 0.456 | 0.489 | 0.4722 | 60 | 0.979 | 0.000 | 195 |
| riemann:xd=1,fb=1/calib | oracle-id | router-psd:riemann | eeg | all | loso | 0.4722 | 0.456 | 0.489 | 0.4722 | 60 | 0.979 | 0.000 | 853 |
| riemann:xd=1,fb=1/calib | oracle-id | router-psd:riemann | eeg | test | last | 0.4500 | 0.425 | 0.475 | 0.4500 | 60 | 0.996 | 0.000 | 87 |
| riemann:xd=1,fb=1/calib | oracle-id | router-psd:riemann | eeg+emg | all | last | 0.4611 | 0.442 | 0.481 | 0.4611 | 60 | 0.978 | 0.000 | 222 |
| riemann:xd=1,fb=1/calib | oracle-id | router-psd:riemann | eeg+eog | all | last | 0.5056 | 0.475 | 0.536 | 0.5056 | 60 | 0.978 | 0.000 | 205 |
| riemann:xd=1,fb=1/calib | oracle-id | router-psdctx:riemann | eeg | all | last | 0.4889 | 0.481 | 0.497 | 0.4889 | 60 | 0.994 | 0.000 | 195 |
| riemann:xd=1,fb=1/calib | router-id | online-64:riemann | eeg | all | last | 0.4750 | 0.450 | 0.500 | 0.4750 | 60 | 0.979 | 0.000 | 201 |
| riemann:xd=1,fb=1/calib | router-id | router-psd:riemann | all | all | last | 0.4750 | 0.464 | 0.486 | 0.4750 | 60 | 0.978 | 0.000 | 267 |
| riemann:xd=1,fb=1/calib | router-id | router-psd:riemann | eeg | all | last | 0.4681 | 0.456 | 0.481 | 0.4681 | 60 | 0.979 | 0.000 | 195 |
| riemann:xd=1,fb=1/calib | router-id | router-psd:riemann | eeg | all | loso | 0.4681 | 0.456 | 0.481 | 0.4681 | 60 | 0.979 | 0.000 | 853 |
| riemann:xd=1,fb=1/calib | router-id | router-psd:riemann | eeg | test | last | 0.4500 | 0.425 | 0.475 | 0.4500 | 60 | 0.996 | 0.000 | 87 |
| riemann:xd=1,fb=1/calib | router-id | router-psd:riemann | eeg+emg | all | last | 0.4583 | 0.442 | 0.475 | 0.4583 | 60 | 0.978 | 0.000 | 222 |
| riemann:xd=1,fb=1/calib | router-id | router-psd:riemann | eeg+eog | all | last | 0.5028 | 0.475 | 0.531 | 0.5028 | 60 | 0.978 | 0.000 | 205 |
| riemann:xd=1,fb=1/calib | router-id | router-psdctx:riemann | eeg | all | last | 0.4861 | 0.481 | 0.492 | 0.4861 | 60 | 0.994 | 0.000 | 195 |
| riemann:xd=1,fb=1/pooled | none | online-64:riemann | eeg | all | last | 0.4931 | 0.444 | 0.542 | 0.4931 | 60 | 0.979 | 0.000 | 201 |
| riemann:xd=1,fb=1/pooled | none | router-psd:riemann | all | all | last | 0.5097 | 0.514 | 0.506 | 0.5097 | 60 | 0.978 | 0.000 | 267 |
| riemann:xd=1,fb=1/pooled | none | router-psd:riemann | eeg | all | last | 0.4806 | 0.453 | 0.508 | 0.4806 | 60 | 0.979 | 0.000 | 195 |
| riemann:xd=1,fb=1/pooled | none | router-psd:riemann | eeg | all | loso | 0.4806 | 0.453 | 0.508 | 0.4806 | 60 | 0.979 | 0.000 | 853 |
| riemann:xd=1,fb=1/pooled | none | router-psd:riemann | eeg | test | last | 0.4403 | 0.411 | 0.469 | 0.4403 | 60 | 0.996 | 0.000 | 87 |
| riemann:xd=1,fb=1/pooled | none | router-psd:riemann | eeg+emg | all | last | 0.4819 | 0.458 | 0.506 | 0.4819 | 60 | 0.978 | 0.000 | 222 |
| riemann:xd=1,fb=1/pooled | none | router-psd:riemann | eeg+eog | all | last | 0.5125 | 0.483 | 0.542 | 0.5125 | 60 | 0.978 | 0.000 | 205 |
| riemann:xd=1,fb=1/pooled | none | router-psdctx:riemann | eeg | all | last | 0.5403 | 0.508 | 0.572 | 0.5403 | 60 | 0.994 | 0.000 | 195 |

## DECISIONS: release-day rules (pre-registered 2026-09-28)

```
recipe: riemann:xd=1,fb=1 / router-psd:riemann / chans eeg / pool all / wcv last: cell 0.4833 (60 cells)
channels  eeg+eog vs eeg: +2.8 pts (95 % CI +1.0, +4.7; 7/10 subjects up): passes (>= +2.0 and CI > 0)
channels  eeg+emg vs eeg: +0.8 pts (95 % CI -2.1, +3.9; 4/10 subjects up): fails (>= +2.0 and CI > 0)
channels  all vs eeg: +2.4 pts (95 % CI +0.4, +4.4; 6/10 subjects up): passes (>= +2.0 and CI > 0)
  -> chans = eeg+eog
context   router-psdctx vs router-psd: +6.4 pts (95 % CI +2.8, +10.0; 8/10 subjects up) -> ADOPT router-psdctx (>= +2.0 and CI > 0)
xDAWN     xd=0 vs xd=1: +1.5 pts (95 % CI -1.4, +4.3; 6/10 subjects up) -> DROP xDAWN (drop if xd=0 >= +1.0)
wcv       loso vs last: +0.0 pts (95 % CI +0.0, +0.0; 0/10 subjects up) (w: loso 0.75, last 0.75) -> wcv = loso (loso unless <= -2.0)
pool      test-only vs all: -3.7 pts (95 % CI -6.8, -0.0; 1/10 subjects up) -> pool = all (test only if >= +2.0 and CI > 0)
online-64 (RULE-DEPENDENT, report only) vs clean: +0.8 pts (95 % CI -1.4, +2.8; 6/10 subjects up)

result: spec riemann:xd=0,fb=1 / align router-psdctx:riemann / chans eeg+eog / pool all / wcv loso
train_sealed.sh: RECIPE_SPEC=riemann:xd=0,fb=1 RECIPE_ALIGN=router-psdctx:riemann CHANS=eeg+eog WCV=loso SPLIT=replica:3 bash ~/codabench/scripts/train_sealed.sh ...
(each rule compared one change with the recipe; train_sealed.sh's harness steps score the combined settings)
NOTE: Riemann-Sealed has no (subject, context) router yet: the adopted alignment needs a solver option before it can be trained
```
