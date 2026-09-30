# Release ablations: f0929_abl_s / mock_sealed_s

Split calib:3 (test subjects 0,1,2,3,4,5,6,7,8,9), router cap 0.5, X=(2880, 43, 480) (after the channel pick), train=1440 test=720 windows, test cells per row [60] (OK: one value), contexts ['brainhero', 'graz']. Cell = balanced accuracy averaged over subject x session x context cells of the test sessions; pooled = over all test windows. Recipe: riemann:xd=1,fb=1,x=bpt4, router-psd:riemann, chans eeg, pool all, wcv last. Rows read from sealed_f0929_abl_s, sealed_f0929_abl_s_A, sealed_f0929_abl_s_B.

## Recipe variants (blend_calib, router ids)

| variant | cell | vs recipe | brainhero | graz | pooled | w | router acc | s |
|---|---|---|---|---|---|---|---|---|
| base (recipe) | 0.5431 |  | 0.544 | 0.542 | 0.5431 | 0.5 | 0.989 | 302 |
| chans eeg+eog | not run |  | | |  |  |  |  |
| chans eeg+emg | not run |  | | |  |  |  |  |
| chans all | not run |  | | |  |  |  |  |
| align router-psdctx:riemann | not run |  | | |  |  |  |  |
| no xDAWN (riemann:xd=0,fb=1,x=bpt4) | not run |  | | |  |  |  |  |
| without blocks (riemann:xd=1,fb=1) | 0.5486 | +0.56 | 0.542 | 0.556 | 0.5486 | 0.75 | 0.989 | 221 |
| with blocks (riemann:xd=1,fb=1,x=bpt4+icoh) | 0.5625 | +1.94 | 0.567 | 0.558 | 0.5625 | 1.0 | 0.989 | 532 |
| wcv loso | not run |  | | |  |  |  |  |
| pool test | not run |  | | |  |  |  |  |
| online-64 (RULE-DEPENDENT) | not run |  | | |  |  |  |  |

## All rows

| spec | mode | align | chans | pool | wcv | cell | brainhero | graz | pooled | n_cells | router acc | fallback | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemann:xd=1,fb=1,x=bpt4+icoh/blend_calib | oracle-id | router-psd:riemann | eeg | all | last | 0.5625 | 0.567 | 0.558 | 0.5625 | 60 | 0.989 | 0.000 | 532 |
| riemann:xd=1,fb=1,x=bpt4+icoh/blend_calib | router-id | router-psd:riemann | eeg | all | last | 0.5625 | 0.567 | 0.558 | 0.5625 | 60 | 0.989 | 0.000 | 532 |
| riemann:xd=1,fb=1,x=bpt4+icoh/calib | oracle-id | router-psd:riemann | eeg | all | last | 0.5042 | 0.519 | 0.489 | 0.5042 | 60 | 0.989 | 0.000 | 532 |
| riemann:xd=1,fb=1,x=bpt4+icoh/calib | router-id | router-psd:riemann | eeg | all | last | 0.5069 | 0.519 | 0.494 | 0.5069 | 60 | 0.989 | 0.000 | 532 |
| riemann:xd=1,fb=1,x=bpt4+icoh/pooled | none | router-psd:riemann | eeg | all | last | 0.5625 | 0.567 | 0.558 | 0.5625 | 60 | 0.989 | 0.000 | 532 |
| riemann:xd=1,fb=1,x=bpt4/blend_calib | oracle-id | router-psd:riemann | eeg | all | last | 0.5444 | 0.544 | 0.544 | 0.5444 | 60 | 0.989 | 0.000 | 302 |
| riemann:xd=1,fb=1,x=bpt4/blend_calib | router-id | router-psd:riemann | eeg | all | last | 0.5431 | 0.544 | 0.542 | 0.5431 | 60 | 0.989 | 0.000 | 302 |
| riemann:xd=1,fb=1,x=bpt4/calib | oracle-id | router-psd:riemann | eeg | all | last | 0.5181 | 0.519 | 0.517 | 0.5181 | 60 | 0.989 | 0.000 | 302 |
| riemann:xd=1,fb=1,x=bpt4/calib | router-id | router-psd:riemann | eeg | all | last | 0.5167 | 0.519 | 0.514 | 0.5167 | 60 | 0.989 | 0.000 | 302 |
| riemann:xd=1,fb=1,x=bpt4/pooled | none | router-psd:riemann | eeg | all | last | 0.5458 | 0.544 | 0.547 | 0.5458 | 60 | 0.989 | 0.000 | 302 |
| riemann:xd=1,fb=1/blend_calib | oracle-id | router-psd:riemann | eeg | all | last | 0.5486 | 0.542 | 0.556 | 0.5486 | 60 | 0.989 | 0.000 | 221 |
| riemann:xd=1,fb=1/blend_calib | router-id | router-psd:riemann | eeg | all | last | 0.5486 | 0.542 | 0.556 | 0.5486 | 60 | 0.989 | 0.000 | 221 |
| riemann:xd=1,fb=1/calib | oracle-id | router-psd:riemann | eeg | all | last | 0.4847 | 0.481 | 0.489 | 0.4847 | 60 | 0.989 | 0.000 | 221 |
| riemann:xd=1,fb=1/calib | router-id | router-psd:riemann | eeg | all | last | 0.4861 | 0.481 | 0.492 | 0.4861 | 60 | 0.989 | 0.000 | 221 |
| riemann:xd=1,fb=1/pooled | none | router-psd:riemann | eeg | all | last | 0.5444 | 0.542 | 0.547 | 0.5444 | 60 | 0.989 | 0.000 | 221 |

## DECISIONS: release-day rules (pre-registered 2026-09-28)

```
recipe: riemann:xd=1,fb=1,x=bpt4 / router-psd:riemann / chans eeg / pool all / wcv last: cell 0.5431 (60 cells)
MISSING: ch_eeg_eog has no row (not run or failed): rule not applied
channels  eeg vs eeg+eog: no row -> keep eeg
MISSING: ch_eeg_emg has no row (not run or failed): rule not applied
channels  eeg vs eeg+emg: no row -> keep eeg
MISSING: ch_all has no row (not run or failed): rule not applied
channels  eeg vs all: no row -> keep eeg
  -> chans = eeg
MISSING: al_ctx has no row (not run or failed): rule not applied
context   router-psdctx: no row -> keep router-psd
MISSING: xd0 has no row (not run or failed): rule not applied
xDAWN     no-xDAWN row missing -> keep xDAWN
blocks    without vs with the recipe's x=: +0.56 pts (95 % CI -2.36, +3.47; 4/10 subjects up) (per context -0.28, +1.39) -> keep the blocks (drop only if >= +1.0 without them)
blocks    adding icoh vs the recipe: +1.94 pts (95 % CI -1.39, +5.28; 6/10 subjects up) (per context +2.22, +1.67) -> ADD icoh (add only if >= +1.0 and no context < -1.0)
MISSING: wcv_loso has no row (not run or failed): rule not applied
wcv       wcv=loso row missing -> wcv = loso (rule 4's default: loso unless measured <= -2.0 vs last)
MISSING: pool_test has no row (not run or failed): rule not applied
pool      pool=test row missing -> keep all
MISSING: online has no row (not run or failed): rule not applied
online-64 (rule-dependent) no row

result: spec riemann:xd=1,fb=1,x=bpt4+icoh / align router-psd:riemann / chans eeg / pool all / wcv loso
MISSING: 8 step(s) without a row (ch_eeg_eog ch_eeg_emg ch_all al_ctx xd0 wcv_loso pool_test online): their rules were NOT applied (defaults kept); rerun them (release_ablations.sh STEPS="ch_eeg_eog ch_eeg_emg ch_all al_ctx xd0 wcv_loso pool_test online") before deciding
train_sealed.sh: RECIPE_SPEC=riemann:xd=1,fb=1,x=bpt4+icoh RECIPE_ALIGN=router-psd:riemann CHANS=eeg WCV=loso BLEND_W=auto SPLIT=calib:3 TEST_SUBJECTS=0,1,2,3,4,5,6,7,8,9 bash ~/codabench/scripts/train_sealed.sh ...
(each rule compared one change with the recipe; train_sealed.sh's harness steps score the combined settings)
```
