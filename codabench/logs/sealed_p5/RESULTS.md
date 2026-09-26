# Results: p5

Cell = balanced accuracy averaged over (subject, test session) cells; pooled = over all test windows. Mean ± SD over seeds (n). Test = each subject's last session.

## scherer2015 (classes 0,1,3), X=(2130, 11, 480), train=1080, test=1050

| model | mode | align | n | cell | pooled | router acc | fallback | epochs | s/run |
|---|---|---|---|---|---|---|---|---|---|
| eegnet_st/scratch@ch11 | persubject | router-psd:euclid | 3 | 0.466 ± 0.032 | 0.468 ± 0.031 | 0.867 ± 0.000 | 0.038 ± 0.000 | 80 | 46 |
| eegnet_st/scratch@ch11 | persubject | online-64:euclid | 3 | 0.438 ± 0.014 | 0.440 ± 0.013 | 0.867 ± 0.000 |  | 80 | 68 |
| eegnet_st/scratch@ch11 | pooled | none | 3 | 0.438 ± 0.028 | 0.438 ± 0.029 |  |  | 53 | 69 |
| riemann_fbts@ch11/ref=scherer | persubject | none | 1 | 0.436 | 0.436 |  |  |  | 0 |
| riemann_fbts@ch11/ref=mi+scherer | pooled | none | 1 | 0.436 | 0.435 |  |  |  | 0 |
| riemann_fbts@ch11/ref=scherer | pooled | none | 1 | 0.436 | 0.435 |  |  |  | 0 |
| riemann_fbts@ch11/ref=mi+scherer | persubject | none | 1 | 0.431 | 0.431 |  |  |  | 0 |
| eegnet_st/scratch@ch11 | pooled | online-64:euclid | 3 | 0.430 ± 0.024 | 0.431 ± 0.023 | 0.867 ± 0.000 |  | 46 | 43 |
| eegnet_st/pretrained@ch11 | persubject | online-64:euclid | 3 | 0.429 ± 0.046 | 0.430 ± 0.046 | 0.867 ± 0.000 |  | 73 | 62 |
| eegnet_st/pretrained@ch11 | persubject | router-psd:euclid | 3 | 0.420 ± 0.042 | 0.421 ± 0.042 | 0.867 ± 0.000 | 0.038 ± 0.000 | 73 | 43 |
| eegnet_st/scratch@ch11 | pooled | router-psd:euclid | 3 | 0.414 ± 0.021 | 0.416 ± 0.020 | 0.867 ± 0.000 | 0.038 ± 0.000 | 46 | 38 |
| eegnet_st/pretrained@ch11 | persubject | none | 3 | 0.410 ± 0.005 | 0.410 ± 0.005 |  |  | 62 | 69 |
| eegnet_st/pretrained@ch11 | pooled | none | 3 | 0.364 ± 0.034 | 0.365 ± 0.034 |  |  | 36 | 54 |
| eegnet_st/scratch@ch11 | persubject | none | 3 | 0.349 ± 0.016 | 0.349 ± 0.016 |  |  | 45 | 52 |
| eegnet_st/pretrained@ch11 | pooled | online-64:euclid | 3 | 0.346 ± 0.016 | 0.346 ± 0.016 | 0.867 ± 0.000 |  | 2 | 9 |
| eegnet_st/pretrained@ch11 | pooled | router-psd:euclid | 3 | 0.342 ± 0.003 | 0.342 ± 0.003 | 0.867 ± 0.000 | 0.038 ± 0.000 | 2 | 8 |

