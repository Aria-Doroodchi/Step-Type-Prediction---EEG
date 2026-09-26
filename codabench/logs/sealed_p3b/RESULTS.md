# Results: p3, p3b

Cell = balanced accuracy averaged over (subject, test session) cells; pooled = over all test windows. Mean ± SD over seeds (n). Test = each subject's last session.

## scherer2015 (classes all), X=(3550, 30, 480), train=1800, test=1750

| model | mode | align | n | cell | pooled | router acc | fallback | epochs | s/run |
|---|---|---|---|---|---|---|---|---|---|
| riemann:xd=1,fb=1/calib | oracle-id | online-64:riemann | 1 | 0.395 | 0.395 | 0.866 |  |  | 917 |
| riemann:xd=1,fb=1/blend_calib | oracle-id | online-64:riemann | 1 | 0.395 | 0.395 | 0.866 |  |  | 917 |
| riemann:xd=1,fb=1/blend | oracle-id | online-64:riemann | 1 | 0.387 | 0.389 | 0.866 |  |  | 917 |
| riemann:xd=1,fb=1/persubject | oracle-id | online-64:riemann | 1 | 0.385 | 0.387 | 0.866 |  |  | 917 |
| riemann:xd=1,fb=1/calib | router-id | online-64:riemann | 1 | 0.375 | 0.375 | 0.866 |  |  | 917 |
| riemann:xd=1,fb=1/blend_calib | router-id | online-64:riemann | 1 | 0.375 | 0.375 | 0.866 |  |  | 917 |
| riemann:xd=1,fb=1/blend | router-id | online-64:riemann | 1 | 0.363 | 0.365 | 0.866 |  |  | 917 |
| riemann:xd=1,fb=1/persubject | router-id | online-64:riemann | 1 | 0.362 | 0.363 | 0.866 |  |  | 917 |
| riemann:xd=1,fb=1/calib | oracle-id | router-psd:riemann | 1 | 0.331 | 0.331 | 0.866 | 0.137 |  | 938 |
| riemann:xd=1,fb=1/blend_calib | oracle-id | router-psd:riemann | 1 | 0.331 | 0.331 | 0.866 | 0.137 |  | 938 |
| riemann:xd=1,fb=1/calib | router-id | router-psd:riemann | 1 | 0.322 | 0.323 | 0.866 | 0.137 |  | 938 |
| riemann:xd=1,fb=1/blend_calib | router-id | router-psd:riemann | 1 | 0.322 | 0.323 | 0.866 | 0.137 |  | 938 |
| riemann:xd=1,fb=1/blend | oracle-id | router-psd:riemann | 1 | 0.311 | 0.313 | 0.866 | 0.137 |  | 938 |
| riemann:xd=1,fb=1/persubject | oracle-id | router-psd:riemann | 1 | 0.309 | 0.311 | 0.866 | 0.137 |  | 938 |
| riemann:xd=1,fb=1/blend | router-id | router-psd:riemann | 1 | 0.304 | 0.305 | 0.866 | 0.137 |  | 938 |
| eegnet_st/calib | router-id | router-psd:euclid | 3 | 0.303 ± 0.014 | 0.305 ± 0.014 | 0.866 ± 0.000 | 0.137 ± 0.000 | 30 | 178 |
| eegnet_st/calib | oracle-id | router-psd:euclid | 3 | 0.302 ± 0.022 | 0.304 ± 0.022 | 0.866 ± 0.000 | 0.137 ± 0.000 | 30 | 178 |
| riemann:xd=1,fb=1/persubject | router-id | router-psd:riemann | 1 | 0.302 | 0.303 | 0.866 | 0.137 |  | 938 |
| riemann:xd=1,fb=1/pooled | none | online-64:riemann | 1 | 0.301 | 0.302 | 0.866 |  |  | 917 |
| riemann:xd=0,fb=1/blend | oracle-id | router-psd:riemann | 1 | 0.300 | 0.301 | 0.866 | 0.137 |  | 448 |
| riemann:xd=0,fb=1/calib | oracle-id | router-psd:riemann | 1 | 0.298 | 0.299 | 0.866 | 0.137 |  | 448 |
| riemann:xd=0,fb=1/blend | router-id | router-psd:riemann | 1 | 0.297 | 0.298 | 0.866 | 0.137 |  | 448 |
| riemann:xd=0,fb=1/persubject | oracle-id | router-psd:riemann | 1 | 0.297 | 0.298 | 0.866 | 0.137 |  | 448 |
| riemann:xd=0,fb=1/calib | router-id | router-psd:riemann | 1 | 0.296 | 0.297 | 0.866 | 0.137 |  | 448 |
| eegnet_st/pooled | none | router-psd:euclid | 3 | 0.295 ± 0.002 | 0.296 ± 0.002 | 0.866 ± 0.000 | 0.137 ± 0.000 | 30 | 178 |
| riemann:xd=0,fb=1/persubject | router-id | router-psd:riemann | 1 | 0.294 | 0.295 | 0.866 | 0.137 |  | 448 |
| riemann:xd=1,fb=1/pooled | none | router-psd:riemann | 1 | 0.277 | 0.278 | 0.866 | 0.137 |  | 938 |
| riemann:xd=0,fb=1/pooled | none | router-psd:riemann | 1 | 0.275 | 0.276 | 0.866 | 0.137 |  | 448 |

## tangermann2012 (classes all), X=(5184, 22, 480), train=2592, test=2592

| model | mode | align | n | cell | pooled | router acc | fallback | epochs | s/run |
|---|---|---|---|---|---|---|---|---|---|
| riemann:xd=1,fb=1/blend_calib | oracle-id | online-64:riemann | 1 | 0.852 | 0.852 | 0.965 |  |  | 354 |
| riemann:xd=1,fb=1/blend | oracle-id | online-64:riemann | 1 | 0.848 | 0.848 | 0.965 |  |  | 354 |
| riemann:xd=1,fb=1/calib | oracle-id | online-64:riemann | 1 | 0.839 | 0.839 | 0.965 |  |  | 354 |
| riemann:xd=1,fb=1/blend_calib | router-id | online-64:riemann | 1 | 0.837 | 0.837 | 0.965 |  |  | 354 |
| riemann:xd=1,fb=1/blend | router-id | online-64:riemann | 1 | 0.836 | 0.836 | 0.965 |  |  | 354 |
| riemann:xd=1,fb=1/persubject | oracle-id | online-64:riemann | 1 | 0.831 | 0.831 | 0.965 |  |  | 354 |
| riemann:xd=1,fb=1/calib | router-id | online-64:riemann | 1 | 0.824 | 0.824 | 0.965 |  |  | 354 |
| riemann:xd=1,fb=1/persubject | router-id | online-64:riemann | 1 | 0.820 | 0.820 | 0.965 |  |  | 354 |
| riemann:xd=1,fb=1/blend_calib | oracle-id | router-psd:riemann | 1 | 0.810 | 0.810 | 0.965 | 0.056 |  | 401 |
| riemann:xd=1,fb=1/blend_calib | router-id | router-psd:riemann | 1 | 0.802 | 0.802 | 0.965 | 0.056 |  | 401 |
| riemann:xd=1,fb=1/blend | oracle-id | router-psd:riemann | 1 | 0.797 | 0.797 | 0.965 | 0.056 |  | 401 |
| riemann:xd=1,fb=1/calib | oracle-id | router-psd:riemann | 1 | 0.796 | 0.796 | 0.965 | 0.056 |  | 401 |
| riemann:xd=1,fb=1/calib | router-id | router-psd:riemann | 1 | 0.792 | 0.792 | 0.965 | 0.056 |  | 401 |
| riemann:xd=1,fb=1/blend | router-id | router-psd:riemann | 1 | 0.792 | 0.792 | 0.965 | 0.056 |  | 401 |
| riemann:xd=1,fb=1/persubject | oracle-id | router-psd:riemann | 1 | 0.780 | 0.780 | 0.965 | 0.056 |  | 401 |
| riemann:xd=1,fb=1/persubject | router-id | router-psd:riemann | 1 | 0.777 | 0.777 | 0.965 | 0.056 |  | 401 |
| riemann:xd=1,fb=1/pooled | none | online-64:riemann | 1 | 0.729 | 0.729 | 0.965 |  |  | 354 |
| riemann:xd=0,fb=1/calib | oracle-id | router-psd:riemann | 1 | 0.699 | 0.699 | 0.965 | 0.056 |  | 318 |
| riemann:xd=0,fb=1/blend | oracle-id | router-psd:riemann | 1 | 0.697 | 0.697 | 0.965 | 0.056 |  | 318 |
| riemann:xd=0,fb=1/calib | router-id | router-psd:riemann | 1 | 0.696 | 0.696 | 0.965 | 0.056 |  | 318 |
| riemann:xd=0,fb=1/blend | router-id | router-psd:riemann | 1 | 0.693 | 0.693 | 0.965 | 0.056 |  | 318 |
| riemann:xd=0,fb=1/persubject | oracle-id | router-psd:riemann | 1 | 0.693 | 0.693 | 0.965 | 0.056 |  | 318 |
| riemann:xd=0,fb=1/persubject | router-id | router-psd:riemann | 1 | 0.689 | 0.689 | 0.965 | 0.056 |  | 318 |
| riemann:xd=1,fb=1/pooled | none | router-psd:riemann | 1 | 0.687 | 0.688 | 0.965 | 0.056 |  | 401 |
| eegnet_st/calib | oracle-id | router-psd:euclid | 3 | 0.682 ± 0.029 | 0.682 ± 0.029 | 0.965 ± 0.000 | 0.056 ± 0.000 | 81 | 389 |
| eegnet_st/calib | router-id | router-psd:euclid | 3 | 0.678 ± 0.029 | 0.678 ± 0.029 | 0.965 ± 0.000 | 0.056 ± 0.000 | 81 | 389 |
| eegnet_st/pooled | none | router-psd:euclid | 3 | 0.653 ± 0.028 | 0.653 ± 0.028 | 0.965 ± 0.000 | 0.056 ± 0.000 | 81 | 389 |
| riemann:xd=0,fb=1/pooled | none | router-psd:riemann | 1 | 0.574 | 0.574 | 0.965 | 0.056 |  | 318 |

## zhou2016 (classes all), X=(1800, 14, 480), train=1200, test=600

| model | mode | align | n | cell | pooled | router acc | fallback | epochs | s/run |
|---|---|---|---|---|---|---|---|---|---|
| riemann:xd=1,fb=1/blend | oracle-id | online-64:riemann | 1 | 0.802 | 0.802 | 1.000 |  |  | 43 |
| riemann:xd=1,fb=1/blend | router-id | online-64:riemann | 1 | 0.802 | 0.802 | 1.000 |  |  | 43 |
| riemann:xd=1,fb=1/blend_calib | oracle-id | online-64:riemann | 1 | 0.797 | 0.797 | 1.000 |  |  | 43 |
| riemann:xd=1,fb=1/blend_calib | router-id | online-64:riemann | 1 | 0.797 | 0.797 | 1.000 |  |  | 43 |
| riemann:xd=1,fb=1/persubject | oracle-id | online-64:riemann | 1 | 0.783 | 0.783 | 1.000 |  |  | 43 |
| riemann:xd=1,fb=1/persubject | router-id | online-64:riemann | 1 | 0.783 | 0.783 | 1.000 |  |  | 43 |
| riemann:xd=1,fb=1/calib | oracle-id | online-64:riemann | 1 | 0.778 | 0.778 | 1.000 |  |  | 43 |
| riemann:xd=1,fb=1/calib | router-id | online-64:riemann | 1 | 0.778 | 0.778 | 1.000 |  |  | 43 |
| riemann:xd=1,fb=1/blend_calib | oracle-id | router-psd:riemann | 1 | 0.778 | 0.778 | 1.000 | 0.002 |  | 48 |
| riemann:xd=1,fb=1/blend_calib | router-id | router-psd:riemann | 1 | 0.778 | 0.778 | 1.000 | 0.002 |  | 48 |
| riemann:xd=1,fb=1/pooled | none | router-psd:riemann | 1 | 0.767 | 0.767 | 1.000 | 0.002 |  | 48 |
| riemann:xd=1,fb=1/blend | oracle-id | router-psd:riemann | 1 | 0.757 | 0.757 | 1.000 | 0.002 |  | 48 |
| riemann:xd=1,fb=1/blend | router-id | router-psd:riemann | 1 | 0.757 | 0.757 | 1.000 | 0.002 |  | 48 |
| riemann:xd=1,fb=1/pooled | none | online-64:riemann | 1 | 0.747 | 0.747 | 1.000 |  |  | 43 |
| eegnet_st/calib | oracle-id | router-psd:euclid | 3 | 0.729 ± 0.116 | 0.729 ± 0.116 | 1.000 ± 0.000 | 0.002 ± 0.000 | 85 | 128 |
| eegnet_st/calib | router-id | router-psd:euclid | 3 | 0.729 ± 0.116 | 0.729 ± 0.116 | 1.000 ± 0.000 | 0.002 ± 0.000 | 85 | 128 |
| riemann:xd=0,fb=1/pooled | none | router-psd:riemann | 1 | 0.720 | 0.720 | 1.000 | 0.002 |  | 44 |
| riemann:xd=1,fb=1/calib | oracle-id | router-psd:riemann | 1 | 0.708 | 0.708 | 1.000 | 0.002 |  | 48 |
| riemann:xd=1,fb=1/calib | router-id | router-psd:riemann | 1 | 0.708 | 0.708 | 1.000 | 0.002 |  | 48 |
| eegnet_st/pooled | none | router-psd:euclid | 3 | 0.704 ± 0.117 | 0.704 ± 0.117 | 1.000 ± 0.000 | 0.002 ± 0.000 | 85 | 128 |
| riemann:xd=1,fb=1/persubject | oracle-id | router-psd:riemann | 1 | 0.688 | 0.688 | 1.000 | 0.002 |  | 48 |
| riemann:xd=1,fb=1/persubject | router-id | router-psd:riemann | 1 | 0.688 | 0.688 | 1.000 | 0.002 |  | 48 |
| riemann:xd=0,fb=1/blend | oracle-id | router-psd:riemann | 1 | 0.675 | 0.675 | 1.000 | 0.002 |  | 44 |
| riemann:xd=0,fb=1/blend | router-id | router-psd:riemann | 1 | 0.675 | 0.675 | 1.000 | 0.002 |  | 44 |
| riemann:xd=0,fb=1/calib | oracle-id | router-psd:riemann | 1 | 0.633 | 0.633 | 1.000 | 0.002 |  | 44 |
| riemann:xd=0,fb=1/calib | router-id | router-psd:riemann | 1 | 0.633 | 0.633 | 1.000 | 0.002 |  | 44 |
| riemann:xd=0,fb=1/persubject | oracle-id | router-psd:riemann | 1 | 0.632 | 0.632 | 1.000 | 0.002 |  | 44 |
| riemann:xd=0,fb=1/persubject | router-id | router-psd:riemann | 1 | 0.632 | 0.632 | 1.000 | 0.002 |  | 44 |


# Decision table, router ids

Personal variants with router-id (pooled needs no id); seed-mean cell score, best in bold.

| study | family | align | pooled | calib | blend | blend_calib | persubject |
|---|---|---|---|---|---|---|---|
| scherer2015 | eegnet_st | router-psd:euclid | 0.295 (n3) | **0.303 (n3)** |  |  |  |
| scherer2015 | riemann:xd=0,fb=1 | router-psd:riemann | 0.275 | 0.296 | **0.297** |  | 0.294 |
| scherer2015 | riemann:xd=1,fb=1 | online-64:riemann | 0.301 | **0.375** | 0.363 | **0.375** | 0.362 |
| scherer2015 | riemann:xd=1,fb=1 | router-psd:riemann | 0.277 | **0.322** | 0.304 | **0.322** | 0.302 |
| tangermann2012 | eegnet_st | router-psd:euclid | 0.653 (n3) | **0.678 (n3)** |  |  |  |
| tangermann2012 | riemann:xd=0,fb=1 | router-psd:riemann | 0.574 | **0.696** | 0.693 |  | 0.689 |
| tangermann2012 | riemann:xd=1,fb=1 | online-64:riemann | 0.729 | 0.824 | 0.836 | **0.837** | 0.820 |
| tangermann2012 | riemann:xd=1,fb=1 | router-psd:riemann | 0.687 | 0.792 | 0.792 | **0.802** | 0.777 |
| zhou2016 | eegnet_st | router-psd:euclid | 0.704 (n3) | **0.729 (n3)** |  |  |  |
| zhou2016 | riemann:xd=0,fb=1 | router-psd:riemann | **0.720** | 0.633 | 0.675 |  | 0.632 |
| zhou2016 | riemann:xd=1,fb=1 | online-64:riemann | 0.747 | 0.778 | **0.802** | 0.797 | 0.783 |
| zhou2016 | riemann:xd=1,fb=1 | router-psd:riemann | 0.767 | 0.708 | 0.757 | **0.778** | 0.688 |

# Decision table, oracle ids

Personal variants with oracle-id (pooled needs no id); seed-mean cell score, best in bold.

| study | family | align | pooled | calib | blend | blend_calib | persubject |
|---|---|---|---|---|---|---|---|
| scherer2015 | eegnet_st | router-psd:euclid | 0.295 (n3) | **0.302 (n3)** |  |  |  |
| scherer2015 | riemann:xd=0,fb=1 | router-psd:riemann | 0.275 | 0.298 | **0.300** |  | 0.297 |
| scherer2015 | riemann:xd=1,fb=1 | online-64:riemann | 0.301 | **0.395** | 0.387 | **0.395** | 0.385 |
| scherer2015 | riemann:xd=1,fb=1 | router-psd:riemann | 0.277 | **0.331** | 0.311 | **0.331** | 0.309 |
| tangermann2012 | eegnet_st | router-psd:euclid | 0.653 (n3) | **0.682 (n3)** |  |  |  |
| tangermann2012 | riemann:xd=0,fb=1 | router-psd:riemann | 0.574 | **0.699** | 0.697 |  | 0.693 |
| tangermann2012 | riemann:xd=1,fb=1 | online-64:riemann | 0.729 | 0.839 | 0.848 | **0.852** | 0.831 |
| tangermann2012 | riemann:xd=1,fb=1 | router-psd:riemann | 0.687 | 0.796 | 0.797 | **0.810** | 0.780 |
| zhou2016 | eegnet_st | router-psd:euclid | 0.704 (n3) | **0.729 (n3)** |  |  |  |
| zhou2016 | riemann:xd=0,fb=1 | router-psd:riemann | **0.720** | 0.633 | 0.675 |  | 0.632 |
| zhou2016 | riemann:xd=1,fb=1 | online-64:riemann | 0.747 | 0.778 | **0.802** | 0.797 | 0.783 |
| zhou2016 | riemann:xd=1,fb=1 | router-psd:riemann | 0.767 | 0.708 | 0.757 | **0.778** | 0.688 |
