# Results: p1, p2

Cell = balanced accuracy averaged over (subject, test session) cells; pooled = over all test windows. Mean ± SD over seeds (n). Test = each subject's last session.

## scherer2015 (classes all), X=(3550, 30, 480), train=1800, test=1750

| model | mode | align | n | cell | pooled | router acc | fallback | epochs | s/run |
|---|---|---|---|---|---|---|---|---|---|
| riemann:xd=1,fb=1 | persubject | online-64:riemann | 1 | 0.385 | 0.387 | 0.866 |  |  | 156 |
| riemann:xd=1,fb=1 | persubject | oracle:riemann | 1 | 0.383 | 0.383 |  |  |  | 153 |
| riemann:xd=1,fb=1 | persubject | online-128:riemann | 1 | 0.382 | 0.384 | 0.866 |  |  | 156 |
| riemann:xd=1,fb=1 | persubject | batch:riemann | 1 | 0.382 | 0.383 |  |  |  | 147 |
| riemann:xd=1,fb=1 | persubject | oracle:euclid | 1 | 0.363 | 0.363 |  |  |  | 158 |
| riemann:xd=0,fb=1 | persubject | online-64:riemann | 1 | 0.360 | 0.361 | 0.866 |  |  | 80 |
| riemann:xd=0,fb=1 | persubject | oracle:riemann | 1 | 0.359 | 0.360 |  |  |  | 52 |
| riemann:xd=0,fb=1 | persubject | online-128:riemann | 1 | 0.352 | 0.352 | 0.866 |  |  | 80 |
| riemann:xd=0,fb=1 | persubject | batch:riemann | 1 | 0.344 | 0.345 |  |  |  | 57 |
| riemann:xd=0,fb=1 | persubject | oracle:euclid | 1 | 0.336 | 0.337 |  |  |  | 50 |
| riemann:xd=1,fb=1 | persubject | router-psd:euclid | 1 | 0.322 | 0.323 | 0.866 | 0.137 |  | 133 |
| riemann:xd=1,fb=1 | persubject | none | 1 | 0.320 | 0.320 |  |  |  | 148 |
| riemann:xd=1,fb=1 | persubject | routerb-psd:riemann | 1 | 0.315 | 0.317 | 0.926 | 0.049 |  | 146 |
| riemann:xd=1,fb=0 | persubject | none | 1 | 0.315 | 0.315 |  |  |  | 17 |
| riemann:xd=1,fb=1 | persubject | router-psd:riemann | 1 | 0.309 | 0.311 | 0.866 | 0.137 |  | 138 |
| riemann:xd=1,fb=1 | persubject | trainonly:euclid | 1 | 0.308 | 0.309 |  |  |  | 148 |
| eegnet_st | pooled | oracle:euclid | 3 | 0.305 ± 0.018 | 0.306 ± 0.018 |  |  | 30 | 130 |
| riemann:xd=1,fb=1 | persubject | trainonly:riemann | 1 | 0.303 | 0.305 |  |  |  | 135 |
| riemann:xd=0,fb=1 | persubject | router-psd:euclid | 1 | 0.302 | 0.303 | 0.866 | 0.137 |  | 56 |
| riemann:xd=1,fb=1 | pooled | online-64:riemann | 1 | 0.301 | 0.302 | 0.866 |  |  | 67 |
| riemann:xd=0,fb=1 | persubject | none | 1 | 0.301 | 0.301 |  |  |  | 52 |
| riemann:xd=1,fb=1 | pooled | batch:riemann | 1 | 0.300 | 0.300 |  |  |  | 56 |
| riemann:xd=0,fb=1 | persubject | routerb-psd:riemann | 1 | 0.298 | 0.299 | 0.926 | 0.049 |  | 56 |
| riemann:xd=0,fb=1 | persubject | router-psd:riemann | 1 | 0.297 | 0.298 | 0.866 | 0.137 |  | 57 |
| riemann:xd=1,fb=1 | pooled | online-128:riemann | 1 | 0.296 | 0.297 | 0.866 |  |  | 64 |
| riemann:xd=0,fb=1 | pooled | online-128:riemann | 1 | 0.296 | 0.296 | 0.866 |  |  | 45 |
| eegnet_bd | pooled | none | 3 | 0.295 ± 0.014 | 0.296 ± 0.015 |  |  |  | 91 |
| eegnet_st | pooled | router-psd:euclid | 3 | 0.295 ± 0.002 | 0.296 ± 0.002 | 0.866 ± 0.000 | 0.137 ± 0.000 | 30 | 123 |
| riemann:xd=0,fb=1 | pooled | online-64:riemann | 1 | 0.295 | 0.295 | 0.866 |  |  | 43 |
| eegnet_st | pooled | online-128:euclid | 3 | 0.294 ± 0.016 | 0.295 ± 0.016 | 0.866 ± 0.000 |  | 30 | 174 |
| riemann:xd=0,fb=0 | persubject | none | 1 | 0.294 | 0.294 |  |  |  | 6 |
| riemann:xd=0,fb=1 | persubject | trainonly:riemann | 1 | 0.294 | 0.295 |  |  |  | 59 |
| riemann:xd=1,fb=1 | pooled | oracle:riemann | 1 | 0.293 | 0.293 |  |  |  | 65 |
| riemann:xd=0,fb=1 | pooled | oracle:riemann | 1 | 0.289 | 0.290 |  |  |  | 38 |
| riemann:xd=0,fb=1 | pooled | batch:riemann | 1 | 0.289 | 0.290 |  |  |  | 30 |
| eegnet_st | pooled | trainonly:euclid | 3 | 0.288 ± 0.005 | 0.289 ± 0.004 |  |  | 30 | 121 |
| riemann:xd=0,fb=1 | pooled | oracle:euclid | 1 | 0.284 | 0.285 |  |  |  | 33 |
| riemann:xd=1,fb=1 | pooled | trainonly:riemann | 1 | 0.283 | 0.284 |  |  |  | 74 |
| riemann:xd=1,fb=1 | pooled | oracle:euclid | 1 | 0.282 | 0.282 |  |  |  | 57 |
| riemann:xd=1,fb=1 | pooled | routerb-psd:riemann | 1 | 0.282 | 0.282 | 0.926 | 0.049 |  | 55 |
| eegnet_st | pooled | none | 3 | 0.281 ± 0.010 | 0.282 ± 0.010 |  |  | 25 | 208 |
| riemann:xd=0,fb=1 | persubject | trainonly:euclid | 1 | 0.280 | 0.280 |  |  |  | 50 |
| riemann:xd=1,fb=1 | pooled | router-psd:riemann | 1 | 0.277 | 0.278 | 0.866 | 0.137 |  | 55 |
| riemann:xd=0,fb=1 | pooled | none | 1 | 0.277 | 0.277 |  |  |  | 32 |
| riemann:xd=0,fb=1 | pooled | router-psd:riemann | 1 | 0.275 | 0.276 | 0.866 | 0.137 |  | 28 |
| riemann:xd=0,fb=1 | pooled | router-psd:euclid | 1 | 0.275 | 0.276 | 0.866 | 0.137 |  | 28 |
| riemann:xd=0,fb=1 | pooled | routerb-psd:riemann | 1 | 0.274 | 0.274 | 0.926 | 0.049 |  | 29 |
| riemann:xd=0,fb=1 | pooled | trainonly:euclid | 1 | 0.271 | 0.271 |  |  |  | 27 |
| riemann:xd=1,fb=1 | pooled | router-psd:euclid | 1 | 0.271 | 0.271 | 0.866 | 0.137 |  | 66 |
| riemann:xd=0,fb=1 | pooled | trainonly:riemann | 1 | 0.266 | 0.267 |  |  |  | 32 |
| riemann:xd=1,fb=1 | pooled | trainonly:euclid | 1 | 0.266 | 0.265 |  |  |  | 68 |
| riemann:xd=0,fb=0 | pooled | none | 1 | 0.264 | 0.264 |  |  |  | 4 |
| eegnet_st | persubject | none | 3 | 0.261 ± 0.015 | 0.262 ± 0.015 |  |  | 51 | 285 |
| riemann:xd=1,fb=1 | pooled | none | 1 | 0.259 | 0.260 |  |  |  | 58 |
| riemann:xd=1,fb=0 | pooled | none | 1 | 0.252 | 0.251 |  |  |  | 14 |
| meanlr | pooled | none | 1 | 0.218 | 0.218 |  |  |  | 0 |
| eegnet_bd | persubject | none | 3 | 0.213 ± 0.016 | 0.213 ± 0.016 |  |  |  | 126 |
| meanlr | persubject | none | 1 | 0.193 | 0.193 |  |  |  | 0 |

## tangermann2012 (classes all), X=(5184, 22, 480), train=2592, test=2592

| model | mode | align | n | cell | pooled | router acc | fallback | epochs | s/run |
|---|---|---|---|---|---|---|---|---|---|
| riemann:xd=1,fb=1 | persubject | online-64:riemann | 1 | 0.831 | 0.831 | 0.965 |  |  | 51 |
| riemann:xd=1,fb=1 | persubject | online-128:riemann | 1 | 0.825 | 0.825 | 0.965 |  |  | 52 |
| riemann:xd=1,fb=1 | persubject | oracle:riemann | 1 | 0.824 | 0.824 |  |  |  | 52 |
| riemann:xd=1,fb=1 | persubject | batch:riemann | 1 | 0.818 | 0.818 |  |  |  | 59 |
| riemann:xd=1,fb=1 | persubject | oracle:euclid | 1 | 0.809 | 0.809 |  |  |  | 55 |
| riemann:xd=1,fb=1 | persubject | router-psd:riemann | 1 | 0.780 | 0.780 | 0.965 | 0.056 |  | 52 |
| riemann:xd=1,fb=1 | persubject | router-psd:euclid | 1 | 0.778 | 0.778 | 0.965 | 0.056 |  | 47 |
| riemann:xd=1,fb=1 | persubject | routerb-psd:riemann | 1 | 0.778 | 0.778 | 0.951 | 0.000 |  | 57 |
| riemann:xd=1,fb=1 | persubject | none | 1 | 0.775 | 0.775 |  |  |  | 54 |
| riemann:xd=0,fb=1 | persubject | online-64:riemann | 1 | 0.755 | 0.755 | 0.965 |  |  | 34 |
| riemann:xd=0,fb=1 | persubject | oracle:riemann | 1 | 0.748 | 0.748 |  |  |  | 35 |
| riemann:xd=0,fb=1 | persubject | batch:riemann | 1 | 0.745 | 0.745 |  |  |  | 34 |
| riemann:xd=0,fb=1 | persubject | online-128:riemann | 1 | 0.745 | 0.745 | 0.965 |  |  | 37 |
| riemann:xd=0,fb=1 | persubject | oracle:euclid | 1 | 0.735 | 0.735 |  |  |  | 33 |
| riemann:xd=1,fb=1 | pooled | online-128:riemann | 1 | 0.730 | 0.730 | 0.965 |  |  | 43 |
| riemann:xd=1,fb=1 | pooled | online-64:riemann | 1 | 0.729 | 0.729 | 0.965 |  |  | 48 |
| riemann:xd=1,fb=1 | persubject | trainonly:riemann | 1 | 0.727 | 0.727 |  |  |  | 52 |
| riemann:xd=0,fb=1 | persubject | none | 1 | 0.726 | 0.726 |  |  |  | 32 |
| riemann:xd=1,fb=1 | pooled | oracle:riemann | 1 | 0.726 | 0.726 |  |  |  | 35 |
| riemann:xd=1,fb=1 | pooled | batch:riemann | 1 | 0.723 | 0.723 |  |  |  | 42 |
| riemann:xd=1,fb=1 | pooled | oracle:euclid | 1 | 0.721 | 0.721 |  |  |  | 36 |
| riemann:xd=1,fb=1 | persubject | trainonly:euclid | 1 | 0.714 | 0.714 |  |  |  | 52 |
| riemann:xd=0,fb=1 | persubject | router-psd:euclid | 1 | 0.693 | 0.693 | 0.965 | 0.056 |  | 32 |
| riemann:xd=0,fb=1 | persubject | router-psd:riemann | 1 | 0.693 | 0.693 | 0.965 | 0.056 |  | 35 |
| eegnet_st | pooled | oracle:euclid | 3 | 0.690 ± 0.032 | 0.690 ± 0.032 |  |  | 81 | 409 |
| riemann:xd=0,fb=1 | persubject | routerb-psd:riemann | 1 | 0.690 | 0.690 | 0.951 | 0.000 |  | 34 |
| riemann:xd=1,fb=1 | pooled | routerb-psd:riemann | 1 | 0.688 | 0.688 | 0.951 | 0.000 |  | 42 |
| riemann:xd=1,fb=1 | pooled | router-psd:riemann | 1 | 0.687 | 0.688 | 0.965 | 0.056 |  | 42 |
| riemann:xd=1,fb=0 | persubject | none | 1 | 0.679 | 0.679 |  |  |  | 16 |
| riemann:xd=1,fb=1 | pooled | router-psd:euclid | 1 | 0.677 | 0.677 | 0.965 | 0.056 |  | 46 |
| riemann:xd=1,fb=0 | pooled | none | 1 | 0.663 | 0.663 |  |  |  | 14 |
| riemann:xd=0,fb=1 | persubject | trainonly:riemann | 1 | 0.659 | 0.659 |  |  |  | 35 |
| eegnet_st | pooled | router-psd:euclid | 3 | 0.653 ± 0.028 | 0.653 ± 0.028 | 0.965 ± 0.000 | 0.056 ± 0.000 | 81 | 442 |
| eegnet_st | pooled | none | 3 | 0.644 ± 0.037 | 0.644 ± 0.037 |  |  | 84 | 531 |
| riemann:xd=1,fb=1 | pooled | none | 1 | 0.639 | 0.639 |  |  |  | 37 |
| riemann:xd=0,fb=1 | persubject | trainonly:euclid | 1 | 0.628 | 0.628 |  |  |  | 33 |
| eegnet_bd | pooled | none | 3 | 0.622 ± 0.026 | 0.622 ± 0.026 |  |  |  | 87 |
| riemann:xd=0,fb=1 | pooled | online-64:riemann | 1 | 0.608 | 0.608 | 0.965 |  |  | 32 |
| riemann:xd=0,fb=1 | pooled | oracle:riemann | 1 | 0.606 | 0.606 |  |  |  | 33 |
| riemann:xd=0,fb=1 | pooled | online-128:riemann | 1 | 0.599 | 0.599 | 0.965 |  |  | 32 |
| riemann:xd=0,fb=0 | persubject | none | 1 | 0.599 | 0.599 |  |  |  | 7 |
| riemann:xd=0,fb=1 | pooled | batch:riemann | 1 | 0.596 | 0.596 |  |  |  | 32 |
| riemann:xd=1,fb=1 | pooled | trainonly:riemann | 1 | 0.588 | 0.588 |  |  |  | 41 |
| riemann:xd=0,fb=1 | pooled | oracle:euclid | 1 | 0.577 | 0.577 |  |  |  | 30 |
| riemann:xd=0,fb=1 | pooled | router-psd:riemann | 1 | 0.574 | 0.574 | 0.965 | 0.056 |  | 32 |
| riemann:xd=1,fb=1 | pooled | trainonly:euclid | 1 | 0.573 | 0.573 |  |  |  | 34 |
| riemann:xd=0,fb=1 | pooled | routerb-psd:riemann | 1 | 0.567 | 0.567 | 0.951 | 0.000 |  | 32 |
| riemann:xd=0,fb=1 | pooled | router-psd:euclid | 1 | 0.562 | 0.562 | 0.965 | 0.056 |  | 29 |
| eegnet_st | persubject | none | 3 | 0.546 ± 0.065 | 0.546 ± 0.065 |  |  | 82 | 422 |
| riemann:xd=0,fb=1 | pooled | none | 1 | 0.544 | 0.544 |  |  |  | 29 |
| riemann:xd=0,fb=0 | pooled | none | 1 | 0.501 | 0.501 |  |  |  | 8 |
| riemann:xd=0,fb=1 | pooled | trainonly:riemann | 1 | 0.474 | 0.474 |  |  |  | 32 |
| riemann:xd=0,fb=1 | pooled | trainonly:euclid | 1 | 0.473 | 0.473 |  |  |  | 29 |
| eegnet_bd | persubject | none | 3 | 0.395 ± 0.008 | 0.395 ± 0.008 |  |  |  | 132 |
| meanlr | persubject | none | 1 | 0.270 | 0.270 |  |  |  | 0 |
| meanlr | pooled | none | 1 | 0.258 | 0.258 |  |  |  | 0 |

## zhou2016 (classes all), X=(1800, 14, 480), train=1200, test=600

| model | mode | align | n | cell | pooled | router acc | fallback | epochs | s/run |
|---|---|---|---|---|---|---|---|---|---|
| eegnet_st | pooled | online-128:euclid | 3 | 0.824 ± 0.026 | 0.824 ± 0.026 | 1.000 ± 0.000 |  | 97 | 111 |
| eegnet_st | pooled | oracle:euclid | 3 | 0.819 ± 0.019 | 0.819 ± 0.019 |  |  | 97 | 122 |
| riemann:xd=1,fb=1 | persubject | online-64:riemann | 1 | 0.783 | 0.783 | 1.000 |  |  | 11 |
| riemann:xd=1,fb=1 | persubject | online-128:riemann | 1 | 0.782 | 0.782 | 1.000 |  |  | 11 |
| riemann:xd=1,fb=1 | pooled | none | 1 | 0.772 | 0.772 |  |  |  | 12 |
| riemann:xd=1,fb=1 | pooled | trainonly:riemann | 1 | 0.770 | 0.770 |  |  |  | 11 |
| riemann:xd=1,fb=1 | pooled | routerb-psd:riemann | 1 | 0.768 | 0.768 | 0.927 | 0.000 |  | 11 |
| riemann:xd=1,fb=1 | pooled | router-psd:riemann | 1 | 0.767 | 0.767 | 1.000 | 0.002 |  | 11 |
| riemann:xd=1,fb=1 | persubject | batch:riemann | 1 | 0.763 | 0.763 |  |  |  | 12 |
| riemann:xd=1,fb=1 | persubject | oracle:riemann | 1 | 0.763 | 0.763 |  |  |  | 12 |
| eegnet_st | pooled | trainonly:euclid | 3 | 0.756 ± 0.031 | 0.756 ± 0.031 |  |  | 97 | 103 |
| eegnet_st | pooled | none | 3 | 0.754 ± 0.063 | 0.754 ± 0.063 |  |  | 99 | 124 |
| riemann:xd=1,fb=1 | pooled | online-128:riemann | 1 | 0.753 | 0.753 | 1.000 |  |  | 12 |
| riemann:xd=1,fb=1 | pooled | router-psd:euclid | 1 | 0.753 | 0.753 | 1.000 | 0.002 |  | 12 |
| riemann:xd=1,fb=1 | pooled | online-64:riemann | 1 | 0.747 | 0.747 | 1.000 |  |  | 11 |
| riemann:xd=1,fb=1 | pooled | batch:riemann | 1 | 0.745 | 0.745 |  |  |  | 11 |
| riemann:xd=1,fb=1 | pooled | oracle:riemann | 1 | 0.742 | 0.742 |  |  |  | 11 |
| riemann:xd=1,fb=1 | persubject | trainonly:riemann | 1 | 0.732 | 0.732 |  |  |  | 12 |
| riemann:xd=0,fb=1 | persubject | online-64:riemann | 1 | 0.730 | 0.730 | 1.000 |  |  | 9 |
| riemann:xd=0,fb=1 | pooled | router-psd:euclid | 1 | 0.728 | 0.728 | 1.000 | 0.002 |  | 8 |
| riemann:xd=0,fb=1 | pooled | none | 1 | 0.728 | 0.728 |  |  |  | 9 |
| riemann:xd=0,fb=1 | persubject | batch:riemann | 1 | 0.725 | 0.725 |  |  |  | 7 |
| riemann:xd=1,fb=1 | pooled | trainonly:euclid | 1 | 0.722 | 0.722 |  |  |  | 11 |
| riemann:xd=0,fb=1 | persubject | oracle:riemann | 1 | 0.722 | 0.722 |  |  |  | 8 |
| riemann:xd=0,fb=1 | pooled | router-psd:riemann | 1 | 0.720 | 0.720 | 1.000 | 0.002 |  | 9 |
| riemann:xd=0,fb=1 | pooled | routerb-psd:riemann | 1 | 0.718 | 0.718 | 0.927 | 0.000 |  | 8 |
| riemann:xd=0,fb=1 | persubject | online-128:riemann | 1 | 0.718 | 0.718 | 1.000 |  |  | 9 |
| riemann:xd=1,fb=1 | persubject | none | 1 | 0.708 | 0.708 |  |  |  | 11 |
| riemann:xd=1,fb=1 | persubject | router-psd:euclid | 1 | 0.707 | 0.707 | 1.000 | 0.002 |  | 11 |
| riemann:xd=0,fb=1 | pooled | trainonly:riemann | 1 | 0.705 | 0.705 |  |  |  | 8 |
| eegnet_st | pooled | router-psd:euclid | 3 | 0.704 ± 0.117 | 0.704 ± 0.117 | 1.000 ± 0.000 | 0.002 ± 0.000 | 85 | 122 |
| riemann:xd=0,fb=1 | pooled | trainonly:euclid | 1 | 0.703 | 0.703 |  |  |  | 8 |
| riemann:xd=0,fb=1 | persubject | trainonly:riemann | 1 | 0.702 | 0.702 |  |  |  | 9 |
| riemann:xd=0,fb=1 | pooled | oracle:riemann | 1 | 0.695 | 0.695 |  |  |  | 9 |
| riemann:xd=1,fb=1 | persubject | router-psd:riemann | 1 | 0.688 | 0.688 | 1.000 | 0.002 |  | 12 |
| riemann:xd=0,fb=1 | pooled | online-128:riemann | 1 | 0.687 | 0.687 | 1.000 |  |  | 9 |
| riemann:xd=0,fb=1 | pooled | online-64:riemann | 1 | 0.687 | 0.687 | 1.000 |  |  | 9 |
| riemann:xd=1,fb=1 | persubject | routerb-psd:riemann | 1 | 0.683 | 0.683 | 0.927 | 0.000 |  | 11 |
| riemann:xd=0,fb=1 | pooled | batch:riemann | 1 | 0.682 | 0.682 |  |  |  | 8 |
| riemann:xd=0,fb=1 | persubject | none | 1 | 0.667 | 0.667 |  |  |  | 8 |
| riemann:xd=1,fb=1 | persubject | trainonly:euclid | 1 | 0.665 | 0.665 |  |  |  | 11 |
| riemann:xd=1,fb=1 | pooled | oracle:euclid | 1 | 0.665 | 0.665 |  |  |  | 11 |
| riemann:xd=1,fb=1 | persubject | oracle:euclid | 1 | 0.663 | 0.663 |  |  |  | 11 |
| riemann:xd=0,fb=1 | persubject | trainonly:euclid | 1 | 0.640 | 0.640 |  |  |  | 8 |
| riemann:xd=0,fb=1 | persubject | router-psd:euclid | 1 | 0.637 | 0.637 | 1.000 | 0.002 |  | 8 |
| riemann:xd=1,fb=0 | pooled | none | 1 | 0.635 | 0.635 |  |  |  | 6 |
| riemann:xd=0,fb=1 | persubject | router-psd:riemann | 1 | 0.632 | 0.632 | 1.000 | 0.002 |  | 8 |
| riemann:xd=0,fb=1 | persubject | routerb-psd:riemann | 1 | 0.630 | 0.630 | 0.927 | 0.000 |  | 7 |
| riemann:xd=1,fb=0 | persubject | none | 1 | 0.622 | 0.622 |  |  |  | 4 |
| riemann:xd=0,fb=1 | pooled | oracle:euclid | 1 | 0.610 | 0.610 |  |  |  | 9 |
| eegnet_bd | pooled | none | 3 | 0.592 ± 0.073 | 0.592 ± 0.073 |  |  |  | 22 |
| eegnet_st | persubject | none | 3 | 0.561 ± 0.003 | 0.561 ± 0.003 |  |  | 52 | 52 |
| riemann:xd=0,fb=0 | pooled | none | 1 | 0.555 | 0.555 |  |  |  | 2 |
| riemann:xd=0,fb=1 | persubject | oracle:euclid | 1 | 0.493 | 0.493 |  |  |  | 8 |
| riemann:xd=0,fb=0 | persubject | none | 1 | 0.478 | 0.478 |  |  |  | 2 |
| eegnet_bd | persubject | none | 3 | 0.477 ± 0.017 | 0.477 ± 0.017 |  |  |  | 19 |
| meanlr | persubject | none | 1 | 0.457 | 0.457 |  |  |  | 0 |
| meanlr | pooled | none | 1 | 0.435 | 0.435 |  |  |  | 0 |


# Decision table (sealed_decide.py)

| study | model | mode | none | orc:E | orc:R | tro:E | tro:R | rt:E | rt:R | rtb:R | batch:R | gain_b | rec(rt best) | rec(tro best) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scherer2015 | eegnet_bd | persubject | 0.213 (n3) |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| scherer2015 | eegnet_bd | pooled | 0.295 (n3) |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| scherer2015 | eegnet_st | persubject | 0.261 (n3) |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| scherer2015 | eegnet_st | pooled | 0.281 (n3) | 0.305 (n3) |  | 0.288 (n3) |  | 0.295 (n3) |  |  |  | +0.024 | 58% | 31% |
| scherer2015 | meanlr | persubject | 0.193 |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| scherer2015 | meanlr | pooled | 0.218 |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| scherer2015 | riemann:xd=0,fb=0 | persubject | 0.294 |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| scherer2015 | riemann:xd=0,fb=0 | pooled | 0.264 |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| scherer2015 | riemann:xd=0,fb=1 | persubject | 0.301 | 0.336 | 0.359 | 0.280 | 0.294 | 0.302 | 0.297 | 0.298 | 0.344 | +0.059 | 3% | -12% |
| scherer2015 | riemann:xd=0,fb=1 | pooled | 0.277 | 0.284 | 0.289 | 0.271 | 0.266 | 0.275 | 0.275 | 0.274 | 0.289 | +0.013 | -10% | -42% |
| scherer2015 | riemann:xd=1,fb=0 | persubject | 0.315 |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| scherer2015 | riemann:xd=1,fb=0 | pooled | 0.252 |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| scherer2015 | riemann:xd=1,fb=1 | persubject | 0.320 | 0.363 | 0.383 | 0.308 | 0.303 | 0.322 | 0.309 | 0.315 | 0.382 | +0.063 | 4% | -18% |
| scherer2015 | riemann:xd=1,fb=1 | pooled | 0.259 | 0.282 | 0.293 | 0.266 | 0.283 | 0.271 | 0.277 | 0.282 | 0.300 | +0.034 | 52% | 70% |
| tangermann2012 | eegnet_bd | persubject | 0.395 (n3) |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| tangermann2012 | eegnet_bd | pooled | 0.622 (n3) |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| tangermann2012 | eegnet_st | persubject | 0.546 (n3) |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| tangermann2012 | eegnet_st | pooled | 0.644 (n3) | 0.690 (n3) |  |  |  | 0.653 (n3) |  |  |  | +0.046 | 19% | n/a |
| tangermann2012 | meanlr | persubject | 0.270 |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| tangermann2012 | meanlr | pooled | 0.258 |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| tangermann2012 | riemann:xd=0,fb=0 | persubject | 0.599 |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| tangermann2012 | riemann:xd=0,fb=0 | pooled | 0.501 |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| tangermann2012 | riemann:xd=0,fb=1 | persubject | 0.726 | 0.735 | 0.748 | 0.628 | 0.659 | 0.693 | 0.693 | 0.690 | 0.745 | +0.021 | -156% | -316% |
| tangermann2012 | riemann:xd=0,fb=1 | pooled | 0.544 | 0.577 | 0.606 | 0.473 | 0.474 | 0.562 | 0.574 | 0.567 | 0.596 | +0.062 | 49% | -111% |
| tangermann2012 | riemann:xd=1,fb=0 | persubject | 0.679 |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| tangermann2012 | riemann:xd=1,fb=0 | pooled | 0.663 |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| tangermann2012 | riemann:xd=1,fb=1 | persubject | 0.775 | 0.809 | 0.824 | 0.714 | 0.727 | 0.778 | 0.780 | 0.778 | 0.818 | +0.049 | 10% | -97% |
| tangermann2012 | riemann:xd=1,fb=1 | pooled | 0.639 | 0.721 | 0.726 | 0.573 | 0.588 | 0.677 | 0.687 | 0.688 | 0.723 | +0.087 | 56% | -60% |
| zhou2016 | eegnet_bd | persubject | 0.477 (n3) |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| zhou2016 | eegnet_bd | pooled | 0.592 (n3) |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| zhou2016 | eegnet_st | persubject | 0.561 (n3) |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| zhou2016 | eegnet_st | pooled | 0.754 (n3) | 0.819 (n3) |  | 0.756 (n3) |  | 0.704 (n3) |  |  |  | +0.065 | -76% | 3% |
| zhou2016 | meanlr | persubject | 0.457 |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| zhou2016 | meanlr | pooled | 0.435 |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| zhou2016 | riemann:xd=0,fb=0 | persubject | 0.478 |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| zhou2016 | riemann:xd=0,fb=0 | pooled | 0.555 |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| zhou2016 | riemann:xd=0,fb=1 | persubject | 0.667 | 0.493 | 0.722 | 0.640 | 0.702 | 0.637 | 0.632 | 0.630 | 0.725 | +0.055 | -55% | 64% |
| zhou2016 | riemann:xd=0,fb=1 | pooled | 0.728 | 0.610 | 0.695 | 0.703 | 0.705 | 0.728 | 0.720 | 0.718 | 0.682 | -0.033 | n/a | n/a |
| zhou2016 | riemann:xd=1,fb=0 | persubject | 0.622 |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| zhou2016 | riemann:xd=1,fb=0 | pooled | 0.635 |  |  |  |  |  |  |  |  | +nan | n/a | n/a |
| zhou2016 | riemann:xd=1,fb=1 | persubject | 0.708 | 0.663 | 0.763 | 0.665 | 0.732 | 0.707 | 0.688 | 0.683 | 0.763 | +0.055 | -3% | 42% |
| zhou2016 | riemann:xd=1,fb=1 | pooled | 0.772 | 0.665 | 0.742 | 0.722 | 0.770 | 0.753 | 0.767 | 0.768 | 0.745 | -0.030 | n/a | n/a |

Router accuracy (per window, before fallback):
- scherer2015 all router-psd:euclid: 0.866
- scherer2015 all router-psd:riemann: 0.866
- scherer2015 all routerb-psd:riemann: 0.866
- tangermann2012 all router-psd:euclid: 0.965
- tangermann2012 all router-psd:riemann: 0.965
- tangermann2012 all routerb-psd:riemann: 0.965
- zhou2016 all router-psd:euclid: 1.000
- zhou2016 all router-psd:riemann: 1.000
- zhou2016 all routerb-psd:riemann: 1.000
