# Results: p1

Cell = balanced accuracy averaged over (subject, test session) cells; pooled = over all test windows. Mean ± SD over seeds (n). Test = each subject's last session.

## scherer2015 (classes all), X=(3550, 30, 480), train=1800, test=1750

| model | mode | align | n | cell | pooled | router acc | fallback | epochs | s/run |
|---|---|---|---|---|---|---|---|---|---|
| riemann:xd=1,fb=1 | persubject | none | 1 | 0.320 | 0.320 |  |  |  | 131 |
| riemann:xd=1,fb=0 | persubject | none | 1 | 0.315 | 0.315 |  |  |  | 17 |
| riemann:xd=0,fb=1 | persubject | none | 1 | 0.301 | 0.301 |  |  |  | 84 |
| eegnet_bd | pooled | none | 3 | 0.295 ± 0.014 | 0.296 ± 0.015 |  |  |  | 91 |
| riemann:xd=0,fb=0 | persubject | none | 1 | 0.294 | 0.294 |  |  |  | 6 |
| eegnet_st | pooled | none | 3 | 0.281 ± 0.010 | 0.282 ± 0.010 |  |  | 25 | 208 |
| riemann:xd=0,fb=1 | pooled | none | 1 | 0.277 | 0.277 |  |  |  | 31 |
| riemann:xd=0,fb=0 | pooled | none | 1 | 0.264 | 0.264 |  |  |  | 4 |
| eegnet_st | persubject | none | 3 | 0.261 ± 0.015 | 0.262 ± 0.015 |  |  | 51 | 285 |
| riemann:xd=1,fb=1 | pooled | none | 1 | 0.259 | 0.260 |  |  |  | 43 |
| riemann:xd=1,fb=0 | pooled | none | 1 | 0.252 | 0.251 |  |  |  | 14 |
| meanlr | pooled | none | 1 | 0.218 | 0.218 |  |  |  | 0 |
| eegnet_bd | persubject | none | 3 | 0.213 ± 0.016 | 0.213 ± 0.016 |  |  |  | 126 |
| meanlr | persubject | none | 1 | 0.193 | 0.193 |  |  |  | 0 |

## tangermann2012 (classes all), X=(5184, 22, 480), train=2592, test=2592

| model | mode | align | n | cell | pooled | router acc | fallback | epochs | s/run |
|---|---|---|---|---|---|---|---|---|---|
| riemann:xd=1,fb=1 | persubject | none | 1 | 0.775 | 0.775 |  |  |  | 68 |
| riemann:xd=0,fb=1 | persubject | none | 1 | 0.726 | 0.726 |  |  |  | 32 |
| riemann:xd=1,fb=0 | persubject | none | 1 | 0.679 | 0.679 |  |  |  | 16 |
| riemann:xd=1,fb=0 | pooled | none | 1 | 0.663 | 0.663 |  |  |  | 14 |
| eegnet_st | pooled | none | 3 | 0.644 ± 0.037 | 0.644 ± 0.037 |  |  | 84 | 531 |
| riemann:xd=1,fb=1 | pooled | none | 1 | 0.639 | 0.639 |  |  |  | 41 |
| eegnet_bd | pooled | none | 3 | 0.622 ± 0.026 | 0.622 ± 0.026 |  |  |  | 87 |
| riemann:xd=0,fb=0 | persubject | none | 1 | 0.599 | 0.599 |  |  |  | 7 |
| eegnet_st | persubject | none | 3 | 0.546 ± 0.065 | 0.546 ± 0.065 |  |  | 82 | 422 |
| riemann:xd=0,fb=1 | pooled | none | 1 | 0.544 | 0.544 |  |  |  | 38 |
| riemann:xd=0,fb=0 | pooled | none | 1 | 0.501 | 0.501 |  |  |  | 8 |
| eegnet_bd | persubject | none | 3 | 0.395 ± 0.008 | 0.395 ± 0.008 |  |  |  | 132 |
| meanlr | persubject | none | 1 | 0.270 | 0.270 |  |  |  | 0 |
| meanlr | pooled | none | 1 | 0.258 | 0.258 |  |  |  | 0 |

## zhou2016 (classes all), X=(1800, 14, 480), train=1200, test=600

| model | mode | align | n | cell | pooled | router acc | fallback | epochs | s/run |
|---|---|---|---|---|---|---|---|---|---|
| riemann:xd=1,fb=1 | pooled | none | 1 | 0.772 | 0.772 |  |  |  | 12 |
| eegnet_st | pooled | none | 3 | 0.754 ± 0.063 | 0.754 ± 0.063 |  |  | 99 | 124 |
| riemann:xd=0,fb=1 | pooled | none | 1 | 0.728 | 0.728 |  |  |  | 8 |
| riemann:xd=1,fb=1 | persubject | none | 1 | 0.708 | 0.708 |  |  |  | 13 |
| riemann:xd=0,fb=1 | persubject | none | 1 | 0.667 | 0.667 |  |  |  | 9 |
| riemann:xd=1,fb=0 | pooled | none | 1 | 0.635 | 0.635 |  |  |  | 6 |
| riemann:xd=1,fb=0 | persubject | none | 1 | 0.622 | 0.622 |  |  |  | 4 |
| eegnet_bd | pooled | none | 3 | 0.592 ± 0.073 | 0.592 ± 0.073 |  |  |  | 22 |
| eegnet_st | persubject | none | 3 | 0.561 ± 0.003 | 0.561 ± 0.003 |  |  | 52 | 52 |
| riemann:xd=0,fb=0 | pooled | none | 1 | 0.555 | 0.555 |  |  |  | 2 |
| riemann:xd=0,fb=0 | persubject | none | 1 | 0.478 | 0.478 |  |  |  | 2 |
| eegnet_bd | persubject | none | 3 | 0.477 ± 0.017 | 0.477 ± 0.017 |  |  |  | 19 |
| meanlr | persubject | none | 1 | 0.457 | 0.457 |  |  |  | 0 |
| meanlr | pooled | none | 1 | 0.435 | 0.435 |  |  |  | 0 |

