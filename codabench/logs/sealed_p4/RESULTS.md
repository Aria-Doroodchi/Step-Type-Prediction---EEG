# Results: p4

Cell = balanced accuracy averaged over (subject, test session) cells; pooled = over all test windows. Mean ± SD over seeds (n). Test = each subject's last session.

## scherer2015 (classes 0,1,3), X=(2130, 30, 480), train=1080, test=1050

| model | mode | align | n | cell | pooled | router acc | fallback | epochs | s/run |
|---|---|---|---|---|---|---|---|---|---|
| riemann:xd=1,fb=1/blend_calib | oracle-id | online-64:riemann | 1 | 0.588 | 0.590 | 0.897 |  |  | 439 |
| riemann:xd=1,fb=1/calib | oracle-id | online-64:riemann | 1 | 0.587 | 0.588 | 0.897 |  |  | 439 |
| riemann:xd=1,fb=1 | persubject | online-64:riemann | 1 | 0.573 | 0.574 | 0.897 |  |  | 93 |
| riemann:xd=1,fb=1/persubject | oracle-id | online-64:riemann | 1 | 0.573 | 0.574 | 0.897 |  |  | 439 |
| riemann:xd=1,fb=1/blend | oracle-id | online-64:riemann | 1 | 0.572 | 0.573 | 0.897 |  |  | 439 |
| riemann:xd=1,fb=1/blend_calib | router-id | online-64:riemann | 1 | 0.561 | 0.563 | 0.897 |  |  | 439 |
| riemann:xd=1,fb=1/calib | router-id | online-64:riemann | 1 | 0.558 | 0.559 | 0.897 |  |  | 439 |
| riemann:xd=0,fb=1 | persubject | online-64:riemann | 1 | 0.556 | 0.556 | 0.897 |  |  | 59 |
| riemann:xd=1,fb=0 | persubject | online-64:riemann | 1 | 0.555 | 0.556 | 0.897 |  |  | 13 |
| riemann:xd=1,fb=1,blocks=fb | persubject | online-64:riemann | 1 | 0.553 | 0.553 | 0.897 |  |  | 130 |
| riemann:xd=1,fb=1/persubject | router-id | online-64:riemann | 1 | 0.546 | 0.548 | 0.897 |  |  | 439 |
| riemann:xd=1,fb=1/blend | router-id | online-64:riemann | 1 | 0.544 | 0.546 | 0.897 |  |  | 439 |
| riemann:xd=1,fb=1/blend | oracle-id | router-psd:riemann | 1 | 0.515 | 0.514 | 0.897 | 0.172 |  | 449 |
| riemann:xd=1,fb=1,blocks=broad+logvar | persubject | online-64:riemann | 1 | 0.509 | 0.510 | 0.897 |  |  | 87 |
| riemann:xd=1,fb=0 | persubject | router-psd:riemann | 1 | 0.501 | 0.500 | 0.897 | 0.172 |  | 11 |
| riemann:xd=1,fb=1 | persubject | router-psd:riemann | 1 | 0.499 | 0.498 | 0.897 | 0.172 |  | 76 |
| riemann:xd=1,fb=1/persubject | oracle-id | router-psd:riemann | 1 | 0.499 | 0.498 | 0.897 | 0.172 |  | 449 |
| riemann:xd=1,fb=1/blend | router-id | router-psd:riemann | 1 | 0.499 | 0.499 | 0.897 | 0.172 |  | 449 |
| riemann:xd=1,fb=1/blend_calib | oracle-id | router-psd:riemann | 1 | 0.497 | 0.497 | 0.897 | 0.172 |  | 449 |
| riemann:xd=1,fb=1 | persubject | none | 1 | 0.493 | 0.492 |  |  |  | 62 |
| riemann:xd=0,fb=1 | persubject | none | 1 | 0.491 | 0.490 |  |  |  | 51 |
| riemann:xd=1,fb=1/calib | oracle-id | router-psd:riemann | 1 | 0.488 | 0.488 | 0.897 | 0.172 |  | 449 |
| riemann:xd=1,fb=1,blocks=fb | pooled | online-64:riemann | 1 | 0.487 | 0.489 | 0.897 |  |  | 37 |
| riemann:xd=0,fb=1/blend | oracle-id | router-psd:riemann | 1 | 0.487 | 0.487 | 0.897 | 0.172 |  | 306 |
| riemann:xd=1,fb=1/blend_calib | router-id | router-psd:riemann | 1 | 0.486 | 0.487 | 0.897 | 0.172 |  | 449 |
| riemann:xd=1,fb=1/persubject | router-id | router-psd:riemann | 1 | 0.486 | 0.486 | 0.897 | 0.172 |  | 449 |
| riemann:xd=0,fb=1/blend_calib | oracle-id | router-psd:riemann | 1 | 0.485 | 0.485 | 0.897 | 0.172 |  | 306 |
| riemann:xd=1,fb=1 | pooled | online-64:riemann | 1 | 0.485 | 0.487 | 0.897 |  |  | 44 |
| riemann:xd=1,fb=1/pooled | none | online-64:riemann | 1 | 0.485 | 0.487 | 0.897 |  |  | 439 |
| riemann:xd=1,fb=1,blocks=fb | persubject | none | 1 | 0.484 | 0.484 |  |  |  | 91 |
| riemann:xd=1,fb=1,blocks=xdawn | persubject | online-64:riemann | 1 | 0.483 | 0.484 | 0.897 |  |  | 105 |
| riemann:xd=1,fb=0 | persubject | none | 1 | 0.479 | 0.478 |  |  |  | 8 |
| riemann:xd=0,fb=1/calib | oracle-id | router-psd:riemann | 1 | 0.479 | 0.479 | 0.897 | 0.172 |  | 306 |
| riemann:xd=0,fb=1/blend | router-id | router-psd:riemann | 1 | 0.479 | 0.479 | 0.897 | 0.172 |  | 306 |
| riemann:xd=0,fb=1/blend_calib | router-id | router-psd:riemann | 1 | 0.477 | 0.477 | 0.897 | 0.172 |  | 306 |
| riemann:xd=0,fb=1 | persubject | router-psd:riemann | 1 | 0.476 | 0.476 | 0.897 | 0.172 |  | 54 |
| riemann:xd=0,fb=1/persubject | oracle-id | router-psd:riemann | 1 | 0.476 | 0.476 | 0.897 | 0.172 |  | 306 |
| riemann:xd=0,fb=1 | pooled | online-64:riemann | 1 | 0.476 | 0.478 | 0.897 |  |  | 24 |
| eegnet_st | pooled | none | 3 | 0.476 ± 0.010 | 0.477 ± 0.010 |  |  | 28 | 101 |
| riemann:xd=1,fb=0 | pooled | online-64:riemann | 1 | 0.473 | 0.474 | 0.897 |  |  | 11 |
| riemann:xd=1,fb=1/calib | router-id | router-psd:riemann | 1 | 0.473 | 0.473 | 0.897 | 0.172 |  | 449 |
| riemann:xd=1,fb=1,blocks=xdawn | persubject | router-psd:riemann | 1 | 0.472 | 0.473 | 0.897 | 0.172 |  | 95 |
| riemann:xd=0,fb=1/persubject | router-id | router-psd:riemann | 1 | 0.469 | 0.470 | 0.897 | 0.172 |  | 306 |
| riemann:xd=0,fb=1/calib | router-id | router-psd:riemann | 1 | 0.469 | 0.470 | 0.897 | 0.172 |  | 306 |
| eegnet_st | pooled | oracle:euclid | 3 | 0.469 ± 0.036 | 0.470 ± 0.036 |  |  | 35 | 112 |
| riemann:xd=1,fb=1,blocks=fb | persubject | router-psd:riemann | 1 | 0.463 | 0.463 | 0.897 | 0.172 |  | 115 |
| eegnet_st | pooled | online-64:euclid | 3 | 0.459 ± 0.032 | 0.460 ± 0.033 | 0.897 ± 0.000 |  | 35 | 110 |
| riemann:xd=1,fb=1,blocks=logvar | persubject | online-64:riemann | 1 | 0.458 | 0.460 | 0.897 |  |  | 82 |
| riemann:xd=1,fb=1 | pooled | router-psd:riemann | 1 | 0.452 | 0.452 | 0.897 | 0.172 |  | 46 |
| riemann:xd=1,fb=1/pooled | none | router-psd:riemann | 1 | 0.452 | 0.452 | 0.897 | 0.172 |  | 449 |
| riemann:xd=1,fb=1,blocks=broad+logvar | pooled | online-64:riemann | 1 | 0.450 | 0.451 | 0.897 |  |  | 39 |
| riemann:xd=1,fb=0 | pooled | router-psd:riemann | 1 | 0.448 | 0.449 | 0.897 | 0.172 |  | 9 |
| riemann:xd=1,fb=1,blocks=broad+logvar | persubject | none | 1 | 0.446 | 0.446 |  |  |  | 87 |
| riemann:xd=1,fb=1,blocks=broad+logvar | persubject | router-psd:riemann | 1 | 0.443 | 0.443 | 0.897 | 0.172 |  | 84 |
| riemann:xd=1,fb=1 | pooled | none | 1 | 0.443 | 0.444 |  |  |  | 41 |
| riemann:xd=1,fb=1,blocks=xdawn | pooled | online-64:riemann | 1 | 0.437 | 0.436 | 0.897 |  |  | 45 |
| riemann:xd=1,fb=1,blocks=fb | pooled | none | 1 | 0.435 | 0.436 |  |  |  | 35 |
| riemann:xd=1,fb=1,blocks=fb | pooled | router-psd:riemann | 1 | 0.433 | 0.432 | 0.897 | 0.172 |  | 43 |
| riemann:xd=1,fb=1,blocks=logvar | persubject | none | 1 | 0.432 | 0.433 |  |  |  | 83 |
| riemann:xd=0,fb=1 | pooled | none | 1 | 0.431 | 0.433 |  |  |  | 20 |
| riemann:xd=0,fb=1 | pooled | router-psd:riemann | 1 | 0.431 | 0.431 | 0.897 | 0.172 |  | 22 |
| riemann:xd=0,fb=1/pooled | none | router-psd:riemann | 1 | 0.431 | 0.431 | 0.897 | 0.172 |  | 306 |
| riemann:xd=1,fb=1,blocks=broad+logvar | pooled | none | 1 | 0.423 | 0.425 |  |  |  | 31 |
| riemann:xd=1,fb=0 | pooled | none | 1 | 0.423 | 0.423 |  |  |  | 6 |
| riemann:xd=1,fb=1,blocks=xdawn | persubject | none | 1 | 0.419 | 0.419 |  |  |  | 105 |
| riemann:xd=1,fb=1,blocks=logvar | pooled | online-64:riemann | 1 | 0.413 | 0.413 | 0.897 |  |  | 36 |
| riemann:xd=1,fb=1,blocks=xdawn | pooled | router-psd:riemann | 1 | 0.413 | 0.412 | 0.897 | 0.172 |  | 34 |
| riemann:xd=1,fb=1,blocks=broad+logvar | pooled | router-psd:riemann | 1 | 0.406 | 0.406 | 0.897 | 0.172 |  | 34 |
| riemann:xd=1,fb=1,blocks=logvar | persubject | router-psd:riemann | 1 | 0.397 | 0.398 | 0.897 | 0.172 |  | 91 |
| riemann:xd=1,fb=1,blocks=xdawn | pooled | none | 1 | 0.386 | 0.386 |  |  |  | 30 |
| riemann:xd=1,fb=1,blocks=logvar | pooled | none | 1 | 0.381 | 0.383 |  |  |  | 30 |
| eegnet_st | persubject | none | 3 | 0.363 ± 0.012 | 0.363 ± 0.012 |  |  | 40 | 123 |
| riemann:xd=1,fb=1,blocks=logvar | pooled | router-psd:riemann | 1 | 0.361 | 0.361 | 0.897 | 0.172 |  | 33 |
| meanlr | pooled | none | 1 | 0.353 | 0.353 |  |  |  | 0 |
| meanlr | pooled | router-psd:riemann | 1 | 0.337 | 0.337 | 0.897 | 0.172 |  | 3 |
| meanlr | persubject | none | 1 | 0.336 | 0.336 |  |  |  | 0 |
| meanlr | pooled | online-64:riemann | 1 | 0.324 | 0.325 | 0.897 |  |  | 5 |
| meanlr | persubject | online-64:riemann | 1 | 0.324 | 0.324 | 0.897 |  |  | 6 |
| meanlr | persubject | router-psd:riemann | 1 | 0.324 | 0.324 | 0.897 | 0.172 |  | 3 |

## zyma2019 (classes all), X=(1659, 19, 600), train=1327, test=1659

| model | mode | align | n | cell | pooled | router acc | fallback | epochs | s/run |
|---|---|---|---|---|---|---|---|---|---|
| riemann:xd=1,fb=1,blocks=fb | xsubject5 | subject:riemann | 1 | 0.779 | 0.779 |  |  |  | 101 |
| riemann:xd=1,fb=1,blocks=xdawn | xsubject5 | subject:riemann | 1 | 0.778 | 0.785 |  |  |  | 104 |
| riemann:xd=1,fb=1 | xsubject5 | subject:riemann | 1 | 0.762 | 0.764 |  |  |  | 64 |
| riemann:xd=0,fb=1 | xsubject5 | subject:riemann | 1 | 0.762 | 0.761 |  |  |  | 47 |
| riemann:xd=1,fb=1,blocks=broad+logvar | xsubject5 | subject:riemann | 1 | 0.753 | 0.757 |  |  |  | 108 |
| riemann:xd=1,fb=0 | xsubject5 | subject:riemann | 1 | 0.751 | 0.757 |  |  |  | 18 |
| riemann:xd=0,fb=1 | xsubject5 | subject:euclid | 1 | 0.747 | 0.746 |  |  |  | 62 |
| riemann:xd=1,fb=1,blocks=xdawn | xsubject5 | subject:euclid | 1 | 0.746 | 0.749 |  |  |  | 108 |
| riemann:xd=1,fb=1 | xsubject5 | subject:euclid | 1 | 0.743 | 0.742 |  |  |  | 74 |
| riemann:xd=1,fb=1,blocks=fb | xsubject5 | subject:euclid | 1 | 0.740 | 0.739 |  |  |  | 102 |
| riemann:xd=1,fb=0 | xsubject5 | subject:euclid | 1 | 0.734 | 0.736 |  |  |  | 21 |
| riemann:xd=1,fb=1,blocks=fb | xsubject5 | none | 1 | 0.734 | 0.731 |  |  |  | 127 |
| riemann:xd=1,fb=1,blocks=broad+logvar | xsubject5 | subject:euclid | 1 | 0.731 | 0.733 |  |  |  | 99 |
| riemann:xd=0,fb=1 | xsubject5 | none | 1 | 0.724 | 0.722 |  |  |  | 90 |
| riemann:xd=1,fb=1 | xsubject5 | none | 1 | 0.723 | 0.721 |  |  |  | 99 |
| riemann:xd=1,fb=1,blocks=broad+logvar | xsubject5 | none | 1 | 0.627 | 0.628 |  |  |  | 106 |
| riemann:xd=1,fb=0 | xsubject5 | none | 1 | 0.625 | 0.626 |  |  |  | 29 |
| riemann:xd=1,fb=1,blocks=xdawn | xsubject5 | none | 1 | 0.622 | 0.625 |  |  |  | 132 |
| meanlr | xsubject5 | subject:riemann | 1 | 0.519 | 0.519 |  |  |  | 0 |
| meanlr | xsubject5 | none | 1 | 0.512 | 0.512 |  |  |  | 1 |
| meanlr | xsubject5 | subject:euclid | 1 | 0.508 | 0.508 |  |  |  | 0 |

