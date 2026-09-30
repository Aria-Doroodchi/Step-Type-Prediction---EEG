## Reverse-time split (--split first): Scherer 3-class (WORD / SUB / HAND), train on the later session, test on the first

Cell-averaged balanced accuracy. Δ in points vs the recipe union (`riemann:xd=1,fb=1`), paired over the 9 subjects: mean (95 % bootstrap CI) subjects improved. Screen = mean of pooled and persubject under the clean router alignment.

| block | fam. | pooled none | persubj. none | pooled router | persubj. router | Δ screen (router) | Δ none (info) |
|---|---|---|---|---|---|---|---|
| baseline | - | 0.456 | 0.516 | 0.429 | 0.505 | — | — |
| icoh | S | 0.446 | 0.529 | 0.461 | 0.517 | +2.22 (+0.56, +4.72) 7/9 | +0.19 (-2.18, +2.22) 5/9 |
| bpt4 | T | 0.446 | 0.542 | 0.438 | 0.524 | +1.44 (-0.42, +3.33) 5/9 | +0.83 (-1.20, +2.69) 6/9 |
| tseg3 | T | 0.513 | 0.571 | 0.465 | 0.537 | +3.43 (+1.53, +5.51) 7/9 | +5.65 (+2.92, +8.01) 7/9 |
| tseg3+icoh | T | 0.517 | 0.570 | 0.476 | 0.542 | +4.21 (+1.81, +6.90) 8/9 | +5.79 (+2.41, +8.84) 6/9 |
