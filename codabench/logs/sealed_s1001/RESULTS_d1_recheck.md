# D1 re-check after the review fixes (sprint 2026-10-01 Phase 4)

`lda_dual_check.py --bench 5`, 4 threads, 18:25-18:36, contended by the Phase 3 lane.

```
[18:26:32] iid n60 p4500: n=60 p=4500 K=3 |dP| chol 2.00e-28 (argmax 1.0000), sklearn 4.00e-28 (argmax 1.0000; chol vs sklearn 6.00e-28); residual sk 8.2e-15 chol 9.0e-16 dual 1.2e-15; dcoef 5.46e-16; shrink 9.87e-16; dual used True; harness bits True; plain True; 0.02 s dual vs 1.69 s chol -> OK
[18:26:46] factor+scales n420 p9373: n=420 p=9373 K=3 |dP| chol 1.99e-23 (argmax 1.0000); dcoef 1.91e-14; shrink 1.86e-16; dual used True; harness bits True; plain True; 0.16 s dual vs 10.67 s chol -> OK
[18:27:31] factor n3000 p4500: n=3000 p=4500 K=3 |dP| chol 1.07e-27 (argmax 1.0000), sklearn 3.54e-27 (argmax 1.0000; chol vs sklearn 4.61e-27); residual sk 1.5e-13 chol 7.4e-14 dual 4.8e-13; dcoef 7.37e-14; shrink 1.23e-15; dual used True; harness bits True; plain True; 1.21 s dual vs 5.03 s chol -> OK
[18:28:04] binary n420 p4500: n=420 p=4500 K=2 |dP| chol 1.43e-39 (argmax 1.0000), sklearn 2.37e-38 (argmax 1.0000; chol vs sklearn 2.22e-38); dcoef 5.64e-14; shrink 2.15e-16; dual used True; harness bits True; plain True; 0.05 s dual vs 1.77 s chol -> OK
[18:28:36] tiny class + constant n62 p4500: n=62 p=4500 K=3 |dP| chol 1.56e-13 (argmax 1.0000), sklearn 3.06e-08 (argmax 1.0000; chol vs sklearn 3.06e-08); residual sk 1.6e-11 chol 1.9e-14 dual 2.4e-14; dcoef 1.29e-14; shrink 3.57e-16; dual used True; harness bits True; plain True; 0.01 s dual vs 1.66 s chol -> OK
[18:29:12] empirical priors n300 p4500: n=300 p=4500 K=3 |dP| chol 2.57e-13 (argmax 1.0000), sklearn 1.16e-12 (argmax 1.0000; chol vs sklearn 9.02e-13); residual sk 1.0e-13 chol 2.1e-14 dual 1.2e-13; dcoef 3.17e-14; shrink 6.48e-16; dual used True; harness bits True; plain True; 0.05 s dual vs 2.35 s chol -> OK
[Riemann-StepType] fitting on X=(1080, 30, 480), classes=[0, 1, 2], features=2655 (xdawn 300 + broad 465 + logvar 30 + fb 1860)
[RiemannX] X=(1080, 30, 480) features=4875 (base 2655 + bpt4 480 + icoh 1740)
[18:30:31] scherer3 bpt4+icoh pooled: n=1080 p=4875 K=3 |dP| chol 4.12e-12 (argmax 1.0000), sklearn 1.40e-09 (argmax 1.0000; chol vs sklearn 1.40e-09); residual sk 2.0e-12 chol 9.2e-14 dual 1.3e-13; dcoef 4.63e-14; shrink 7.83e-16; dual used True; harness bits True; plain True; 0.23 s dual vs 3.34 s chol -> OK
[18:31:13] scherer3 bpt4+icoh subj 0: n=120 p=4875 K=3 |dP| chol 7.96e-13 (argmax 1.0000), sklearn 4.34e-11 (argmax 1.0000; chol vs sklearn 4.30e-11); residual sk 2.2e-13 chol 1.1e-14 dual 1.5e-14; dcoef 2.60e-15; shrink 8.08e-16; dual used True; harness bits True; plain True; 0.02 s dual vs 2.07 s chol -> OK
[18:32:01] scherer3 bpt4+icoh subj 1: n=120 p=4875 K=3 |dP| chol 6.75e-13 (argmax 1.0000), sklearn 2.15e-11 (argmax 1.0000; chol vs sklearn 2.17e-11); residual sk 1.3e-13 chol 9.5e-15 dual 1.2e-14; dcoef 2.55e-15; shrink 7.12e-16; dual used True; harness bits True; plain True; 0.01 s dual vs 2.02 s chol -> OK
[18:32:46] scherer3 bpt4+icoh subj 2: n=120 p=4875 K=3 |dP| chol 6.62e-13 (argmax 1.0000), sklearn 4.92e-11 (argmax 1.0000; chol vs sklearn 4.92e-11); residual sk 1.2e-13 chol 1.7e-14 dual 2.9e-14; dcoef 7.06e-15; shrink 2.15e-16; dual used True; harness bits True; plain True; 0.02 s dual vs 2.73 s chol -> OK
[18:33:30] scherer3 bpt4+icoh subj 3: n=120 p=4875 K=3 |dP| chol 6.64e-13 (argmax 1.0000), sklearn 5.60e-10 (argmax 1.0000; chol vs sklearn 5.60e-10); residual sk 2.7e-13 chol 1.8e-14 dual 2.8e-14; dcoef 5.19e-15; shrink 1.12e-15; dual used True; harness bits True; plain True; 0.02 s dual vs 2.07 s chol -> OK
[Riemann-StepType] fitting on X=(2160, 47, 480), classes=[0, 1, 2], features=5987 (xdawn 300 + broad 1128 + logvar 47 + fb 4512)
[18:34:40] mock_s recipe subj 0: n=144 p=5987 K=3 |dP| chol 2.93e-90 (argmax 1.0000); dcoef 5.24e-15; shrink 1.00e-15; dual used True; harness bits True; plain True; 0.05 s dual vs 5.74 s chol -> OK
[18:34:45] mock_s recipe subj 1: n=144 p=5987 K=3 |dP| chol 2.09e-104 (argmax 1.0000); dcoef 6.52e-15; shrink 3.62e-16; dual used True; harness bits True; plain True; 0.05 s dual vs 4.55 s chol -> OK
[18:34:52] mock_s recipe subj 2: n=144 p=5987 K=3 |dP| chol 1.63e-80 (argmax 1.0000); dcoef 2.10e-15; shrink 8.08e-16; dual used True; harness bits True; plain True; 0.02 s dual vs 4.98 s chol -> OK
[18:35:02] mock_s recipe subj 3: n=144 p=5987 K=3 |dP| chol 2.08e-85 (argmax 1.0000); dcoef 3.83e-15; shrink 4.85e-16; dual used True; harness bits True; plain True; 0.07 s dual vs 8.15 s chol -> OK
[18:35:57] bench 5 x (n=420, p=9373): dual 1.7 s, Cholesky 88.1 s (from 3 fits) -> 50.6x
LDA_STATS {'fast': 18, 'fallback': 0, 'dual': 20, 'dual_fallback': 0} (harness copy {'fast': 0, 'fallback': 0, 'dual': 17, 'dual_fallback': 0})
D1 (dual LDA equivalence + speed): PASS
```
