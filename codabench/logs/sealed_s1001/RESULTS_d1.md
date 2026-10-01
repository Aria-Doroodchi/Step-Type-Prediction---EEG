# Sprint 2026-10-01 Phase 1: D1 dual-LDA equivalence gate

`python ~/codabench/analysis/lda_dual_check.py --bench 20`, 10 threads, 17:57:46-18:05:42 (8 min; est. 15). Rule: brief § 5 Phase 1 D1 (vs-sklearn deviation for ill-conditioned cases: LOG 2026-10-01).

```
[17:58:17] iid n60 p4500: n=60 p=4500 K=3 |dP| chol 5.01e-29 (argmax 1.0000), sklearn 1.60e-27 (argmax 1.0000; chol vs sklearn 1.55e-27); residual sk 7.9e-15 chol 9.0e-16 dual 1.2e-15; dcoef 5.41e-16; shrink 9.87e-16; dual used True; harness bits True; plain True; 0.02 s dual vs 1.62 s chol -> OK
[17:58:27] factor+scales n420 p9373: n=420 p=9373 K=3 |dP| chol 2.24e-24 (argmax 1.0000); dcoef 1.92e-14; shrink 1.86e-16; dual used True; harness bits True; plain True; 0.10 s dual vs 7.63 s chol -> OK
[17:59:04] factor n3000 p4500: n=3000 p=4500 K=3 |dP| chol 1.01e-27 (argmax 1.0000), sklearn 6.63e-27 (argmax 1.0000; chol vs sklearn 5.62e-27); residual sk 1.5e-13 chol 7.4e-14 dual 5.7e-13; dcoef 7.29e-14; shrink 1.23e-15; dual used True; harness bits True; plain True; 0.90 s dual vs 3.55 s chol -> OK
[17:59:32] binary n420 p4500: n=420 p=4500 K=2 |dP| chol 5.20e-39 (argmax 1.0000), sklearn 8.45e-39 (argmax 1.0000; chol vs sklearn 3.25e-39); dcoef 5.58e-14; shrink 2.15e-16; dual used True; harness bits True; plain True; 0.06 s dual vs 1.27 s chol -> OK
[18:00:00] tiny class + constant n62 p4500: n=62 p=4500 K=3 |dP| chol 1.28e-13 (argmax 1.0000), sklearn 4.68e-08 (argmax 1.0000; chol vs sklearn 4.68e-08); residual sk 1.4e-11 chol 2.0e-14 dual 2.4e-14; dcoef 1.28e-14; shrink 3.57e-16; dual used True; harness bits True; plain True; 0.01 s dual vs 1.37 s chol -> OK
[18:00:29] empirical priors n300 p4500: n=300 p=4500 K=3 |dP| chol 1.77e-13 (argmax 1.0000), sklearn 8.88e-13 (argmax 1.0000; chol vs sklearn 8.43e-13); residual sk 1.0e-13 chol 2.1e-14 dual 1.2e-13; dcoef 3.18e-14; shrink 6.48e-16; dual used True; harness bits True; plain True; 0.04 s dual vs 1.49 s chol -> OK
[Riemann-StepType] fitting on X=(1080, 30, 480), classes=[0, 1, 2], features=2655 (xdawn 300 + broad 465 + logvar 30 + fb 1860)
[RiemannX] X=(1080, 30, 480) features=4875 (base 2655 + bpt4 480 + icoh 1740)
[18:01:37] scherer3 bpt4+icoh pooled: n=1080 p=4875 K=3 |dP| chol 3.71e-12 (argmax 1.0000), sklearn 1.29e-09 (argmax 1.0000; chol vs sklearn 1.29e-09); residual sk 2.1e-12 chol 9.8e-14 dual 1.4e-13; dcoef 4.19e-14; shrink 1.04e-15; dual used True; harness bits True; plain True; 0.19 s dual vs 2.77 s chol -> OK
[18:02:14] scherer3 bpt4+icoh subj 0: n=120 p=4875 K=3 |dP| chol 3.94e-13 (argmax 1.0000), sklearn 1.16e-10 (argmax 1.0000; chol vs sklearn 1.16e-10); residual sk 2.2e-13 chol 1.1e-14 dual 1.5e-14; dcoef 2.96e-15; shrink 8.08e-16; dual used True; harness bits True; plain True; 0.02 s dual vs 1.67 s chol -> OK
[18:02:50] scherer3 bpt4+icoh subj 1: n=120 p=4875 K=3 |dP| chol 7.69e-13 (argmax 1.0000), sklearn 2.97e-11 (argmax 1.0000; chol vs sklearn 2.89e-11); residual sk 1.3e-13 chol 9.8e-15 dual 1.1e-14; dcoef 2.42e-15; shrink 5.34e-16; dual used True; harness bits True; plain True; 0.02 s dual vs 1.68 s chol -> OK
[18:03:26] scherer3 bpt4+icoh subj 2: n=120 p=4875 K=3 |dP| chol 1.77e-12 (argmax 1.0000), sklearn 6.22e-11 (argmax 1.0000; chol vs sklearn 6.09e-11); residual sk 1.2e-13 chol 1.8e-14 dual 3.2e-14; dcoef 6.47e-15; shrink 3.40e-16; dual used True; harness bits True; plain True; 0.01 s dual vs 1.70 s chol -> OK
[18:04:02] scherer3 bpt4+icoh subj 3: n=120 p=4875 K=3 |dP| chol 4.89e-13 (argmax 1.0000), sklearn 5.99e-10 (argmax 1.0000; chol vs sklearn 5.99e-10); residual sk 2.7e-13 chol 1.8e-14 dual 1.9e-14; dcoef 4.39e-15; shrink 9.63e-16; dual used True; harness bits True; plain True; 0.02 s dual vs 1.83 s chol -> OK
[Riemann-StepType] fitting on X=(2160, 47, 480), classes=[0, 1, 2], features=5987 (xdawn 300 + broad 1128 + logvar 47 + fb 4512)
[18:04:58] mock_s recipe subj 0: n=144 p=5987 K=3 |dP| chol 4.89e-91 (argmax 1.0000); dcoef 5.68e-15; shrink 1.00e-15; dual used True; harness bits True; plain True; 0.02 s dual vs 2.60 s chol -> OK
[18:05:01] mock_s recipe subj 1: n=144 p=5987 K=3 |dP| chol 4.19e-104 (argmax 1.0000); dcoef 8.28e-15; shrink 5.43e-16; dual used True; harness bits True; plain True; 0.03 s dual vs 2.65 s chol -> OK
[18:05:04] mock_s recipe subj 2: n=144 p=5987 K=3 |dP| chol 2.22e-80 (argmax 1.0000); dcoef 2.17e-15; shrink 4.96e-16; dual used True; harness bits True; plain True; 0.03 s dual vs 2.65 s chol -> OK
[18:05:07] mock_s recipe subj 3: n=144 p=5987 K=3 |dP| chol 2.43e-85 (argmax 1.0000); dcoef 4.60e-15; shrink 8.08e-16; dual used True; harness bits True; plain True; 0.02 s dual vs 3.00 s chol -> OK
[18:05:38] bench 20 x (n=420, p=9373): dual 2.9 s, Cholesky 178.4 s (from 3 fits) -> 60.7x
LDA_STATS {'fast': 18, 'fallback': 0, 'dual': 35, 'dual_fallback': 0} (harness copy {'fast': 0, 'fallback': 0, 'dual': 17, 'dual_fallback': 0})
D1 (dual LDA equivalence + speed): PASS
```
