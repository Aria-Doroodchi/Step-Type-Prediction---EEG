# Results (5 runs from 5 files)

Dreyer 2023 test split (participants 61-81), balanced accuracy, chance 0.50.

| solver | config | bal. acc. mean | SD | n | time (s) |
|---|---|---|---|---|---|
| Riemann-StepType | bandpass=none,estimator=oas,filterbank=True,max_batches=None,nfilter=4,reference=none,slow_block=False,use_xdawn=True | 0.770 |  | 1 | 165 |
| Riemann-StepType | bandpass=none,estimator=oas,filterbank=True,max_batches=None,nfilter=4,reference=none,slow_block=True,use_xdawn=True | 0.769 |  | 1 | 167 |
| Riemann-StepType | bandpass=none,estimator=oas,filterbank=False,max_batches=None,nfilter=4,reference=none,slow_block=True,use_xdawn=True | 0.756 |  | 1 | 50 |
| Riemann-StepType | bandpass=none,estimator=oas,filterbank=False,max_batches=None,nfilter=4,reference=none,slow_block=False,use_xdawn=True | 0.754 |  | 1 | 50 |
| Riemann-StepType | bandpass=none,estimator=oas,filterbank=True,max_batches=None,nfilter=4,reference=none,slow_block=True,use_xdawn=False | 0.718 |  | 1 | 144 |
