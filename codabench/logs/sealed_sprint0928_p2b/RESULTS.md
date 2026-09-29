# Results (3 runs from 3 files)

Dreyer 2023 test split (participants 61-81), balanced accuracy, chance 0.50.

| solver | config | bal. acc. mean | SD | n | time (s) |
|---|---|---|---|---|---|
| Riemann-Sealed | adapt=none,align=subject,bandpass=none,blend_w=0.5,buffer=64,chans=eeg,estimator=oas,filterbank=True,kind=riemann,max_batches=None,nfilter=4,personal=blend,reference=none,slow_block=False,use_xdawn=True | 0.613 | 0.000 | 2 | 0 |
| Riemann-Sealed | adapt=none,align=subject,bandpass=none,blend_w=auto,buffer=64,chans=eeg,estimator=oas,filterbank=True,kind=riemann,max_batches=None,nfilter=4,personal=blend,reference=none,slow_block=False,use_xdawn=True | 0.613 |  | 1 | 1826 |
