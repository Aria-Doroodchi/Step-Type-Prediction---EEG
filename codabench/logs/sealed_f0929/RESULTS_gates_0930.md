[gate 1] zhou2016 X=(1800, 14, 480) train=1200 test=600
[Riemann-Sealed] fitting on X=(1200, 14, 480), subjects=4, classes=[0, 1, 2], features=839, align=subject/riemann, personal=blend, router_thr=0.500, session_ids=no, chans=eeg (14 kept)
[Riemann-Sealed] fit seconds: collect 0.0, router 1.8, align 0.7, features 9.0, lda 1.8, wcv 0.0, total 13.2; chunk=1200/1200 windows; maxrss 0.83 GB
[Riemann-Sealed] fitting on X=(1200, 14, 480), subjects=4, classes=[0, 1, 2], features=839, align=subject/riemann, personal=blend, router_thr=0.500, session_ids=no, chans=eeg (14 kept)
[Riemann-Sealed] fit seconds: collect 0.0, router 2.8, align 1.0, features 10.8, lda 1.7, wcv 0.0, total 16.4; chunk=1200/1200 windows; maxrss 0.90 GB
[Riemann-Sealed] fitting on X=(1200, 14, 480), subjects=4, classes=[0, 1, 2], features=839, align=subject/riemann, personal=blend, router_thr=0.500, session_ids=no, chans=eeg (14 kept)
[Riemann-Sealed] fit seconds: collect 0.0, router 1.5, align 0.6, features 7.4, lda 2.2, wcv 0.0, total 11.8; chunk=1200/1200 windows; maxrss 0.90 GB
[gate 1] max |dP| ref vs new = 0.0
[gate 1] max |dP| ref vs new-empty = 0.0
GATE 1 (defaults bit-identical): PASS
[gate 2] scherer2015 [0,1,3] X=(2130, 30, 480) train=1080 test=1050 xblocks=bpt4
[Riemann-StepType] fitting on X=(1080, 30, 480), classes=[0, 1, 2], features=2655 (xdawn 300 + broad 465 + logvar 30 + fb 1860)
[RiemannX] X=(1080, 30, 480) features=3135 (base 2655 + bpt4 480)
[Riemann-Sealed] fitting on X=(1080, 30, 480), subjects=9, classes=[0, 1, 2], features=3135, align=none/riemann, personal=pooled, router_thr=0.500, session_ids=no, chans=eeg (30 kept)
[Riemann-Sealed] fit seconds: collect 0.0, router 10.1, align 0.0, features 20.5, lda 27.6, wcv 0.0, total 58.2; chunk=1080/1080 windows; maxrss 1.66 GB
[gate 2] features (1050, 3135) vs (1050, 3135); max rel |dF| = 0.000e+00; max |dP| = 0.000e+00; acc harness 0.4752 solver 0.4752; harness 101 s, solver 79 s
GATE 2 (harness parity, bpt4): PASS
[gate 2] scherer2015 [0,1,3] X=(2130, 30, 480) train=1080 test=1050 xblocks=icoh
[Riemann-StepType] fitting on X=(1080, 30, 480), classes=[0, 1, 2], features=2655 (xdawn 300 + broad 465 + logvar 30 + fb 1860)
[RiemannX] X=(1080, 30, 480) features=4395 (base 2655 + icoh 1740)
[Riemann-Sealed] fitting on X=(1080, 30, 480), subjects=9, classes=[0, 1, 2], features=4395, align=none/riemann, personal=pooled, router_thr=0.500, session_ids=no, chans=eeg (30 kept)
[Riemann-Sealed] fit seconds: collect 0.1, router 12.3, align 0.0, features 17.0, lda 4.9, wcv 0.0, total 34.2; chunk=1080/1080 windows; maxrss 1.91 GB
[gate 2] features (1050, 4395) vs (1050, 4395); max rel |dF| = 0.000e+00; max |dP| = 0.000e+00; acc harness 0.4352 solver 0.4352; harness 70 s, solver 53 s
GATE 2 (harness parity, icoh): PASS
ALL GATES PASS
