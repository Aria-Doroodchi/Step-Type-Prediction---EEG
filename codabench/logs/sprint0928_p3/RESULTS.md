# Sprint 0928 Phase 3: online re-centring on a realistic test stream

Metric: **cell-averaged balanced accuracy** (mean over subject × test-session cells) unless a column says otherwise. Splits: *last* = each subject's last session is the test session (the harness split); *stream* = Zhou calibrated on session 0 only, test sessions 1 and 2 fed in recording order (the sealed structure has 3 consecutive test sessions). Recipe: Riemann xDAWN + FB, log-PSD router, blend_calib with router ids, w chosen on training data only. **clean** = router alignment (stateless per window; the recipe default). **online-N (RULE-DEPENDENT)** = each routed subject's whitening reference is the mean covariance of its last N routed test windows (training reference until N/4 are seen), fed 64 per predict() call; never a default. Single deterministic fits; uncertainty is across subjects (BOOTSTRAP.md). Code: `analysis/sealed_stream.py`.

## Sanity: reproduction of the committed Phase 3 rows (last split, recording order)

| proxy | clean committed | clean here | online-64 committed (rule-dep.) | online-64 here (rule-dep.) | max abs. diff | w clean / online |
|---|---|---|---|---|---|---|
| Tangermann (4 cl., last session) | 0.802 | 0.802 | 0.837 | 0.837 | 0.0000 | 0.5 / 0.5 |
| Scherer 3-class (WORD/SUB/HAND, last session) | 0.486 | 0.486 | 0.561 | 0.561 | 0.0000 | 0.5 / 0.5 |
| Zhou (3 cl., last session) | 0.778 | 0.778 | 0.797 | 0.797 | 0.0000 | 0.75 / 0.5 |

## 3a: multi-session stream (Zhou, calibrate on session 0, test sessions 1 → 2, recording order)

n_train=614, n_test=1186, router accuracy 0.534 (per test session: session 1 0.718, session 2 0.353; chance 0.25), fallback to pooled 0.490 (threshold 1.000: with one calibration session the out-of-fold posteriors saturate). blend_calib w = 0.5 (training chronological halves, CV [0.706, 0.7, 0.748, 0.738, 0.731]). The router is the same for clean and online, so the comparison is fair, but on session 2 most windows go to the wrong subject's reference and personal LDA in both conditions.

Cell score overall and per test session; per-subject = mean of its 2 cells. Online columns are rule-dependent; resets = buffer resets by the guard.

| variant | cell (all) | session 1 | session 2 | subj 0 | subj 1 | subj 2 | subj 3 | resets |
|---|---|---|---|---|---|---|---|---|
| clean | 0.616 | 0.608 | 0.623 | 0.643 | 0.675 | 0.568 | 0.577 |  |
| online-32 (rule-dep.) | 0.660 | 0.654 | 0.667 | 0.630 | 0.773 | 0.529 | 0.710 |  |
| online-64 (rule-dep.) | 0.651 | 0.654 | 0.647 | 0.630 | 0.780 | 0.515 | 0.677 |  |
| online-128 (rule-dep.) | 0.667 | 0.679 | 0.655 | 0.673 | 0.767 | 0.552 | 0.677 |  |
| online-32 + guard (rule-dep.) | 0.660 | 0.652 | 0.667 | 0.630 | 0.776 | 0.522 | 0.710 | 1 |
| online-64 + guard (rule-dep.) | 0.650 | 0.653 | 0.647 | 0.630 | 0.784 | 0.509 | 0.677 | 1 |
| online-128 + guard (rule-dep.) | 0.658 | 0.665 | 0.652 | 0.673 | 0.771 | 0.515 | 0.673 | 1 |

Session-boundary check (second test session = session 2): plain **accuracy** on its first 32 windows in stream order vs the rest of that session and the whole session, mean over subjects (per subject in brackets); first − session mean < 0 would be a post-boundary drop; gain = variant − clean on the same windows; the last two columns use the session's balanced accuracy.

| variant | first 32 | rest | session mean | first − session mean | gain first 32 | gain rest | session-2 BA gain per subject | subjects with gain < 0 |
|---|---|---|---|---|---|---|---|---|
| clean | 0.641 (0.78; 0.69; 0.47; 0.62) | 0.619 | 0.623 | +0.017 | +0.000 | +0.000 | +0.000, +0.000, +0.000, +0.000 | - |
| online-32 (rule-dep.) | 0.703 (0.72; 0.84; 0.53; 0.72) | 0.657 | 0.667 | +0.036 | +0.062 | +0.038 | -0.087, +0.113, +0.027, +0.120 | 1/4 |
| online-64 (rule-dep.) | 0.680 (0.78; 0.78; 0.47; 0.69) | 0.638 | 0.647 | +0.033 | +0.039 | +0.019 | -0.100, +0.107, -0.020, +0.107 | 2/4 |
| online-128 (rule-dep.) | 0.695 (0.78; 0.72; 0.50; 0.78) | 0.644 | 0.655 | +0.040 | +0.055 | +0.025 | -0.020, +0.080, -0.033, +0.100 | 2/4 |
| online-32 + guard (rule-dep.) | 0.703 (0.72; 0.84; 0.53; 0.72) | 0.657 | 0.667 | +0.036 | +0.062 | +0.038 | -0.087, +0.113, +0.027, +0.120 | 1/4 |
| online-64 + guard (rule-dep.) | 0.680 (0.78; 0.78; 0.47; 0.69) | 0.638 | 0.647 | +0.033 | +0.039 | +0.019 | -0.100, +0.107, -0.020, +0.107 | 2/4 |
| online-128 + guard (rule-dep.) | 0.688 (0.78; 0.72; 0.50; 0.75) | 0.642 | 0.652 | +0.036 | +0.047 | +0.023 | -0.020, +0.080, -0.040, +0.093 | 2/4 |

## 3b: order robustness (online-N minus clean, cell score)

Orders: **rec** = recording order; **inter** = subjects interleaved round-robin window by window, each subject's windows in order; **shuf** = fully shuffled, mean ± SD over seeds 0, 1, 2 (sessions mixed on the stream). The clean score is the same for every order. Retained = inter gain / rec gain. All online columns are rule-dependent.

| proxy | N | clean | online rec | online inter | online shuf | gain rec | gain inter | gain shuf | retained (inter) |
|---|---|---|---|---|---|---|---|---|---|
| Tangermann (4 cl., last session) | 32 | 0.802 | 0.836 | 0.835 | 0.835 ± 0.002 (n3) | +0.034 | +0.033 | +0.033 | 98% |
| Tangermann (4 cl., last session) | 64 | 0.802 | 0.837 | 0.832 | 0.833 ± 0.002 (n3) | +0.035 | +0.030 | +0.031 | 85% |
| Tangermann (4 cl., last session) | 128 | 0.802 | 0.831 | 0.829 | 0.833 ± 0.001 (n3) | +0.029 | +0.027 | +0.031 | 95% |
| Scherer 3-class (WORD/SUB/HAND, last session) | 32 | 0.486 | 0.556 | 0.546 | 0.537 ± 0.006 (n3) | +0.070 | +0.060 | +0.051 | 86% |
| Scherer 3-class (WORD/SUB/HAND, last session) | 64 | 0.486 | 0.561 | 0.530 | 0.532 ± 0.003 (n3) | +0.075 | +0.044 | +0.046 | 58% |
| Scherer 3-class (WORD/SUB/HAND, last session) | 128 | 0.486 | 0.549 | 0.517 | 0.525 ± 0.006 (n3) | +0.063 | +0.031 | +0.039 | 49% |
| Zhou (3 cl., last session) | 32 | 0.778 | 0.800 | 0.795 | 0.788 ± 0.006 (n3) | +0.022 | +0.017 | +0.010 | 77% |
| Zhou (3 cl., last session) | 64 | 0.778 | 0.797 | 0.783 | 0.794 ± 0.002 (n3) | +0.018 | +0.005 | +0.016 | 27% |
| Zhou (3 cl., last session) | 128 | 0.778 | 0.787 | 0.758 | 0.792 ± 0.001 (n3) | +0.008 | -0.020 | +0.013 | -240% |
| Zhou stream (calib session 0; test sessions 1+2) | 32 | 0.616 | 0.660 | 0.641 | 0.634 ± 0.004 (n3) | +0.044 | +0.025 | +0.018 | 57% |
| Zhou stream (calib session 0; test sessions 1+2) | 64 | 0.616 | 0.651 | 0.638 | 0.633 ± 0.007 (n3) | +0.035 | +0.022 | +0.017 | 65% |
| Zhou stream (calib session 0; test sessions 1+2) | 128 | 0.616 | 0.667 | 0.630 | 0.627 ± 0.008 (n3) | +0.051 | +0.014 | +0.011 | 28% |

Batch composition per order (64 windows per predict() call): share of batches holding one true subject, mean number of routed subjects per batch and mean windows per routed subject per batch. In recording order a batch is mostly one subject, so online-64's buffer is essentially the current batch (its own statistics, look-ahead within the batch); interleaved or shuffled, each subject gets a few windows per batch and its buffer is mostly past windows.

| proxy | order | single-subject batches | routed subjects / batch | windows / routed subject |
|---|---|---|---|---|
| Tangermann (4 cl., last session) | rec | 0.90 | 1.7 | 47.9 |
| Tangermann (4 cl., last session) | inter | 0.00 | 8.9 | 7.1 |
| Tangermann (4 cl., last session) | shuf0 | 0.00 | 9.0 | 7.0 |
| Tangermann (4 cl., last session) | shuf1 | 0.00 | 9.0 | 7.0 |
| Tangermann (4 cl., last session) | shuf2 | 0.00 | 9.0 | 7.0 |
| Scherer 3-class (WORD/SUB/HAND, last session) | rec | 0.53 | 3.1 | 24.0 |
| Scherer 3-class (WORD/SUB/HAND, last session) | inter | 0.00 | 8.8 | 7.0 |
| Scherer 3-class (WORD/SUB/HAND, last session) | shuf0 | 0.00 | 9.0 | 6.9 |
| Scherer 3-class (WORD/SUB/HAND, last session) | shuf1 | 0.00 | 8.9 | 6.9 |
| Scherer 3-class (WORD/SUB/HAND, last session) | shuf2 | 0.00 | 9.0 | 6.9 |
| Zhou (3 cl., last session) | rec | 0.70 | 1.3 | 50.4 |
| Zhou (3 cl., last session) | inter | 0.00 | 4.0 | 15.0 |
| Zhou (3 cl., last session) | shuf0 | 0.00 | 4.0 | 15.0 |
| Zhou (3 cl., last session) | shuf1 | 0.00 | 4.0 | 15.0 |
| Zhou (3 cl., last session) | shuf2 | 0.00 | 4.0 | 15.0 |
| Zhou stream (calib session 0; test sessions 1+2) | rec | 0.84 | 2.1 | 33.5 |
| Zhou stream (calib session 0; test sessions 1+2) | inter | 0.00 | 3.5 | 18.0 |
| Zhou stream (calib session 0; test sessions 1+2) | shuf0 | 0.00 | 4.0 | 15.6 |
| Zhou stream (calib session 0; test sessions 1+2) | shuf1 | 0.00 | 4.0 | 15.6 |
| Zhou stream (calib session 0; test sessions 1+2) | shuf2 | 0.00 | 4.0 | 15.6 |

Buffer-reset guard (rule-dependent, like online): a routed subject's buffer is emptied when the median Riemannian distance of its windows in the current batch to the buffer mean exceeds the 99th percentile of training windows' distances to their own (subject, session) mean (training data only).

| proxy | N | order | online | online + guard | guard − online | resets | threshold |
|---|---|---|---|---|---|---|---|
| Tangermann (4 cl., last session) | 64 | rec | 0.837 | 0.838 | +0.001 | 3 | 4.216 |
| Tangermann (4 cl., last session) | 64 | inter | 0.832 | 0.831 | -0.001 | 4 | 4.216 |
| Tangermann (4 cl., last session) | 64 | shuf0 | 0.829 | 0.829 | +0.000 | 0 | 4.216 |
| Tangermann (4 cl., last session) | 64 | shuf1 | 0.833 | 0.833 | +0.000 | 0 | 4.216 |
| Tangermann (4 cl., last session) | 64 | shuf2 | 0.835 | 0.836 | +0.000 | 1 | 4.216 |
| Scherer 3-class (WORD/SUB/HAND, last session) | 64 | rec | 0.561 | 0.561 | -0.000 | 2 | 14.202 |
| Scherer 3-class (WORD/SUB/HAND, last session) | 64 | inter | 0.530 | 0.530 | +0.000 | 0 | 14.202 |
| Scherer 3-class (WORD/SUB/HAND, last session) | 64 | shuf0 | 0.530 | 0.530 | +0.000 | 0 | 14.202 |
| Scherer 3-class (WORD/SUB/HAND, last session) | 64 | shuf1 | 0.529 | 0.529 | +0.000 | 0 | 14.202 |
| Scherer 3-class (WORD/SUB/HAND, last session) | 64 | shuf2 | 0.536 | 0.536 | +0.000 | 0 | 14.202 |
| Zhou (3 cl., last session) | 64 | rec | 0.797 | 0.797 | +0.000 | 0 | 11.167 |
| Zhou (3 cl., last session) | 64 | inter | 0.783 | 0.783 | +0.000 | 0 | 11.167 |
| Zhou (3 cl., last session) | 64 | shuf0 | 0.792 | 0.792 | +0.000 | 0 | 11.167 |
| Zhou (3 cl., last session) | 64 | shuf1 | 0.793 | 0.793 | +0.000 | 0 | 11.167 |
| Zhou (3 cl., last session) | 64 | shuf2 | 0.797 | 0.797 | +0.000 | 0 | 11.167 |
| Zhou stream (calib session 0; test sessions 1+2) | 32 | rec | 0.660 | 0.660 | -0.001 | 1 | 8.713 |
| Zhou stream (calib session 0; test sessions 1+2) | 32 | inter | 0.641 | 0.641 | +0.000 | 1 | 8.713 |
| Zhou stream (calib session 0; test sessions 1+2) | 32 | shuf0 | 0.630 | 0.630 | +0.000 | 0 | 8.713 |
| Zhou stream (calib session 0; test sessions 1+2) | 32 | shuf1 | 0.639 | 0.639 | +0.000 | 0 | 8.713 |
| Zhou stream (calib session 0; test sessions 1+2) | 32 | shuf2 | 0.632 | 0.632 | +0.000 | 0 | 8.713 |
| Zhou stream (calib session 0; test sessions 1+2) | 64 | rec | 0.651 | 0.650 | -0.001 | 1 | 8.713 |
| Zhou stream (calib session 0; test sessions 1+2) | 64 | inter | 0.638 | 0.638 | +0.000 | 1 | 8.713 |
| Zhou stream (calib session 0; test sessions 1+2) | 64 | shuf0 | 0.625 | 0.625 | +0.000 | 0 | 8.713 |
| Zhou stream (calib session 0; test sessions 1+2) | 64 | shuf1 | 0.643 | 0.643 | +0.000 | 0 | 8.713 |
| Zhou stream (calib session 0; test sessions 1+2) | 64 | shuf2 | 0.630 | 0.630 | +0.000 | 0 | 8.713 |
| Zhou stream (calib session 0; test sessions 1+2) | 128 | rec | 0.667 | 0.658 | -0.009 | 1 | 8.713 |
| Zhou stream (calib session 0; test sessions 1+2) | 128 | inter | 0.630 | 0.630 | +0.000 | 1 | 8.713 |
| Zhou stream (calib session 0; test sessions 1+2) | 128 | shuf0 | 0.622 | 0.622 | +0.000 | 0 | 8.713 |
| Zhou stream (calib session 0; test sessions 1+2) | 128 | shuf1 | 0.638 | 0.638 | +0.000 | 0 | 8.713 |
| Zhou stream (calib session 0; test sessions 1+2) | 128 | shuf2 | 0.621 | 0.621 | +0.000 | 0 | 8.713 |

Guard resets on the Zhou stream: N=32 rec: routed subject 2 at stream position 384, its windows in that batch mostly true subject 1, session 1 (a routing error, not a session change); N=32 inter: routed subject 1 at stream position 576, its windows in that batch mostly true subject 1, session 2; N=64 rec: routed subject 2 at stream position 384, its windows in that batch mostly true subject 1, session 1 (a routing error, not a session change); N=64 inter: routed subject 1 at stream position 576, its windows in that batch mostly true subject 1, session 2; N=128 rec: routed subject 2 at stream position 384, its windows in that batch mostly true subject 1, session 1 (a routing error, not a session change); N=128 inter: routed subject 1 at stream position 576, its windows in that batch mostly true subject 1, session 2.

## Pre-registered decisions (brief § 5, Phase 3)

- **Order robustness** (online-64 keeps ≥ 75 % of its recording-order gain under the interleaved order on every proxy): Tangermann (4 cl., last session): rec +0.035, inter +0.030, retained 85%; Scherer 3-class (WORD/SUB/HAND, last session): rec +0.075, inter +0.044, retained 58%; Zhou (3 cl., last session): rec +0.018, inter +0.005, retained 27%; Zhou stream (calib session 0; test sessions 1+2): rec +0.035, inter +0.022, retained 65%. → **NOT order-robust** (fails on: Scherer 3-class (WORD/SUB/HAND, last session), Zhou (3 cl., last session), Zhou stream (calib session 0; test sessions 1+2)).
- **Session-boundary lag** (3a, online-64, recording order): second-session gain < 0 for 2/4 subjects (trigger ≥ 3); first-32 accuracy 0.680 vs session mean 0.647, i.e. 3.3 points above the mean (no drop) (trigger: drop > 5 points; clean on the same windows: +1.7 points vs its session mean). → **no session-boundary lag risk**.
- **Buffer-reset guard** (online-64 + guard vs online-64 on the stream, recording order; adopt if ≥ +0.020 with the CI excluding 0): -0.001 (-0.005, +0.003), 1/4 subjects. → **no gain (not adopted)** (evaluated because the order-robustness rule failed: the brief says the runbook then adds the guard; the lag rule itself did not trigger).
- Descriptive, not a decision (choosing N on test sessions would be tuning on test data): best buffer per order and its gain over clean — Tangermann (4 cl., last session) rec: N=64 (+0.035); Tangermann (4 cl., last session) inter: N=32 (+0.033); Tangermann (4 cl., last session) shuf: N=32 (+0.033); Scherer 3-class (WORD/SUB/HAND, last session) rec: N=64 (+0.075); Scherer 3-class (WORD/SUB/HAND, last session) inter: N=32 (+0.060); Scherer 3-class (WORD/SUB/HAND, last session) shuf: N=32 (+0.051); Zhou (3 cl., last session) rec: N=32 (+0.022); Zhou (3 cl., last session) inter: N=32 (+0.017); Zhou (3 cl., last session) shuf: N=64 (+0.016); Zhou stream (calib session 0; test sessions 1+2) rec: N=128 (+0.051); Zhou stream (calib session 0; test sessions 1+2) inter: N=32 (+0.025); Zhou stream (calib session 0; test sessions 1+2) shuf: N=32 (+0.018).

## Timings (4 threads, other agents sharing the CPU)

Weight CV: *choose_w* = the state was written by `sealed_personal.choose_w` itself (which also fits the per-subject Riemann models only its blend variant uses); *calib-only* = `calib_fold_scores` (the same blend_calib computation without them; `--verify_w` found identical CV scores and w on the three Zhou states).

| proxy | fit + weight CV clean (s) | fit + weight CV online (s) | weight CV | per-variant predict (s, median) | rows |
|---|---|---|---|---|---|
| Tangermann (4 cl., last session) | 301.3 | same fit | choose_w | 22.05 | 21 |
| Scherer 3-class (WORD/SUB/HAND, last session) | 393.2 | same fit | calib-only | 14.05 | 21 |
| Zhou (3 cl., last session) | 23.3 | 21.9 | choose_w | 2.6 | 21 |
| Zhou stream (calib session 0; test sessions 1+2) | 33.0 | same fit | choose_w | 4.6 | 31 |
