# EEG/EMG Foundation Challenge 2026 — competition reference

Everything about the competition itself: dates, tracks, rules, limits and
baselines. For our setup see [SETUP.md](SETUP.md); for the track we're entering
see [TRACK2_BCI.md](TRACK2_BCI.md).

*Compiled 2026-09-23 from the official site, the competition repo and the
Codabench pages. Re-check the site for changes, especially the Track 2 data
release.*

## At a glance

| | |
|---|---|
| **Name** | EEG/EMG Foundation Challenge 2026, Brain & Body Workshop at NeurIPS 2026 |
| **Site** | <https://neural-interfaces26.github.io/> |
| **Platform** | Codabench: one competition per track, each needing its own registration |
| **Code** | <https://github.com/neural-interfaces26/2026-competition>, cloned here as `2026-competition/` |
| **Framework** | NeuralBench / NeuralSet 0.3.1 (Meta, `facebookresearch/neuroai`) + benchopt |
| **Prizes** | $20,000 pool plus internship opportunities; AWS (the compute partner) gives finalists free instance hours |

## Dates

| Date | Event |
|---|---|
| Sep 21, 2026 | Warm-up phase opens; leaderboard scores become the official comparison |
| **Oct 24, 2026** | **Registration closes** |
| Oct 25, 2026 | Warm-up ends |
| Oct 28 – Nov 21, 2026 | **Sealed final phase**, scored on private cohorts |
| Dec 11–12, 2026 | Winners' ceremony, NeurIPS Brain & Body Workshop (Sydney) |

## The four tracks

| # | Track | Input | `predict(X)` returns | Ranking metric |
|---|---|---|---|---|
| 1 | EEG-to-Image | one EEG epoch | `(B, 1536)` DINOv2-giant embedding | top-5 retrieval accuracy ↑ |
| **2** | **BCI decoding** | **47-ch window (43 EEG + 2 EMG + 2 EOG, 500 Hz)** | **`(B,)` class index** | **balanced accuracy ↑** |
| 3 | Sleep onset | 4-ch 128 Hz home EEG window | `(B,)` seconds to first N2 (≤ 600) | weighted binned MAE ↓ |
| 4 | EMG-to-Pose | 16-ch wrist sEMG window | `(B, 20, T)` joint angles in **radians** | mean angular MAE (°) ↓ |

Warm-up and sealed data differ for every track. The warm-up uses public data,
so leakage is possible and warm-up scores are **indicative only**. Only the
sealed phase decides the ranking.

## Rules that matter

- **Training compute is uncapped.** "Train on whatever you have." CPU-only
  training is allowed.
- **Inference limit:** the scoring container must finish a full test pass
  in **under 60 minutes on one A100 GPU**. Any download a submission does at
  load time counts toward that hour.
- **Submission = a trained model.** A ZIP holding `submission.py` plus weights,
  scored *inference-only*. The platform never trains. See
  [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md).
- **Single file, fixed dependencies.** `submission.py` holds *all* Python
  code; extra Python modules aren't supported and nothing is installed at
  evaluation. Only packages already in the scoring image can be imported (see
  [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md#what-submissionpy-may-import)).
- **Storage:** 15 GB of submission storage per Codabench profile, shared
  across all tracks. A submission shown on a leaderboard can't be deleted.
- **Data:** pre-training on *any publicly available, redistributable*
  dataset is allowed. **Never** the sealed test split, and **no closed
  clinical datasets**. Original dataset and model licences still apply.
- **Declare** every external corpus and a **compute estimate** in the method
  description that ships with the final submission. Finalists are audited:
  the organisers re-run the submitted training pipeline and declared
  configuration.
- **Cheating** (closed clinical data, touching the sealed split, modifying
  the scoring harness) means disqualification.
- **IP:** you keep it. Finalists grant audit-only access; open-sourcing
  afterwards is encouraged but optional.
- Submission caps, eligibility and team rules are on the site's **Rules &
  FAQ** page (not copied here; check it before the sealed phase).

## Published baselines (public development data, not warm-up scores)

Training cost is in GPU-hours. On this CPU-only machine expect several
times longer.

| Track | Dataset | Chance | EEGNet (0.04 M params) | REVE foundation model (14 M, probe) |
|---|---|---|---|---|
| 1 | THINGS-EEG2 | top-5 2.22 % | 28.13 % (2 GPU-h) | 84.75 % (0.5 GPU-h) |
| **2** | **Stieger 2021 (4-class MI)** | **24.81 %** | **58.58 % (4 GPU-h)** | **68.04 % (1 GPU-h)** |
| 3 | Sleep-EDF | bMAE 205.4 s | 143.3 s (4 GPU-h) | 134.9 s (1 GPU-h) |
| 4 | EMG2Pose | — | NeuroPose (paper): 17.5° | — |

Source: NeuralBench (Banville et al., 2026), Table 1, via the participant guide.

## Useful links

- Participant guide: <https://neural-interfaces26.github.io/participant-guide.html>
- Tracks & datasets: <https://neural-interfaces26.github.io/tracks.html>
- Rules: <https://neural-interfaces26.github.io/rules.html>
- Submission guide (in repo): `2026-competition/codabench/pages/participate.md`
- Track 2 NeuralBench start kit:
  <https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track2_eeg_to_bci.html>
- NeuralBench install docs: <https://facebookresearch.github.io/neuroai/neuralbench/install.html>
