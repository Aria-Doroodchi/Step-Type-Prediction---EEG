# Track 2 — BCI decoding (cross-session)

The track we're entering. Task spec, the data we have, how to train, and
where things stand.

## The task

Decode a user's cued mental command from a short EEG window, reliably
**across recording sessions**.

| | Warm-up (now → Oct 25) | Sealed phase (Oct 28 → Nov 21), decides the ranking |
|---|---|---|
| Data | **Dreyer 2023** (public), Part B participants **61–81** = test set | private **2026 Graz + BrainHero** longitudinal cohort |
| Classes | **2** (left / right motor imagery) | **3**: kinesthetic MI, mental calculation, word association |
| Channels | 27 EEG, 512 Hz raw → delivered as **4-s windows of 27 × 480** (120 Hz) | **47** = 43 EEG + 2 EMG + 2 EOG at **500 Hz** |
| Split | cross-*subject* (unseen test participants) | cross-*session*: labelled early sessions of each participant for calibration; later sessions hidden |
| Metric | balanced accuracy, pooled over all windows | balanced accuracy **averaged over subject × session × context cells** |

The public Graz/BrainHero *training* data is "coming soon". When it's
released, warm-up switches to it (3 classes, cell-averaged metric). Watch
the Track 2 page. Until then Dreyer is the proxy.

Contract: `predict(X)` gets `(B, C, T)` and returns `(B,)` class indices.
See [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md).

## Data on disk (WSL `~/neuralbench/benchopt_data/neural_compet/`)

| benchopt study | Dataset | Role | Status |
|---|---|---|---|
| `tangermann2012` | BNCI2014_001: 4-class MI, 22 ch, 9 subjects | quick real-data smoke tests | ✅ prepared (1.2 GB) |
| `dreyer2023` | Dreyer2023Large: left/right-hand MI, 27 ch, 87 participants, 520 recordings | **warm-up proxy**: train on 1–60 + 82–87, test on 61–81 | ✅ prepared (≈21 GB + cache; subject 59 fixed by hand, see SETUP gotcha 4) |
| `stieger2021` | Stieger2021Continuous: 4-class MI, 62 subjects | published baseline (EEGNet 58.6 %, REVE 68.0 %); large pre-training pool | ⛔ **not downloaded: 399 GB**, C: has ~106 GB free (see below) |

**Stieger 2021 needs a bigger disk.** WSL's virtual disk lives on C:, which
has ~106 GB free after Dreyer. Dreyer (~21 GB on disk) is enough for warm-up work. To use Stieger
later, move the WSL distro to a drive with 450+ GB free (`wsl --export` /
`wsl --import`; G: has ~112 GB free, so it's too small too), or point
`BENCHOPT_DATA_HOME` in `env.sh` at a large drive. X:, Y: and Z: report
~14 TB free, probably network shares: check what they are and whether
lab data policy allows it first. Access over `/mnt/...` is slower but works.
Keep ≥ 30 GB free on C: for Windows itself.

(Re)download or resume any of them:
`bash ~/codabench/scripts/prepare_track2_data.sh [study ...]` (idempotent).

## Our solvers (`codabench/solvers/bci_decoding/`)

| Solver | From | What it is |
|---|---|---|
| `eegnet_steptype.py` → **EEGNet-StepType** | thesis `eegnet_torch.py` | Keras-faithful EEGNet, K-class, early stopping on held-out subjects |
| `riemann_steptype.py` → **Riemann-StepType** | thesis `riemannian.py` | xDAWN-TS + broadband-TS + log-var → shrinkage LDA |

Upstream baselines (in `2026-competition/tracks/bci_decoding/solvers/`):
`Constant`, `MeanLogReg`, `Torch-Linear`, `EEGNet` (braindecode).

## Commands (in WSL, after `source ~/codabench/env.sh`; run from `2026-competition/`)

```bash
# quick pipeline test on real data (minutes)
bash ~/codabench/scripts/test_track2_solvers.sh dreyer2023 40

# full training of one of our solvers on the warm-up proxy
benchopt run tracks/bci_decoding -d "BCI[study=dreyer2023]" \
    -s ../solvers/bci_decoding/eegnet_steptype.py -o "BCI-decoding[training=True]"

# hyperparameter sweep inline
benchopt run tracks/bci_decoding -d "BCI[study=dreyer2023]" \
    -s "../solvers/bci_decoding/riemann_steptype.py[use_xdawn=[True,False]]" \
    -o "BCI-decoding[training=True]"

# compare against the upstream EEGNet (delete its stale outputs folder first)
rm -rf tracks/bci_decoding/outputs/EEGNet
benchopt run tracks/bci_decoding -d "BCI[study=dreyer2023]" -s EEGNet -o "BCI-decoding[training=True]"
```

Results: `tracks/bci_decoding/outputs/benchopt_run_*.parquet` (columns
`solver_name`, `objective_balanced_accuracy`, ...). Trained, ready-to-zip
submissions: `tracks/bci_decoding/outputs/<SolverName>/`.

**Long runs:** full Dreyer training on CPU can take hours. Launch it with
`nohup ... > ~/codabench/logs/<name>.log 2>&1 &` and keep an Ubuntu window
open. Note the start time and check the log for the last epoch line.

## Test results (2026-09-23)

Pipeline checks, **not tuned results**. All on CPU.

**tangermann2012** (4-class, chance 0.25), full training set:

| Solver | Bal. acc. | Train+eval time |
|---|---|---|
| EEGNet (upstream braindecode) | **0.580** | 60 s (20 epochs) |
| Riemann-StepType (ours) | **0.536** | < 30 s |
| EEGNet-StepType (ours) | 0.454 | 39 s (early stop @ 15) |
| MeanLogReg (upstream floor) | 0.266 | — |

**dreyer2023** (2-class warm-up proxy, chance 0.50), the test batch
(`test_track2_solvers.sh dreyer2023 40`): our solvers trained on only the
first **40 batches = 2,560 windows** (EEGNet: 3 epochs), scored on the
**full** test split (participants 61–81). Whole run 47 s.

| Solver | Training data | Bal. acc. |
|---|---|---|
| Riemann-StepType (ours) | 2,560 windows | **0.716** |
| MeanLogReg (upstream floor) | full train split (no cap) | 0.681 |
| EEGNet-StepType (ours) | 2,560 windows, 3 epochs | 0.651 |

**Platform replay:** both exported submission folders were copied read-only
and re-run inference-only with `COMPET_SUBMISSION_DIR` (what Codabench
does). Both load and reproduce their scores exactly (Riemann's `joblib`
unpickles fine). The upload path works end to end locally.

⚠️ The folders now in `tracks/bci_decoding/outputs/{EEGNet,Riemann}-StepType/`
hold these **test-batch models**. Don't upload them; retrain on the full data
first (next steps).

## Next steps (suggested)

1. **Full Dreyer runs** of Riemann-StepType and EEGNet-StepType (no
   `max_batches`) to get real warm-up-proxy numbers. Riemann holds all
   training windows in memory as float64: 27 × 480 × 8 B ≈ 0.1 MB per window,
   so 100 k windows ≈ 10 GB against WSL's 31 GB. Check the window count first
   and watch RAM. EEGNet trained at ~2.7 s per epoch per 2,560 windows on
   this CPU; scale by the real window count to estimate. Then submit the
   better one to the warm-up leaderboard to validate the upload path end to end.
2. **Close the EEGNet gap** (ours 0.45 vs upstream 0.58 on tangermann). Likely
   causes to test: (a) held-out-subject early stopping fires too early on a
   9-subject dataset, so try `patience=[20]` or a larger `n_epochs`; (b) the
   kernel conversion via `sfreq`; (c) `standardize=True`.
3. **Riemann:** sweep `use_xdawn=[True,False]` (xDAWN targets ERPs, not MI).
   Add band-pass filter banks (mu 8–13 Hz, beta 13–30 Hz): the classic MI win.
4. **Cross-session drift** (sealed phase): per-subject re-centring / Euclidean
   alignment, which needs a way to group windows at predict time (only `X`
   is given). Design question for later.
5. Register for Track 2 on Codabench (3-step approval; **closes Oct 24**).
