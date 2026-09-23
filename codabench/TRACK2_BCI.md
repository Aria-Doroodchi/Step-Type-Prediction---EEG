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
| Channels | Dreyer montage (≈27 EEG) | **47** = 43 EEG + 2 EMG + 2 EOG at **500 Hz** |
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
| `dreyer2023` | Dreyer2023Large: 2-class MI, 87 participants | **warm-up proxy**: train on 1–60 + 82–87, test on 61–81 | see [LOG.md](LOG.md) |
| `stieger2021` | Stieger2021Continuous: 4-class MI, 62 subjects | published baseline (EEGNet 58.6 %, REVE 68.0 %); large pre-training pool | see [LOG.md](LOG.md) |

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
| Riemann-StepType (ours) | **0.536** | ~20 s |
| EEGNet-StepType (ours) | 0.454 | 39 s (early stop @ 15) |
| MeanLogReg (upstream floor) | 0.266 | — |

**dreyer2023** (2-class warm-up proxy, chance 0.50), 40-batch test batch: see
[LOG.md](LOG.md).

## Next steps (suggested)

1. **Full Dreyer runs** of Riemann-StepType (minutes) and EEGNet-StepType
   (likely hours on CPU) to get real warm-up-proxy numbers. Then submit the
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
