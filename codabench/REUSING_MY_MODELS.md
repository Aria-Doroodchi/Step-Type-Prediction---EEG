# Reusing the thesis models — compatibility audit

Which parts of this repo's EEG step-type pipeline (`src/eeg_steptype/`) can be
used in the competition, what it takes, and what's already been done.
Audited 2026-09-23 against the Track 2 contract.

## The constraints that decide it

1. **Input is a raw window** `(B, C, T)` as a torch tensor (Track 2: 47 channels
   at 500 Hz in the sealed phase; warm-up data differ). No epoch metadata, no
   hand-built feature table, no participant ID at prediction time.
2. **One self-contained `submission.py`.** Nothing from `eeg_steptype` can be
   imported, so code has to be copied in.
3. **Only the scoring image's packages:** `torch`, `numpy`, `scipy`,
   `sklearn`, `mne`, `pyriemann`, `braindecode`, ... **No `xgboost`, no
   `tensorflow`/`keras`/`scikeras`.**
4. **Multi-class:** 2 classes in warm-up (Dreyer), 3 in the sealed phase.
   The thesis models are binary (straight vs diagonal step).
5. Training happens locally (any code, any packages); only *inference* is
   constrained.

## Verdict per model

| Thesis model | File | Verdict | Status |
|---|---|---|---|
| **EEGNet (PyTorch port)** | `models/eegnet_torch.py` | ✅ **Compatible with small changes**: multi-class head, drop the tabular branch, inline into one file | **Done**: `solvers/bci_decoding/eegnet_steptype.py`, tested |
| **Riemannian** (xDAWN-TS + broadband-TS + log-var → shrinkage LDA) | `models/riemannian.py` | ✅ **Compatible with small changes**: LDA priors `[0.5, 0.5]` → uniform over K classes; save only pyriemann/sklearn objects | **Done**: `solvers/bci_decoding/riemann_steptype.py`, tested |
| Logistic / SVM | `models/logistic.py`, `svm.py` | ⚠️ The classifiers are fine (sklearn), but they need features from `features/*.py`, which work on MNE Epochs / DataFrames with the thesis's CNV channel sets and bins | Not ported. Easiest path: put them on top of the Riemann features (swap the LDA) |
| PSD / amplitude / slopes features | `features/psd.py`, `amplitude.py`, `slopes.py` | ⚠️ **Medium rework**: MNE/numpy only (allowed), but tied to epoch objects, thesis channel names and CNV time bins | Not ported |
| XGBoost | `models/xgb.py` | ❌ `xgboost` isn't in the scoring image | Could be replaced by sklearn `HistGradientBoostingClassifier` on ported features |
| EEGNet (Keras), CNN, EEGNeXt, BiLSTM | `models/eegnet.py`, `cnn.py`, `eegnext.py`, `lstm.py` | ❌ as is (TensorFlow/Keras not in image); ⚠️ **port to PyTorch** | EEGNeXt is the best next candidate: `eegnet_torch.py` shows the porting pattern |
| Shrinkage-LDA ERP benchmark | `features/cnv_benchmark.py` | ⚠️ CNV-specific bins | Superseded by Riemann-StepType's LDA |
| Nested CV driver, feature selection, pooling | `models/train.py`, `feature_selection.py`, `pooling.py` | ➖ Not needed: benchopt drives train/eval | Reuse ideas only (subject-level validation is built into EEGNet-StepType) |

## What changed in the two ported models

Full reasoning is in each file's docstring. In short:

**EEGNet-StepType** (`eegnet_steptype.py`), from `eegnet_torch.py` @ `48fc3fa`:
- K-class logits + cross-entropy (was a single logit + BCE); no tabular branch.
- Kernel lengths set in **seconds** (0.5 s / 0.125 s = the thesis's 64/16 samples
  at 128 Hz) and converted with `meta["sfreq"]`.
- Early stopping on **held-out subjects** (every 5th `subject_id`), because the
  loader reshuffles every epoch.
- Kept: layer stack, TF "same" padding, Keras init/BN/Adam constants, max-norm
  after every step, best-weight restore.
- Optional `standardize=True`: per-window z-scoring (off by default).

**Riemann-StepType** (`riemann_steptype.py`), from `riemannian.py`
(`make_riemannian`):
- LDA priors uniform over K classes.
- Feature union rebuilt as functions over fitted pyriemann/sklearn objects;
  only those are saved (`riemann.joblib`), so it unpickles on the scoring worker.
- `use_xdawn=False` drops the xDAWN block, which is ERP-oriented and may just
  add noise for motor imagery. Worth a sweep.

## Test results (2026-09-23)

See [TRACK2_BCI.md § Test results](TRACK2_BCI.md#test-results-2026-09-23).

## Ideas that carry over from the thesis

- **Participant-level thinking.** The sealed phase is *within-user,
  cross-session*: labelled early sessions of each evaluation participant are
  in the training data. Per-subject alignment (e.g. Riemannian re-centring per
  subject, Euclidean alignment) is the classic fix for session drift. Note that
  at `predict` time the model sees only `X`, with no subject ID.
- **Motor preparation.** The CNV and the readiness potential are
  movement-preparation signals; motor imagery lives in mu/beta band power
  (ERD/ERS). The thesis's PSD-band features are the natural bridge if ported.
