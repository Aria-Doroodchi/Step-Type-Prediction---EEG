# Reusing the thesis models — compatibility audit and strategy

Which parts of this repo's EEG step-type pipeline (`src/eeg_steptype/`) can be
used in the competition, what each one takes, what has been done, and whether
to keep adapting thesis code or build on the ecosystem the scoring image
ships. First audited 2026-09-23 against the Track 2 contract; sections 2, 4, 5
and 6 added 2026-09-24 after the Dreyer EDA and the preprocessing sweep.

## 1. The constraints that decide it

1. **Input is a raw window** `(B, C, T)` as a torch tensor. No epoch metadata,
   no hand-built feature table, **no participant or session id at prediction
   time**: the objective calls `model.predict(X)` and nothing else
   (`2026-competition/tracks/bci_decoding/objective.py`). `info["subject_id"]`
   exists only in the training loader.
2. **One self-contained `submission.py`.** Nothing from `eeg_steptype` can be
   imported, so code has to be copied in.
3. **Only the scoring image's packages:** `torch`, `numpy`, `scipy`, `sklearn`,
   `mne`, `pyriemann`, `braindecode`, ... **No `xgboost`, no
   `tensorflow`/`keras`/`scikeras`.**
4. **Multi-class:** 2 classes in warm-up (Dreyer), 3 in the sealed phase. The
   thesis models are binary (straight vs diagonal step).
5. Training happens locally (any code, any packages); only *inference* is
   constrained (60 min on one A100 for the full test pass, downloads included).

## 2. What a window looks like when a solver receives it

NeuralBench's `EegExtractor` runs before anything we control
(`neuralbench/defaults/config.yaml` + `tasks/eeg/motor_imagery/config.yaml`):

| Step | Setting | Consequence |
|---|---|---|
| Channel picks | `[eeg]` | EMG/EOG in the sealed data reach us only if the sealed task config picks them; unknown until it is published |
| Resample | 120 Hz | 4 s = 480 samples. Kernel lengths given in seconds (as in EEGNet-StepType) convert correctly |
| Filter | 0.1–75 Hz | the 75 Hz low-pass sits above the 60 Hz Nyquist and is dropped, so effectively a **0.1 Hz high-pass**; notch at 50 and 60 Hz |
| Baseline | none | slow drifts within the window survive; 78 % of Dreyer power is < 4 Hz |
| Scaling | `RobustScaler` per recording × channel, clamp ±20 | **absolute amplitude is gone**; only within-window shape, spectrum and spatial covariance carry information |
| Window | 4 s from the cue (`start 0.0`, `duration 4.0`) | one label per window, no pre-cue baseline |
| Test loader | `shuffle=False`, subject-level split | windows probably arrive subject-contiguous, but nothing guarantees it; do not rely on batch statistics |

So recording-level cleaning is impossible at inference. The only preprocessing
a solver can add is stateless and per-window: linear re-referencing, band-pass,
per-window scaling. That is exactly what `WindowPreproc` implements.

## 3. Verdict per thesis asset

### 3a. Applied — done and tested

| Asset | Where | Evidence | Note |
|---|---|---|---|
| **EEGNet (PyTorch port)** `models/eegnet_torch.py` | `solvers/bci_decoding/eegnet_steptype.py` | Dreyer 40-batch: 0.651, **0.711 with CAR**. tangermann (4-class, full data): **0.454 vs 0.580** for the stock braindecode EEGNet | trails the stock implementation on the one full-data comparison; the cause (held-out-subject early stopping on 9 subjects, kernel conversion, Keras constants) was never isolated |
| **Riemannian union → shrinkage LDA** `models/riemannian.py` | `solvers/bci_decoding/riemann_steptype.py` | Dreyer 40-batch: 0.716, **0.718 with CAR**, trained on 2,560 windows, beating the MeanLogReg floor (0.681) trained on all 12,392. tangermann: 0.536 | best solver so far. In the thesis it was the weakest model (AUC ~0.53): this family was built for motor-imagery covariance, not slow potentials |
| **CAR / CSD Laplacian referencing** `preprocessing/reference.py` | `WindowPreproc` in both solvers | CAR +0.02 to +0.04 on both models; Laplacian neutral to negative | thesis `lambda2`/`stiffness` reused; non-10-05 channels pass through |
| **Participant-level validation** | early stopping on held-out subjects | — | mirrors the thesis's participant-level CV and matches NeuralBench's subject-level validation split |
| **Working method** | `analysis/dreyer_eda.py`, `LOG.md`, `SUBMISSIONS.md` | — | EDA before modelling, ETA and time-log, paired comparisons, the overfit-gap guardrail. The most transferable asset of the thesis |

### 3b. Applicable with modification

| Asset | What it would take | Expected value | Verdict |
|---|---|---|---|
| **FBCSP log-variance placeholder → real filter-bank tangent space** (`FBCSPLogVariance` never filtered) | `WindowPreproc` already band-passes. Fit one `TangentSpace` per band (4–8, 8–13, 13–30, 30–45 Hz) plus broadband, concatenate, LDA or logistic. About one session | **High.** The field-standard MI method. The EDA gives mu power alone 0.645 and beta 0.609 cross-subject; combining bands in tangent space is the classic step up | **Do first** |
| **Logistic / SVM heads** `models/logistic.py`, `svm.py` | sklearn, drop-in on any feature block | Low on its own (LDA is already there); a C sweep on logistic is cheap | Cheap try |
| **Partial pooling** `models/pooling.py` | The sealed setup *is* partial pooling: train on everyone's sessions 1–3, test on the same people's sessions 4–6. The missing piece is per-subject adaptation at predict time without a subject id | **High for the sealed phase** | Design in § 6: subject fingerprint from the window covariance → per-subject Riemannian re-centring |
| **PSD band power** `features/psd.py` (Morlet, binned) | Rewrite as per-window Welch or filtered log-power in numpy (no MNE epochs, no thesis channel names) | Medium. Overlaps with the filter-bank tangent space; useful as an interpretable side model | Later, only if the filter bank stalls |
| **Binned amplitude / slopes** `features/amplitude.py`, `slopes.py` | numpy on windows: 0.25 s bins × C channels | Warm-up only. The slow (< 4 Hz) waveform is Dreyer's strongest cross-subject cue (0.769 with all 27 channels) but it is lateralised and cue-locked, likely eye or cue related. The sealed classes (MI, calculation, word association) are not a lateralised pair | Skip. MeanLogReg is already the one-bin version and scores 0.681 |
| **Hybrid tensor + tabular fusion** (CNN / EEGNet / EEGNeXt tabular branch) | The tabular branch would have to be computed from `X` inside the model (e.g. log band power) | Uncertain | Not now |
| **EEGNeXt, CNN** (Keras) `models/eegnext.py`, `cnn.py` | Port to PyTorch, ~250 lines each; `eegnet_torch.py` shows the pattern | **Low.** EEGNeXt was never run in the thesis, so there is no evidence to protect. braindecode 1.8.1 already ships EEGNeX, ATCNet, EEGConformer, EEGInceptionMI, ShallowFBCSPNet, tested and importable in the scoring image | **Do not port.** Use braindecode's |
| **XGBoost** `models/xgb.py` | Replace with sklearn `HistGradientBoostingClassifier` on ported features | Low. Tree models on hand features rarely beat tangent space + linear on MI | Skip |
| **3-class state module** `src/eeg_statetype/` | Multinomial logistic and macro-OVR evaluation pattern | Pattern only; both solvers already have K-class heads | Nothing to port. Its lesson matters: in the thesis, standing was easy for an uninteresting reason (gross movement). Expect the same trap with EMG/EOG in the sealed data |
| **eLORETA source features** `source_localization/` | A fixed inverse operator is a linear map (parcels × C) and could become a `WindowPreproc`-style matrix | Low. The thesis found `src` lower-value and dropped it from the rich set | Skip |

### 3c. Not feasible

| Asset | Why |
|---|---|
| Keras EEGNet, CNN, BiLSTM as they are | no TensorFlow, Keras or scikeras in the image |
| Preprocessing pipeline (ZapLine, PyPREP, ASR, ICA + ICLabel, AutoReject) | fitted on continuous recordings; `predict` sees isolated 4-s windows with no recording id. It could clean *training* windows only, and the loader already filters, notches and robust-scales them |
| Nested-CV driver, feature funnel, stability selection, halving search (`models/train.py`, `feature_selection.py`) | benchopt owns train and eval. And the regime is n ≫ p here (12k windows against hundreds of features); the p ≫ n problem the funnel was built for does not exist |
| Feature caches, epoch tensors, config system, CNV window logic | tied to the thesis data layout, trigger codes and channel names |
| The thesis's headline model (XGBoost on binned CNV features, 0.655–0.714 AUC) | inapplicable end to end. What transfers is the lesson, not the code: window and preprocessing choices moved AUC more than any model tuning |

## 4. Adapt the thesis code, or build on the ecosystem?

"From scratch" here does not mean writing new architectures. It means building
on what the scoring image ships: braindecode 1.8.1 (60+ models, including
EEGNet, EEGNeX, ATCNet, ShallowFBCSPNet, EEGConformer and the pretrained
encoders REVE, EEGPT, LaBraM, CBraMod), pyriemann 0.12, scikit-learn 1.9. The
organisers' own Track 2 baselines on Stieger (4-class): EEGNet 58.6 %, REVE
68.0 %.

| Criterion | Keep adapting thesis code | Build on the ecosystem |
|---|---|---|
| Effort | both ports are done (~600 lines); further ports (EEGNeXt, CNN, features) cost a session each | one braindecode solver is ~80 lines: the upstream `solvers/eegnet.py` is the template |
| Evidence | EEGNet-StepType 0.454 vs stock EEGNet 0.580 on tangermann. Riemann-StepType is the best thing we have | the stock model wins the only full-data comparison; REVE is +9.5 points over EEGNet in the organisers' table |
| Fit to the task | the thesis code was built for a slow evoked potential: xDAWN (an ERP method), 0.1 Hz content, long kernels, Keras constants tuned on 80-epoch subjects | MI and cognitive-task decoding is band power plus spatial covariance (filter-bank tangent space), and pretrained encoders trained on exactly this kind of data |
| Risk | the port's held-out-subject early stopping and Keras-faithful details are unvalidated on MI | pretrained weights must be shipped in the ZIP (a download at load time counts against the 60-minute budget) and their licence declared |
| CPU cost (this machine) | small models: minutes per full Dreyer run | REVE fine-tuning end to end is a GPU job. A frozen encoder plus a linear or Riemannian head is CPU-feasible: embed ~20k windows once |
| What is actually *ours* | nothing in the two ported models is thesis-specific in a way that helps | the harness, docs, EDA, ledgers and discipline, which stay either way |

**Verdict: hybrid, with a clear split.**

- **Keep and extend the classical line.** Riemann-StepType is the best solver,
  it is cheap, it is robust, and it is the one place where the thesis's code
  structure (feature union → shrinkage LDA) is genuinely reused. Next: filter
  bank, xDAWN off.
- **Stop investing in the deep-line port.** Do not port EEGNeXt or the CNN.
  Use braindecode models through the upstream template. Keep
  `eegnet_steptype.py` as a record and as a control.
- **Add one line the thesis has no analogue for:** a pretrained encoder
  (REVE) with a frozen probe.
- **Everything around the models stays:** `WindowPreproc`, the test scripts,
  the docs, the EDA, the ledgers.

**Caveat that outranks all of the above.** Warm-up is a proxy. The sealed
phase is 3 classes, 47 channels (43 EEG + 2 EMG + 2 EOG) recorded at 500 Hz
with an unknown delivery config, within-subject and cross-session (sessions
1–3 labelled, 4–6 hidden, 20 participants), scored per subject × session ×
context cell. Its training data is still "coming soon" (checked 2026-09-24).
Anything tuned to Dreyer's quirks (the slow lateralised waveform) is wasted;
anything that buys cross-session robustness is not.

## 5. Practical next steps

Ordered by value per hour on this CPU-only machine. Sizes are for the Dreyer
train split (12,392 windows; validation 3,360; test 5,040).

1. **EDA within-subject result (done 2026-09-24, `reports/dreyer_eda/summary.json`).**
   Median over 87 subjects, 5-fold within-subject: slow-waveform LDA 0.725,
   mu/beta Riemann 0.696, with 92 % and 85 % of subjects above chance
   (0.563). Cross-subject, mu power alone gave 0.645, so **per-subject
   adaptation is worth about +0.05 on the band-power features**. The two
   feature families are uncorrelated across subjects (r = −0.02): they carry
   independent information, so a combined feature union (step 3) should add,
   not duplicate.
2. **Full Dreyer runs, then the first upload.** Riemann-StepType with
   `reference=car` and `use_xdawn` on/off; the upstream EEGNet for 20 epochs.
   Riemann holds ~1.3 GB in RAM and fits in minutes; EEGNet ran at ~2.7 s per
   epoch per 2,560 windows, so ~13 s/epoch here: 20–50 epochs is 5–11 min.
   Zip the best, upload (5 per day in warm-up), record in `SUBMISSIONS.md`.
   Register on Codabench (closes Oct 24).
3. **Filter-bank tangent space** in Riemann-StepType. One session of code,
   15 min of runs. Expected to match or beat 0.72 on Dreyer and to be the
   robust choice for the sealed task.
4. **braindecode sweep** through the upstream template: EEGNet with early
   stopping on the validation split, EEGNeX, ATCNet, ShallowFBCSPNet. About
   10 min per full run. Keep the best two.
5. **Build the local cross-session proxy now.** Dreyer is cross-subject and
   cannot predict the sealed phase. Two MOABB datasets already in NeuralBench
   can: tangermann2012 has 2 sessions per subject (write a dataset overlay
   with `PredefinedSplit` on session, pattern in
   `neuralbench/tasks/eeg/motor_imagery/datasets/zhou2016.yaml`), and
   zhou2016 is 3-class, 3 sessions, 4 subjects, with the within-subject
   session hold-out already configured. Add both to
   `tracks/bci_decoding/datasets/bci_studies.py`. Small downloads.
6. **Per-subject alignment without a subject id** (§ 6), validated on the
   step-5 proxies.
7. **REVE frozen-encoder probe.** Check how braindecode loads REVE weights,
   whether they can be shipped in the ZIP, and the licence. Extract embeddings
   on CPU once, fit an LDA or logistic head, measure inference time.
8. **When the Graz/BrainHero training data is released:** parametrise
   `dreyer_eda.py` by study and rerun it; ablate the EMG/EOG channels; add a
   local cell-averaged metric (the objective pools windows); retrain the top
   two or three lines. One submission per day in the sealed phase.
9. **Housekeeping.** Commit `analysis/` and `reports/` (untracked), keep
   `LOG.md` and `SUBMISSIONS.md` current.

## 6. Open design questions for the sealed phase

- **Per-subject alignment.** Riemannian re-centring per subject is the
  standard fix for session drift, but `predict` gets no subject id. Proposal:
  at training time store one reference covariance (Riemannian mean) per
  training subject; at prediction, assign each window to the nearest training
  subject and re-centre with that reference, falling back to the global mean
  when the distance is large. Uses training statistics only, so it is not
  test-time adaptation. Batch-level statistics of the test loader would be
  cheaper but depend on batch composition and arguably use the test set;
  avoid.
- **EMG and EOG channels.** If delivered, they may separate mental
  calculation, word association and MI through eye and muscle activity: a
  real, allowed signal, but also the thesis's "standing is easy for an
  uninteresting reason" trap, and one likely to drift across sessions.
  Decide by ablation on the released training data, with cross-session
  validation.
- **Delivery format.** Whether the sealed loader keeps 120 Hz and EEG-only
  picks or delivers 47 channels at 500 Hz. Both solvers read `meta["sfreq"]`
  and `meta["n_chans"]`, so they adapt; runtime and memory estimates do not.
- **Pretrained weights and licences.** Any encoder weights must be in the
  ZIP (15 GB per profile across tracks) and declared with a compute estimate.

## 7. What changed in the two ported models

Full reasoning is in each file's docstring. In short:

**EEGNet-StepType** (`eegnet_steptype.py`), from `eegnet_torch.py` @ `48fc3fa`:
- K-class logits + cross-entropy (was a single logit + BCE); no tabular branch.
- Kernel lengths set in **seconds** (0.5 s / 0.125 s = the thesis's 64/16
  samples at 128 Hz) and converted with `meta["sfreq"]`.
- Early stopping on **held-out subjects** (every 5th `subject_id`), because
  the loader reshuffles every epoch.
- Kept: layer stack, TF "same" padding, Keras init/BN/Adam constants,
  max-norm after every step, best-weight restore.
- Optional `standardize=True`: per-window z-scoring (off by default).
- `bandpass` / `reference` parameters → `WindowPreproc` (2026-09-24).

**Riemann-StepType** (`riemann_steptype.py`), from `riemannian.py`
(`make_riemannian`):
- LDA priors uniform over K classes.
- Feature union rebuilt as functions over fitted pyriemann/sklearn objects;
  only those are saved (`riemann.joblib`), so it unpickles on the scoring
  worker.
- `use_xdawn=False` drops the xDAWN block, which is ERP-oriented and may just
  add noise for motor imagery.
- `bandpass` / `reference` parameters → `WindowPreproc` (2026-09-24).

## 8. Test results

See [TRACK2_BCI.md § Test results](TRACK2_BCI.md#test-results-2026-09-23)
and the per-window preprocessing sweep below it.
