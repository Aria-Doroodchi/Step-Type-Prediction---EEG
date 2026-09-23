# Submission guide — how a Codabench submission works

A condensed, standalone version of the official guide
(`2026-competition/codabench/pages/participate.md`, the authoritative source),
plus the gotchas found while setting this up.

## What gets uploaded

```text
my_submission.zip          <- every file at the ZIP ROOT, no subfolder
├── submission.py          <- required: ALL Python code (model + inference)
├── weights.pt             <- trained parameters, any name/format
└── config.json            <- optional non-Python artefacts
```

Codabench mounts the extracted files **read-only**, runs **inference only**
and scores the result. It never trains. In `load_model` / `predict` you must
not train, download competition data, or write into the submission
directory.

## The contract

```python
import torch
from benchmark_utils.base_solver import CompetSolver   # provided by the platform

class Solver(CompetSolver):
    name = "MyModel"

    def load_model(self, meta):            # REQUIRED
        model = MyModel(n_chans=meta["n_chans"], n_times=meta["n_times"],
                        n_classes=meta["n_classes"])
        w = meta["submission_dir"] / "weights.pt"
        if w.exists():
            model.load_state_dict(torch.load(w, map_location="cpu"))
        return model.to(meta["device"])    # must expose .predict(X)

    def fit(self, model, train_loader):    # optional: LOCAL training only
        ...

    def save_model(self, model, path):     # optional: writes weights for export
        torch.save(model.state_dict(), path / "weights.pt")
```

- **`meta`** is a plain dict: `sfreq, ch_names, chs_info, n_chans, n_times,
  n_classes` (classification) or `n_outputs` (regression), `device`,
  `submission_dir`.
- **`predict(X)`**: `X` is a torch tensor `(B, C, T)` already on
  `meta["device"]`. For **Track 2** it returns **`(B,)` class indices**.
- **`train_loader`** yields `(X, y, info)`, already on the device. `info` holds
  `subject_id` per window for the real NeuralBench datasets.
- The scoring server has a GPU (A100), so `meta["device"]` may be `cuda`
  there even though it's `cpu` here. Always `.to(meta["device"])`; never
  hard-code `"cpu"` or `"cuda"`.

## What `submission.py` may import

Only what the scoring image carries (`2026-competition/requirements.txt` +
`tools/Dockerfile`):

| Available | **Not** available |
|---|---|
| `torch`, `torchvision`, `torchaudio` | `xgboost`, `lightgbm`, `catboost` |
| `numpy`, `scipy`, `pandas`, `scikit-learn`, `joblib` | `tensorflow`, `keras`, `scikeras` |
| `mne`, `moabb`, `braindecode` | our repo's `eeg_steptype` package (copy code in) |
| `pyriemann` (comes in with `moabb`) | any other `pip install` |
| `neuralset`, `neuralbench`, `benchopt` | |

`pyriemann` is only there because `moabb` depends on it. Pickled objects
are version-sensitive, so re-check the image's versions before the sealed
phase.

**Pickling gotcha:** don't pickle instances of classes *defined inside
`submission.py`* (benchopt loads it as a dynamically named module, so the
class may not resolve on unpickle). Save plain tensors / arrays, or
`sklearn`/`pyriemann` objects, and rebuild your own classes in `load_model`.
`solvers/bci_decoding/riemann_steptype.py` shows the pattern.

**Parameters gotcha:** the exported `submission.py` is the solver file as
is, so on the platform it runs with the **defaults** in `parameters`. If you
trained with a non-default sweep value (e.g. `-s "file.py[lr=3e-4]"`), edit
the defaults to match before zipping. Otherwise the architecture may not match
the saved weights.

## Build → test → zip → upload

1. **Train locally** with the training variant (writes the submission folder):
   ```bash
   benchopt run tracks/bci_decoding -d "BCI[study=dreyer2023]" \
       -s ../solvers/bci_decoding/eegnet_steptype.py -o "BCI-decoding[training=True]"
   ```
   The result lands in `2026-competition/tracks/bci_decoding/outputs/<SolverName>/`
   (`submission.py` + weights).
2. **Re-test inference-only**, exactly as the platform does:
   ```bash
   COMPET_SUBMISSION_DIR="$PWD/tracks/bci_decoding/outputs/EEGNet-StepType" \
   benchopt run tracks/bci_decoding -d "BCI[study=dreyer2023]" \
       -s tracks/bci_decoding/outputs/EEGNet-StepType/submission.py
   ```
3. **Zip the folder's contents** (files at the root):
   ```bash
   cd tracks/bci_decoding/outputs/EEGNet-StepType && zip -r ~/codabench/submissions/eegnet_steptype_$(date +%F).zip . && cd -
   ```
4. **Upload** on the Track 2 Codabench page → *My Submissions*. Record it in
   [SUBMISSIONS.md](SUBMISSIONS.md).

## Common errors (from the official guide)

- `class Solver` missing, renamed or not importable
- the weight filename in `load_model` doesn't match the uploaded file
- importing a package that isn't in the image
- model or input on the wrong device
- `predict(X)` returns the wrong shape, type or unit
- trying to train, download data or write into the read-only folder
