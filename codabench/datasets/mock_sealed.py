"""MockSealed: a mock sealed study served through the Track 2 dataset contract.

The Graz + BrainHero sealed data is not released yet. ``analysis/mock_sealed.py``
writes caches with its structure (20 participants x 6 sessions x 4 runs,
43 EEG + 2 EMG + 2 EOG, 3 classes CALC / MI / WORD, Graz vs BrainHero
contexts) to ``~/neuralbench/xsess_cache/<study>/``; this dataset hands one
to a solver exactly like ``BCI[...]`` does through ``benchmark_utils.nb_task``,
so the release pipeline (solver fit, save, replay) runs end to end today.
Its accuracies mean nothing for the recipe.

- ``train_loader``: every window with ``split != 2`` (the 10 fully labelled
  participants' 6 sessions + the 10 evaluation participants' calibration
  sessions 0..2), shuffled; ``test_loader``: ``split == 2`` (evaluation
  participants' sessions 3..5) in recording order, unshuffled.
- Batches are ``(X, y, info)``: X float32 ``(B, C, T)``, y long ``(B,)``,
  info ``{"subject_id", "record_id", "onset"}`` (onset = window start in
  samples of its run). As on NeuralBench, the loaders' dataset exposes
  ``.seg_ds.triggers``: a DataFrame in dataset index order with columns
  subject, session, run, context, start (s), which is where
  ``Riemann-Sealed._collect`` reads session ids from.
- meta: n_classes, sfreq, ch_names, chs_info, n_chans, n_times, device
  (the ``load_task`` keys; channel types are not part of the contract, the
  names EMG1/EMG2/EOG1/EOG2 identify the non-EEG channels).

X is memory-mapped (``mock_sealed_500`` is ~5.4 GB), so each window is read
from the page cache on access, never held twice in RAM.

Build a cache first (inside WSL, after ``source ~/codabench/env.sh``):

    python ~/codabench/analysis/mock_sealed.py mock_sealed_s

then run from ~/codabench/2026-competition (benchopt 1.10 loads a dataset
straight from a file path; no copy into the track's datasets/ is needed):

    benchopt run tracks/bci_decoding \\
        -d "../datasets/mock_sealed.py[study=mock_sealed_s]" \\
        -s MeanLogReg -o "BCI-decoding[training=True]" --no-plot --no-html
"""

import json
from pathlib import Path

import numpy as np
import torch
from benchopt import BaseDataset
from torch.utils.data import DataLoader, default_collate

from benchmark_utils.data import chs_info_from_names, get_device

CACHE_ROOT = Path.home() / "neuralbench/xsess_cache"


class _Segments:
    """Stand-in for the neuralset ``SegmentDataset`` behind a NeuralBench
    loader: only the per-window ``triggers`` table (dataset index order)."""

    def __init__(self, triggers):
        self.triggers = triggers

    def __len__(self):
        return len(self.triggers)


class _MockWindows(torch.utils.data.Dataset):
    """Rows ``rows`` of a cache as ``(X, y, info)`` items (X read lazily)."""

    def __init__(self, X, y, rows, subj, record_id, onset, triggers):
        self.X, self.rows = X, rows
        self.y = torch.as_tensor(y[rows], dtype=torch.long)
        self.subj, self.record_id, self.onset = subj[rows], record_id[rows], onset[rows]
        self.seg_ds = _Segments(triggers)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        X = torch.from_numpy(np.array(self.X[self.rows[i]], dtype=np.float32))
        info = {"subject_id": int(self.subj[i]), "record_id": int(self.record_id[i]),
                "onset": int(self.onset[i])}
        return X, self.y[i], info


def _collate(device):
    """nb_task's collate: default collate, X / y moved onto ``device``."""
    def collate(batch):
        X, y, info = default_collate(batch)
        return X.to(device), y.to(device), info
    return collate


class Dataset(BaseDataset):

    name = "MockSealed"

    requirements = []

    parameters = {
        "study": ["mock_sealed_s"],
        "batch_size": [64],
    }

    test_parameters = {
        "study": ["mock_sealed_s"],
        "batch_size": [32],
    }

    def _dir(self):
        d = CACHE_ROOT / self.study
        if not (d / "meta.json").exists():
            raise RuntimeError(
                f"no mock cache at {d}: build it first with "
                f"`python ~/codabench/analysis/mock_sealed.py {self.study}`")
        return d

    def prepare(self):
        # Synthetic data: nothing to download; the cache is built by
        # analysis/mock_sealed.py. Fail early if it is missing.
        self._dir()

    def get_data(self):
        d = self._dir()
        meta = json.loads((d / "meta.json").read_text())
        X = np.load(d / "X.npy", mmap_mode="r")
        a = {k: np.load(d / f"{k}.npy") for k in
             ("y", "subj", "session", "run", "split", "onset", "context")}
        sfreq = float(meta["sfreq"])
        n_sess = int(a["session"].max()) + 1
        n_runs = int(a["run"].max()) + 1
        # one id per recording (subject x session x run), as record_id is
        # meant to trace a window back to its recording
        record_id = (a["subj"] * n_sess + a["session"]) * n_runs + a["run"]
        onset = np.round(a["onset"] * sfreq).astype(np.int64)
        device = get_device()
        try:
            seed = self.get_seed()
        except Exception:        # get_data called outside a benchopt run
            seed = 0

        def windows(rows):
            import pandas as pd
            trig = pd.DataFrame({
                "subject": a["subj"][rows], "session": a["session"][rows],
                "run": a["run"][rows], "context": a["context"][rows],
                "start": a["onset"][rows]})
            return _MockWindows(X, a["y"], rows, a["subj"], record_id, onset, trig)

        train = windows(np.flatnonzero(a["split"] != 2))
        test = windows(np.flatnonzero(a["split"] == 2))
        ch_names = list(meta["ch_names"])
        return dict(
            train_loader=DataLoader(
                train, batch_size=self.batch_size, shuffle=True,
                generator=torch.Generator().manual_seed(int(seed)),
                collate_fn=_collate(device)),
            test_loader=DataLoader(test, batch_size=self.batch_size,
                                   shuffle=False, collate_fn=_collate(device)),
            n_classes=len(meta["classes"]),
            sfreq=sfreq,
            ch_names=ch_names,
            chs_info=chs_info_from_names(ch_names),
            n_chans=int(X.shape[1]),
            n_times=int(X.shape[2]),
            device=device,
        )
