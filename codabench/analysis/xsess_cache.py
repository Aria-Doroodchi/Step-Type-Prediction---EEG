#!/usr/bin/env python
"""Cross-session window caches for the Track 2 sealed-phase proxies.

Runs a study through the official NeuralBench task pipeline (same windows,
filters, resampling and robust scaling a Track 2 solver receives) and saves
*every* window with its subject / session / run metadata, so alignment and
pooling experiments can run on plain numpy arrays without benchopt.

    python ~/codabench/analysis/xsess_cache.py zhou2016 [tangermann2012 ...]

Output: ~/neuralbench/xsess_cache/<study>/
    X.npy        (n, C, T) float32 windows, as delivered to a solver
    y.npy        (n,) int class index (NeuralBench LabelEncoder = sorted codes)
    subj.npy     (n,) int subject index 0..S-1 (sorted original ids)
    session.npy  (n,) int session index within subject, chronological
    run.npy      (n,) int run index within session
    split.npy    (n,) int NeuralBench split of the task default (0 train, 1 val, 2 test)
    onset.npy    (n,) float trigger onset in its recording (s)
    code.npy     (n,) str raw event code (the label name)
    meta.json    sfreq, ch_names, classes, subjects, sessions, counts, config

Rows are sorted by (subject, session, run, onset) = recording order, which is
the order the Track 2 test loader delivers windows in (shuffle=False).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HOME = Path.home()
CACHE_ROOT = HOME / "neuralbench/xsess_cache"
sys.path.insert(0, str(HOME / "codabench/2026-competition/tracks/bci_decoding"))

# study key -> (modality, task, dataset overlay or None)
STUDIES = {
    "tangermann2012": ("eeg", "motor_imagery", "tangermann2012"),
    "zhou2016": ("eeg", "motor_imagery", "zhou2016"),
    "scherer2015": ("eeg", "mental_imagery", None),
    "zyma2019": ("eeg", "mental_arithmetic", None),
    "dreyer2023": ("eeg", "motor_imagery", "dreyer2023"),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _session_key(s):
    """Chronological sort key for MOABB session labels ('0', '1test', ...)."""
    s = str(s)
    digits = "".join(ch for ch in s if ch.isdigit())
    return (int(digits) if digits else 0, s)


def build(study):
    import torch
    from neuralbench.data import Data
    from benchmark_utils.nb_task import _quiet_neuro_logs, _task_data_config

    modality, task, overlay = STUDIES[study]
    out = CACHE_ROOT / study
    if (out / "meta.json").exists():
        log(f"{study}: cache exists at {out}, skipping")
        return
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(os.environ["BENCHOPT_DATA_HOME"]) / "neural_compet"
    _quiet_neuro_logs()
    cfg = _task_data_config(modality, task, overlay, data_dir, {
        "batch_size": 256, "seed": 33, "num_workers": 0,
        "pin_memory": False, "persistent_workers": False})
    log(f"{study}: preparing {modality}/{task} overlay={overlay} "
        f"start={cfg.get('start')} duration={cfg.get('duration')}")
    t0 = time.time()
    loaders = Data(**cfg).prepare()
    log(f"{study}: prepared in {time.time() - t0:.0f} s")

    Xs, ys, trig = [], [], []
    for k, name in enumerate(("train", "val", "test")):
        ds = loaders[name].dataset
        if len(ds) == 0:
            continue
        df = ds.triggers.copy()
        df["_split"] = k
        dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False,
                                         collate_fn=ds.collate_fn, num_workers=0)
        for b in dl:
            X = np.asarray(b.data["neuro"], dtype=np.float32)
            if X.ndim == 4:
                X = X[:, 0]
            Xs.append(X)
            ys.append(np.asarray(b.data["target"]).reshape(len(X), -1).argmax(-1))
        trig.append(df)
        log(f"  {name}: {len(ds)} windows")
    import pandas as pd
    df = pd.concat(trig, ignore_index=True)
    X, y = np.concatenate(Xs), np.concatenate(ys).astype(np.int64)
    assert len(df) == len(X), (len(df), len(X))

    # metadata columns (names differ slightly between studies)
    subj_raw = df["subject"].astype(str).to_numpy()
    sess_raw = (df["session"].astype(str).to_numpy() if "session" in df
                else np.full(len(df), "0"))
    run_raw = (df["run"].astype(str).to_numpy() if "run" in df
               else np.full(len(df), "0"))
    onset = df["start"].astype(float).to_numpy()
    code = df["code"].astype(str).to_numpy() if "code" in df else (
        df["task"].astype(str).to_numpy() if "task" in df else y.astype(str))

    # MOABB's "-100" sentinel (one end-of-run marker per run, e.g. Zhou 2016)
    # is not a class: drop those windows and re-index the remaining labels.
    keep = code != "-100"
    if not keep.all():
        log(f"{study}: dropping {int((~keep).sum())} windows with code -100")
        X, y, df = X[keep], y[keep], df[keep].reset_index(drop=True)
        subj_raw, sess_raw, run_raw = subj_raw[keep], sess_raw[keep], run_raw[keep]
        onset, code = onset[keep], code[keep]
        y = np.searchsorted(np.unique(y), y)

    subjects = sorted(set(subj_raw), key=lambda s: (len(s), s))
    subj = np.array([subjects.index(s) for s in subj_raw])
    session = np.zeros(len(df), dtype=np.int64)
    run = np.zeros(len(df), dtype=np.int64)
    sess_map = {}
    for si, s in enumerate(subjects):
        m = subj == si
        sess_sorted = sorted(set(sess_raw[m]), key=_session_key)
        sess_map[s] = sess_sorted
        for j, v in enumerate(sess_sorted):
            mm = m & (sess_raw == v)
            session[mm] = j
            runs = sorted(set(run_raw[mm]), key=_session_key)
            for r, rv in enumerate(runs):
                run[mm & (run_raw == rv)] = r

    order = np.lexsort((onset, run, session, subj))
    X, y, subj, session, run = X[order], y[order], subj[order], session[order], run[order]
    split, onset, code = df["_split"].to_numpy()[order], onset[order], code[order]

    # label index -> code name (LabelEncoder sorts names); verify consistency
    classes = {}
    for k in np.unique(y):
        names = sorted(set(code[y == k]))
        classes[int(k)] = names[0] if len(names) == 1 else names
    neuro = loaders["test"].dataset.extractors["neuro"]
    chans = getattr(neuro, "_channels", None) or {}
    ch_names = [n for n, _ in sorted(chans.items(), key=lambda kv: kv[1])]
    counts = {f"{subjects[s]}|{v}": int(((subj == s) & (session == v)).sum())
              for s in range(len(subjects)) for v in np.unique(session[subj == s])}
    meta = {
        "study": study, "modality": modality, "task": task, "overlay": overlay,
        "sfreq": float(neuro.frequency), "ch_names": ch_names,
        "shape": list(X.shape), "classes": classes,
        "class_counts": np.bincount(y).tolist(),
        "subjects": subjects, "sessions": sess_map,
        "windows_per_subject_session": counts,
        "window": {"start": cfg.get("start"), "duration": cfg.get("duration"),
                   "stride": cfg.get("stride")},
        "default_split_counts": np.bincount(split, minlength=3).tolist(),
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for k, v in dict(X=X, y=y, subj=subj, session=session, run=run,
                     split=split, onset=onset, code=code).items():
        np.save(out / f"{k}.npy", v)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"{study}: X={X.shape} subjects={len(subjects)} "
        f"sessions/subject={[len(v) for v in sess_map.values()]} "
        f"classes={classes} counts={np.bincount(y).tolist()} -> {out}")


def load(study):
    """(X, y, subj, session, run, meta) from a built cache."""
    d = CACHE_ROOT / study
    meta = json.loads((d / "meta.json").read_text())
    arrs = [np.load(d / f"{k}.npy", allow_pickle=True)
            for k in ("X", "y", "subj", "session", "run")]
    return (*arrs, meta)


if __name__ == "__main__":
    for s in sys.argv[1:] or ["zhou2016"]:
        build(s)
