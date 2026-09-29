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
    meta.json    sfreq, ch_names, ch_types, classes, subjects, sessions, counts,
                 config
    context.npy  (n,) str context per window, only when the study has one
                 (``load_context``; the sealed Graz / BrainHero contexts)

Rows are sorted by (subject, session, run, onset) = recording order, which is
the order the Track 2 test loader delivers windows in (shuffle=False).

meta["ch_types"] ("eeg" | "emg" | "eog" | "ecg" per channel) comes from the
submission's own name rule (riemann_sealed._channel_types: a name containing
EMG / EOG / ECG), so the harness's --chans and the solver's ``chans`` pick
the same channels. ``--picks eeg,emg,eog`` overrides the neuro extractor's
channel types (NeuralBench default: eeg only), e.g. for an EMG/EOG EDA cache
under a second study key.

Release structure (the sealed Graz + BrainHero data: a study with a context
column, or ``--sealed``): meta also gets hidden_split (split 2 rows = the
organisers' hidden test sessions, whose labels are not the true ones),
hidden_rows, eval_subjects (subjects with split 2 rows, else those with fewer
sessions than the most; ``--eval_subjects`` overrides), full_subjects and
calib_sessions (``--calib_sessions``, default 3). The harness keeps hidden
rows out of every training set (xsess_lib.xsess_split).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HOME = Path.home()
# XSESS_CACHE_ROOT (opt-in, e.g. a scratch folder for a test build)
CACHE_ROOT = Path(os.environ.get("XSESS_CACHE_ROOT") or HOME / "neuralbench/xsess_cache")
sys.path.insert(0, str(HOME / "codabench/2026-competition/tracks/bci_decoding"))
sys.path.insert(0, str(HOME / "codabench/solvers/bci_decoding"))

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


def release_structure(subj, session, split, eval_subjects=None, calib_sessions=3):
    """meta keys of the sealed release structure: hidden_split / hidden_rows
    (split 2 = the organisers' hidden test rows), eval_subjects (``eval_subjects``
    if given, else the subjects with split 2 rows, else the subjects with fewer
    sessions than the most; none of these -> no eval_subjects key),
    full_subjects (the rest) and calib_sessions."""
    subjects = np.unique(subj)
    hidden = split == 2
    ev = eval_subjects
    if ev is None and hidden.any():
        ev = np.unique(subj[hidden]).tolist()
    if ev is None:
        n_sess = np.array([len(np.unique(session[subj == s])) for s in subjects])
        if 0 < (n_sess < n_sess.max()).sum() < len(subjects):
            ev = subjects[n_sess < n_sess.max()].tolist()
    out = {"hidden_split": True, "hidden_rows": int(hidden.sum()),
           "calib_sessions": int(calib_sessions)}
    if ev is not None:
        ev = sorted(int(s) for s in ev)
        out.update(eval_subjects=ev,
                   full_subjects=[int(s) for s in subjects if int(s) not in ev])
    return out


def build(study, picks=None, sealed=False, eval_subjects=None, calib_sessions=3,
          hidden_labelled=False):
    """``picks``: neuro extractor channel types (e.g. ("eeg", "emg", "eog");
    None = the task's, NeuralBench's default eeg only). ``sealed`` (or a
    context column): record the release structure in meta (module doc);
    ``hidden_labelled``: its split 2 rows carry their true labels (meta
    hidden_labelled: the harness may then test on them)."""
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
    over = {"batch_size": 256, "seed": 33, "num_workers": 0,
            "pin_memory": False, "persistent_workers": False}
    if picks:
        over["neuro.picks"] = tuple(picks)
    cfg = _task_data_config(modality, task, overlay, data_dir, over)
    log(f"{study}: preparing {modality}/{task} overlay={overlay} "
        f"start={cfg.get('start')} duration={cfg.get('duration')}"
        + (f" picks={list(picks)}" if picks else ""))
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
    arrays = dict(X=X, y=y, subj=subj, session=session, run=run,
                  split=split, onset=onset, code=code)
    # The sealed metric averages over subject x session x *context* cells:
    # keep any context-like column the study provides (none in the proxies).
    for col in ("context", "condition", "paradigm"):
        if col in df.columns:
            arrays["context"] = df[col].astype(str).to_numpy()[order]
            meta["context_column"] = col
            break
    # channel types by the submission's own rule (the extractor keeps names only)
    import riemann_sealed
    meta["ch_types"] = riemann_sealed._channel_types(ch_names)
    meta["ch_types_source"] = "names"
    if picks:
        meta["picks"] = list(picks)
    if sealed or "context" in arrays:
        meta.update(release_structure(subj, session, split, eval_subjects, calib_sessions))
        if hidden_labelled:
            meta["hidden_labelled"] = True
        log(f"{study}: release structure: hidden (split 2) rows={meta['hidden_rows']} "
            f"eval_subjects={meta.get('eval_subjects')} full_subjects="
            f"{meta.get('full_subjects')} calib_sessions={meta['calib_sessions']}")
    for k, v in arrays.items():
        np.save(out / f"{k}.npy", v)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    types = {t: meta["ch_types"].count(t) for t in sorted(set(meta["ch_types"]))}
    log(f"{study}: X={X.shape} subjects={len(subjects)} "
        f"sessions/subject={[len(v) for v in sess_map.values()]} "
        f"classes={classes} counts={np.bincount(y).tolist()} ch_types={types} "
        f"context={meta.get('context_column')} -> {out}")


def load(study, mmap=False):
    """(X, y, subj, session, run, meta) from a built cache. ``mmap``: X is a
    read-only memory map (np.load mmap_mode="r"; the 500 Hz sealed data is
    ~5.4 GB), read on demand; the values are the same."""
    d = CACHE_ROOT / study
    meta = json.loads((d / "meta.json").read_text())
    arrs = [np.load(d / f"{k}.npy", allow_pickle=True,
                    mmap_mode="r" if (mmap and k == "X") else None)
            for k in ("X", "y", "subj", "session", "run")]
    return (*arrs, meta)


def load_context(study):
    """(n,) str context label per window (e.g. "graz" / "brainhero"), in the
    cache's row order, or None when the study has no context column (every
    proxy). The sealed metric averages over subject x session x context."""
    p = CACHE_ROOT / study / "context.npy"
    return np.load(p, allow_pickle=True).astype(str) if p.exists() else None


def load_split(study):
    """(n,) int NeuralBench split code per window (0 train, 1 val, 2 test; on
    the sealed release 2 = the organisers' hidden test rows), in the cache's
    row order, or None when the cache has no split.npy."""
    p = CACHE_ROOT / study / "split.npy"
    return np.load(p) if p.exists() else None


if __name__ == "__main__":
    # New study without editing STUDIES (e.g. the Graz + BrainHero release):
    #   python xsess_cache.py graz2026 --task eeg/<task> [--overlay <name>] [--sealed]
    #   EMG/EOG EDA cache under a second key:
    #   python xsess_cache.py graz2026_exg --task eeg/<task> --overlay <name> \
    #       --picks eeg,emg,eog --sealed
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("studies", nargs="*", default=["zhou2016"])
    ap.add_argument("--task", default=None, help="modality/task, e.g. eeg/motor_imagery")
    ap.add_argument("--overlay", default=None, help="dataset overlay in the task's datasets/")
    ap.add_argument("--picks", default=None,
                    help="neuro extractor channel types, e.g. eeg,emg,eog (default: the "
                         "task's, i.e. eeg)")
    ap.add_argument("--sealed", action="store_true",
                    help="record the release structure in meta (automatic with a context column)")
    ap.add_argument("--eval_subjects", default=None,
                    help="comma list of evaluation-participant indices (default: subjects "
                         "with split 2 rows)")
    ap.add_argument("--calib_sessions", type=int, default=3)
    ap.add_argument("--hidden_labelled", action="store_true",
                    help="the split 2 rows carry their true labels (the harness may test "
                         "on them)")
    a = ap.parse_args()
    ev = (None if a.eval_subjects is None
          else [int(s) for s in a.eval_subjects.split(",") if s != ""])
    for s in a.studies:
        if a.task is not None:
            mod, task = a.task.split("/")
            STUDIES[s] = (mod, task, a.overlay)
        build(s, picks=a.picks.split(",") if a.picks else None,
              sealed=a.sealed or ev is not None, eval_subjects=ev,
              calib_sessions=a.calib_sessions, hidden_labelled=a.hidden_labelled)
