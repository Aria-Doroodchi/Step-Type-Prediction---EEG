"""Cross-session evaluation harness for the Track 2 sealed-phase proxies.

Plain numpy / torch on the caches built by ``xsess_cache.py``; no benchopt.
The sealed phase is within-subject, cross-session: every participant's early
sessions are labelled (calibration), later sessions are hidden, and the score
is balanced accuracy averaged over subject x session (x context) cells. The
proxies mimic that: for each subject the LAST session is the test session,
earlier ones are training data (the second-to-last doubles as a validation
session where a subject has >= 3 sessions, e.g. Zhou 2016). The sealed
structure itself (fully labelled training participants + evaluation
participants whose first K sessions are calibration and the rest test) is
``xsess_split(d, "replica:K")``.

Pieces:
- ``load_study`` / ``xsess_split`` / ``restrict_pool`` / ``select_channels``:
  arrays (+ context codes) and boolean masks; the organisers' hidden-test rows
  of a release cache (``hidden_rows``) never enter a training set, and channel
  types fall back to the submission's own name rule (``ch_types``);
- ``score``: cell-averaged balanced accuracy (subject x session x context;
  the proxies have no context, i.e. one context), pooled balanced accuracy,
  per-subject table;
- models with ``fit(X, y, ...)`` / ``predict_proba(X)``: MeanLogReg,
  Riemann-StepType (the solver's own feature code, imported), EEGNet-StepType
  (the solver's own network, imported; ES on a held-out split, then refit),
  upstream braindecode EEGNet (20 epochs, Adam 1e-3, batch 64);
- alignment: per-group whitening X <- R^(-1/2) X with R the Euclidean
  (He & Wu 2020 EA) or Riemannian mean of the group's window covariances,
  i.e. Riemannian re-centring applied at the signal level, so it propagates
  to every covariance block (broadband, filter bank, xDAWN) and to EEGNet.

Every shrinkage LDA (Riemann models, the subject router, sealed_personal's
per-subject LDAs) is ``riemann_steptype.fit_shrinkage_lda``, the solvers' own
solve rule: sklearn's fit up to 4000 features (every proxy, bit-identical),
a Cholesky solve above (43-47 ch filter-bank unions; 70-100 s -> 2-4 s per
fit, predict_proba within ~1e-10 of sklearn's). For 500 Hz data,
``load_study(mmap=True)`` memory-maps X, and the router PSD, the alignment
and the Riemann feature passes run per chunk of windows when X exceeds one
512 MiB float64 chunk (the proxies are one chunk: their code path is as
before).
"""

import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

HOME = Path.home()
sys.path.insert(0, str(HOME / "codabench/analysis"))
sys.path.insert(0, str(HOME / "codabench/2026-competition/tracks/bci_decoding"))
sys.path.insert(0, str(HOME / "codabench/solvers/bci_decoding"))

N_THREADS = int(os.environ.get("XS_THREADS", "10"))
torch.set_num_threads(N_THREADS)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_study(study, classes=None, mmap=False):
    """dict(X, y, subj, sess, run, ctx, ctx_names, has_ctx, split, meta).
    ``classes``: keep only these label indices (in the cache's numbering) and
    re-index them 0..K-1 in order. ``ctx``: int context code per window, an
    index into ``ctx_names`` (sorted names); a cache without context.npy (every
    proxy) gets ctx = 0 everywhere, ctx_names = ["none"], has_ctx = False, so
    its cells stay subject x session. ``split``: the cache's NeuralBench split
    code (zeros without split.npy); only ``hidden_rows`` reads it. ``mmap``: X
    stays a read-only memory map of the cache (500 Hz: no 5.4 GB read up
    front; ``select_channels`` then makes the only in-memory copy); same
    values."""
    from xsess_cache import load, load_context, load_split
    X, y, subj, sess, run, meta = load(study, mmap=mmap)
    ctx_raw = load_context(study)
    split = load_split(study)
    split = np.zeros(len(y), dtype=np.int64) if split is None else split.astype(np.int64)
    if ctx_raw is None:
        ctx, ctx_names = np.zeros(len(y), dtype=np.int64), ["none"]
    else:
        names, ctx = np.unique(ctx_raw, return_inverse=True)
        ctx, ctx_names = ctx.reshape(-1).astype(np.int64), [str(c) for c in names]
    if classes is not None:
        keep = np.isin(y, classes)
        X, y, subj, sess, run = X[keep], y[keep], subj[keep], sess[keep], run[keep]
        ctx, split = ctx[keep], split[keep]
        y = np.array([list(classes).index(v) for v in y])
        meta = dict(meta, classes={str(i): meta["classes"][str(c)]
                                   for i, c in enumerate(classes)})
    return dict(X=np.ascontiguousarray(X, dtype=np.float32), y=y.astype(np.int64),
                subj=subj, sess=sess, run=run, ctx=ctx, ctx_names=ctx_names,
                has_ctx=ctx_raw is not None, split=split, meta=meta)


def hidden_rows(d):
    """Boolean mask of the organisers' hidden-test rows: split == 2 of a
    release-structure cache (meta hidden_split; by default any cache whose
    meta names eval_subjects or a context column, i.e. the sealed data and
    the mock). On the released data their labels are placeholders, so no
    split may train on them. A proxy's split 2 is NeuralBench's default test
    split of labelled public data: never hidden (all False)."""
    meta, n = d["meta"], len(d["y"])
    release = meta.get("hidden_split", "eval_subjects" in meta or "context_column" in meta)
    if not release or "split" not in d:
        return np.zeros(n, bool)
    return np.asarray(d["split"]) == 2


def _drop_hidden(d, mode, sp):
    """``sp`` with the hidden-test rows (``hidden_rows``) removed from train /
    val. A test set holding hidden rows is refused unless meta marks them as
    labelled (hidden_labelled; default: the mock, whose labels are known).
    No hidden rows (every proxy): ``sp`` itself."""
    hid = hidden_rows(d)
    if not hid.any():
        return sp
    bad = sp["test"] & hid
    if bad.any() and not d["meta"].get("hidden_labelled", "mock" in d["meta"]):
        subs = np.unique(d["subj"][bad]).tolist()
        raise ValueError(
            f"split {mode!r}: the test set holds {int(bad.sum())} hidden-test rows "
            f"(cache split 2: subjects {subs}, sessions {np.unique(d['sess'][bad]).tolist()}) "
            "whose labels are not the true ones; validate on calib:K --test_subjects "
            "<the fully labelled participants> (if they are labelled: meta hidden_labelled, "
            "xsess_cache.py --hidden_labelled)")
    return dict(sp, train=sp["train"] & ~hid, val=sp["val"] & ~hid)


def xsess_split(d, mode="last", test_subjects=None):
    """Boolean masks train / test / val, plus the ``test_subjects`` array.

    mode="last" (default; the proxies): test = each subject's last session,
      train = the rest, val = second-to-last session where a subject has
      >= 3. ``test_subjects`` given: only those subjects are tested (the
      others are training data entirely).
    mode="calib:K": each subject in ``test_subjects`` (default: every
      subject) trains on its sessions < K (calibration) and is tested on its
      sessions >= K; every other subject is training data entirely.
    mode="replica:K": calib:K with ``test_subjects`` defaulting to
      meta["eval_subjects"] (the sealed structure: 10 fully labelled training
      participants + 10 evaluation participants with K = 3 calibration
      sessions). calib / replica have no val session (val all False).
      replica:K is refused on a cache whose meta has no "mock" key
      (2026-09-29): on the released data use calib:K with test_subjects =
      the fully labelled participants.
    Every mode keeps the organisers' hidden-test rows (``hidden_rows``: cache
    split 2 of a release-structure cache, e.g. the evaluation participants'
    sessions 3..5) out of train / val, and refuses a test set holding them
    unless they are labelled (the mock). On the released data the internal
    replica is therefore calib:K with test_subjects = the fully labelled
    participants; on the mock, replica:K tests exactly the hidden rows. The
    proxies have no hidden rows: their masks are unchanged."""
    subj, sess = d["subj"], d["sess"]
    name, _, k = mode.partition(":")
    if name == "last":
        last = np.array([sess[subj == s].max() for s in range(subj.max() + 1)])
        test = sess == last[subj]
        val = (sess == last[subj] - 1) & (last[subj] >= 2)
        if test_subjects is None:
            return _drop_hidden(d, mode, dict(train=~test, test=test, val=val,
                                              test_subjects=np.unique(subj)))
        ts = np.isin(subj, test_subjects)
        test, val = test & ts, val & ts
        return _drop_hidden(d, mode, dict(
            train=~test, test=test, val=val,
            test_subjects=np.unique(np.asarray(test_subjects, dtype=np.int64))))
    if name not in ("calib", "replica") or not k.isdigit():
        raise ValueError(f"split {mode!r}: expected last | calib:K | replica:K")
    if name == "replica" and "mock" not in d["meta"]:
        # replica:K = the organisers' own split; only the mock has true labels
        # on its hidden rows. On the released data it would test nothing (the
        # evaluation participants' sessions >= K absent) or score placeholder
        # labels (meta hidden_labelled), so it is refused whatever the meta says.
        full = d["meta"].get("full_subjects")
        raise ValueError(
            f"split {mode!r} runs only on a mock cache (meta has no \"mock\" key: "
            f"{d['meta'].get('study', '?')}): on the released data validate on "
            f"calib:{k} --test_subjects <the fully labelled participants> "
            f"(train_sealed.sh / release_ablations.sh: SPLIT=calib:{k} TEST_SUBJECTS=<full>"
            + (f"; meta full_subjects: {','.join(map(str, full))}" if full else "") + ")")
    if test_subjects is None:
        test_subjects = (np.unique(subj) if name == "calib"
                         else d["meta"].get("eval_subjects"))
        if test_subjects is None:
            raise ValueError(f"split {mode!r}: meta has no eval_subjects; "
                             "pass test_subjects")
    test_subjects = np.unique(np.asarray(test_subjects, dtype=np.int64))
    missing = set(test_subjects.tolist()) - set(np.unique(subj).tolist())
    if missing:
        raise ValueError(f"split {mode!r}: test subjects {sorted(missing)} not in the data")
    test = np.isin(subj, test_subjects) & (sess >= int(k))
    return _drop_hidden(d, mode, dict(train=~test, test=test, val=np.zeros(len(subj), bool),
                                      test_subjects=test_subjects))


def restrict_pool(d, sp, pool="all"):
    """pool="all": ``sp`` unchanged (train on every subject's training
    windows). pool="test": train only on the test subjects' own training
    windows (the "evaluation participants only" ablation)."""
    if pool == "all":
        return sp
    if pool != "test":
        raise ValueError(f"pool {pool!r}: expected all | test")
    return dict(sp, train=sp["train"] & np.isin(d["subj"], sp["test_subjects"]))


def split_info(d, sp):
    """Counts for the [data] line / result rows: n_train, n_test, test
    subjects and sessions, the number of test cells (subject x session x
    context) the metric will average over, and the hidden-test rows
    (``hidden_rows``) in train (must be 0) / kept out of train and test."""
    te = sp["test"]
    ctx = d.get("ctx", np.zeros(len(te), dtype=np.int64))
    cells = set(zip(d["subj"][te].tolist(), d["sess"][te].tolist(), ctx[te].tolist()))
    hid = hidden_rows(d)
    return dict(n_train=int(sp["train"].sum()), n_test=int(te.sum()),
                test_subjects=np.unique(d["subj"][te]).tolist(),
                test_sessions=np.unique(d["sess"][te]).tolist(),
                n_test_cells=len(cells),
                hidden_in_train=int((sp["train"] & hid).sum()),
                hidden_excluded=int((hid & ~sp["train"] & ~te).sum()))


CHAN_SETS = {"eeg": ("eeg",), "eeg+eog": ("eeg", "eog"), "eeg+emg": ("eeg", "emg"),
             "all": None}


def ch_types(meta):
    """Per-channel type ("eeg" | "emg" | "eog" | "ecg"): meta ch_types (the
    mock; caches built since 2026-09-28), else the submission's own name rule
    (riemann_sealed._channel_types: a name containing EMG / EOG / ECG), so the
    harness's --chans and the solver's ``chans`` keep the same channels.
    Every proxy channel is EEG under both."""
    if meta.get("ch_types"):
        return list(meta["ch_types"])
    import riemann_sealed
    return riemann_sealed._channel_types(list(meta["ch_names"]))


def ch_type_counts(meta):
    """'eeg:43,emg:2,eog:2(meta)': channel count per type and where the
    types come from (meta | names), for the [data] line."""
    t = ch_types(meta)
    src = meta.get("ch_types_source", "meta") if meta.get("ch_types") else "names"
    return ",".join(f"{k}:{t.count(k)}" for k in sorted(set(t))) + f"({src})"


def select_channels(d, drop=(), chans="eeg", chunk=1024):
    """Keep the channels whose type is in ``chans`` (eeg | eeg+eog | eeg+emg |
    all) and whose name is not in ``drop``; X, meta ch_names / ch_types follow
    (in place). Returns the dropped channel names (X untouched when none).
    The kept channels are copied into one new array ``chunk`` windows at a
    time (no whole-array temporary; with ``load_study(mmap=True)`` it is the
    only in-memory copy of X)."""
    if chans not in CHAN_SETS:
        raise ValueError(f"chans {chans!r}: expected one of {list(CHAN_SETS)}")
    meta, ok = d["meta"], CHAN_SETS[chans]
    names, types = meta["ch_names"], ch_types(meta)
    keep = [i for i, (c, t) in enumerate(zip(names, types))
            if c not in drop and (ok is None or t in ok)]
    dropped = [c for i, c in enumerate(names) if i not in set(keep)]
    if dropped:
        X = d["X"]
        out = np.empty((X.shape[0], len(keep)) + X.shape[2:], dtype=X.dtype)
        for i in range(0, len(X), chunk):
            out[i:i + chunk] = X[i:i + chunk][:, keep]
        d["X"] = out
        del X
        upd = dict(ch_names=[names[i] for i in keep])
        if "ch_types" in meta:
            upd["ch_types"] = [types[i] for i in keep]
        d["meta"] = dict(meta, **upd)
    return dropped


# ---------------------------------------------------------------------------
# metric
# ---------------------------------------------------------------------------
def score(y, yhat, subj, sess, ctx=None):
    """Cell-averaged balanced accuracy over (subject, session[, context])
    cells (the sealed metric; the proxies have no context), pooled balanced
    accuracy, and per-subject balanced accuracy (its cells averaged)."""
    import warnings
    ctx = np.zeros(len(y), int) if ctx is None else np.asarray(ctx)
    cells, per_subj = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in np.unique(subj):
            ms = subj == s
            cell_ids = sorted({(v, c) for v, c in zip(sess[ms], ctx[ms])})
            vals = [balanced_accuracy_score(y[ms & (sess == v) & (ctx == c)],
                                            yhat[ms & (sess == v) & (ctx == c)])
                    for v, c in cell_ids]
            cells += vals
            per_subj[int(s)] = float(np.mean(vals))
        pooled = balanced_accuracy_score(y, yhat)
    return dict(cell=float(np.mean(cells)), pooled=float(pooled),
                per_subject=per_subj, n_cells=len(cells))


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------
def window_covs(X, shrink=1e-3, chunk=1024):
    """(n, C, C) covariances X X^T / T, trace-scaled ridge for stability.
    Computed per chunk of windows (each window's covariance is independent of
    the others), so float64 memory stays bounded at 500 Hz x 47 channels."""
    out = []
    for i in range(0, len(X), chunk):
        Xc = np.asarray(X[i:i + chunk], dtype=np.float64)
        C = np.einsum("nct,ndt->ncd", Xc, Xc) / Xc.shape[-1]
        tr = np.trace(C, axis1=1, axis2=2)[:, None, None] / C.shape[1]
        out.append(C + shrink * tr * np.eye(C.shape[1])[None])
    return np.concatenate(out) if out else np.zeros((0, X.shape[1], X.shape[1]))


def mean_cov(covs, kind="euclid"):
    if kind == "euclid":
        return covs.mean(0)
    from pyriemann.utils.mean import mean_riemann
    return mean_riemann(covs, maxiter=50)


def inv_sqrtm(R):
    w, V = np.linalg.eigh((R + R.T) / 2)
    return (V * (1.0 / np.sqrt(np.maximum(w, 1e-12)))) @ V.T


def apply_W(X, W):
    return np.einsum("ij,njt->nit", W.astype(np.float32), X).astype(np.float32)


def align_groups(X, groups, kind="euclid", covs=None):
    """Whiten each group with its own reference: returns aligned X and the
    {group: W} dict."""
    covs = window_covs(X) if covs is None else covs
    Xa = np.empty_like(X)
    Ws = {}
    for g in np.unique(groups):
        m = groups == g
        Ws[g] = inv_sqrtm(mean_cov(covs[m], kind))
        Xa[m] = apply_W(X[m], Ws[g])
    return Xa, Ws


def align_rows(X, rows, groups, kind, covs, out, chunk=1024):
    """``align_groups(X[rows], groups, kind, covs[rows])`` written straight
    into ``out[rows]``, without the X[rows] / aligned copies (500 Hz memory).
    ``groups`` is aligned with ``rows``; ``covs`` is full-size (row-indexed as
    X). Each group's windows are whitened ``chunk`` at a time, so a group of
    <= chunk windows (every proxy subject / session) gets exactly
    align_groups' call. Returns {group: W}."""
    rows = np.asarray(rows)
    Ws = {}
    for g in np.unique(groups):
        idx = rows[groups == g]
        Ws[g] = inv_sqrtm(mean_cov(covs[idx], kind))
        for i in range(0, len(idx), chunk):
            c = idx[i:i + chunk]
            out[c] = apply_W(X[c], Ws[g])
    return Ws


# ---------------------------------------------------------------------------
# subject routing without ids (fingerprints)
# ---------------------------------------------------------------------------
FP_BANDS = {
    "ts": [None],                                            # broadband TS
    "fbts": [None, "1to4", "4to8", "8to13", "13to30", "30to45"],  # filter-bank TS
    "lv": [None, "1to4", "4to8", "8to13", "13to30", "30to45"],    # log-var per band
    "psd": [None],                    # log Welch PSD per channel, 1-Hz bins 1-45 Hz
    "psdf": [None],                   # same, 0.5-Hz bins (2-s Welch segments)
}


def _band_filter(X, sfreq, band):
    if band is None:
        return X
    import riemann_steptype as RS
    f = RS.WindowPreproc([], sfreq, bandpass=band)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            out.append(f(torch.from_numpy(X[i:i + 1024])).numpy())
    return np.concatenate(out)


class SubjectRouter:
    """Predict the training subject of an unlabelled window (no ids at test).

    fp = "ts" | "fbts" | "lv": features = tangent space of OAS covariances
    (per band for fbts) at the training mean, or per-band per-channel
    log-variance (lv); classifier = shrinkage LDA over training subjects.
    The fallback threshold on the max posterior is set on training data only:
    out-of-fold (leave-one-session-out when every subject has >= 2 training
    sessions, else 5-fold) max posteriors; t = their 1st percentile, i.e. a
    window less confident than 99 % of training windows is treated as an
    outlier and whitened with the global reference (the same semantics as the
    distance router's 99th-percentile distance). ``cap`` (opt-in, e.g. 0.5 =
    the solver's rule): thr = min(t, cap), since within-training OOF
    posteriors can saturate at 1.0 and then send most test windows to the
    fallback; ``thr_raw`` keeps t."""

    def __init__(self, fp, sfreq, cap=None):
        self.fp, self.sfreq = fp, float(sfreq)
        self.cap = None if cap is None else float(cap)

    def _feats(self, X, fit=False):
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        if "+" in self.fp:      # concatenation, e.g. "lv+fbts"
            if fit:
                self.parts = [SubjectRouter(p, self.sfreq) for p in self.fp.split("+")]
            return np.concatenate([p._feats(X, fit) for p in self.parts], 1)
        if self.fp in ("psd", "psdf"):
            # per chunk of windows when X exceeds one float64 chunk (500 Hz;
            # Welch's segment copies + FFT are ~3.5x X); one call otherwise
            import riemann_steptype as RS
            from scipy.signal import welch
            seg = int(self.sfreq) * (2 if self.fp == "psdf" else 1)
            out = []
            for s in RS._row_chunks(len(X), RS._chunk_len(X.shape)) or [slice(None)]:
                f, P = welch(X[s], fs=self.sfreq, nperseg=seg, axis=-1)
                m = (f >= 1) & (f <= 45)
                out.append(np.log(P[..., m] + 1e-12).reshape(len(P), -1))
            return out[0] if len(out) == 1 else np.concatenate(out)
        blocks = []
        if fit:
            self.ts = []
        for k, band in enumerate(FP_BANDS[self.fp]):
            Xb = _band_filter(X, self.sfreq, band).astype(np.float64)
            if self.fp == "lv":
                blocks.append(np.log(Xb.var(-1) + 1e-12))
                continue
            C = Covariances("oas").transform(Xb)
            if fit:
                self.ts.append(TangentSpace(metric="riemann").fit(C))
            blocks.append(self.ts[k].transform(C))
        return np.concatenate(blocks, 1)

    @staticmethod
    def _fit_lda(F, y):
        """Shrinkage LDA (class-count priors) with the solvers' solve rule
        (riemann_steptype.fit_shrinkage_lda: sklearn's own fit at <= 4000
        features, i.e. every log-PSD router up to 88 channels)."""
        import riemann_steptype as RS
        return RS.fit_shrinkage_lda(F, y)

    def fit(self, X, subj, sess):
        F = self._feats(X, fit=True)
        self.clf = self._fit_lda(F, subj)
        self.subjects = self.clf.classes_
        # OOF threshold (features reuse the full-train tangent point: a mild
        # optimism, only used to place the fallback threshold)
        from sklearn.model_selection import StratifiedKFold
        n_sess = min(len(np.unique(sess[subj == s])) for s in self.subjects)
        oof = np.zeros((len(subj), len(self.subjects)))
        if n_sess >= 2:
            folds = [(np.where(sess != v)[0], np.where(sess == v)[0])
                     for v in np.unique(sess)]
        else:
            folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(F, subj))
        for tr, va in folds:
            clf = self._fit_lda(F[tr], subj[tr])
            oof[np.ix_(va, np.searchsorted(self.subjects, clf.classes_))] = clf.predict_proba(F[va])
        mp, ok = oof.max(1), self.subjects[oof.argmax(1)] == subj
        self.thr_raw = float(np.percentile(mp, 1))
        self.thr = self.thr_raw if self.cap is None else min(self.thr_raw, self.cap)
        self.oof_acc = float(ok.mean())
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(self._feats(X))


# ---------------------------------------------------------------------------
# models: fit(X, y, ...) / predict_proba(X)
# ---------------------------------------------------------------------------
class MeanLogRegModel:
    """Upstream MeanLogReg: per-channel window mean -> StandardScaler + LR."""
    name = "MeanLogReg"

    def fit(self, X, y, **_):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        self.clf.fit(X.mean(-1), y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X.mean(-1))


class RiemannModel:
    """Riemann-StepType through the solver's own code (feature union ->
    shrinkage LDA, uniform priors). Blocks: xdawn / broad / logvar are the
    thesis union; fb = 4-band filter-bank tangent space; ``blocks`` restricts
    the union for ablations (e.g. ("fb",) or ("broad", "logvar"))."""
    name = "Riemann"

    def __init__(self, meta, use_xdawn=True, filterbank=False, reference="none",
                 blocks=None, estimator="oas"):
        import riemann_steptype as RS
        self.RS = RS
        self.meta = meta
        self.kw = dict(use_xdawn=use_xdawn, filterbank=filterbank)
        self.reference, self.blocks, self.estimator = reference, blocks, estimator

    def _feats(self, X):
        # one float64 call when X fits in one chunk (every proxy), else per
        # chunk of windows (500 Hz): the same per-window values
        blocks = self.RS.feature_blocks_chunked(self.model.parts,
                                                self.model._prep(X, np.float32))
        if self.blocks is not None:
            blocks = {k: v for k, v in blocks.items() if k in self.blocks}
        return np.concatenate(list(blocks.values()), axis=1)

    def fit(self, X, y, **_):
        RS = self.RS
        pre = RS.WindowPreproc(self.meta["ch_names"], self.meta["sfreq"], "none",
                               self.reference)
        self.model = RS.RiemannStepTypeModel(
            None, nfilter=4, estimator=self.estimator, preproc=pre, **self.kw)
        self.model.fit([(torch.from_numpy(X), torch.from_numpy(y), {})])
        if self.blocks is not None:   # refit the LDA on the selected blocks only
            k = len(np.unique(y))
            self.model.parts["lda"] = RS.fit_shrinkage_lda(self._feats(X), y,
                                                           np.full(k, 1 / k))
        return self

    def predict_proba(self, X):
        return self.model.parts["lda"].predict_proba(self._feats(X))

    # personalisation helpers (Phase 3)
    def features(self, X):
        return self._feats(X)


def _batches(n, bs, rng):
    idx = rng.permutation(n)
    return [idx[i:i + bs] for i in range(0, n, bs)]


class EEGNetSTModel:
    """EEGNet-StepType network (imported from the solver), K-class.

    Training protocol: up to ``n_epochs`` with early stopping (patience) on a
    held-out validation set -> best epoch E*; then, if ``refit``, re-train from
    scratch on train + val for E* epochs so the final model sees every
    training window. Same optimiser / max-norm constants as the solver."""
    name = "EEGNet-ST"

    def __init__(self, meta, seed=33, n_epochs=100, patience=20, lr=1e-3,
                 batch_size=64, refit=True, verbose=False, init_state=None):
        self.meta, self.seed = meta, seed
        self.n_epochs, self.patience, self.lr = n_epochs, patience, lr
        self.bs, self.refit, self.verbose = batch_size, refit, verbose
        self.init_state = init_state    # pretrained trunk weights (Phase 5)

    def _new_net(self, C, T, K):
        import eegnet_steptype as ES
        torch.manual_seed(self.seed)
        sf = float(self.meta["sfreq"])
        net = ES.EEGNetTorch(n_channels=C, n_times=T, n_classes=K,
                             kernel_length=round(0.5 * sf),
                             separable_kernel_length=round(0.125 * sf))
        if self.init_state is not None:   # everything but the classifier head
            trunk = {k: v for k, v in self.init_state.items()
                     if not k.startswith("classifier.")}
            missing, unexpected = net.load_state_dict(trunk, strict=False)
            assert all(k.startswith("classifier.") for k in missing), missing
        return net

    def _train(self, X, y, K, epochs, Xv=None, yv=None):
        import eegnet_steptype as ES
        net = self._new_net(X.shape[1], X.shape[2], K)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, eps=ES._KERAS_ADAM_EPSILON)
        rng = np.random.default_rng(self.seed)
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        best, best_ep, best_state, wait = math.inf, epochs, None, 0
        for ep in range(epochs):
            net.train()
            for b in _batches(len(y), self.bs, rng):
                if len(b) < 2:
                    continue
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(net(Xt[b]), yt[b])
                loss.backward()
                opt.step()
                net.apply_max_norm()
            if (ep + 1) % 10 == 0:      # heartbeat for the watchdog (log growth)
                log(f"    [EEGNet-ST] epoch {ep + 1}/{epochs} n={len(y)}"
                    + ("" if Xv is None else f" best_val={best:.4f}@{best_ep}"))
            if Xv is None:
                continue
            vl = self._loss(net, Xv, yv)
            if vl < best:
                best, best_ep, wait = vl, ep + 1, 0
                best_state = copy.deepcopy(net.state_dict())
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        return net, best_ep

    @torch.no_grad()
    def _loss(self, net, X, y):
        net.eval()
        out = torch.cat([net(torch.from_numpy(X[i:i + 512])) for i in range(0, len(X), 512)])
        return torch.nn.functional.cross_entropy(out, torch.from_numpy(y)).item()

    def fit(self, X, y, Xval=None, yval=None, **_):
        K = int(self.meta.get("n_classes", len(np.unique(y))))
        if Xval is None or len(Xval) == 0:     # fixed schedule, no ES
            self.net, self.best_epoch = self._train(X, y, K, self.n_epochs)
            return self
        net, ep = self._train(X, y, K, self.n_epochs, Xval, yval)
        self.best_epoch = ep
        if self.refit:
            Xa, ya = np.concatenate([X, Xval]), np.concatenate([y, yval])
            net, _ = self._train(Xa, ya, K, ep)
        self.net = net
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        self.net.eval()
        out = [torch.softmax(self.net(torch.from_numpy(X[i:i + 512])), 1)
               for i in range(0, len(X), 512)]
        return torch.cat(out).numpy()


class BraindecodeEEGNetModel:
    """Upstream baseline: braindecode EEGNet, Adam 1e-3, CE, 20 epochs,
    batch 64, no early stopping (tracks/bci_decoding/solvers/eegnet.py)."""
    name = "EEGNet-BD"

    def __init__(self, meta, seed=33, n_epochs=20, lr=1e-3, batch_size=64):
        self.meta, self.seed, self.n_epochs, self.lr, self.bs = meta, seed, n_epochs, lr, batch_size

    def fit(self, X, y, **_):
        from braindecode.models import EEGNet
        K = int(self.meta.get("n_classes", len(np.unique(y))))
        torch.manual_seed(self.seed)
        self.net = EEGNet(n_chans=X.shape[1], n_outputs=K, n_times=X.shape[2])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        rng = np.random.default_rng(self.seed)
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        self.net.train()
        for _ in range(self.n_epochs):
            for b in _batches(len(y), self.bs, rng):
                if len(b) < 2:
                    continue
                opt.zero_grad()
                torch.nn.functional.cross_entropy(self.net(Xt[b]), yt[b]).backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        self.net.eval()
        out = [torch.softmax(self.net(torch.from_numpy(X[i:i + 512])), 1)
               for i in range(0, len(X), 512)]
        return torch.cat(out).numpy()


def make_model(spec, meta, seed=33):
    """spec: 'meanlr' | 'riemann[:xd=1,fb=0]' | 'eegnet_st' | 'eegnet_bd'."""
    name, _, opts = spec.partition(":")
    kw = dict(kv.split("=") for kv in opts.split(",") if kv)
    if name == "meanlr":
        return MeanLogRegModel()
    if name == "riemann":
        blocks = tuple(kw["blocks"].split("+")) if "blocks" in kw else None
        return RiemannModel(meta, use_xdawn=kw.get("xd", "1") == "1",
                            filterbank=kw.get("fb", "0") == "1",
                            reference=kw.get("ref", "none"), blocks=blocks)
    if name == "eegnet_st":
        return EEGNetSTModel(meta, seed=seed, n_epochs=int(kw.get("ep", 100)),
                             patience=int(kw.get("pat", 20)))
    if name == "eegnet_bd":
        return BraindecodeEEGNetModel(meta, seed=seed, n_epochs=int(kw.get("ep", 20)))
    raise ValueError(spec)


def is_neural(spec):
    return spec.startswith("eegnet")


# ---------------------------------------------------------------------------
# validation sets for early stopping (never the test session)
# ---------------------------------------------------------------------------
def es_split(d, train_idx, mode, seed):
    """(fit_idx, val_idx) within the training windows for early stopping.

    A val *session* when the subjects have >= 3 sessions (Zhou 2016: session
    1). Otherwise pooled: 2 held-out training subjects; per-subject: a
    stratified 20 % of that subject's training windows. The model is then
    refit on fit + val for the best epoch count (EEGNetSTModel.refit)."""
    subj, sess, y = d["subj"][train_idx], d["sess"][train_idx], d["y"][train_idx]
    subjects = np.unique(subj)
    if min(len(np.unique(sess[subj == s])) for s in subjects) >= 2:
        last = np.zeros(subj.max() + 1, dtype=np.int64)
        for s in subjects:
            last[s] = sess[subj == s].max()
        val = sess == last[subj]
    elif mode == "pooled" and len(subjects) > 3:
        rng = np.random.default_rng(seed)
        held = rng.choice(np.unique(subj), 2, replace=False)
        val = np.isin(subj, held)
    else:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(1, test_size=0.2, random_state=seed)
        _, vi = next(sss.split(np.zeros(len(y)), y))
        val = np.zeros(len(y), bool)
        val[vi] = True
    return train_idx[~val], train_idx[val]


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
def append_result(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def read_results(path):
    """Rows of ``path`` plus its per-study siblings (``results_<study>.jsonl``
    next to it); malformed lines (e.g. an interleaved concurrent append) are
    skipped, so that config simply re-runs on resume."""
    path = Path(path)
    rows = []
    for p in [path] + sorted(path.parent.glob("results_*.jsonl")):
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
