"""Riemann-Sealed — Riemann-StepType plus what the sealed phase needs: a
subject router and per-subject alignment inside ``predict(X)``, with no
subject or session id at prediction time.

Sealed phase (Track 2): the same participants across sessions; early sessions
are labelled calibration data, later ones are scored, and ``predict`` receives
bare ``(B, C, T)`` windows. Built from the cross-session proxy study
(codabench/SEALED_RECIPE.md, codabench/LOG.md 2026-09-25/26).

Training (``fit``; the train loader supplies ``info["subject_id"]``):
  1. per-window preprocessing (WindowPreproc, as Riemann-StepType);
  2. subject router: log-PSD (Welch, 1 s segments, 1-45 Hz, every channel)
     -> shrinkage LDA over training subjects. Fallback threshold = min(0.5, 1st
     percentile of 5-fold out-of-fold max posteriors on training windows);
  3. alignment (``align="subject"``): one whitening matrix per training
     subject, W_s = R_s^-1/2 with R_s the Euclidean (``kind="euclid"``, EA) or
     Riemannian mean of that subject's window covariances; plus a global W
     for fallback windows. Every training window is whitened with its own
     subject's W_s;
  4. Riemann-StepType feature union on the whitened windows (xDAWN,
     broadband TS, log-variance, optional filter bank / slow block) ->
     pooled shrinkage LDA (uniform priors);
  5. ``personal="calib"``: additionally one LDA per subject on the shared
     features (pooled extractor, own LDA); ``personal="blend"``: prediction =
     w * pooled + (1 - w) * per-subject LDA probabilities (``blend_w``).
Prediction: route each window (argmax posterior; below threshold -> global W
and pooled LDA), whiten with the routed subject's W, features, LDA(s).
``adapt="online"`` (RULE-DEPENDENT: uses unlabelled test windows; off by
default until the organisers confirm it is allowed): the routed subject's W
comes from the mean covariance of the last ``buffer`` test windows routed to
that subject, kept across predict() calls (training W until buffer/4 seen).
On the proxies this recovers the whole session-drift gain (+5.6 to +7.5
points per-subject); without it, per-subject whitening is ~invisible to a
tangent space at the subject's own mean (affine invariance).

Everything saved is numpy arrays and sklearn/pyriemann objects (joblib), so it
unpickles on the scoring worker.

Train locally (from ~/codabench/2026-competition, after ``source env.sh``):

    benchopt run tracks/bci_decoding -d "BCI[study=zhou2016_xsess]" \\
        -s ../solvers/bci_decoding/riemann_sealed.py \\
        -o "BCI-decoding[training=True]"
"""


import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pyriemann.estimation import Covariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy


# ---------------------------------------------------------------------------
# Per-window preprocessing (identical block in every thesis solver).
# Applied inside the model, so training and Codabench scoring see exactly the
# same transform. Stateless: nothing is fitted on data, so it is safe to run
# on test windows. Both steps are linear (they commute).
# ---------------------------------------------------------------------------
def _parse_band(bandpass):
    """"none" -> None; "8to30" -> (8.0, 30.0). (No "-": benchopt would parse
    "8-30" as arithmetic.)"""
    if bandpass in (None, "none"):
        return None
    lo, hi = (float(v) for v in str(bandpass).split("to"))
    if not 0 < lo < hi:
        raise ValueError(f"bad bandpass {bandpass!r}")
    return lo, hi


def _spatial_matrix(ch_names, reference, lambda2=1e-5, stiffness=4):
    """(C, C) matrix M so that re-referenced X = M @ X, or None.

    Only channels with a standard 10-05 position are re-referenced; any other
    channel (e.g. EMG/EOG in the sealed data) passes through unchanged.
    "laplacian" = MNE spherical-spline surface Laplacian (CSD) with the thesis
    pipeline's lambda2/stiffness; being linear, its matrix is read off by
    transforming an identity "recording". Rescaled to unit median gain.
    """
    if reference in (None, "none"):
        return None
    import mne
    montage = mne.channels.make_standard_montage("standard_1005")
    known = {c.lower() for c in montage.ch_names}
    n = len(ch_names)
    idx = [i for i, c in enumerate(ch_names) if c.lower() in known]
    if len(idx) < 3:
        idx = list(range(n))
    M = np.eye(n)
    if reference == "car":
        M[np.ix_(idx, idx)] -= 1.0 / len(idx)
        return M
    if reference == "laplacian":
        info = mne.create_info([ch_names[i] for i in idx], 100.0, "eeg")
        epochs = mne.EpochsArray(np.eye(len(idx))[None], info, verbose=False)
        epochs.set_montage(montage, match_case=False, verbose=False)
        csd = mne.preprocessing.compute_current_source_density(
            epochs, lambda2=lambda2, stiffness=stiffness, verbose=False)
        sub = csd.get_data()[0]
        M[np.ix_(idx, idx)] = sub / np.median(np.abs(np.diag(sub)))
        return M
    raise ValueError(f"unknown reference {reference!r}")


class WindowPreproc(nn.Module):
    """Re-reference then zero-phase FFT band-pass, per window, on any device.

    Band-pass: reflect-pad each window by T/2 on both sides, multiply the
    spectrum by a mask that is 1 on [lo, hi] with cosine skirts (min(2, lo/2)
    Hz below, 2 Hz above), inverse FFT, crop. No state, no fitting.
    """

    def __init__(self, ch_names, sfreq, bandpass="none", reference="none"):
        super().__init__()
        M = _spatial_matrix(list(ch_names), reference)
        self.register_buffer(
            "M", None if M is None else torch.as_tensor(M, dtype=torch.float32),
            persistent=False)   # rebuilt from ch_names in load_model
        self.band = _parse_band(bandpass)
        self.sfreq = float(sfreq)
        self._masks = {}

    def _mask(self, n, device):
        key = (n, str(device))
        if key not in self._masks:
            f = np.fft.rfftfreq(n, 1.0 / self.sfreq)
            lo, hi = self.band
            tw_lo, tw_hi = min(2.0, lo / 2), 2.0
            r_lo = np.clip((f - (lo - tw_lo)) / tw_lo, 0.0, 1.0)
            r_hi = np.clip(((hi + tw_hi) - f) / tw_hi, 0.0, 1.0)
            mask = (0.5 - 0.5 * np.cos(np.pi * r_lo)) * (0.5 - 0.5 * np.cos(np.pi * r_hi))
            self._masks[key] = torch.as_tensor(mask, dtype=torch.float32,
                                               device=device)
        return self._masks[key]

    def forward(self, x):
        if self.M is not None:
            x = torch.einsum("ij,bjt->bit", self.M.to(x.device, x.dtype), x)
        if self.band is not None:
            T = x.shape[-1]
            p = min(T // 2, T - 1)
            xp = F.pad(x, (p, p), mode="reflect")
            spec = torch.fft.rfft(xp, dim=-1) * self._mask(xp.shape[-1], x.device)
            x = torch.fft.irfft(spec, n=xp.shape[-1], dim=-1)[..., p:p + T]
        return x


FB_BANDS = ["4to8", "8to13", "13to30", "30to45"]   # filter-bank block, Hz


def _band_chunks(X, sfreq, band, chunk=2048):
    """Yield (n, C, T) float64 windows band-passed with WindowPreproc's FFT
    mask (no re-reference), in chunks to bound memory on the full train set."""
    filt = WindowPreproc([], sfreq, bandpass=band)
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            xb = torch.as_tensor(X[i:i + chunk], dtype=torch.float32)
            yield filt(xb).numpy().astype(np.float64)


def _slow_bins(parts, X):
    """< 4 Hz waveform, averaged in fixed time bins -> (n, C * n_bins)."""
    s = parts["slow"]
    out = []
    for xb in _band_chunks(X, parts["sfreq"], s["band"]):
        seg = xb[:, :, s["start"]:s["start"] + s["n_bins"] * s["width"]]
        out.append(seg.reshape(*seg.shape[:2], s["n_bins"], s["width"])
                   .mean(axis=-1).reshape(len(seg), -1))
    return np.concatenate(out)


def _band_covs(parts, X, band):
    cov = Covariances(estimator=parts["estimator"])
    return np.concatenate([cov.transform(xb)
                           for xb in _band_chunks(X, parts["sfreq"], band)])


def _feature_blocks(parts, X):
    """Named feature blocks for (n, C, T) float64 windows, in union order."""
    blocks = {}
    if parts.get("xdawn") is not None:
        blocks["xdawn"] = parts["xdawn_ts"].transform(parts["xdawn"].transform(X))
    covs = Covariances(estimator=parts["estimator"]).transform(X)
    blocks["broad"] = parts["broad_ts"].transform(covs)
    blocks["logvar"] = np.log(np.var(X, axis=2) + 1e-12)
    if parts.get("slow") is not None:
        blocks["slow"] = _slow_bins(parts, X)
    if parts.get("fb_ts") is not None:
        blocks["fb"] = np.concatenate(
            [ts.transform(_band_covs(parts, X, band))
             for band, ts in zip(parts["fb_bands"], parts["fb_ts"])], axis=1)
    return blocks


def _features(parts, X):
    """Feature union for (n, C, T) float64 windows -> (n, n_features)."""
    return np.concatenate(list(_feature_blocks(parts, X).values()), axis=1)




# ---------------------------------------------------------------------------
# Sealed-phase additions: router + per-subject whitening + personal LDAs.
# ---------------------------------------------------------------------------
def _window_covs(X, shrink=1e-3):
    C = np.einsum("nct,ndt->ncd", X, X) / X.shape[-1]
    tr = np.trace(C, axis1=1, axis2=2)[:, None, None] / C.shape[1]
    return C + shrink * tr * np.eye(C.shape[1])[None]


def _mean_cov(covs, kind):
    if kind == "euclid":
        return covs.mean(0)
    from pyriemann.utils.mean import mean_riemann
    return mean_riemann(covs, maxiter=50)


def _inv_sqrtm(R):
    w, V = np.linalg.eigh((R + R.T) / 2)
    return (V * (1.0 / np.sqrt(np.maximum(w, 1e-12)))) @ V.T


def _psd_features(X, sfreq):
    from scipy.signal import welch
    f, P = welch(X, fs=sfreq, nperseg=int(sfreq), axis=-1)
    m = (f >= 1) & (f <= 45)
    return np.log(P[..., m] + 1e-12).reshape(len(X), -1)


def _shrink_lda(priors=None):
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=priors)


class RiemannSealedModel:

    def __init__(self, parts=None, preproc=None, nfilter=4, estimator="oas",
                 use_xdawn=True, filterbank=True, slow_block=False,
                 align="subject", kind="riemann", personal="pooled",
                 blend_w=0.5, max_batches=None, adapt="none", buffer=64):
        self.parts, self.preproc = parts, preproc
        self.nfilter, self.estimator = nfilter, estimator
        self.use_xdawn, self.filterbank, self.slow_block = use_xdawn, filterbank, slow_block
        self.align, self.kind, self.personal = align, kind, personal
        self.blend_w, self.max_batches = float(blend_w), max_batches
        # adapt="online" (RULE-DEPENDENT, transductive): each routed subject's
        # whitening reference = mean covariance of the last `buffer` test
        # windows routed to it (training reference until buffer/4 are seen).
        # State persists across predict() calls, i.e. across test batches.
        self.adapt, self.buffer = adapt, int(buffer)
        self._buf = {}

    def _prep(self, X):
        X = torch.as_tensor(X, dtype=torch.float32).cpu()
        if self.preproc is not None:
            with torch.no_grad():
                X = self.preproc(X)
        return X.numpy().astype(np.float64)

    # -- routing / whitening -------------------------------------------------
    def _route(self, X):
        """Index into parts["subjects"] per window, -1 = fallback (outlier)."""
        r = self.parts["router"]
        P = r["lda"].predict_proba(_psd_features(X, self.parts["sfreq"]))
        idx = P.argmax(1)
        idx[P.max(1) < r["thr"]] = -1
        return idx

    def _whiten(self, X, idx, parts=None):
        parts = self.parts if parts is None else parts
        Ws, Wg = parts["W"], parts["W_global"]
        out = np.empty_like(X)
        for k in np.unique(idx):
            m = idx == k
            W = Wg if k < 0 else Ws[k]
            out[m] = np.einsum("ij,njt->nit", W, X[m])
        return out

    # -- training -------------------------------------------------------------
    def _collect(self, train_loader):
        """(X, y, subject, session-or-None) from the train loader.

        The competition loader gives ``info["subject_id"]`` only. NeuralBench's
        dataset underneath keeps a per-window trigger table with ``session``;
        when it is reachable (local training), read the dataset in index order
        so session ids line up with the windows. Otherwise iterate the loader
        (shuffled) and return session=None."""
        ds = getattr(train_loader, "dataset", None)
        seg = getattr(ds, "seg_ds", None)
        trig = None
        try:
            trig = seg.triggers if seg is not None else None
        except Exception:        # no triggers table: fall back to the loader
            trig = None
        if trig is not None and "session" in trig.columns and self.max_batches is None:
            import torch.utils.data as tud
            dl = tud.DataLoader(ds, batch_size=256, shuffle=False,
                                collate_fn=getattr(train_loader, "collate_fn", None))
            Xs, ys, ss = [], [], []
            for X, y, info in dl:
                Xs.append(self._prep(X))
                ys.append(to_numpy(y))
                ss.append(np.asarray(to_numpy(info["subject_id"])).reshape(-1))
            sess = trig["session"].astype(str).to_numpy()
            X, y, subj = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss)
            if len(sess) == len(y):
                return X, y, subj, sess
            return X, y, subj, None
        Xs, ys, ss = [], [], []
        for i, (X, y, info) in enumerate(train_loader):
            if self.max_batches is not None and i >= self.max_batches:
                break
            Xs.append(self._prep(X))
            ys.append(to_numpy(y))
            ss.append(np.asarray(to_numpy(info["subject_id"])).reshape(-1))
        return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss), None

    def fit(self, train_loader):
        X, y, subj, sess = self._collect(train_loader)
        subjects = np.unique(subj)
        sidx = np.searchsorted(subjects, subj)
        n_classes = len(np.unique(y))
        sfreq = self.preproc.sfreq
        parts = {"estimator": self.estimator, "xdawn": None, "sfreq": sfreq,
                 "subjects": subjects, "n_classes": n_classes}

        # router: log-PSD -> shrinkage LDA; OOF threshold (1st percentile)
        from sklearn.model_selection import StratifiedKFold
        Fp = _psd_features(X, sfreq)
        oof = np.zeros((len(y), len(subjects)))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(Fp, sidx):
            lda = _shrink_lda().fit(Fp[tr], sidx[tr])
            oof[np.ix_(va, lda.classes_)] = lda.predict_proba(Fp[va])
        parts["router"] = {"lda": _shrink_lda().fit(Fp, sidx),
                           # capped at 0.5: within-session OOF posteriors can saturate
                           # at 1.0 (Zhou: thr=1.000 -> every window fell back)
                           "thr": min(float(np.percentile(oof.max(1), 1)), 0.5)}

        # per-subject whitening (identity everywhere when align="none")
        if self.align == "subject":
            covs = _window_covs(X)
            parts["W_global"] = _inv_sqrtm(_mean_cov(covs, self.kind))
            parts["W"] = np.stack([_inv_sqrtm(_mean_cov(covs[sidx == k], self.kind))
                                   for k in range(len(subjects))])
        else:
            parts["W_global"] = np.eye(X.shape[1])
            parts["W"] = np.stack([np.eye(X.shape[1])] * len(subjects))
        if self.adapt == "online" and self.align == "subject" and sess is not None:
            # online test windows are centred on their own session's recent
            # statistics, so centre the training data per (subject, session)
            # too (the harness's online condition). Without session ids the
            # training data stays centred per subject, which cost 2.7 points
            # on Zhou (two training sessions) in the 2026-09-25 check.
            g = sidx * 1000 + np.unique(sess, return_inverse=True)[1]
            Xw = np.empty_like(X)
            for gg in np.unique(g):
                m = g == gg
                Xw[m] = np.einsum("ij,njt->nit",
                                  _inv_sqrtm(_mean_cov(covs[m], self.kind)), X[m])
            X = Xw
        else:
            X = self._whiten(X, sidx, parts)

        # Riemann-StepType feature union on the whitened windows
        if self.use_xdawn:
            parts["xdawn"] = XdawnCovariances(
                nfilter=self.nfilter, estimator=self.estimator,
                xdawn_estimator=self.estimator).fit(X, y)
            parts["xdawn_ts"] = TangentSpace(metric="riemann").fit(
                parts["xdawn"].transform(X), y)
        c = Covariances(estimator=self.estimator).transform(X)
        parts["broad_ts"] = TangentSpace(metric="riemann").fit(c, y)
        if self.slow_block:
            start, width = round(0.5 * sfreq), round(0.25 * sfreq)
            parts["slow"] = {"band": "0.1to4", "start": start, "width": width,
                             "n_bins": min(14, (X.shape[2] - start) // width)}
        if self.filterbank:
            parts["fb_bands"] = list(FB_BANDS)
            parts["fb_ts"] = [TangentSpace(metric="riemann").fit(_band_covs(parts, X, b), y)
                              for b in FB_BANDS]
        feats = _features(parts, X)
        priors = np.full(n_classes, 1.0 / n_classes)
        parts["lda"] = _shrink_lda(priors).fit(feats, y)
        if self.personal in ("calib", "blend"):
            parts["lda_subject"] = [
                _shrink_lda(priors).fit(feats[sidx == k], y[sidx == k])
                if len(np.unique(y[sidx == k])) == n_classes else None
                for k in range(len(subjects))]
        print(f"[Riemann-Sealed] fitting on X={X.shape}, subjects={len(subjects)}, "
              f"classes={np.unique(y).tolist()}, features={feats.shape[1]}, "
              f"align={self.align}/{self.kind}, personal={self.personal}, "
              f"router_thr={parts['router']['thr']:.3f}, "
              f"session_ids={'yes' if sess is not None else 'no'}", flush=True)
        self.parts = parts
        return self

    # -- prediction -----------------------------------------------------------
    def _online_whiten(self, X, idx):
        """Whiten with running per-subject test references (see __init__)."""
        covs = _window_covs(X)
        out = self._whiten(X, idx)            # fallback windows: global W
        for k in np.unique(idx[idx >= 0]):
            m = idx == k
            buf = self._buf.setdefault(int(k), [])
            buf.extend(list(covs[m]))
            del buf[:-self.buffer]
            if len(buf) >= self.buffer // 4:
                W = _inv_sqrtm(_mean_cov(np.stack(buf), self.kind))
                out[m] = np.einsum("ij,njt->nit", W, X[m])
        return out

    def predict_proba(self, X):
        X = self._prep(X)
        idx = self._route(X)
        Xw = (self._online_whiten(X, idx) if self.adapt == "online"
              else self._whiten(X, idx))
        F_ = _features(self.parts, Xw)
        P = self.parts["lda"].predict_proba(F_)
        if self.personal in ("calib", "blend"):
            for k in np.unique(idx[idx >= 0]):
                lda = self.parts["lda_subject"][k]
                if lda is None:
                    continue
                m = idx == k
                Ps = lda.predict_proba(F_[m])
                P[m] = Ps if self.personal == "calib" else (
                    self.blend_w * P[m] + (1 - self.blend_w) * Ps)
        return P

    def predict(self, X):
        return torch.as_tensor(self.predict_proba(X).argmax(1))


class Solver(CompetSolver):

    name = "Riemann-Sealed"

    parameters = {
        "nfilter": [4],
        "estimator": ["oas"],
        "use_xdawn": [True],
        "filterbank": [True],
        "slow_block": [False],
        "align": ["subject"],       # "subject" (router + per-subject W) | "none"
        "kind": ["riemann"],        # reference mean: "riemann" | "euclid"
        "personal": ["pooled"],     # "pooled" | "calib" | "blend"
        "blend_w": [0.5],           # pooled weight when personal="blend"
        # "online" = rule-dependent test-time re-centring (see RiemannSealedModel)
        "adapt": ["none"],
        "buffer": [64],
        "bandpass": ["none"],
        "reference": ["none"],
        "max_batches": [None],
    }

    def load_model(self, meta):
        weights = meta["submission_dir"] / "riemann_sealed.joblib"
        parts = (joblib.load(weights)
                 if weights.exists() and self.train_loader is None else None)
        preproc = WindowPreproc(meta["ch_names"], meta["sfreq"],
                                self.bandpass, self.reference)
        return RiemannSealedModel(parts, preproc, self.nfilter, self.estimator,
                                  self.use_xdawn, self.filterbank, self.slow_block,
                                  self.align, self.kind, self.personal,
                                  self.blend_w, self.max_batches,
                                  self.adapt, self.buffer)

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        joblib.dump(model.parts, path / "riemann_sealed.joblib")
