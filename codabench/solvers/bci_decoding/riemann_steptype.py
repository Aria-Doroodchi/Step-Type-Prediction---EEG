"""Riemann-StepType — the thesis repo's Riemannian comparator as a Track 2
(BCI decoding) submission.

Ported from ``src/eeg_steptype/models/riemannian.py`` (``make_riemannian``):
the feature union of

  1. xDAWN covariances -> tangent space   (pyriemann XdawnCovariances + TS)
  2. broadband OAS covariance -> tangent space
  3. per-channel log-variance              (the thesis "FBCSP" placeholder)

plus two optional blocks (both off by default, so the default config is the
thesis model unchanged):

  4. ``slow_block``: < 4 Hz waveform. Each window is band-passed 0.1-4 Hz
     (WindowPreproc's FFT mask), then averaged in 0.25 s bins over 0.5-4.0 s
     (14 bins x C channels at 4 s windows). Targets the slow lateralised
     potential from the Dreyer EDA (LDA on it alone: 0.77 cross-subject);
  5. ``filterbank``: filter-bank tangent space. Band-pass 4-8, 8-13, 13-30,
     30-45 Hz, OAS covariance, one TangentSpace per band, concatenated.
     Targets mu/beta power (EDA: 0.64 / 0.61).

Both extra blocks filter on top of the solver's own ``bandpass``/``reference``
preprocessing; blocks 1-3 see exactly what they saw before.

All blocks are concatenated and piped into shrinkage LDA with uniform class
priors. Changes vs the thesis model, and why:

- LDA priors are uniform over ``n_classes`` (the thesis hard-codes
  ``[0.5, 0.5]`` for its binary step-type task; Track 2 is 2-class in
  warm-up, 3-class in the sealed phase);
- the feature blocks are plain functions over fitted pyriemann/sklearn
  objects, and only those objects are saved (joblib). A custom class defined
  in a dynamically loaded ``submission.py`` may not unpickle on the scoring
  worker; pyriemann/sklearn classes always do (both ship in the image —
  pyriemann comes in with moabb);
- the streamed train loader is materialised into memory once (covariance
  methods need all windows); ``max_batches`` caps it for quick tests;
- above 4000 features (43-47 ch with the filter bank) the shrinkage LDA
  solves with a Cholesky factorisation instead of sklearn's SVD lstsq
  (``fit_shrinkage_lda``: same coefficients, a plain sklearn object; ~10x
  faster); a training set larger than one 512 MiB float64 chunk (500 Hz) is
  kept as float32 and processed per chunk. Every proxy has <= 3175 features
  and fits in one chunk, so its numbers are unchanged. The analysis harness
  (analysis/xsess_lib.py) fits its Riemann models and LDAs through this file.

Train locally (from ~/codabench/2026-competition, after ``source env.sh``):

    benchopt run tracks/bci_decoding -d "BCI[study=dreyer2023]" \
        -s ../solvers/bci_decoding/riemann_steptype.py \
        -o "BCI-decoding[training=True]"
"""

import time

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
# Chunked passes for large training sets (43-47 ch x 500 Hz; the same helpers
# as riemann_sealed.py). Windows stay float32 (the preprocessing output) and
# each chunk is converted to float64 before any math. Data that fit in one
# chunk (every proxy: <= 0.45 GB as float64) run the original one-shot code.
# ---------------------------------------------------------------------------
_CHUNK_BYTES = 512 * 2 ** 20      # float64 bytes of windows per chunk


def _chunk_len(shape):
    """Windows per chunk for (n, C, T) data: all n when they fit in one."""
    n, per = int(shape[0]), 8 * int(np.prod(shape[1:]))
    return max(1, n if n * per <= _CHUNK_BYTES else _CHUNK_BYTES // per)


def _row_chunks(rows, size):
    """Chunks of <= size rows: slices for ``rows`` = a count n (all rows),
    else pieces of the index array ``rows``."""
    if isinstance(rows, (int, np.integer)):
        return [slice(i, min(i + size, rows)) for i in range(0, rows, size)]
    return [rows[i:i + size] for i in range(0, len(rows), size)]


def _oas_shrink(emp, n_samples):
    """sklearn's OAS shrinkage (covariance._shrunk_covariance._oas) applied to
    a precomputed centred empirical covariance of ``n_samples`` samples."""
    p = emp.shape[0]
    alpha = np.mean(emp ** 2)
    mu = np.trace(emp) / p
    num = alpha + mu ** 2
    den = (n_samples + 1) * (alpha - mu ** 2 / p)
    shrink = 1.0 if den == 0 else min(num / den, 1.0)
    out = (1.0 - shrink) * emp
    out.flat[::p + 1] += shrink * mu
    return out


def _fit_xdawn(nfilter, estimator, get, y, rows, size):
    """XdawnCovariances fitted on the windows ``get(rows)``: pyriemann's own fit
    when they are one chunk (bit-identical), else the same filters from chunked
    statistics: the signal covariance over all samples (Chan et al. pairwise
    merge of per-chunk means/scatters, then the OAS shrinkage) as
    ``baseline_cov``, and each class's mean response as its only "trial"."""
    kw = dict(nfilter=nfilter, estimator=estimator, xdawn_estimator=estimator)
    chunks = _row_chunks(rows, size)
    if len(chunks) == 1:
        return XdawnCovariances(**kw).fit(get(chunks[0]), y[chunks[0]])
    if estimator not in ("oas", "scm"):
        raise ValueError(f"chunked xDAWN supports estimator oas|scm, not {estimator!r}")
    classes = np.unique(y if isinstance(rows, (int, np.integer)) else y[rows])
    sums, counts = {c: 0.0 for c in classes}, {c: 0 for c in classes}
    n_tot, mean, M2 = 0, 0.0, 0.0
    for s in chunks:
        xb, yb = get(s), y[s]
        for c in classes:
            sums[c] = sums[c] + xb[yb == c].sum(axis=0)
            counts[c] += int(np.sum(yb == c))
        nb = xb.shape[0] * xb.shape[2]
        mb = xb.mean(axis=(0, 2)) if estimator == "oas" else np.zeros(xb.shape[1])
        xc = xb - mb[None, :, None] if estimator == "oas" else xb
        Sb = np.tensordot(xc, xc, axes=([0, 2], [0, 2]))
        d, tot = mb - mean, n_tot + nb
        M2 = M2 + Sb + np.outer(d, d) * (n_tot * nb / tot)
        mean, n_tot = mean + d * (nb / tot), tot
    Cx = M2 / n_tot
    if estimator == "oas":
        Cx = _oas_shrink(Cx, n_tot)
    P = np.stack([sums[c] / counts[c] for c in classes])
    return XdawnCovariances(**kw, baseline_cov=Cx).fit(P, classes)


def _window_blocks(parts, get, rows, size):
    """One chunked pass over the windows ``get(chunk)`` (float64): the
    per-window input of every feature block before its tangent space (xDAWN,
    broadband and filter-bank covariances; log-variance; slow bins), as in
    ``_feature_blocks``."""
    acc = {}
    for s in _row_chunks(rows, size):
        xb = get(s)
        out = {}
        if parts.get("xdawn") is not None:
            out["xdawn"] = parts["xdawn"].transform(xb)
        out["broad"] = Covariances(estimator=parts["estimator"]).transform(xb)
        out["logvar"] = np.log(np.var(xb, axis=2) + 1e-12)
        if parts.get("slow") is not None:
            out["slow"] = _slow_bins(parts, xb)
        for band in parts.get("fb_bands") or []:
            out["fb:" + band] = _band_covs(parts, xb, band)
        for k, v in out.items():
            acc.setdefault(k, []).append(v)
    return {k: v[0] if len(v) == 1 else np.concatenate(v) for k, v in acc.items()}


def _fit_tangent(parts, raw, y):
    """Tangent spaces of the covariance blocks of ``_window_blocks`` output."""
    def ts(covs):
        return TangentSpace(metric="riemann").fit(covs, y)
    if parts.get("xdawn") is not None:
        parts["xdawn_ts"] = ts(raw["xdawn"])
    parts["broad_ts"] = ts(raw["broad"])
    if parts.get("fb_bands"):
        parts["fb_ts"] = [ts(raw["fb:" + b]) for b in parts["fb_bands"]]


def _raw_blocks(parts, raw):
    """Named feature blocks from ``_window_blocks`` output (the order and
    values of ``_feature_blocks``)."""
    blocks = {}
    if parts.get("xdawn") is not None:
        blocks["xdawn"] = parts["xdawn_ts"].transform(raw["xdawn"])
    blocks["broad"] = parts["broad_ts"].transform(raw["broad"])
    blocks["logvar"] = raw["logvar"]
    if parts.get("slow") is not None:
        blocks["slow"] = raw["slow"]
    if parts.get("fb_ts") is not None:
        blocks["fb"] = np.concatenate(
            [ts.transform(raw["fb:" + b]) for b, ts in zip(parts["fb_bands"], parts["fb_ts"])],
            axis=1)
    return blocks


def feature_blocks_chunked(parts, X):
    """``_feature_blocks`` for (n, C, T) windows of any float dtype: one call
    on the float64 windows when they fit in one chunk (exactly the original
    code), else per chunk of windows (every block is per-window) and
    concatenated, so float64 memory stays bounded."""
    n, size = len(X), _chunk_len(X.shape)
    if size >= n:
        return _feature_blocks(parts, np.asarray(X, dtype=np.float64))
    acc = {}
    for s in _row_chunks(n, size):
        for k, v in _feature_blocks(parts, np.asarray(X[s], dtype=np.float64)).items():
            acc.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in acc.items()}


# ---------------------------------------------------------------------------
# Shrinkage LDA with a fast solve (identical block in riemann_sealed.py).
# sklearn's LinearDiscriminantAnalysis(solver="lsqr") solves
# covariance_ @ coef_.T = means_.T with scipy's SVD lstsq, O(p^3): 70-100 s
# per fit at p = 5073 features (fb=1 union at 43 ch) on 4 threads. The shrunk
# covariance is SPD, so a Cholesky solve gives the same coefficients (to
# ~1e-14 relative) in ~3 s. Above LDA_FAST_P features the fit is sklearn's own
# fit with only that solve replaced; every proxy has p <= 3175 (3595 with
# slow_block) and keeps sklearn's lstsq bit-for-bit.
# Above LDA_FAST_P, a fit with fewer rows than features (n < p: every
# per-subject LDA, and the pooled one once p exceeds the training windows)
# never forms the p x p matrix (LDA_DUAL, sprint 2026-10-01). sklearn's
# Ledoit-Wolf estimate shrinks each class towards mu * I in its standardised
# space, so in the original space it is diag(lam) + U.T diag(c) U with U the
# n x p centred class rows: _DualLsqrLDA solves that in the n x n (Woodbury)
# form. Same coefficients to ~1e-12 relative (analysis/lda_dual_check.py); 20
# per-subject LDAs at p = 9373, n = 420 took 200 s with the Cholesky solve.
# ---------------------------------------------------------------------------
LDA_FAST_P = 4000
LDA_MAX_RESID = 1e-8      # relative residual above which the solve falls back
LDA_DUAL = True           # n < p above LDA_FAST_P: the n x n form (_DualLsqrLDA)
# diagnostics: Cholesky solves / lstsq fallbacks / dual solves / dual -> Cholesky
LDA_STATS = {"fast": 0, "fallback": 0, "dual": 0, "dual_fallback": 0}

try:
    from sklearn.discriminant_analysis import _class_cov, _class_means
except ImportError:       # pragma: no cover (a future sklearn): plain fits only
    _class_cov = _class_means = None


class _CholeskyLsqrLDA(LinearDiscriminantAnalysis):
    """LinearDiscriminantAnalysis whose lsqr step is a Cholesky solve. Used
    only inside ``fit_shrinkage_lda``, which turns the fitted object back into
    a plain LinearDiscriminantAnalysis: nothing pickled refers to this class."""

    def _solve_lstsq(self, X, y, shrinkage, covariance_estimator):
        # sklearn 1.9 LinearDiscriminantAnalysis._solve_lstsq, except the solve
        import warnings
        from scipy import linalg
        self.means_ = _class_means(X, y)
        self.covariance_ = _class_cov(X, y, self.priors_, shrinkage, covariance_estimator)
        B = self.means_.T
        coef = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", linalg.LinAlgWarning)   # ill-conditioned
                coef = linalg.solve(self.covariance_, B, assume_a="pos")
            resid = np.linalg.norm(self.covariance_ @ coef - B) / np.linalg.norm(B)
            if not resid <= LDA_MAX_RESID:          # also catches NaN
                coef = None
        except (linalg.LinAlgError, linalg.LinAlgWarning, ValueError):
            coef = None
        if coef is None:                            # sklearn's own solve
            LDA_STATS["fallback"] += 1
            coef = linalg.lstsq(self.covariance_, B)[0]
        else:
            LDA_STATS["fast"] += 1
        self.coef_ = coef.T
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )


def _lw_lowrank(X, y, priors):
    """sklearn's _class_cov(X, y, priors, shrinkage="auto") (per class:
    StandardScaler, LedoitWolf, rescale; prior-weighted sum) as (lam, U, c),
    the estimate being diag(lam) + U.T @ diag(c) @ U with U (n x p) the
    centred class rows in the original scale. Each class's Ledoit-Wolf
    shrinkage is ledoit_wolf_shrinkage's formula on the n_g x n_g Gram matrix
    (sum of <Z.T, Z>**2 = sum of <Z, Z.T>**2; sum of <Z2.T, Z2> = sum of the
    squared row norms)."""
    from sklearn.preprocessing import StandardScaler
    p = X.shape[1]
    lam = np.zeros(p)
    Us, cs = [], []
    for idx, group in enumerate(np.unique(y)):
        Xg = np.asarray(X[y == group, :], dtype=np.float64)
        n = Xg.shape[0]
        sc = StandardScaler().fit(Xg)
        Z = sc.transform(Xg)
        Z -= Z.mean(0)                          # LedoitWolf.fit: X - location_
        r = np.einsum("ij,ij->i", Z, Z)
        trace = r.sum() / n                     # trace of the empirical covariance
        mu = trace / p
        beta_ = np.sum(r ** 2)
        delta_ = np.sum((Z @ Z.T) ** 2) / n ** 2
        beta = 1.0 / (p * n) * (beta_ / n - delta_)
        delta = (delta_ - 2.0 * mu * trace + p * mu ** 2) / p
        beta = min(beta, delta)
        shrink = 0 if beta == 0 else beta / delta
        lam += priors[idx] * shrink * mu * sc.scale_ ** 2
        Us.append(Z * sc.scale_)
        cs.append(np.full(n, priors[idx] * (1.0 - shrink) / n))
    return lam, np.concatenate(Us), np.concatenate(cs)


class _DualLsqrLDA(_CholeskyLsqrLDA):
    """_CholeskyLsqrLDA for n < p: the same estimate and solve without the
    p x p matrix. With V = C^1/2 U Lam^-1/2 the estimate is
    Lam^1/2 (I + V.T V) Lam^1/2, so coef = Lam^-1/2 (B~ - V.T t) with
    B~ = Lam^-1/2 means_.T and (I_n + V V.T) t = V B~ (Cholesky). Falls back
    to the Cholesky path when lam is not positive, the n x n solve fails or
    the full system's relative residual (in low-rank form) exceeds
    LDA_MAX_RESID. covariance_ is not formed."""

    def _solve_lstsq(self, X, y, shrinkage, covariance_estimator):
        import warnings
        from scipy import linalg
        coef = None
        if shrinkage == "auto" and covariance_estimator is None:
            means = _class_means(X, y)
            B = np.asarray(means.T, dtype=np.float64)
            lam, V, c = _lw_lowrank(X, y, np.asarray(self.priors_, dtype=np.float64))
            if (np.all(np.isfinite(lam)) and lam.min() > 0
                    and np.all(np.isfinite(c)) and c.min() >= 0):
                d = 1.0 / np.sqrt(lam)
                V *= np.sqrt(c)[:, None]
                V *= d[None, :]
                Bt = d[:, None] * B
                M = V @ V.T
                M.flat[::M.shape[0] + 1] += 1.0
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", linalg.LinAlgWarning)
                        t = linalg.solve(M, V @ Bt, assume_a="pos")
                    yv = Bt - V.T @ t               # Lam^1/2 coef
                    coef = d[:, None] * yv
                    r = (yv + V.T @ (V @ yv)) / d[:, None] - B
                    resid = np.linalg.norm(r) / np.linalg.norm(B)
                    if not resid <= LDA_MAX_RESID:  # also catches NaN
                        coef = None
                except (linalg.LinAlgError, linalg.LinAlgWarning, ValueError):
                    coef = None
        if coef is None:                            # the Cholesky path
            LDA_STATS["dual_fallback"] += 1
            return super()._solve_lstsq(X, y, shrinkage, covariance_estimator)
        LDA_STATS["dual"] += 1
        self.means_ = means
        self.coef_ = coef.T
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )


def fit_shrinkage_lda(X, y, priors=None, fast=None, dual=None):
    """LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors)
    fitted on (X, y), returned as a plain sklearn object (it unpickles
    anywhere sklearn does). Up to LDA_FAST_P features: sklearn's own fit.
    Above: the same fit (validation, classes_, priors_, class means,
    Ledoit-Wolf class covariance, 2-class coef reduction, n_features_in_)
    with the lstsq step as a Cholesky solve; sklearn's lstsq is kept when the
    matrix is not numerically SPD / well conditioned or the solve's relative
    residual exceeds LDA_MAX_RESID. With fewer rows than features (and
    LDA_DUAL) the solve is the n x n form of _DualLsqrLDA, which does not
    form covariance_. ``fast`` = True / False forces the first choice and
    ``dual`` = True / False the second (equivalence checks)."""
    use_fast = (np.shape(X)[1] > LDA_FAST_P) if fast is None else bool(fast)
    if not use_fast or _class_cov is None:
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
                                          priors=priors).fit(X, y)
    use_dual = ((LDA_DUAL and np.shape(X)[0] < np.shape(X)[1]) if dual is None
                else bool(dual))
    lda = (_DualLsqrLDA if use_dual else _CholeskyLsqrLDA)(
        solver="lsqr", shrinkage="auto", priors=priors)
    try:
        lda.fit(X, y)
    finally:
        lda.__class__ = LinearDiscriminantAnalysis
    return lda


class RiemannStepTypeModel:

    def __init__(self, parts=None, nfilter=4, estimator="oas",
                 use_xdawn=True, max_batches=None, preproc=None,
                 slow_block=False, filterbank=False):
        self.parts = parts
        self.preproc = preproc          # WindowPreproc or None
        self.nfilter, self.estimator = nfilter, estimator
        self.use_xdawn, self.max_batches = use_xdawn, max_batches
        self.slow_block, self.filterbank = slow_block, filterbank

    def fit(self, train_loader):
        Xs, ys = [], []
        for i, (X, y, _info) in enumerate(train_loader):
            if self.max_batches is not None and i >= self.max_batches:
                break
            Xs.append(self._prep(X, np.float32))
            ys.append(to_numpy(y))
        X = Xs[0] if len(Xs) == 1 else np.concatenate(Xs)
        y = np.concatenate(ys)
        del Xs
        if _chunk_len(X.shape) < len(X):    # > one chunk (500 Hz): float32 + chunks
            return self._fit_chunked(X, y)
        X = X.astype(np.float64)             # the same values as a float64 _prep

        # Only plain values and pyriemann/sklearn objects go into parts.
        sfreq = self.preproc.sfreq
        parts = {"estimator": self.estimator, "xdawn": None, "sfreq": sfreq}
        if self.use_xdawn:
            parts["xdawn"] = XdawnCovariances(
                nfilter=self.nfilter, estimator=self.estimator,
                xdawn_estimator=self.estimator).fit(X, y)
            parts["xdawn_ts"] = TangentSpace(metric="riemann").fit(
                parts["xdawn"].transform(X), y)
        covs = Covariances(estimator=self.estimator).transform(X)
        parts["broad_ts"] = TangentSpace(metric="riemann").fit(covs, y)
        if self.slow_block:
            start, width = round(0.5 * sfreq), round(0.25 * sfreq)
            n_bins = min(14, (X.shape[2] - start) // width)   # 0.5-4.0 s
            if n_bins < 1:
                raise ValueError(f"window too short for slow_block: T={X.shape[2]}")
            parts["slow"] = {"band": "0.1to4", "start": start,
                             "width": width, "n_bins": n_bins}
        if self.filterbank:
            parts["fb_bands"] = list(FB_BANDS)
            parts["fb_ts"] = [
                TangentSpace(metric="riemann").fit(_band_covs(parts, X, band), y)
                for band in FB_BANDS]

        blocks = _feature_blocks(parts, X)
        sizes = " + ".join(f"{k} {v.shape[1]}" for k, v in blocks.items())
        feats = np.concatenate(list(blocks.values()), axis=1)
        print(f"[Riemann-StepType] fitting on X={X.shape}, "
              f"classes={np.unique(y).tolist()}, "
              f"features={feats.shape[1]} ({sizes})", flush=True)

        n_classes = len(np.unique(y))
        parts["lda"] = fit_shrinkage_lda(feats, y, np.full(n_classes, 1.0 / n_classes))
        self.parts = parts
        return self

    def _fit_chunked(self, X, y):
        """``fit`` for a float32 training set larger than one chunk: the same
        parts from chunked passes (float64 per chunk); xDAWN from chunked
        statistics (~1e-12 relative vs pyriemann's one-shot fit, as the
        Riemann-Sealed solver), every other block per window as in ``fit``."""
        n, size = len(X), _chunk_len(X.shape)
        sfreq = self.preproc.sfreq
        t0 = time.time()

        def lap(stage):         # heartbeat: long 500 Hz fits keep the log growing
            print(f"[Riemann-StepType] [{time.strftime('%H:%M:%S')}] chunked fit: "
                  f"{stage} ({time.time() - t0:.0f} s)", flush=True)

        def get(s):
            return np.asarray(X[s], dtype=np.float64)
        parts = {"estimator": self.estimator, "xdawn": None, "sfreq": sfreq}
        if self.use_xdawn:
            parts["xdawn"] = _fit_xdawn(self.nfilter, self.estimator, get, y, n, size)
            lap("xDAWN")
        if self.slow_block:
            start, width = round(0.5 * sfreq), round(0.25 * sfreq)
            n_bins = min(14, (X.shape[2] - start) // width)   # 0.5-4.0 s
            if n_bins < 1:
                raise ValueError(f"window too short for slow_block: T={X.shape[2]}")
            parts["slow"] = {"band": "0.1to4", "start": start,
                             "width": width, "n_bins": n_bins}
        if self.filterbank:
            parts["fb_bands"] = list(FB_BANDS)
        raw = _window_blocks(parts, get, n, size)
        lap("window blocks")
        _fit_tangent(parts, raw, y)
        blocks = _raw_blocks(parts, raw)
        del raw
        lap("tangent spaces")
        sizes = " + ".join(f"{k} {v.shape[1]}" for k, v in blocks.items())
        feats = np.concatenate(list(blocks.values()), axis=1)
        del blocks
        print(f"[Riemann-StepType] fitting on X={X.shape}, "
              f"classes={np.unique(y).tolist()}, "
              f"features={feats.shape[1]} ({sizes}), chunked {size}/{n} windows",
              flush=True)
        n_classes = len(np.unique(y))
        parts["lda"] = fit_shrinkage_lda(feats, y, np.full(n_classes, 1.0 / n_classes))
        lap("LDA")
        self.parts = parts
        return self

    def _prep(self, X, dtype=np.float64):
        """Per-window preprocessing on CPU, then float64 for pyriemann
        (``dtype=np.float32``: the preprocessing output as is, for chunking)."""
        X = torch.as_tensor(X, dtype=torch.float32).cpu()
        if self.preproc is not None:
            with torch.no_grad():
                X = self.preproc(X)
        return X.numpy().astype(dtype, copy=dtype != np.float32)

    def predict(self, X):
        feats = np.concatenate(list(feature_blocks_chunked(
            self.parts, self._prep(X, np.float32)).values()), axis=1)
        return torch.as_tensor(self.parts["lda"].predict(feats))


class Solver(CompetSolver):

    name = "Riemann-StepType"

    parameters = {
        "nfilter": [4],
        "estimator": ["oas"],
        "use_xdawn": [True],     # xDAWN is ERP-oriented; try False for MI
        "slow_block": [False],   # + binned < 4 Hz waveform block
        "filterbank": [False],   # + 4-band filter-bank tangent space
        # Per-window preprocessing (see WindowPreproc): "none" or e.g. "8to30"
        # Hz; reference "none" | "car" | "laplacian".
        "bandpass": ["none"],
        "reference": ["none"],
        "max_batches": [None],   # small int = quick pipeline test
    }

    def load_model(self, meta):
        weights = meta["submission_dir"] / "riemann.joblib"
        # Inference only: a training run starts fresh (see eegnet_steptype).
        parts = (joblib.load(weights)
                 if weights.exists() and self.train_loader is None else None)
        preproc = WindowPreproc(meta["ch_names"], meta["sfreq"],
                                self.bandpass, self.reference)
        return RiemannStepTypeModel(parts, self.nfilter, self.estimator,
                                    self.use_xdawn, self.max_batches, preproc,
                                    self.slow_block, self.filterbank)

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        joblib.dump(model.parts, path / "riemann.joblib")
