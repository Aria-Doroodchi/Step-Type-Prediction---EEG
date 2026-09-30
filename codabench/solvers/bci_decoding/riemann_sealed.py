"""Riemann-Sealed — Riemann-StepType plus what the sealed phase needs: a
subject router and per-subject alignment inside ``predict(X)``, with no
subject or session id at prediction time.

Sealed phase (Track 2): the same participants across sessions; early sessions
are labelled calibration data, later ones are scored, and ``predict`` receives
bare ``(B, C, T)`` windows. Built from the cross-session proxy study
(codabench/SEALED_RECIPE.md, codabench/LOG.md 2026-09-25/26).

Training (``fit``; the train loader supplies ``info["subject_id"]``):
  0. channel pick (``chans``): "eeg" (default) drops EMG/EOG/ECG channels,
     typed by ``meta["ch_types"]``, else the ``chs_info`` kind, else the
     name (EMG / EOG / ECG anywhere in it); "eeg+eog" / "eeg+emg" keep that type
     too; "all" keeps every channel. The index is stored in the joblib and
     applied identically at prediction (a joblib without one keeps all
     channels, as it was trained). The proxies are EEG-only: no change there.
     The joblib also stores the training channel names (``ch_names``, all
     channels, before the pick); ``load_model`` matches the scoring
     ``meta["ch_names"]`` to them by name (``_match_channels``): the same
     list -> nothing changes; another order -> the windows are reordered to
     the training order before preprocessing and the pick; extra scoring
     channels are dropped (a printed note); a trained channel missing ->
     ValueError naming it. A joblib without names (trained before this
     check) or ``meta["ch_names"]`` None -> by position, as before, with a
     printed warning;
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
  5. personalisation (``personal``): "pooled" = the pooled LDA alone;
     "calib" = the routed subject's own LDA, fitted on the pooled feature
     extractor's features; "blend" = **blend_calib** of SEALED_RECIPE.md,
     i.e. the pooled feature extractor with a per-subject LDA, P = w *
     P_pooled + (1 - w) * P_subject (the pooled LDA's and the routed
     subject's LDA's probabilities). w = ``blend_w``: a number, or "auto"
     (chosen in ``fit``, see below, and stored in the joblib).
Prediction: route each window (argmax posterior; below threshold -> global W
and pooled LDA), whiten with the routed subject's W, features, LDA(s).
``align="subject_context"`` (opt-in; the harness's ``router-psdctx:<kind>``,
analysis/sealed_run.aligned_data + sealed_personal.py): the unit of routing
and whitening is the (subject, context) pair, e.g. Graz vs BrainHero, whose
ids ``fit`` reads from the train loader's trigger table (column "context",
else "condition" / "paradigm", as analysis/xsess_cache.py; the same row-order
check as the session ids). The router is the same log-PSD shrinkage LDA, fed
pair labels (subject-major, contexts in sorted order, like the harness's
subj * n_ctx + ctx), with the same threshold rule and 0.5 cap; each pair gets
W_p from its own training windows (a pair with fewer than ``ctx_min``
windows keeps its subject's W, with a warning), and the pooled model is
trained on pair-whitened windows. Personal LDAs stay per subject: a routed
pair selects its subject's LDA (below threshold: global W, pooled LDA, as
before). blend_w="auto" folds use the pair-whitened data (the harness's
choose_w under router-psdctx). No context ids (e.g. the proxies, or a
trigger table that ``_collect``'s row-order check rejects) -> a warning and
exactly ``align="subject"``. That check sees a table misordered ACROSS
subjects (via its subject column), but a table misordered WITHIN subjects
only when the loader's info carries record_id / onset (datasets/mock_sealed.py
does; NeuralBench's nb_task loader gives subject_id only, and its table is
aligned by construction, seg_ds[i] <-> row i). A within-subject misorder that
passes trains the pairs, and blend_w="auto"'s sessions, on wrong ids with no
warning. The tell: when it breaks the (subject, context) grouping, the fit
log's "router OOF pair acc" drops well below the router-psdctx OOF / test
pair accuracy that analysis/release_eda.py reports from the harness cache
(mock_sealed_s, solver: 1.000 aligned, 0.500 for a within-subject shuffle,
0.813 for a table sorted by context; harness psdctx router 0.994 on test;
release_eda OOF 1.000 on mock_sealed_120). A table that swaps the context
names consistently keeps 1.000, but then the grouping, hence every W_p and
the routing, is unchanged.
With ``adapt="online"`` it raises NotImplementedError (no per-pair online
re-centring: the harness has none to check it against). Everything added
to the joblib (pair -> subject map, pair contexts / sizes, W per pair) is
plain numpy; ``predict`` reads the mode from the joblib, not from the
solver's parameters. Checked 2026-09-28 on mock_sealed_s (replica:3, 43 EEG
ch, w = 0.75): benchopt train = read-only replay = the harness's
blend_calib router-id row under router-psdctx:riemann (0.547222; router
pair accuracy 0.9944 on both sides); blend_w="auto" LOSO fold scores
bit-identical to the harness's choose_w (--wcv loso) under router-psdctx
(w = 0.75 on both). The default align="subject" is unchanged (bit-identical
probabilities to the previous version on zhou2016 and mock_sealed_s).
``adapt="online"`` (RULE-DEPENDENT: uses unlabelled test windows; off by
default until the organisers confirm it is allowed): the routed subject's W
comes from the mean covariance of the last ``buffer`` test windows routed to
that subject, kept across predict() calls (training W until buffer/4 seen).
On the proxies this recovers the whole session-drift gain (+5.6 to +7.5
points per-subject); without it, per-subject whitening is ~invisible to a
tangent space at the subject's own mean (affine invariance).

``blend_w="auto"`` needs session ids (local training: NeuralBench's trigger
table, see ``_collect``; without them w = 0.5 and a warning). It is the
harness's ``sealed_personal.py --wcv loso`` rule, leave-one-calibration-
session-out: for each chronological session index k, validate on session k of
every subject with >= 2 training sessions and refit on all other training
windows (xDAWN, tangent spaces, pooled LDA, per-subject LDAs); score w in
{0, .25, .5, .75, 1} on the validation rows with each row's TRUE subject's
LDA, cell metric over subject x session (x context) cells; mean over folds,
ties to the larger (more pooled) w. No subject with 2 training sessions ->
two chronological halves per subject (the harness's fallback). As in the
harness, the folds keep the whitening references of the whole training set
(each subject's W from all its training windows, the validation session
included) and whiten in float32 like ``xsess_lib.apply_W``, so the solver's
weight equals the harness's when both see the same windows in the same order.
Residual differences: row order (the train loader's index order vs the
cache's recording order: last-bit differences in the xDAWN/LDA fits), which
windows are "training" (the benchopt train split vs the harness's split),
preprocessing (the solver re-references/band-passes before whitening, the
harness after; both are off by default), and above one chunk the chunked
xDAWN statistics (~1e-12 relative vs the harness's one-shot pyriemann fit).
Every LDA (solver and harness) is ``fit_shrinkage_lda`` with the same solve
rule (Cholesky above LDA_FAST_P features, sklearn's lstsq below), so the LDAs
add no solver-harness difference at any size.
Checked 2026-09-28: identical weights and bit-identical fold scores on
zhou2016 (with and without contexts, adapt none/online) and mock_sealed_s
(replica:3, 43 EEG ch, contexts; harness LDAs on the same Cholesky solve).
The strict alternative, whitening references refitted on each fold's fit
rows (a held-out session then drifts like a real test session), is not
implemented: it would no longer equal the harness's weight.

Memory and time (43-47 ch x 500 Hz): training windows are kept as float32
(exact: the preprocessing output is float32); every float64 step (router PSD,
window covariances, whitening, feature blocks) runs in chunks of <=
_CHUNK_BYTES, and xDAWN is fitted from chunked statistics (signal covariance +
class-mean responses) when the training set exceeds one chunk. LDAs with more
than LDA_FAST_P features (the sealed data: ~5-6k) solve with a Cholesky
factorisation instead of sklearn's SVD lstsq (86 s -> 2 s per LDA at p = 5987;
sklearn's lstsq is kept if the matrix is ill-conditioned or the residual is
not tiny) and drop their unused p x p covariance_ (287 MB each). Every proxy is one
chunk and <= 3175 features, so their numbers are bit-identical to the plain
code. Prediction memory is bounded by the batch. ``fit`` prints the seconds
per stage and the peak RSS (plus a heartbeat line for any stage or blend fold
slower than 30 s); ``predict_proba`` prints a timing summary every 20 calls.

Everything saved is numpy arrays and sklearn/pyriemann objects (joblib), so it
unpickles on the scoring worker.

Train locally (from ~/codabench/2026-competition, after ``source env.sh``):

    benchopt run tracks/bci_decoding -d "BCI[study=zhou2016_xsess]" \\
        -s ../solvers/bci_decoding/riemann_sealed.py \\
        -o "BCI-decoding[training=True]"
"""


import re
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

try:
    import resource          # peak RSS; not available on Windows
except ImportError:          # pragma: no cover
    resource = None


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
    if not ch_names:
        raise ValueError(f"reference={reference!r} needs the channel names "
                         "(meta['ch_names'] is None or empty)")
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
        M = _spatial_matrix(list(ch_names or []), reference)   # None: no names
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


# ---------------------------------------------------------------------------
# Extra temporal blocks (opt-in ``xblocks``, sprint 2026-09-29; the harness's
# analysis/xfeat_temporal.py blocks of the same ids, same values):
#   tseg<K>  per FB band: band-pass the whole window, K equal contiguous
#            segments (remainder dropped), OAS covariance each -> one tangent
#            space per (band, segment); 4 * K * C(C+1)/2 features;
#   bpt<K>   per FB band: band-pass, square, mean over K equal contiguous time
#            bins per channel, log(. + 1e-12) (stateless); 4 * K * C features;
#   icoh     per FB band imaginary coherence of every channel pair (i < j,
#            signed; Welch within the window: 1 s periodic-Hann segments, 50 %
#            overlap, per-segment mean removed, band = the FFT bins in [lo, hi])
#            (stateless); 4 * C(C-1)/2 features.
# Appended after the filter bank, in ``xblocks`` order. Default "" = none (the
# model is then exactly the recipe).
# ---------------------------------------------------------------------------
_XBLOCK_RX = re.compile(r"(tseg|bpt)([1-9][0-9]*)")


def _parse_xblocks(xblocks):
    """"" / None -> []; "tseg3_bpt4" (or "tseg3+bpt4") -> ["tseg3", "bpt4"].
    ("_" is the separator to use on a benchopt command line.)"""
    names = [s for s in re.split(r"[_+,]", str(xblocks or "")) if s]
    for s in names:
        if not (s == "icoh" or _XBLOCK_RX.fullmatch(s)):
            raise ValueError(f"xblocks: unknown block {s!r} (expected tseg<K> / bpt<K> / icoh)")
    if len(set(names)) != len(names):
        raise ValueError(f"xblocks: duplicate block in {xblocks!r}")
    return names


def _seg_slices(T, k):
    w = T // k
    if w < 2:
        raise ValueError(f"{k} segments of a {T}-sample window are too short")
    return [slice(i * w, (i + 1) * w) for i in range(k)]


def _icoh(X, sfreq, chunk_bytes=256 * 2 ** 20):
    """(n, C, T) float64 -> (n, 4 C(C-1)/2) imaginary coherence per FB band
    (the harness's analysis/xfeat_spatial.ImagCoherence, same values):
    icoh_ij = Im(S_ij) / sqrt(S_ii S_jj), S_ij = sum over Welch segments and
    in-band FFT bins of conj(Z_i) Z_j; band-major, pairs in triu order."""
    from numpy.lib.stride_tricks import sliding_window_view
    from scipy.signal import get_window
    n, C, T = X.shape
    nper = min(int(round(sfreq)), T)
    step = max(1, nper // 2)
    win = get_window("hann", nper)
    f = np.fft.rfftfreq(nper, 1.0 / sfreq)
    sel = []
    for band in FB_BANDS:
        lo, hi = (float(v) for v in band.split("to"))
        sel.append(np.flatnonzero((f >= lo) & (f <= hi)))
    iu = np.triu_indices(C, 1)
    n_seg = (T - nper) // step + 1
    rows = max(1, chunk_bytes // (8 * C * n_seg * nper * 3))
    out = []
    for i in range(0, n, rows):
        seg = sliding_window_view(X[i:i + rows], nper, axis=2)[:, :, ::step]
        seg = seg - seg.mean(axis=-1, keepdims=True)
        Z = np.fft.rfft(seg * win, axis=-1)                      # (b, C, S, F)
        blk = []
        for m in sel:
            Zb = Z[..., m].reshape(len(seg), C, -1)
            S = np.conj(Zb) @ Zb.transpose(0, 2, 1)
            p = np.real(np.einsum("nii->ni", S))
            den = np.sqrt(p[:, :, None] * p[:, None, :])
            ic = np.divide(S.imag, den, out=np.zeros_like(den), where=den > 0)
            blk.append(ic[:, iu[0], iu[1]])
        out.append(np.concatenate(blk, axis=1))
    return np.concatenate(out)


def _xblock_raw(parts, X):
    """Per-window inputs of the extra blocks for (n, C, T) float64 windows:
    "<tsegK>:<band>:<i>" -> (n, C, C) covariances, "<bptK>" -> (n, 4 K C),
    "icoh" -> (n, 4 C(C-1)/2)."""
    out = {}
    names = parts.get("xblocks") or []
    if not names:
        return out
    if "icoh" in names:
        out["icoh"] = _icoh(X, parts["sfreq"])
    cov = Covariances(estimator=parts["estimator"])
    bpt = {s: [] for s in names if s.startswith("bpt")}
    for band in FB_BANDS:
        if not any(_XBLOCK_RX.fullmatch(s) for s in names):
            break
        Xb = np.concatenate(list(_band_chunks(X, parts["sfreq"], band)))
        for s in names:
            if s == "icoh":
                continue
            k = int(_XBLOCK_RX.fullmatch(s).group(2))
            segs = _seg_slices(Xb.shape[-1], k)
            if s.startswith("tseg"):
                for i, sl in enumerate(segs):
                    out[f"{s}:{band}:{i}"] = cov.transform(Xb[..., sl])
            else:
                P = Xb ** 2
                bpt[s] += [np.log(P[..., sl].mean(axis=2) + 1e-12) for sl in segs]
    for s, v in bpt.items():
        out[s] = np.concatenate(v, axis=1)
    return out


def _xblock_keys(s):
    k = int(_XBLOCK_RX.fullmatch(s).group(2))
    return [f"{s}:{band}:{i}" for band in FB_BANDS for i in range(k)]


def _xblock_features(parts, raw, rows=slice(None)):
    """Extra-block features of ``rows`` from ``_xblock_raw`` output."""
    out = []
    for s in parts.get("xblocks") or []:
        if s.startswith("tseg"):
            out.append(np.concatenate(
                [ts.transform(raw[key][rows])
                 for key, ts in zip(_xblock_keys(s), parts["xts:" + s])], axis=1))
        else:
            out.append(raw[s][rows])
    return out


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
    if parts.get("xblocks"):
        for s, f in zip(parts["xblocks"], _xblock_features(parts, _xblock_raw(parts, X))):
            blocks["x:" + s] = f
    return blocks


def _features(parts, X):
    """Feature union for (n, C, T) float64 windows -> (n, n_features)."""
    return np.concatenate(list(_feature_blocks(parts, X).values()), axis=1)


# ---------------------------------------------------------------------------
# Chunked training passes (43-47 ch x 500 Hz). The windows are stored as
# float32; each chunk is converted to float64 before any math, so a single
# chunk reproduces the unchunked float64 code exactly.
# ---------------------------------------------------------------------------
# float64 bytes of windows per chunk. Every proxy (<= 0.45 GB as float64) and
# mock_sealed_s (0.48 GiB) is one chunk; 47 ch x 2000 samples -> 713 windows.
_CHUNK_BYTES = 512 * 2 ** 20


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


def _maxrss_gb():
    if resource is None:
        return float("nan")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20   # KB -> GB


_HEARTBEAT_S = 30     # stages / blend folds slower than this print a progress line


def _lap(tm, stage, t0):
    """Record a fit stage's seconds; a slow stage prints a heartbeat line (so
    a long 500 Hz fit keeps its log growing)."""
    tm[stage] = time.time() - t0
    if tm[stage] >= _HEARTBEAT_S:
        print(f"[Riemann-Sealed] [{time.strftime('%H:%M:%S')}] {stage} done in "
              f"{tm[stage]:.0f} s; maxrss {_maxrss_gb():.2f} GB", flush=True)


def _whiten_rows(X, idx, Ws, Wg, f32=False):
    """Row i <- W[idx[i]] @ X[i] (idx -1 -> Wg). float64 (the solver's own
    whitening), or f32=True: the harness's float32 ``xsess_lib.apply_W``,
    used only to choose blend_w="auto" on the harness's aligned data."""
    X = np.asarray(X, dtype=np.float32 if f32 else np.float64)
    out = np.empty_like(X)
    for k in np.unique(idx):
        m = idx == k
        W = Wg if k < 0 else Ws[k]
        if f32:
            out[m] = np.einsum("ij,njt->nit", W.astype(np.float32), X[m]).astype(np.float32)
        else:
            out[m] = np.einsum("ij,njt->nit", W, X[m])
    return out


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


def _window_blocks(parts, get, rows, size, which=None):
    """One chunked pass over the windows ``get(chunk)`` (float64): the
    per-window input of every feature block before its tangent space (xDAWN,
    broadband and filter-bank covariances; log-variance; slow bins), as in
    ``_feature_blocks``. ``which``: only these block names."""
    acc = {}

    def want(k):
        return which is None or k in which
    for s in _row_chunks(rows, size):
        xb = get(s)
        out = {}
        if parts.get("xdawn") is not None and want("xdawn"):
            out["xdawn"] = parts["xdawn"].transform(xb)
        if want("broad"):
            out["broad"] = Covariances(estimator=parts["estimator"]).transform(xb)
        if want("logvar"):
            out["logvar"] = np.log(np.var(xb, axis=2) + 1e-12)
        if parts.get("slow") is not None and want("slow"):
            out["slow"] = _slow_bins(parts, xb)
        if want("fb"):
            for band in parts.get("fb_bands") or []:
                out["fb:" + band] = _band_covs(parts, xb, band)
        if want("x"):
            out.update(_xblock_raw(parts, xb))
        for k, v in out.items():
            acc.setdefault(k, []).append(v)
    return {k: v[0] if len(v) == 1 else np.concatenate(v) for k, v in acc.items()}


def _fit_tangent(parts, raw, y, rows=slice(None)):
    """Tangent spaces of the covariance blocks, fitted on ``rows``."""
    def ts(covs):
        return TangentSpace(metric="riemann").fit(covs, y[rows])
    if parts.get("xdawn") is not None:
        parts["xdawn_ts"] = ts(raw["xdawn"][rows])
    parts["broad_ts"] = ts(raw["broad"][rows])
    if parts.get("fb_bands"):
        parts["fb_ts"] = [ts(raw["fb:" + b][rows]) for b in parts["fb_bands"]]
    for s in parts.get("xblocks") or []:
        if s.startswith("tseg"):
            parts["xts:" + s] = [ts(raw[key][rows]) for key in _xblock_keys(s)]


def _raw_features(parts, raw, rows=slice(None)):
    """Feature union of ``rows`` from ``_window_blocks`` output (the order of
    ``_feature_blocks``)."""
    blocks = []
    if parts.get("xdawn") is not None:
        blocks.append(parts["xdawn_ts"].transform(raw["xdawn"][rows]))
    blocks.append(parts["broad_ts"].transform(raw["broad"][rows]))
    blocks.append(raw["logvar"][rows])
    if parts.get("slow") is not None:
        blocks.append(raw["slow"][rows])
    if parts.get("fb_ts") is not None:
        blocks.append(np.concatenate(
            [ts.transform(raw["fb:" + b][rows])
             for b, ts in zip(parts["fb_bands"], parts["fb_ts"])], axis=1))
    blocks += _xblock_features(parts, raw, rows)
    return np.concatenate(blocks, axis=1)


# ---------------------------------------------------------------------------
# Sealed-phase additions: router + per-subject whitening + personal LDAs.
# ---------------------------------------------------------------------------
def _window_covs(X, shrink=1e-3):
    """(n, C, C) float64 window covariances X X^T / T + trace-scaled ridge,
    chunked over windows."""
    out = []
    for s in _row_chunks(len(X), _chunk_len(X.shape)):
        Xb = np.asarray(X[s], dtype=np.float64)
        C = np.einsum("nct,ndt->ncd", Xb, Xb) / Xb.shape[-1]
        tr = np.trace(C, axis1=1, axis2=2)[:, None, None] / C.shape[1]
        out.append(C + shrink * tr * np.eye(C.shape[1])[None])
    return out[0] if len(out) == 1 else np.concatenate(out)


def _mean_cov(covs, kind):
    if kind == "euclid":
        return covs.mean(0)
    from pyriemann.utils.mean import mean_riemann
    return mean_riemann(covs, maxiter=50)


def _inv_sqrtm(R):
    w, V = np.linalg.eigh((R + R.T) / 2)
    return (V * (1.0 / np.sqrt(np.maximum(w, 1e-12)))) @ V.T


def _psd_features(X, sfreq):
    """Router features (float64 math, chunked over windows)."""
    from scipy.signal import welch
    out = []
    for s in _row_chunks(len(X), _chunk_len(X.shape)):
        f, P = welch(np.asarray(X[s], dtype=np.float64), fs=sfreq,
                     nperseg=int(sfreq), axis=-1)
        m = (f >= 1) & (f <= 45)
        out.append(np.log(P[..., m] + 1e-12).reshape(len(P), -1))
    return out[0] if len(out) == 1 else np.concatenate(out)


# ---------------------------------------------------------------------------
# Shrinkage LDA with a fast solve (identical block in riemann_steptype.py,
# which the harness uses, so solver and harness LDAs agree at every size).
# sklearn's LinearDiscriminantAnalysis(solver="lsqr") solves
# covariance_ @ coef_.T = means_.T with scipy's SVD lstsq, O(p^3): 70-100 s
# per fit at p = 5073 features (fb=1 union at 43 ch) on 4 threads. The shrunk
# covariance is SPD, so a Cholesky solve gives the same coefficients (to
# ~1e-14 relative) in ~3 s. Above LDA_FAST_P features the fit is sklearn's own
# fit with only that solve replaced; every proxy has p <= 3175 (3595 with
# slow_block) and keeps sklearn's lstsq bit-for-bit.
# ---------------------------------------------------------------------------
LDA_FAST_P = 4000
LDA_MAX_RESID = 1e-8      # relative residual above which the solve falls back
LDA_STATS = {"fast": 0, "fallback": 0}   # diagnostics: fast solves / lstsq fallbacks

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


def fit_shrinkage_lda(X, y, priors=None, fast=None):
    """LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors)
    fitted on (X, y), returned as a plain sklearn object (it unpickles
    anywhere sklearn does). Up to LDA_FAST_P features: sklearn's own fit.
    Above: the same fit (validation, classes_, priors_, class means,
    Ledoit-Wolf class covariance, 2-class coef reduction, n_features_in_)
    with the lstsq step as a Cholesky solve; sklearn's lstsq is kept when the
    matrix is not numerically SPD / well conditioned or the solve's relative
    residual exceeds LDA_MAX_RESID. ``fast`` = True / False forces the choice
    (equivalence checks)."""
    use_fast = (np.shape(X)[1] > LDA_FAST_P) if fast is None else bool(fast)
    if not use_fast or _class_cov is None:
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
                                          priors=priors).fit(X, y)
    lda = _CholeskyLsqrLDA(solver="lsqr", shrinkage="auto", priors=priors)
    try:
        lda.fit(X, y)
    finally:
        lda.__class__ = LinearDiscriminantAnalysis
    return lda


def _fit_lda(F, y, priors=None):
    """The solver's shrinkage LDA: ``fit_shrinkage_lda``; above LDA_FAST_P
    features the fitted p x p ``covariance_`` (206 MB at p = 5073, 287 MB at
    5987; one per subject) is then dropped: lsqr prediction uses only
    coef_ / intercept_ / classes_, and 21 of them would put ~4-6 GB in the
    model and the submission joblib."""
    lda = fit_shrinkage_lda(F, y, priors)
    if F.shape[1] > LDA_FAST_P:
        del lda.covariance_
    return lda


# -- channel types (chans) ------------------------------------------------------
# anywhere in the name (HEOG, VEOG, LEOG, Chin_EMG, EOG_L, EMG1, ...); also the
# harness's rule for caches without meta ch_types (analysis/xsess_lib.ch_types)
_NON_EEG = re.compile(r"EMG|EOG|ECG", re.IGNORECASE)
_FIFF_KINDS = {2: "eeg", 202: "eog", 302: "emg", 402: "ecg"}   # MNE FIFFV_*_CH
_CHANS = {"eeg": {"eeg"}, "eeg+eog": {"eeg", "eog"}, "eeg+emg": {"eeg", "emg"}}


def _channel_types(ch_names, chs_info=None, ch_types=None):
    """'eeg' | 'emg' | 'eog' | 'ecg' per channel: an explicit ``ch_types``
    list, else each ``chs_info`` entry's kind (MNE FIFF code or a type
    string), else the name (EMG / EOG / ECG anywhere in it, case-insensitive;
    anything else is EEG, e.g. ExG1)."""
    if ch_types is not None and len(ch_types) == len(ch_names):
        return [str(t).lower() for t in ch_types]
    out = []
    for i, name in enumerate(ch_names):
        kind = None
        info = chs_info[i] if chs_info is not None and i < len(chs_info) else None
        k = (info.get("kind", info.get("ch_type", info.get("type")))
             if isinstance(info, dict) else None)
        if k is not None:
            try:
                kind = _FIFF_KINDS.get(int(k))
            except (TypeError, ValueError):
                kind = next((t for t in ("emg", "eog", "ecg", "eeg")
                             if t in str(k).lower()), None)
        if kind is None:
            m = _NON_EEG.search(str(name))
            kind = m.group(0).lower() if m else "eeg"
        out.append(kind)
    return out


def _chan_index(types, chans):
    """Kept channel indices for ``chans``, or None when every channel is kept."""
    if chans == "all":
        return None
    idx = [i for i, t in enumerate(types) if t in _CHANS[chans]]
    return None if len(idx) == len(types) else idx


def _match_channels(train_names, score_names):
    """(perm, names): how the scoring windows' channels map onto the channels
    the model was trained on (``parts["ch_names"]``, before the pick).

    perm None: X is used as it comes, by position: the same names in the same
    order (bit-identical to before), or nothing to compare (a joblib without
    names, scoring names None, duplicate training names), with a warning.
    Else ``X[:, perm]`` holds the training channels in the training order
    (extra scoring channels dropped, with a note). ``names``: the channel
    names of that X, for WindowPreproc (its re-reference matrix). A trained
    channel missing, or present twice, at scoring -> ValueError."""
    tag = "[Riemann-Sealed]"
    if train_names is None:
        print(f"{tag} WARNING: the model stores no training channel names (a joblib "
              "from before the channel check): the scoring channels are used by "
              "position, unchecked", flush=True)
        return None, score_names
    train = [str(c) for c in train_names]
    if score_names is None:
        print(f"{tag} WARNING: meta['ch_names'] is None at scoring: the windows' "
              f"channels are assumed to be the {len(train)} training channels in "
              "the training order (by position, unchecked)", flush=True)
        return None, train
    score = [str(c) for c in score_names]
    if score == train:
        return None, score_names
    if len(set(train)) != len(train):
        print(f"{tag} WARNING: the training channel names are not unique and the "
              "scoring ones differ from them: channels used by position, "
              "unchecked", flush=True)
        return None, score_names
    pos = {}
    for i, c in enumerate(score):
        pos.setdefault(c, []).append(i)
    missing = [c for c in train if c not in pos]
    if missing:
        raise ValueError(
            f"{tag} {len(missing)} of the {len(train)} channels the model was trained "
            f"on are missing from the scoring meta['ch_names'] ({len(score)} "
            f"channels): {missing}")
    twice = [c for c in train if len(pos[c]) > 1]
    if twice:
        raise ValueError(f"{tag} trained channels found more than once in the "
                         f"scoring meta['ch_names']: {twice}")
    keep = set(train)
    extra = [c for c in score if c not in keep]
    print(f"{tag} note: the scoring channels ({len(score)}) differ from the "
          f"{len(train)} training channels in order or number: reordered by name "
          "to the training order"
          + (f"; {len(extra)} extra channel(s) dropped: {extra}" if extra else ""),
          flush=True)
    return [pos[c][0] for c in train], train


# -- blend_w="auto": leave-one-calibration-session-out ---------------------------
W_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
_CTX_COLUMNS = ("context", "condition", "paradigm")   # as analysis/xsess_cache.py
# align="subject_context": a (subject, context) pair with fewer training windows
# keeps its subject's whitening reference. 16 = the smallest reference this file
# already trusts (adapt="online": buffer/4 of 64). The harness has no minimum;
# 2 * n_channels (86 at 43 ch) would put every mock_sealed_s pair (36 / 72
# windows) on its subject's W, where the harness gains +6.4 points with them.
# The full-size mocks (60 windows per pair and session) are far above either.
CTX_MIN_WINDOWS = 16


def _session_key(s):
    """Chronological sort key for session labels ('0', '1test', ...), the
    order analysis/xsess_cache.py gives sessions."""
    s = str(s)
    digits = "".join(ch for ch in s if ch.isdigit())
    return (int(digits) if digits else 0, s)


def _session_rank(sidx, sess):
    """Chronological index of each window's session among its subject's
    training sessions (= the cache's session index when training sessions
    precede test sessions, as in the sealed split)."""
    rank = np.zeros(len(sidx), dtype=np.int64)
    for k in np.unique(sidx):
        m = sidx == k
        order = {v: j for j, v in enumerate(sorted(set(sess[m]), key=_session_key))}
        rank[m] = [order[v] for v in sess[m]]
    return rank


def _blend_folds(sidx, rank):
    """(fit_idx, val_idx) pairs: leave-one-session-out over chronological
    session indices, validating only subjects with >= 2 training sessions
    (empty folds skipped); else two chronological halves of each subject's
    windows (rows in recording order), fit on one and score the other."""
    n_sess = np.array([len(np.unique(rank[sidx == k])) for k in range(sidx.max() + 1)])
    multi = n_sess[sidx] >= 2
    folds = []
    for k in np.unique(rank):
        v = multi & (rank == k)
        if v.any():
            folds.append((np.where(~v)[0], np.where(v)[0]))
    if folds:
        return folds, "loso"
    first = np.zeros(len(sidx), bool)
    for k in np.unique(sidx):
        i = np.where(sidx == k)[0]
        first[i[: len(i) // 2]] = True
    a, b = np.where(first)[0], np.where(~first)[0]
    return [(a, b), (b, a)], "halves"


def _cell_score(y, yhat, subj, sess, ctx):
    """Mean balanced accuracy over subject x session x context cells, in the
    order of analysis/xsess_lib.score (its "cell")."""
    import warnings
    from sklearn.metrics import balanced_accuracy_score
    cells = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in np.unique(subj):
            ms = subj == s
            for v, c in sorted({(v, c) for v, c in zip(sess[ms], ctx[ms])}):
                m = ms & (sess == v) & (ctx == c)
                cells.append(balanced_accuracy_score(y[m], yhat[m]))
    return float(np.mean(cells))


def _trigger_table(ds):
    """The train dataset's per-window trigger table (index order), or None:
    NeuralBench's ``dataset.seg_ds.triggers``, or a ``dataset.triggers``."""
    for obj in (getattr(ds, "seg_ds", None), ds):
        try:
            trig = getattr(obj, "triggers", None) if obj is not None else None
        except Exception:        # no triggers table: try the next / fall back
            trig = None
        if trig is not None and hasattr(trig, "columns"):
            return trig
    return None


# onset vs table start tolerance: max(2 samples, 20 ms); rounding of either
# is <= 0.5 sample (or a few ms), a misplaced window is off by its jitter or more
_ROW_TOL = (2.0, 0.02)


def _table_row_mismatch(trig, record_id, onset):
    """Why the trigger table does not follow the dataset's index order, from
    the loader's own ``info["record_id"]`` (one id per source recording) and
    ``info["onset"]`` (window start, samples) row by row; None when they
    agree or cannot be compared (no ids, a single recording id as
    benchmark_utils' ArrayWindows default, onsets constant within every
    recording, non-numeric values).
    - each recording id must map to one (subject, session) of the table;
    - within each recording, the table's ``start`` (s) and the onsets must be
      one affine map with a common positive slope (the sample rate), every
      residual within _ROW_TOL. This catches rows moved within a recording
      and whole recordings swapped for one another (their jittered window
      starts differ). Blind spots: swapped recordings whose window starts
      coincide up to a shift (a regular grid) and, without a start column,
      any swap of whole recordings (only rows that split a recording across
      subjects / sessions are seen then)."""
    if record_id is None or len(record_id) != len(trig):
        return None
    try:
        rec, g, cnt = np.unique(np.asarray(record_id), return_inverse=True,
                                return_counts=True)
    except TypeError:                    # unsortable ids: not comparable
        return None
    if len(rec) < 2:
        return None
    g = g.reshape(-1)
    import pandas as pd
    cols = [c for c in ("subject", "session") if c in trig.columns]
    if cols:
        key = np.zeros(len(trig), dtype=np.int64)
        for c in cols:
            codes, uniq = pd.factorize(trig[c].astype(str))
            key = key * len(uniq) + codes
        n_pairs = len(np.unique(g * (int(key.max()) + 1) + key))
        if n_pairs != len(rec):
            return (f"{len(rec)} recording ids map to {n_pairs} (recording, "
                    f"{' / '.join(cols)}) pairs")
    if onset is None or len(onset) != len(trig) or "start" not in trig.columns:
        return None
    try:
        st = trig["start"].to_numpy(dtype=np.float64)
        on = np.asarray(onset, dtype=np.float64)
    except (TypeError, ValueError):      # non-numeric: not comparable
        return None
    if not (np.isfinite(st).all() and np.isfinite(on).all()):
        return None
    dst = st - (np.bincount(g, st) / cnt)[g]      # centred within each recording
    don = on - (np.bincount(g, on) / cnt)[g]
    ss = float(dst @ dst)
    if ss == 0.0 or float(don @ don) == 0.0:      # nothing varies within recordings
        return None
    a = float(dst @ don) / ss
    res = float(np.abs(don - a * dst).max())
    tol = max(_ROW_TOL[0], _ROW_TOL[1] * a)
    if a <= 0 or res > tol:
        return (f"window starts do not follow the onsets within recordings: slope "
                f"{a:.4g} samples/s, max residual {res:.1f} samples > {tol:.1f}")
    return None


class RiemannSealedModel:

    def __init__(self, parts=None, preproc=None, nfilter=4, estimator="oas",
                 use_xdawn=True, filterbank=True, slow_block=False,
                 align="subject", kind="riemann", personal="pooled",
                 blend_w=0.5, max_batches=None, adapt="none", buffer=64,
                 chans="eeg", ch_names=None, chs_info=None, ch_types=None,
                 ctx_min=CTX_MIN_WINDOWS, ch_perm=None, xblocks=""):
        self.parts, self.preproc = parts, preproc
        self.xblocks = _parse_xblocks(xblocks)    # validated early (opt-in, default none)
        self.nfilter, self.estimator = nfilter, estimator
        self.use_xdawn, self.filterbank, self.slow_block = use_xdawn, filterbank, slow_block
        self.align, self.kind, self.personal = align, kind, personal
        # align="subject_context": min training windows for a pair's own W
        self.ctx_min = int(ctx_min)
        if align == "subject_context" and adapt == "online":
            raise NotImplementedError(
                "align='subject_context' with adapt='online' is not implemented (no "
                "per-(subject, context) online re-centring; the harness has none to "
                "check it against): use align='subject' with adapt='online', or "
                "adapt='none'")
        # a number, or "auto": chosen in fit (leave-one-calibration-session-out)
        self.blend_w = "auto" if str(blend_w) == "auto" else float(blend_w)
        self.max_batches = max_batches
        # adapt="online" (RULE-DEPENDENT, transductive): each routed subject's
        # whitening reference = mean covariance of the last `buffer` test
        # windows routed to it (training reference until buffer/4 are seen).
        # State persists across predict() calls, i.e. across test batches.
        self.adapt, self.buffer = adapt, int(buffer)
        self._buf = {}
        if chans != "all" and chans not in _CHANS:
            raise ValueError(f"chans must be eeg|eeg+eog|eeg+emg|all, got {chans!r}")
        self.chans = chans
        self.ch_names, self.chs_info, self.ch_types = ch_names, chs_info, ch_types
        # scoring channel -> training channel order (``_match_channels``; set by
        # load_model when the scoring names differ from the training ones)
        self._ch_perm = (None if ch_perm is None else
                         torch.as_tensor(np.asarray(ch_perm, dtype=np.int64)))
        self._chan_idx = None
        self._pred = {"calls": 0, "windows": 0, "seconds": 0.0}
        self._warned_w = False

    def _prep(self, X, chan_idx=None, dtype=np.float64):
        X = torch.as_tensor(X, dtype=torch.float32).cpu()
        if self._ch_perm is not None:        # scoring order -> training order
            n_in = len(self.ch_names) if self.ch_names is not None else None
            if X.shape[1] != n_in:
                raise ValueError(f"[Riemann-Sealed] windows have {X.shape[1]} channels "
                                 f"but meta['ch_names'] lists {n_in}")
            X = X[:, self._ch_perm]
        if self.preproc is not None:
            with torch.no_grad():
                X = self.preproc(X)
        if chan_idx is not None:
            X = X[:, chan_idx]
        return X.numpy().astype(dtype)

    def _pick_channels(self):
        """Channel index for ``chans`` from the meta channel info (None = all)."""
        if self.ch_names is None:
            return None
        return _chan_index(_channel_types(self.ch_names, self.chs_info, self.ch_types),
                           self.chans)

    # -- routing / whitening -------------------------------------------------
    def _route(self, X):
        """Index into parts["subjects"] per window (into the pairs when the
        model was trained with align="subject_context"), -1 = fallback
        (outlier)."""
        r = self.parts["router"]
        P = r["lda"].predict_proba(_psd_features(X, self.parts["sfreq"]))
        idx = P.argmax(1)
        idx[P.max(1) < r["thr"]] = -1
        return idx

    def _whiten(self, X, idx, parts=None):
        parts = self.parts if parts is None else parts
        return _whiten_rows(X, idx, parts["W"], parts["W_global"])

    # -- training -------------------------------------------------------------
    def _collect(self, train_loader):
        """(X float32 (n, C, T), y, subject, session-or-None, context-or-None)
        from the train loader, channels already picked.

        The competition loader gives ``info["subject_id"]`` only. NeuralBench's
        dataset underneath keeps a per-window trigger table with ``session``
        (and a context column where the study has one); when it is reachable
        (local training), read the dataset in index order so session ids line
        up with the windows. Otherwise iterate the loader (shuffled) and
        return session=None. X is float32 because the preprocessing output
        is: the float64 copies made per chunk later hold the same values.

        Row-order check (a failure -> a warning and session = context = None):
        the table's subject column must map one-to-one onto the loader's
        subject ids (catches a table misordered across subjects); and, when
        the loader's info also carries record_id / onset (the mock does,
        NeuralBench's does not), ``_table_row_mismatch`` compares them with
        the table's subject / session / start row by row (catches a table
        misordered within subjects). On NeuralBench a within-subject misorder
        is not detectable here."""
        ds = getattr(train_loader, "dataset", None)
        trig = _trigger_table(ds)
        ci = self._chan_idx
        if trig is not None and "session" in trig.columns and self.max_batches is None:
            import torch.utils.data as tud
            dl = tud.DataLoader(ds, batch_size=256, shuffle=False,
                                collate_fn=getattr(train_loader, "collate_fn", None))
            X, ys, ss, i = None, [], [], 0
            ids = {"record_id": [], "onset": []}     # for the row-order check
            for Xb, yb, info in dl:
                xb = self._prep(Xb, ci, np.float32)
                if X is None:           # preallocated: no second full copy
                    X = np.empty((len(ds), *xb.shape[1:]), dtype=np.float32)
                X[i:i + len(xb)] = xb
                i += len(xb)
                ys.append(to_numpy(yb))
                ss.append(np.asarray(to_numpy(info["subject_id"])).reshape(-1))
                for k in ids:
                    try:
                        ids[k].append(np.asarray(to_numpy(info[k])).reshape(-1))
                    except Exception:    # absent (None / KeyError) or unusable
                        ids[k] = None
            X, y, subj = X[:i], np.concatenate(ys), np.concatenate(ss)
            sess = trig["session"].astype(str).to_numpy()
            if len(sess) != len(y):
                return X, y, subj, None, None
            # the table must follow the dataset's index order: its subject column
            # has to map one-to-one onto the loader's subject ids, row by row
            if "subject" in trig.columns:
                ts = trig["subject"].astype(str).to_numpy()
                pairs = set(zip(ts.tolist(), subj.tolist()))
                if not len(pairs) == len(set(ts.tolist())) == len(set(subj.tolist())):
                    print("[Riemann-Sealed] WARNING: the trigger table's subject column "
                          "does not match the loader's subject ids row by row "
                          f"({len(pairs)} (table, id) pairs for {len(set(subj.tolist()))} "
                          "subjects): session / context ids not used", flush=True)
                    return X, y, subj, None, None
            # within subjects: the loader's record_id / onset, when it has them
            why = _table_row_mismatch(
                trig, *(None if ids[k] is None else np.concatenate(ids[k])
                        for k in ("record_id", "onset")))
            if why is not None:
                print("[Riemann-Sealed] WARNING: the trigger table does not match the "
                      f"loader's record_id / onset row by row ({why}): session / "
                      "context ids not used", flush=True)
                return X, y, subj, None, None
            ctx = next((trig[c].astype(str).to_numpy() for c in _CTX_COLUMNS
                        if c in trig.columns), None)
            return X, y, subj, sess, ctx
        Xs, ys, ss = [], [], []
        for i, (Xb, yb, info) in enumerate(train_loader):
            if self.max_batches is not None and i >= self.max_batches:
                break
            Xs.append(self._prep(Xb, ci, np.float32))
            ys.append(to_numpy(yb))
            ss.append(np.asarray(to_numpy(info["subject_id"])).reshape(-1))
        return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss), None, None

    def fit(self, train_loader):
        tm, t0 = {}, time.time()
        self._chan_idx = self._pick_channels()
        X, y, subj, sess, ctx = self._collect(train_loader)
        _lap(tm, "collect", t0)
        subjects = np.unique(subj)
        sidx = np.searchsorted(subjects, subj)
        n_classes = len(np.unique(y))
        sfreq = self.preproc.sfreq
        parts = {"estimator": self.estimator, "xdawn": None, "sfreq": sfreq,
                 "subjects": subjects, "n_classes": n_classes,
                 "chan_idx": self._chan_idx,
                 # every training channel's name (before the pick): load_model
                 # matches the scoring meta["ch_names"] to them (_match_channels)
                 "ch_names": (None if self.ch_names is None
                              else [str(c) for c in self.ch_names])}

        # align="subject_context": (subject, context) pairs, numbered as the
        # harness's subj * n_ctx + ctx (subject-major, contexts sorted)
        align, pair = self.align, None
        if align == "subject_context" and ctx is None:
            print("[Riemann-Sealed] WARNING: align='subject_context' needs context ids "
                  f"(trigger-table column {' / '.join(_CTX_COLUMNS)}), which this train "
                  "loader does not expose: falling back to align='subject'", flush=True)
            align = "subject"
        if align == "subject_context":
            cnames, cidx = np.unique(np.asarray(ctx).astype(str), return_inverse=True)
            pair_ids, pidx = np.unique(sidx * len(cnames) + cidx.reshape(-1),
                                       return_inverse=True)
            pair = {"idx": pidx.reshape(-1), "subject": pair_ids // len(cnames),
                    "context": cnames[pair_ids % len(cnames)]}

        # router: log-PSD -> shrinkage LDA over subjects (over pairs under
        # subject_context); OOF threshold (1st percentile)
        t0 = time.time()
        from sklearn.model_selection import StratifiedKFold
        rlab, n_r = (sidx, len(subjects)) if pair is None else (pair["idx"], len(pair_ids))
        Fp = _psd_features(X, sfreq)
        oof = np.zeros((len(y), n_r))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(Fp, rlab):
            lda = _fit_lda(Fp[tr], rlab[tr])
            oof[np.ix_(va, lda.classes_)] = lda.predict_proba(Fp[va])
        parts["router"] = {"lda": _fit_lda(Fp, rlab),
                           # capped at 0.5: within-session OOF posteriors can saturate
                           # at 1.0 (Zhou: thr=1.000 -> every window fell back)
                           "thr": min(float(np.percentile(oof.max(1), 1)), 0.5)}
        if pair is not None:     # training-window pair accuracy (OOF), diagnostics
            parts["router"]["oof_acc"] = float(np.mean(oof.argmax(1) == rlab))
        del Fp, oof
        _lap(tm, "router", t0)

        # per-subject whitening (identity everywhere when align="none"): row i
        # of the training set is whitened with Wset[gidx[i]]
        t0 = time.time()
        if align in ("subject", "subject_context"):
            covs = _window_covs(X)
            parts["W_global"] = _inv_sqrtm(_mean_cov(covs, self.kind))
            parts["W"] = np.stack([_inv_sqrtm(_mean_cov(covs[sidx == k], self.kind))
                                   for k in range(len(subjects))])
        else:
            parts["W_global"] = np.eye(X.shape[1])
            parts["W"] = np.stack([np.eye(X.shape[1])] * len(subjects))
        gidx, Wset = sidx, parts["W"]
        if pair is not None:
            # one W per pair from its own training windows; a pair below
            # ctx_min windows keeps its subject's W (parts["W"])
            cnt = np.bincount(pair["idx"], minlength=len(pair_ids))
            small = cnt < self.ctx_min
            Wp = np.stack([parts["W"][pair["subject"][p]] if small[p] else
                           _inv_sqrtm(_mean_cov(covs[pair["idx"] == p], self.kind))
                           for p in range(len(pair_ids))])
            if small.any():
                print(f"[Riemann-Sealed] WARNING: {int(small.sum())} of {len(pair_ids)} "
                      f"(subject, context) pairs have < ctx_min={self.ctx_min} training "
                      "windows and use their subject's reference: "
                      + ", ".join(f"{subjects[pair['subject'][p]]}/{pair['context'][p]} "
                                  f"({cnt[p]})" for p in np.where(small)[0]), flush=True)
            parts.update(align="subject_context", W_pair=Wp,
                         pair_subject=pair["subject"].astype(np.int64),
                         pair_context=pair["context"], pair_n=cnt,
                         pair_fallback=small, contexts=cnames)
            gidx, Wset = pair["idx"], Wp
        if self.adapt == "online" and align == "subject" and sess is not None:
            # online test windows are centred on their own session's recent
            # statistics, so centre the training data per (subject, session)
            # too (the harness's online condition). Without session ids the
            # training data stays centred per subject, which cost 2.7 points
            # on Zhou (two training sessions) in the 2026-09-25 check.
            g = sidx * 1000 + np.unique(sess, return_inverse=True)[1]
            groups, gidx = np.unique(g, return_inverse=True)
            Wset = np.stack([_inv_sqrtm(_mean_cov(covs[g == gg], self.kind))
                             for gg in groups])
        _lap(tm, "align", t0)

        # Riemann-StepType feature union on the whitened windows: one chunked
        # pass for the xDAWN statistics, one for every per-window block
        t0 = time.time()
        n, size = len(X), _chunk_len(X.shape)
        if size >= n:               # one chunk (every proxy): whiten once
            Xw = _whiten_rows(X, gidx, Wset, parts["W_global"])

            def get(s):
                return Xw[s]
        else:
            Xw = None

            def get(s):
                return _whiten_rows(X[s], gidx[s], Wset, parts["W_global"])
        if self.use_xdawn:
            parts["xdawn"] = _fit_xdawn(self.nfilter, self.estimator, get, y, n, size)
        if self.slow_block:
            start, width = round(0.5 * sfreq), round(0.25 * sfreq)
            parts["slow"] = {"band": "0.1to4", "start": start, "width": width,
                             "n_bins": min(14, (X.shape[2] - start) // width)}
        if self.filterbank:
            parts["fb_bands"] = list(FB_BANDS)
        if self.xblocks:
            parts["xblocks"] = list(self.xblocks)
        raw = _window_blocks(parts, get, n, size)
        _fit_tangent(parts, raw, y)
        feats = _raw_features(parts, raw)
        del raw, Xw
        _lap(tm, "features", t0)

        t0 = time.time()
        priors = np.full(n_classes, 1.0 / n_classes)
        parts["lda"] = _fit_lda(feats, y, priors)
        if self.personal in ("calib", "blend"):
            parts["lda_subject"] = [
                _fit_lda(feats[sidx == k], y[sidx == k], priors)
                if len(np.unique(y[sidx == k])) == n_classes else None
                for k in range(len(subjects))]
        n_feats = feats.shape[1]
        del feats
        _lap(tm, "lda", t0)
        print(f"[Riemann-Sealed] fitting on X={X.shape}, subjects={len(subjects)}, "
              f"classes={np.unique(y).tolist()}, features={n_feats}, "
              f"align={align}/{self.kind}, personal={self.personal}, "
              f"router_thr={parts['router']['thr']:.3f}, "
              f"session_ids={'yes' if sess is not None else 'no'}, "
              f"chans={self.chans} ({X.shape[1]} kept)"
              + ("" if pair is None else
                 f", pairs={len(pair_ids)} (contexts {cnames.tolist()}; "
                 f"{int(parts['pair_fallback'].sum())} on the subject W; windows per "
                 f"pair {int(parts['pair_n'].min())}-{int(parts['pair_n'].max())}; "
                 f"router OOF pair acc {parts['router']['oof_acc']:.3f})"), flush=True)

        t0 = time.time()
        if self.personal == "blend" and self.blend_w == "auto":
            if sess is None:
                parts["blend_w"] = 0.5
                print("[Riemann-Sealed] WARNING: blend_w='auto' needs session ids, "
                      "which this train loader does not expose; using w=0.5", flush=True)
            else:
                # overwrites X in place (float32 harness-style whitening)
                w, cv, how, nf = self._choose_blend_w(parts, X, y, sidx, sess, ctx,
                                                      gidx, Wset)
                parts.update(blend_w=w, blend_cv=cv, blend_cv_folds=how)
                print(f"[Riemann-Sealed] blend_w=auto -> {w} ({how}, {nf} folds, "
                      f"context={'yes' if ctx is not None else 'no'}"
                      + ("" if pair is None else ", pair-whitened") + "; cell scores "
                      f"{np.round(cv, 4).tolist()} for w={W_GRID})", flush=True)
        _lap(tm, "wcv", t0)
        print("[Riemann-Sealed] fit seconds: "
              + ", ".join(f"{k} {v:.1f}" for k, v in tm.items())
              + f", total {sum(tm.values()):.1f}; chunk={size}/{n} windows; "
              f"maxrss {_maxrss_gb():.2f} GB", flush=True)
        self.parts = parts
        return self

    def _choose_blend_w(self, parts, X, y, sidx, sess, ctx, gidx, Wset):
        """blend_w="auto": (w, cell scores per W_GRID, fold kind, n folds).

        The harness's choose_w (sealed_personal.py) under ``--wcv loso``, on
        the harness's aligned data: every training window whitened with the
        whole training set's reference of its group (the true subject, the
        (subject, context) pair under align="subject_context" as the harness's
        router-psdctx, or (subject, session) under adapt="online"), in float32 as
        ``xsess_lib.apply_W``. X is overwritten in place with it. The
        stateless blocks (broadband / filter-bank covariances, log-variance,
        slow bins) are computed once; xDAWN, the tangent spaces, the pooled
        LDA and the per-subject LDAs are refitted on each fold's fit rows."""
        n, size = len(X), _chunk_len(X.shape)
        for s in _row_chunks(n, size):
            X[s] = _whiten_rows(X[s], gidx[s], Wset, parts["W_global"], f32=True)

        def get(s):
            return np.asarray(X[s], dtype=np.float64)
        rank = _session_rank(sidx, sess)
        cc = (np.zeros(n, dtype=np.int64) if ctx is None
              else np.unique(ctx, return_inverse=True)[1])
        folds, how = _blend_folds(sidx, rank)
        K = int(parts["n_classes"])
        base = {k: parts[k] for k in ("estimator", "sfreq", "slow", "fb_bands", "xblocks")
                if k in parts}
        base["xdawn"] = None
        raw = _window_blocks(base, get, n, size)
        scores = np.zeros(len(W_GRID))
        for i, (fit_idx, val_idx) in enumerate(folds):
            t0 = time.time()
            fp = dict(base)
            if self.use_xdawn:
                fp["xdawn"] = _fit_xdawn(self.nfilter, self.estimator, get, y, fit_idx, size)
                raw["xdawn"] = _window_blocks(fp, get, n, size, which=("xdawn",))["xdawn"]
            _fit_tangent(fp, raw, y, fit_idx)
            F_fit, F_val = _raw_features(fp, raw, fit_idx), _raw_features(fp, raw, val_idx)
            yf = y[fit_idx]
            kf = len(np.unique(yf))
            P_pool = _fit_lda(F_fit, yf, np.full(kf, 1.0 / kf)).predict_proba(F_val)
            P_own = np.array(P_pool)
            for k in np.unique(sidx[fit_idx]):
                mf, rows = sidx[fit_idx] == k, sidx[val_idx] == k
                if not rows.any() or len(np.unique(yf[mf])) < K:
                    continue
                P_own[rows] = _fit_lda(F_fit[mf], yf[mf],
                                       np.full(K, 1.0 / K)).predict_proba(F_val[rows])
            for j, w in enumerate(W_GRID):
                Pb = w * P_pool + (1 - w) * P_own
                scores[j] += _cell_score(y[val_idx], Pb.argmax(1), sidx[val_idx],
                                         rank[val_idx], cc[val_idx])
            if time.time() - t0 >= _HEARTBEAT_S:
                print(f"[Riemann-Sealed] [{time.strftime('%H:%M:%S')}] blend_w fold "
                      f"{i + 1}/{len(folds)} ({len(fit_idx)} fit / {len(val_idx)} val "
                      f"windows) in {time.time() - t0:.0f} s", flush=True)
        scores = scores / len(folds)
        # ties -> the larger (more pooled) w, as sealed_personal.choose_w
        j = max(range(len(W_GRID)), key=lambda i: (round(scores[i], 6), W_GRID[i]))
        return W_GRID[j], scores.tolist(), how, len(folds)

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

    def _blend_weight(self):
        """The pooled weight: the one chosen in fit (joblib), else blend_w."""
        w = self.parts.get("blend_w", self.blend_w)
        if isinstance(w, str):          # "auto", but the joblib holds no weight
            if not self._warned_w:
                print("[Riemann-Sealed] WARNING: blend_w='auto' but the model has "
                      "no chosen weight; using w=0.5", flush=True)
                self._warned_w = True
            return 0.5
        return float(w)

    def _predict_block(self, X):
        idx = self._route(X)
        if self.parts.get("align") == "subject_context":
            # routed pair -> that pair's W (fallback: global W), then its subject
            if self.adapt == "online":
                raise NotImplementedError(
                    "adapt='online' with a model trained under align='subject_context' "
                    "is not implemented (no per-pair online re-centring)")
            Xw = _whiten_rows(X, idx, self.parts["W_pair"], self.parts["W_global"])
            sub = np.full(len(idx), -1, dtype=np.int64)
            ok = idx >= 0
            sub[ok] = self.parts["pair_subject"][idx[ok]]
            idx = sub
        else:
            Xw = (self._online_whiten(X, idx) if self.adapt == "online"
                  else self._whiten(X, idx))
        F_ = _features(self.parts, Xw)
        P = self.parts["lda"].predict_proba(F_)
        if self.personal in ("calib", "blend"):
            w = self._blend_weight()
            for k in np.unique(idx[idx >= 0]):
                lda = self.parts["lda_subject"][k]
                if lda is None:
                    continue
                m = idx == k
                Ps = lda.predict_proba(F_[m])
                P[m] = Ps if self.personal == "calib" else (
                    w * P[m] + (1 - w) * Ps)
        return P

    def predict_proba(self, X):
        """(B, n_classes) probabilities. Windows are independent unless
        adapt="online", so a batch larger than one chunk is split (memory is
        bounded by the chunk); online batches are kept whole (the buffer
        advances once per call)."""
        t0 = time.time()
        ci = self.parts.get("chan_idx")
        n = len(X)
        c = X.shape[1] if self._ch_perm is None else len(self._ch_perm)
        shape = (n, c if ci is None else len(ci), X.shape[-1])
        size = n if self.adapt == "online" else _chunk_len(shape)
        chunks = _row_chunks(n, max(size, 1)) or [slice(0, n)]
        Ps = [self._predict_block(self._prep(X[s], ci)) for s in chunks]
        P = Ps[0] if len(Ps) == 1 else np.concatenate(Ps)
        st = self._pred
        st["calls"] += 1
        st["windows"] += n
        st["seconds"] += time.time() - t0
        if st["calls"] % 20 == 0:
            print(f"[Riemann-Sealed] predict: {st['calls']} calls, {st['windows']} "
                  f"windows, {st['seconds']:.1f} s "
                  f"({1000 * st['seconds'] / max(st['windows'], 1):.1f} ms/window), "
                  f"maxrss {_maxrss_gb():.2f} GB", flush=True)
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
        # extra temporal blocks, e.g. "bpt4" or "tseg3_bpt4" (sprint 2026-09-29;
        # "" = none, the recipe)
        "xblocks": [""],
        # "subject" (router + per-subject W) | "none" | "subject_context"
        # (router + W per (subject, context) pair: the harness's router-psdctx)
        "align": ["subject"],
        "ctx_min": [CTX_MIN_WINDOWS],   # subject_context: min windows for a pair's own W
        "kind": ["riemann"],        # reference mean: "riemann" | "euclid"
        # the recipe: "blend" = blend_calib (SEALED_RECIPE.md): pooled feature
        # extractor + per-subject LDA. blend_w is set per dataset from
        # calibration data by scripts/train_sealed.sh, or "auto" = chosen in
        # fit by leave-one-calibration-session-out (needs session ids)
        "personal": ["blend"],      # "pooled" | "calib" | "blend"
        "blend_w": [0.5],           # pooled weight when personal="blend"
        # "online" = rule-dependent test-time re-centring (see RiemannSealedModel)
        "adapt": ["none"],
        "buffer": [64],
        "bandpass": ["none"],
        "reference": ["none"],
        # channel types kept: "eeg" | "eeg+eog" | "eeg+emg" | "all"
        "chans": ["eeg"],
        "max_batches": [None],
    }

    def load_model(self, meta):
        weights = meta["submission_dir"] / "riemann_sealed.joblib"
        parts = (joblib.load(weights)
                 if weights.exists() and self.train_loader is None else None)
        # scoring channels vs the trained ones, by name (see _match_channels)
        ch_names, perm = meta.get("ch_names"), None
        names = ch_names
        if parts is not None:
            perm, names = _match_channels(parts.get("ch_names"), ch_names)
        preproc = WindowPreproc(names, meta["sfreq"],
                                self.bandpass, self.reference)
        return RiemannSealedModel(parts, preproc, self.nfilter, self.estimator,
                                  self.use_xdawn, self.filterbank, self.slow_block,
                                  self.align, self.kind, self.personal,
                                  self.blend_w, self.max_batches,
                                  self.adapt, self.buffer, chans=self.chans,
                                  ch_names=ch_names,
                                  chs_info=meta.get("chs_info"),
                                  ch_types=meta.get("ch_types"),
                                  ctx_min=self.ctx_min, ch_perm=perm,
                                  xblocks=self.xblocks)

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        joblib.dump(model.parts, path / "riemann_sealed.joblib")
