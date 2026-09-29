"""Extra spatial feature blocks for the harness (see xfeat.py for the interface;
sprint 2026-09-29, brief prompts/2026-09-29_temporal_spatial_features.md).

TS + shrinkage LDA is (nearly) invariant to a fixed full-rank spatial filter
(re-referencing, Laplacian), so these blocks change the *structure* of the
spatial representation instead (brief section 1):

    fblv    per FB4 band x channel log-variance (band-power topography)  4*C
    fbrlv   the same minus the channel's broadband log-variance           4*C
    reg     regional covariances (3x3 scalp grid by 10-05 position) -> TS
                                                   4 * sum_r c_r(c_r+1)/2
    csp<k>  per band, k multi-class CSP filters -> k x k covariances -> TS
                                                   4 * k(k+1)/2, k <= C
    icoh    per band imaginary coherence, upper triangle (signed)   4*C(C-1)/2

(C = channels, c_r = channels in region r.) Everything with state (tangent-space
reference means, CSP filters) is fitted on the training windows in ``fit``;
``transform`` is a per-window map given that state.
"""

import numpy as np

from xfeat import FB4, Block, band_pass, covs, register, tangent_space

_CHUNK_BYTES = 256 * 2 ** 20    # float64 bytes of windows per chunk (43-47 ch x 500 Hz)


def _chunks(X, row_bytes=None):
    """Row slices of X so that each chunk's per-window working copy (default:
    the (C, T) float64 window itself) stays under _CHUNK_BYTES."""
    per = row_bytes or 8 * int(np.prod(X.shape[1:]))
    size = max(1, _CHUNK_BYTES // per)
    return [slice(i, min(i + size, len(X))) for i in range(0, len(X), size)]


def _band_covs(X, sfreq, groups=None):
    """OAS covariances of the band-passed windows, band-major: for each FB4
    band and each channel group in ``groups`` (default: all channels) one
    (n, c, c) array. Band-passed chunk by chunk to bound memory."""
    groups = [slice(None)] if groups is None else groups
    parts = []
    for s in _chunks(X):
        row = []
        for band in FB4:
            xb = band_pass(X[s], sfreq, band)
            row += [covs(xb[:, g]) for g in groups]
        parts.append(row)
    return [np.concatenate(p) for p in zip(*parts)]


# ---------------------------------------------------------------------------
# fblv / fbrlv: explicit band-power topographies
# ---------------------------------------------------------------------------
class BandLogVar(Block):
    """Per FB4 band, per-channel log-variance of the band-passed window,
    band-major. ``relative``: minus the channel's broadband log-variance (the
    window as given), i.e. the fraction of the channel's power in the band.
    No fitting. Features: 4*C."""

    def __init__(self, meta, relative=False):
        super().__init__(meta)
        self.relative = relative

    def _feats(self, X):
        out = [np.log(np.var(band_pass(X, self.sfreq, b), axis=2) + 1e-12) for b in FB4]
        if self.relative:
            broad = np.log(np.var(X, axis=2) + 1e-12)
            out = [o - broad for o in out]
        return np.concatenate(out, axis=1)

    def transform(self, X):
        return np.concatenate([self._feats(X[s]) for s in _chunks(X)])


@register(r"fblv")
def _fblv(meta):
    return BandLogVar(meta, relative=False)


@register(r"fbrlv")
def _fbrlv(meta):
    return BandLogVar(meta, relative=True)


# ---------------------------------------------------------------------------
# reg: regional covariances
# ---------------------------------------------------------------------------
# Old 10-20 names -> 10-10 names (Scherer 2015 and Zyma 2019 use T3/T4/T5/T6).
_ALIASES = {"t3": "t7", "t4": "t8", "t5": "p7", "t6": "p8"}
# Grid cut points in MNE's standard_1005 head coordinates (m; x to the right,
# y to the nose). |x| < 0.02 is midline: the 1/2-columns (x ~ +-0.035) are
# lateral, the 10-05 "h" positions next to the midline (|x| ~ 0.015-0.018)
# count as midline. Rows (y of the standard rows): Fp/AF/F ~ +0.042..+0.088,
# FC/FT ~ +0.014..+0.027, C/T7/T8 ~ -0.009..-0.016, CP/TP ~ -0.046..-0.047,
# P ~ -0.073..-0.081, PO/O < -0.097. Anterior = y > +0.035 (Fp, AF and F rows,
# incl. F7/F8), posterior = y < -0.060 (P, PO, O rows, incl. P7/P8), central =
# the FC, C and CP rows in between (the sensorimotor strip, incl. T7/T8).
_MIDLINE_X = 0.02
_Y_ANTERIOR, _Y_POSTERIOR = 0.035, -0.060
_ROWS, _COLS = ("ant", "cen", "post"), ("L", "M", "R")


def scalp_regions(ch_names, min_size=3, min_positioned=6):
    """[(region name, [channel indices])] from the channel NAMES only.

    Positions: MNE standard_1005 (case-insensitive, old 10-20 aliases);
    channels without a position (EMG/EOG/ECG, "A2-A1", ...) are left out.
    3x3 grid (anterior/central/posterior x left/midline/right, cut points
    above), empty cells dropped. Then, while some region has < ``min_size``
    channels, the smallest one (ties: grid order) is merged into the region
    with the nearest centroid (3-D Euclidean). With < ``min_positioned``
    positioned channels: a single region.
    """
    import mne
    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    pos = {k.lower(): np.asarray(v) for k, v in pos.items()}
    idx, xyz = [], []
    for i, c in enumerate(ch_names):
        key = c.strip().lower()
        key = _ALIASES.get(key, key)
        if key in pos:
            idx.append(i)
            xyz.append(pos[key])
    if not idx:
        raise ValueError(f"reg: no channel with a standard_1005 position in {ch_names}")
    xyz = np.array(xyz)
    if len(idx) < min_positioned:
        return [("all", idx)]
    row = np.where(xyz[:, 1] > _Y_ANTERIOR, 0, np.where(xyz[:, 1] < _Y_POSTERIOR, 2, 1))
    col = np.where(np.abs(xyz[:, 0]) < _MIDLINE_X, 1, np.where(xyz[:, 0] < 0, 0, 2))
    cell = 3 * row + col
    # regions as (name, positions into idx/xyz), in grid order
    regs = [(f"{_ROWS[g // 3]}-{_COLS[g % 3]}", list(np.flatnonzero(cell == g)))
            for g in range(9) if np.any(cell == g)]
    while len(regs) > 1 and min(len(r) for _, r in regs) < min_size:
        i = int(np.argmin([len(r) for _, r in regs]))          # first smallest
        cent = np.array([xyz[r].mean(axis=0) for _, r in regs])
        dist = np.linalg.norm(cent - cent[i], axis=1)
        dist[i] = np.inf
        j = int(np.argmin(dist))
        regs[j] = (f"{regs[j][0]}+{regs[i][0]}", sorted(regs[j][1] + regs[i][1]))
        del regs[i]
    return [(name, [idx[k] for k in r]) for name, r in regs]


_REG_PRINTED = set()    # channel layouts whose regions were already logged


class RegionalCov(Block):
    """Per scalp region and FB4 band: OAS covariance of the region's channels
    -> tangent space (reference = training mean). Drops the cross-region
    terms and shrinks each TS to the region's dimension. Features:
    4 * sum_r c_r(c_r+1)/2."""

    def __init__(self, meta):
        super().__init__(meta)
        self.regions = scalp_regions(self.ch_names)
        self.groups = [idx for _, idx in self.regions]

    def fit(self, X, y):
        key = tuple(self.ch_names)
        if key not in _REG_PRINTED:
            _REG_PRINTED.add(key)
            desc = "; ".join(f"{name}: {','.join(self.ch_names[i] for i in idx)}"
                             for name, idx in self.regions)
            print(f"[reg] {len(self.regions)} regions ({sum(map(len, self.groups))}"
                  f"/{len(self.ch_names)} ch positioned): {desc}", flush=True)
        self.ts = [tangent_space().fit(C) for C in _band_covs(X, self.sfreq, self.groups)]
        return self

    def transform(self, X):
        return np.concatenate([ts.transform(C) for ts, C in
                               zip(self.ts, _band_covs(X, self.sfreq, self.groups))],
                              axis=1)


@register(r"reg")
def _reg(meta):
    return RegionalCov(meta)


# ---------------------------------------------------------------------------
# csp<k>: supervised low-rank spatial subspace
# ---------------------------------------------------------------------------
class BandCSP(Block):
    """Per FB4 band: OAS covariances -> k CSP filters (pyriemann, Riemannian
    class means; > 2 classes: AJD of the class means (ajd_pham), filters ranked
    by the approximate mutual information -- works with 3-5 classes in
    pyriemann 0.12) fitted on the training covariances and labels -> the k x k
    filtered covariances -> tangent space fitted on the training ones.
    k = min(k, C). Features: 4 * k(k+1)/2."""

    def __init__(self, meta, k):
        super().__init__(meta)
        self.k = min(int(k), len(self.ch_names))

    def fit(self, X, y):
        from pyriemann.spatialfilters import CSP
        self.csp, self.ts = [], []
        for C in _band_covs(X, self.sfreq):
            csp = CSP(nfilter=self.k, metric="riemann", log=False).fit(C, np.asarray(y))
            self.csp.append(csp)
            self.ts.append(tangent_space().fit(csp.transform(C)))
        return self

    def transform(self, X):
        return np.concatenate([ts.transform(csp.transform(C)) for csp, ts, C in
                               zip(self.csp, self.ts, _band_covs(X, self.sfreq))],
                              axis=1)


@register(r"csp(\d+)")
def _csp(meta, k):
    return BandCSP(meta, int(k))


# ---------------------------------------------------------------------------
# icoh: lagged connectivity
# ---------------------------------------------------------------------------
class ImagCoherence(Block):
    """Per FB4 band, per window: imaginary coherence of every channel pair
    (i < j, signed). Welch cross-spectra within the window: 1 s Hann segments
    (round(sfreq) samples, periodic Hann as scipy.signal.csd), 50 % overlap,
    constant detrend per segment; S_ij = mean over segments and over the
    frequency bins in [lo, hi] Hz of conj(Z_i) Z_j (scipy.signal.csd's sign
    convention); icoh_ij = Im(S_ij) / sqrt(S_ii S_jj). The band is selected
    in the frequency domain, so the window is not band-passed first (with
    1 Hz bins the FFT mask would leave the in-band bins almost unchanged).
    Zero-lag (volume-conducted) coupling has no imaginary part. No fitting.
    Features: 4 * C(C-1)/2."""

    def __init__(self, meta):
        super().__init__(meta)
        self.nper = int(round(self.sfreq))

    def _feats(self, X, nper, step, win, sel, iu):
        from numpy.lib.stride_tricks import sliding_window_view
        seg = sliding_window_view(X, nper, axis=2)[:, :, ::step]     # (n, C, S, nper)
        seg = seg - seg.mean(axis=-1, keepdims=True)
        Z = np.fft.rfft(seg * win, axis=-1)                          # (n, C, S, F)
        out = []
        for m in sel:
            Zb = Z[..., m].reshape(len(X), X.shape[1], -1)           # (n, C, S*Fb)
            S = np.conj(Zb) @ Zb.transpose(0, 2, 1)                  # sum conj(Z_i) Z_j
            p = np.real(np.einsum("nii->ni", S))
            den = np.sqrt(p[:, :, None] * p[:, None, :])
            ic = np.divide(S.imag, den, out=np.zeros_like(den), where=den > 0)
            out.append(ic[:, iu[0], iu[1]])
        return np.concatenate(out, axis=1)

    def transform(self, X):
        from scipy.signal import get_window
        n, C, T = X.shape
        nper = min(self.nper, T)
        step = max(1, nper // 2)
        win = get_window("hann", nper)
        f = np.fft.rfftfreq(nper, 1.0 / self.sfreq)
        sel = []
        for band in FB4:
            lo, hi = (float(v) for v in band.split("to"))
            sel.append(np.flatnonzero((f >= lo) & (f <= hi)))
        iu = np.triu_indices(C, 1)
        n_seg = (T - nper) // step + 1
        seg_bytes = 8 * C * n_seg * nper * 3        # segments + windowed copy + spectrum
        return np.concatenate([self._feats(X[s], nper, step, win, sel, iu)
                               for s in _chunks(X, seg_bytes)])


@register(r"icoh")
def _icoh(meta):
    return ImagCoherence(meta)
