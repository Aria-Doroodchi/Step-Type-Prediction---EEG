"""Extra temporal feature blocks for the harness (see xfeat.py for the interface;
sprint 2026-09-29, brief prompts/2026-09-29_temporal_spatial_features.md).

Every covariance in the recipe union spans the whole 4 s window, so the only
temporal structure it sees is xDAWN's class templates. These blocks add it back:

    tseg<K>     FB4 covariances of K contiguous time segments -> TS each
                (how the spatial pattern evolves: cue response vs sustained task)
    acm<p>x<L>  augmented (time-delay-embedded) broadband covariance, order p,
                lag L samples at 120 Hz -> TS (Carrara & Papadopoulo)
    fb8         8-band filter-bank covariances -> TS (meant to REPLACE the base
                4-band FB: ``blocks=xdawn+broad+logvar,x=fb8``)
    fbd         the 1-4 Hz band covariance -> TS (adds delta to the base FB)
    bpt<K>      FB4 per-channel log-power in K time bins (ERD/ERS time course)
    tcut<ms>    FB4 covariances of [0, ms) and [ms, end) -> TS each (EDA-motivated:
                the cue second vs the sustained task period)
    fbfrom<ms>  FB4 covariances of [ms, end) only -> TS (control: the FB without
                the cue second; replaces the base FB)

Feature counts are given per block as a formula in C (channels); at 14 / 30 /
47 ch, C(C+1)/2 = 105 / 465 / 1128.

All blocks see the preprocessed (float64, possibly whitened) windows. Anything
with state (the tangent-space reference means) is fitted in ``fit`` on the
training windows only; ``transform`` is a per-window map given that state.
"""

import numpy as np

from xfeat import FB4, Block, band_pass, covs, lag_samples, register, tangent_space

FB8 = ["1to4", "4to8", "8to10", "10to13", "13to18", "18to25", "25to30", "30to45"]


# ---------------------------------------------------------------------------
# shared plumbing
# ---------------------------------------------------------------------------
def _segments(X, k):
    """K equal contiguous non-overlapping pieces of the last (time) axis; the
    T % K remainder samples at the end are dropped."""
    w = X.shape[-1] // k
    if w < 2:
        raise ValueError(f"{k} segments of a {X.shape[-1]}-sample window are too short")
    return [X[..., i * w:(i + 1) * w] for i in range(k)]


class _CovTS(Block):
    """A block made of several OAS covariance stacks, each mapped to its own
    tangent space. Subclasses define ``_covsets(X) -> [(n, c, c), ...]`` (a
    fixed number of stacks, in a fixed order); ``fit`` fits one TangentSpace
    (Riemannian mean reference) per stack on the training windows and
    ``transform`` projects each stack at its fitted reference and concatenates.
    """

    def _covsets(self, X):
        raise NotImplementedError

    def fit(self, X, y):
        self.ts = [tangent_space().fit(C) for C in self._covsets(X)]
        return self

    def transform(self, X):
        sets = self._covsets(X)
        return np.concatenate([ts.transform(C) for ts, C in zip(self.ts, sets)],
                              axis=1)


# ---------------------------------------------------------------------------
# tseg<K>: filter-bank covariances of K time segments
# ---------------------------------------------------------------------------
class TimeSegCov(_CovTS):
    """For each FB4 band: band-pass the whole window (so the segment cuts add
    no filter edge effects), split it into K equal contiguous segments, OAS
    covariance per segment; one TangentSpace per (band, segment).

    Features: K * 4 * C(C+1)/2, ordered band-major then segment (tseg2 at 30 ch
    = 3720; tseg3 at 47 ch = 13536). Short segments carry few degrees of
    freedom (measured with white noise through this band-pass, at any sfreq:
    4-8 Hz over 4/3 s ~ 16, 8-13 Hz ~ 20), so those segment covariances are
    rank-deficient already at 30 ch; the OAS shrinkage keeps them SPD.
    """

    def __init__(self, meta, k):
        super().__init__(meta)
        if k < 1:
            raise ValueError(f"tseg needs K >= 1, got {k}")
        self.k = k

    def _covsets(self, X):
        out = []
        for band in FB4:
            Xb = band_pass(X, self.sfreq, band)
            out += [covs(seg) for seg in _segments(Xb, self.k)]
        return out


@register(r"tseg(\d+)")
def _tseg(meta, k):
    return TimeSegCov(meta, int(k))


class CutSegCov(_CovTS):
    """EDA-motivated (LOG 2026-09-29 18:45): FB4 covariances of the window cut
    at ``ms`` milliseconds (scaled with sfreq) -> TS per (band, piece).

    ``keep="both"`` (tcut<ms>): the pieces [0, cut) and [cut, T), i.e. the cue
    second vs the sustained task period; 2 * 4 * C(C+1)/2 features.
    ``keep="after"`` (fbfrom<ms>): only [cut, T), the recipe's FB without the
    cue second (a control, meant to REPLACE the base FB:
    ``blocks=xdawn+broad+logvar,x=fbfrom1000``); 4 * C(C+1)/2 features.
    The whole window is band-passed before the cut, as in tseg.
    """

    def __init__(self, meta, ms, keep):
        super().__init__(meta)
        self.cut = int(round(float(ms) / 1000.0 * self.sfreq))
        self.keep = keep

    def _covsets(self, X):
        if not 2 <= self.cut <= X.shape[-1] - 2:
            raise ValueError(f"cut at {self.cut} samples of a {X.shape[-1]}-sample window")
        out = []
        for band in FB4:
            Xb = band_pass(X, self.sfreq, band)
            if self.keep == "both":
                out.append(covs(Xb[..., :self.cut]))
            out.append(covs(Xb[..., self.cut:]))
        return out


@register(r"tcut(\d+)")
def _tcut(meta, ms):
    return CutSegCov(meta, int(ms), "both")


@register(r"fbfrom(\d+)")
def _fbfrom(meta, ms):
    return CutSegCov(meta, int(ms), "after")


# ---------------------------------------------------------------------------
# acm<p>x<L>: augmented covariance (time-delay embedding)
# ---------------------------------------------------------------------------
def _delay_embed(X, p, lag):
    """(n, C, T) -> (n, C * p, T - (p - 1) * lag): row block d is the signal
    delayed by d * lag samples, x(t - d * lag), over the common valid range
    t in [(p - 1) * lag, T)."""
    T2 = X.shape[-1] - (p - 1) * lag
    if T2 < 2:
        raise ValueError(f"acm order {p} x lag {lag} leaves {T2} samples of {X.shape[-1]}")
    s = (p - 1) * lag
    return np.concatenate([X[..., s - d * lag:s - d * lag + T2] for d in range(p)],
                          axis=1)


class AugmentedCov(_CovTS):
    """Augmented covariance matrix (Carrara & Papadopoulo): the BROADBAND
    window (no extra band-pass) stacked with p - 1 delayed copies of itself,
    at multiples of ``lag_samples(sfreq, L)`` (L samples at 120 Hz, scaled so
    the lag in seconds is fixed); OAS covariance of the (C p, T') signal ->
    TangentSpace. Its off-diagonal blocks are lagged auto- and
    cross-covariances: temporal structure and lagged coupling that the
    zero-lag covariance cannot represent.

    Features: Cp(Cp+1)/2 (acm3x2 at 30 ch: 4095; at 47 ch: 10011).
    """

    def __init__(self, meta, p, lag_at_120):
        super().__init__(meta)
        if p < 1 or lag_at_120 < 1:
            raise ValueError(f"acm needs p >= 1 and L >= 1, got {p}x{lag_at_120}")
        self.p = p
        self.lag = lag_samples(self.sfreq, lag_at_120)

    def _covsets(self, X):
        return [covs(_delay_embed(X, self.p, self.lag))]


@register(r"acm(\d+)x(\d+)")
def _acm(meta, p, lag):
    return AugmentedCov(meta, int(p), int(lag))


# ---------------------------------------------------------------------------
# fb8 / fbd: filter-bank covariances on other band sets
# ---------------------------------------------------------------------------
class FilterBankCov(_CovTS):
    """For each band: band-pass, OAS covariance -> one TangentSpace per band.

    Features: len(bands) * C(C+1)/2. fb8 (all 8 bands, so it can replace the
    base FB) = 8 * C(C+1)/2 (3720 at 30 ch); fbd (1-4 Hz only) = C(C+1)/2.
    Narrow bands (1-4, 8-10 Hz) span ~32 degrees of freedom over 4 s
    (measured, any sfreq), so from ~32 ch on the OAS shrinkage is what keeps
    their covariances SPD.
    """

    def __init__(self, meta, bands):
        super().__init__(meta)
        self.bands = list(bands)

    def _covsets(self, X):
        return [covs(band_pass(X, self.sfreq, band)) for band in self.bands]


@register(r"fb8")
def _fb8(meta):
    return FilterBankCov(meta, FB8)


@register(r"fbd")
def _fbd(meta):
    return FilterBankCov(meta, ["1to4"])


# ---------------------------------------------------------------------------
# bpt<K>: band-power time course
# ---------------------------------------------------------------------------
class BandPowerTime(Block):
    """For each FB4 band: band-pass the whole window, square, mean over K
    equal contiguous time bins per channel, log. Stateless (nothing fitted):
    an explicit ERD/ERS time course per band x channel.

    Features: 4 * C * K, ordered band, bin, channel (bpt4 at 30 ch = 480).
    """

    def __init__(self, meta, k):
        super().__init__(meta)
        if k < 1:
            raise ValueError(f"bpt needs K >= 1, got {k}")
        self.k = k

    def transform(self, X):
        out = []
        for band in FB4:
            P = band_pass(X, self.sfreq, band) ** 2
            out += [np.log(seg.mean(axis=2) + 1e-12) for seg in _segments(P, self.k)]
        return np.concatenate(out, axis=1)


@register(r"bpt(\d+)")
def _bpt(meta, k):
    return BandPowerTime(meta, int(k))
