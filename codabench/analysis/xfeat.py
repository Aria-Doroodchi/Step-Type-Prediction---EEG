"""Extra feature blocks for Riemann-StepType in the cross-session harness
(sprint 2026-09-29, brief ``prompts/2026-09-29_temporal_spatial_features.md``).

Harness spec: ``riemann:xd=1,fb=1,x=<id>[+<id>...]`` (``blocks=`` still
restricts the *base* union, e.g. ``blocks=xdawn+broad+logvar,x=fb8`` swaps the
4-band filter bank for the 8-band one). A spec without ``x=`` never reaches
this module: ``xsess_lib.make_model`` builds the plain ``RiemannModel``, so
every committed number keeps its exact code path.

A block is fitted on the training windows only and maps (n, C, T) float64
windows -- the same preprocessed (and, under router alignment, whitened)
windows the base union sees -- to an (n, d) float64 feature matrix.

    class MyBlock(Block):
        def fit(self, X, y): ...; return self
        def transform(self, X): return F

Blocks register a regex over their id; the factory gets ``meta`` (ch_names,
sfreq, ...) and the regex groups::

    @register(r"tseg(\\d+)")
    def _tseg(meta, k):
        return TimeSegCov(meta, int(k))

Block families live in ``xfeat_temporal.py`` and ``xfeat_spatial.py``;
``make_block`` imports them on first use.
"""

import re
import sys
from pathlib import Path

import numpy as np

HOME = Path.home()
sys.path.insert(0, str(HOME / "codabench/solvers/bci_decoding"))

FB4 = ["4to8", "8to13", "13to30", "30to45"]      # riemann_steptype.FB_BANDS
_REGISTRY = []                                     # (compiled regex, factory)


def register(pattern):
    """Decorator: ``factory(meta, *groups)`` builds the block whose id fully
    matches ``pattern``."""
    rx = re.compile(pattern)

    def deco(factory):
        _REGISTRY.append((rx, factory))
        return factory
    return deco


def _load_families():
    import xfeat_spatial  # noqa: F401  (registers on import)
    import xfeat_temporal  # noqa: F401


def make_block(name, meta):
    _load_families()
    for rx, factory in _REGISTRY:
        m = rx.fullmatch(name)
        if m:
            blk = factory(meta, *m.groups())
            blk.name = name
            return blk
    raise ValueError(f"unknown feature block {name!r}; known: "
                     f"{[rx.pattern for rx, _ in _REGISTRY]}")


class Block:
    """Base class. ``meta``: dict with at least ch_names, sfreq."""
    name = "?"

    def __init__(self, meta):
        self.meta = meta
        self.sfreq = float(meta["sfreq"])
        self.ch_names = list(meta["ch_names"])

    def fit(self, X, y):
        return self

    def transform(self, X):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# shared helpers (the solver's own filters and estimators)
# ---------------------------------------------------------------------------
def band_pass(X, sfreq, band):
    """(n, C, T) float64 band-passed with the solver's FFT mask
    (riemann_steptype._band_chunks: WindowPreproc, no re-reference)."""
    import riemann_steptype as RS
    return np.concatenate(list(RS._band_chunks(X, sfreq, band)))


def covs(X, estimator="oas"):
    from pyriemann.estimation import Covariances
    return Covariances(estimator=estimator).transform(X)


def tangent_space():
    from pyriemann.tangentspace import TangentSpace
    return TangentSpace(metric="riemann")


def lag_samples(sfreq, lag_at_120):
    """A lag given in samples at 120 Hz, scaled to ``sfreq`` (>= 1)."""
    return max(1, int(round(lag_at_120 * float(sfreq) / 120.0)))


# ---------------------------------------------------------------------------
# harness model
# ---------------------------------------------------------------------------
class RiemannXModel:
    """Base Riemann-StepType union (``xsess_lib.RiemannModel``, optionally
    restricted by ``blocks``) + extra blocks -> one shrinkage LDA (uniform
    priors), refitted on the full union."""
    name = "RiemannX"

    def __init__(self, meta, xblocks, **base_kw):
        import xsess_lib as L
        self.base = L.RiemannModel(meta, **base_kw)
        self.meta = meta
        self.xnames = tuple(xblocks)

    def _prep64(self, X):
        return self.base.model._prep(X, np.float32).astype(np.float64)

    def fit(self, X, y, **_):
        import riemann_steptype as RS
        self.base.fit(X, y)
        Xp = self._prep64(X)
        self.xb = [make_block(n, self.meta).fit(Xp, y) for n in self.xnames]
        parts = self._parts(X, Xp)
        F = np.concatenate(parts, axis=1)
        k = len(np.unique(y))
        self.lda = RS.fit_shrinkage_lda(F, y, np.full(k, 1 / k))
        sizes = " + ".join(f"{b.name} {f.shape[1]}" for b, f in zip(self.xb, parts[1:]))
        print(f"[RiemannX] X={X.shape} features={F.shape[1]} "
              f"(base {parts[0].shape[1]} + {sizes})", flush=True)
        return self

    def _parts(self, X, Xp=None):
        """[base union, extra block 1, ...] feature matrices."""
        Xp = self._prep64(X) if Xp is None else Xp
        out = [self.base._feats(X)]
        for b in self.xb:
            f = np.asarray(b.transform(Xp), dtype=np.float64)
            if f.ndim != 2 or len(f) != len(X) or not np.all(np.isfinite(f)):
                raise FloatingPointError(f"block {b.name}: bad features {f.shape}")
            out.append(f)
        return out

    def _feats(self, X, Xp=None):
        return np.concatenate(self._parts(X, Xp), axis=1)

    def predict_proba(self, X):
        return self.lda.predict_proba(self._feats(X))

    def features(self, X):
        return self._feats(X)
