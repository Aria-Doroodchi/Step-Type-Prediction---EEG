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
  methods need all windows); ``max_batches`` caps it for quick tests.

Train locally (from ~/codabench/2026-competition, after ``source env.sh``):

    benchopt run tracks/bci_decoding -d "BCI[study=dreyer2023]" \
        -s ../solvers/bci_decoding/riemann_steptype.py \
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
            Xs.append(self._prep(X))
            ys.append(to_numpy(y))
        X, y = np.concatenate(Xs), np.concatenate(ys)

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
        lda = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto",
            priors=np.full(n_classes, 1.0 / n_classes))
        parts["lda"] = lda.fit(feats, y)
        self.parts = parts
        return self

    def _prep(self, X):
        """Per-window preprocessing on CPU, then float64 for pyriemann."""
        X = torch.as_tensor(X, dtype=torch.float32).cpu()
        if self.preproc is not None:
            with torch.no_grad():
                X = self.preproc(X)
        return X.numpy().astype(np.float64)

    def predict(self, X):
        X = self._prep(X)
        return torch.as_tensor(self.parts["lda"].predict(_features(self.parts, X)))


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
