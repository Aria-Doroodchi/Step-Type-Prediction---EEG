"""EEGNet-StepType — the thesis repo's Keras-faithful PyTorch EEGNet, as a
Track 2 (BCI decoding) submission.

Ported from ``src/eeg_steptype/models/eegnet_torch.py`` (repo commit 48fc3fa)
and made self-contained, because a Codabench submission must be ONE
``submission.py`` importing only what the scoring image carries (torch,
numpy, sklearn, braindecode, pyriemann, mne, ...). What changed vs the thesis
model, and why:

- multi-class head: ``n_classes`` logits + cross-entropy, instead of the
  step-type task's single sigmoid logit + BCE;
- no tabular/fusion branch: the competition only feeds raw ``(B, C, T)``
  windows, never hand-crafted features;
- kernel lengths scale with ``meta["sfreq"]``: the thesis grid's 64 / 16
  samples are EEGNet's canonical 0.5 s / 0.125 s at 128 Hz, so they are
  expressed in seconds here and converted per dataset;
- validation for early stopping holds out every 5th *subject* (``subject_id
  % 5 == 4``; ``record_id`` when a dataset has no subject ids): the loader
  reshuffles windows every epoch, so the Keras "trailing 20 % before
  shuffling" split is not available, and a subject-level hold-out also
  mirrors the thesis's participant-level validation;
- optional per-window, per-channel z-scoring inside the model
  (``standardize``; off by default = the thesis model as-is). Stateless and
  robust to cross-session amplitude drift, but it deletes channel-mean
  information -- which is exactly what the Simulated smoke dataset encodes
  its classes in, so leave it off for Simulated.

Kept exactly: the layer stack, TF "same" padding, channels-last flatten,
glorot-uniform init with Keras fans, BatchNorm momentum/eps, Adam eps, the
max-norm constraints re-applied after every step, and early stopping on
validation loss with best-weight restore.

Train locally (from ~/codabench/2026-competition, after ``source env.sh``):

    benchopt run tracks/bci_decoding -d "BCI[study=dreyer2023]" \
        -s ../solvers/bci_decoding/eegnet_steptype.py \
        -o "BCI-decoding[training=True]"

The trained submission folder (this file as ``submission.py`` + weights.pt)
lands in ``tracks/bci_decoding/outputs/EEGNet-StepType/``.
"""

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from benchmark_utils.base_solver import CompetSolver

_KERAS_EPSILON = 1e-7
_KERAS_BATCHNORM = {"eps": 1e-3, "momentum": 0.01}   # Keras momentum=0.99
_KERAS_ADAM_EPSILON = 1e-7


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


def _keras_max_norm_(weight, max_value, dims):
    with torch.no_grad():
        norms = weight.pow(2).sum(dim=dims, keepdim=True).sqrt()
        weight.mul_(norms.clamp(0.0, max_value) / (_KERAS_EPSILON + norms))


def _glorot_uniform_(weight, *, fan_in, fan_out):
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    nn.init.uniform_(weight, -limit, limit)


def _same_padding(kernel):
    left = (kernel - 1) // 2
    return nn.ZeroPad2d((left, kernel - 1 - left, 0, 0))


def _val_mask(info, n, device):
    """Per-window bool mask: True = held-out validation (every 5th group)."""
    for key in ("subject_id", "record_id"):
        if isinstance(info, dict) and key in info:
            ids = torch.as_tensor(info[key]).reshape(-1)
            if len(ids) == n and bool((ids >= 0).all()):
                return (ids % 5 == 4).to(device)
    return None


class EEGNetTorch(nn.Module):
    """The thesis EEGNet layer stack, multi-class, no tabular branch."""

    def __init__(self, n_channels, n_times, n_classes, *, f1=8,
                 depth_multiplier=2, f2=16, kernel_length=64,
                 separable_kernel_length=16, dropout_rate=0.5,
                 norm_rate=0.25, standardize=False, preproc=None):
        super().__init__()
        self.standardize = bool(standardize)
        self.preproc = preproc          # WindowPreproc or None (no weights)
        kernel_length = max(1, min(int(kernel_length), n_times))
        separable_kernel_length = max(1, min(int(separable_kernel_length),
                                             n_times))
        spatial_filters = f1 * depth_multiplier
        self.norm_rate = float(norm_rate)

        self.temporal_pad = _same_padding(kernel_length)
        self.temporal_conv = nn.Conv2d(1, f1, (1, kernel_length), bias=False)
        self.temporal_bn = nn.BatchNorm2d(f1, **_KERAS_BATCHNORM)
        self.spatial_depthwise = nn.Conv2d(
            f1, spatial_filters, (n_channels, 1), groups=f1, bias=False)
        self.spatial_bn = nn.BatchNorm2d(spatial_filters, **_KERAS_BATCHNORM)
        self.pool_1 = nn.AvgPool2d((1, 4))
        self.dropout_1 = nn.Dropout(float(dropout_rate))

        self.separable_pad = _same_padding(separable_kernel_length)
        self.separable_depthwise = nn.Conv2d(
            spatial_filters, spatial_filters, (1, separable_kernel_length),
            groups=spatial_filters, bias=False)
        self.separable_pointwise = nn.Conv2d(spatial_filters, f2, 1,
                                             bias=False)
        self.separable_bn = nn.BatchNorm2d(f2, **_KERAS_BATCHNORM)
        self.pool_2 = nn.AvgPool2d((1, 8))
        self.dropout_2 = nn.Dropout(float(dropout_rate))

        n_flat = f2 * ((n_times // 4) // 8)
        if n_flat == 0:
            raise ValueError(f"n_times={n_times} < 32: too short for EEGNet")
        self.classifier = nn.Linear(n_flat, int(n_classes))
        self._reset_parameters(n_channels, f1, depth_multiplier)

    def _reset_parameters(self, n_channels, f1, depth_multiplier):
        k_t = self.temporal_conv.kernel_size[1]
        _glorot_uniform_(self.temporal_conv.weight, fan_in=k_t,
                         fan_out=k_t * f1)
        _glorot_uniform_(self.spatial_depthwise.weight,
                         fan_in=n_channels * f1,
                         fan_out=n_channels * depth_multiplier)
        sf = self.separable_depthwise.in_channels
        k_s = self.separable_depthwise.kernel_size[1]
        _glorot_uniform_(self.separable_depthwise.weight, fan_in=sf * k_s,
                         fan_out=k_s)
        _glorot_uniform_(self.separable_pointwise.weight, fan_in=sf,
                         fan_out=self.separable_pointwise.out_channels)
        _glorot_uniform_(self.classifier.weight,
                         fan_in=self.classifier.in_features,
                         fan_out=self.classifier.out_features)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        if self.preproc is not None:
            x = self.preproc(x)
        if self.standardize:   # per-window, per-channel z-score
            x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)
        x = x.unsqueeze(1)
        x = self.temporal_bn(self.temporal_conv(self.temporal_pad(x)))
        x = F.elu(self.spatial_bn(self.spatial_depthwise(x)))
        x = self.dropout_1(self.pool_1(x))
        x = self.separable_pointwise(
            self.separable_depthwise(self.separable_pad(x)))
        x = F.elu(self.separable_bn(x))
        x = self.dropout_2(self.pool_2(x))
        return self.classifier(x.permute(0, 2, 3, 1).flatten(1))

    def apply_max_norm(self):
        _keras_max_norm_(self.spatial_depthwise.weight, 1.0, dims=(1, 2, 3))
        _keras_max_norm_(self.classifier.weight, self.norm_rate, dims=(1,))


class EEGNetStepTypeModel:
    """Owns the network + device; exposes the contract's ``predict``."""

    def __init__(self, net, device, lr, n_epochs, patience, max_batches):
        self.net = net.to(device)
        self.device = device
        self.lr, self.n_epochs, self.patience = lr, n_epochs, patience
        self.max_batches = max_batches   # None = full data; int = quick test

    def fit(self, train_loader):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               eps=_KERAS_ADAM_EPSILON)
        best, best_state, wait = math.inf, None, 0
        for epoch in range(self.n_epochs):
            self.net.train()
            val_loss, val_n, tr_loss, tr_n = 0.0, 0, 0.0, 0
            val_batches = []
            for i, (X, y, info) in enumerate(train_loader):
                if self.max_batches is not None and i >= self.max_batches:
                    break
                val = _val_mask(info, len(y), X.device)
                if val is not None and val.any():
                    val_batches.append((X[val], y[val]))
                    X, y = X[~val], y[~val]
                if len(y) < 2:          # BatchNorm needs > 1 sample
                    continue
                opt.zero_grad()
                loss = F.cross_entropy(self.net(X), y)
                loss.backward()
                opt.step()
                self.net.apply_max_norm()
                tr_loss += loss.item() * len(y)
                tr_n += len(y)
            self.net.eval()
            with torch.no_grad():
                for X, y in val_batches:
                    val_loss += F.cross_entropy(self.net(X), y,
                                                reduction="sum").item()
                    val_n += len(y)
            # No held-out subjects in this data: monitor train loss instead.
            val_loss = val_loss / val_n if val_n else tr_loss / max(tr_n, 1)
            print(f"[EEGNet-StepType] epoch {epoch + 1}/{self.n_epochs} "
                  f"train_loss={tr_loss / max(tr_n, 1):.4f} "
                  f"val_loss={val_loss:.4f} (n_train={tr_n}, n_val={val_n})",
                  flush=True)
            if val_loss < best:
                best, wait = val_loss, 0
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"[EEGNet-StepType] early stop at epoch {epoch + 1}",
                          flush=True)
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    @torch.inference_mode()
    def predict(self, X):
        self.net.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        return self.net(X).argmax(dim=1)


class Solver(CompetSolver):

    name = "EEGNet-StepType"

    # Sweep inline, e.g. -s "eegnet_steptype.py[lr=[1e-3,3e-4]]".
    # max_batches: None = full training; a small int = quick pipeline test.
    parameters = {
        "lr": [1e-3],
        "n_epochs": [50],
        "patience": [10],
        "kernel_s": [0.5],            # 64 samples at 128 Hz (thesis default)
        "separable_kernel_s": [0.125],  # 16 samples at 128 Hz
        "dropout_rate": [0.5],
        "standardize": [False],
        # Per-window preprocessing (see WindowPreproc): "none" or e.g. "8to30"
        # Hz; reference "none" | "car" | "laplacian".
        "bandpass": ["none"],
        "reference": ["none"],
        "max_batches": [None],
        # torch seed for init + dropout; sweep e.g. seed=[33,34,35] to measure
        # run-to-run noise (the data split itself is fixed by NeuralBench).
        "seed": [33],
    }

    def load_model(self, meta):
        torch.manual_seed(int(self.seed))       # reproducible weight init
        sfreq = float(meta["sfreq"])
        net = EEGNetTorch(
            n_channels=meta["n_chans"], n_times=meta["n_times"],
            n_classes=meta["n_classes"],
            kernel_length=round(self.kernel_s * sfreq),
            separable_kernel_length=round(self.separable_kernel_s * sfreq),
            dropout_rate=self.dropout_rate,
            standardize=self.standardize,
            preproc=WindowPreproc(meta["ch_names"], sfreq, self.bandpass,
                                  self.reference),
        )
        # Load shipped weights for inference only: a training run shares the
        # same outputs/<name>/ folder, which may hold weights from a run on
        # another dataset (different channel count) -> start fresh instead.
        weights = meta["submission_dir"] / "weights.pt"
        if weights.exists() and self.train_loader is None:
            net.load_state_dict(torch.load(weights, map_location="cpu"))
        return EEGNetStepTypeModel(net, self.device, self.lr, self.n_epochs,
                                   self.patience, self.max_batches)

    def fit(self, model, train_loader):
        torch.manual_seed(int(self.seed))
        model.fit(train_loader)

    def save_model(self, model, path):
        torch.save(model.net.state_dict(), path / "weights.pt")
