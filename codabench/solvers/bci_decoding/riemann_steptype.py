"""Riemann-StepType — the thesis repo's Riemannian comparator as a Track 2
(BCI decoding) submission.

Ported from ``src/eeg_steptype/models/riemannian.py`` (``make_riemannian``):
the feature union of

  1. xDAWN covariances -> tangent space   (pyriemann XdawnCovariances + TS)
  2. broadband OAS covariance -> tangent space
  3. per-channel log-variance              (the thesis "FBCSP" placeholder)

piped into shrinkage LDA with uniform class priors. Changes vs the thesis
model, and why:

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
import torch
import numpy as np
from pyriemann.estimation import Covariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy


def _features(parts, X):
    """Feature union for (n, C, T) float64 windows -> (n, n_features)."""
    blocks = []
    if parts.get("xdawn") is not None:
        blocks.append(parts["xdawn_ts"].transform(parts["xdawn"].transform(X)))
    covs = Covariances(estimator=parts["estimator"]).transform(X)
    blocks.append(parts["broad_ts"].transform(covs))
    blocks.append(np.log(np.var(X, axis=2) + 1e-12))
    return np.concatenate(blocks, axis=1)


class RiemannStepTypeModel:

    def __init__(self, parts=None, nfilter=4, estimator="oas",
                 use_xdawn=True, max_batches=None):
        self.parts = parts
        self.nfilter, self.estimator = nfilter, estimator
        self.use_xdawn, self.max_batches = use_xdawn, max_batches

    def fit(self, train_loader):
        Xs, ys = [], []
        for i, (X, y, _info) in enumerate(train_loader):
            if self.max_batches is not None and i >= self.max_batches:
                break
            Xs.append(to_numpy(X).astype(np.float64))
            ys.append(to_numpy(y))
        X, y = np.concatenate(Xs), np.concatenate(ys)
        print(f"[Riemann-StepType] fitting on X={X.shape}, "
              f"classes={np.unique(y).tolist()}", flush=True)

        parts = {"estimator": self.estimator, "xdawn": None}
        if self.use_xdawn:
            parts["xdawn"] = XdawnCovariances(
                nfilter=self.nfilter, estimator=self.estimator,
                xdawn_estimator=self.estimator).fit(X, y)
            parts["xdawn_ts"] = TangentSpace(metric="riemann").fit(
                parts["xdawn"].transform(X), y)
        covs = Covariances(estimator=self.estimator).transform(X)
        parts["broad_ts"] = TangentSpace(metric="riemann").fit(covs, y)

        n_classes = len(np.unique(y))
        lda = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto",
            priors=np.full(n_classes, 1.0 / n_classes))
        parts["lda"] = lda.fit(_features(parts, X), y)
        self.parts = parts
        return self

    def predict(self, X):
        X = to_numpy(X).astype(np.float64)
        return torch.as_tensor(self.parts["lda"].predict(_features(self.parts, X)))


class Solver(CompetSolver):

    name = "Riemann-StepType"

    parameters = {
        "nfilter": [4],
        "estimator": ["oas"],
        "use_xdawn": [True],     # xDAWN is ERP-oriented; try False for MI
        "max_batches": [None],   # small int = quick pipeline test
    }

    def load_model(self, meta):
        weights = meta["submission_dir"] / "riemann.joblib"
        # Inference only: a training run starts fresh (see eegnet_steptype).
        parts = (joblib.load(weights)
                 if weights.exists() and self.train_loader is None else None)
        return RiemannStepTypeModel(parts, self.nfilter, self.estimator,
                                    self.use_xdawn, self.max_batches)

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        joblib.dump(model.parts, path / "riemann.joblib")
