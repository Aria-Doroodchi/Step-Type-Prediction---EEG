"""CNN normalization utilities.

Braindecode-style exponential moving standardization is useful for future CNN
comparators on epoch tensors. Keeping it as a scikit-learn transformer ensures
state is fit only on training folds when wrapped in ``GridSearchCV``.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ExponentialMovingStandardizer(BaseEstimator, TransformerMixin):
    """Per-channel exponential moving standardization for EEG tensors."""

    def __init__(
        self,
        factor_new: float = 0.001,
        init_block_size: int = 1000,
        eps: float = 1e-4,
    ):
        self.factor_new = factor_new
        self.init_block_size = init_block_size
        self.eps = eps

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 3:
            raise ValueError(
                "CNN standardization expects shape "
                "(n_epochs, n_channels, n_times)"
            )
        try:
            from braindecode.preprocessing import exponential_moving_standardize
        except Exception as exc:                            # noqa: BLE001
            raise RuntimeError(
                "CNN standardization requires braindecode. "
                "Install project dependencies with `pip install -e .`."
            ) from exc

        out = np.empty_like(arr, dtype=float)
        for epoch_idx, epoch in enumerate(arr):
            out[epoch_idx] = exponential_moving_standardize(
                epoch,
                factor_new=float(self.factor_new),
                init_block_size=int(self.init_block_size),
                eps=float(self.eps),
            )
        return out


def make_normalizer(cfg: dict):
    ccfg = cfg.get("modeling", {}).get("cnn", {}).get("standardize", {})
    return ExponentialMovingStandardizer(
        factor_new=float(ccfg.get("factor_new", 0.001)),
        init_block_size=int(ccfg.get("init_block_size", 1000)),
        eps=float(ccfg.get("eps", 1e-4)),
    )
