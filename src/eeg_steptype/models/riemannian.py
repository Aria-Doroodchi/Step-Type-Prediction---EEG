"""Riemannian-model normalization and classifier utilities.

This module is infrastructure for future covariance-based comparators. It
keeps shrinkage estimation fold-local when used inside ``GridSearchCV``.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline


class OASCovarianceVectorizer(BaseEstimator, TransformerMixin):
    """Estimate OAS-shrinkage covariance matrices and vectorize triangles."""

    def __init__(self, covariance_estimator: str = "oas"):
        self.covariance_estimator = covariance_estimator

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.asarray(X)
        if arr.ndim != 3:
            raise ValueError(
                "Riemannian normalization expects shape "
                "(n_epochs, n_channels, n_times)"
            )
        try:
            from pyriemann.estimation import Covariances
        except Exception as exc:                            # noqa: BLE001
            raise RuntimeError(
                "Riemannian normalization requires the optional 'riemannian' extra. "
                "Install it with `pip install -e .[riemannian]`."
            ) from exc

        covs = Covariances(estimator=self.covariance_estimator).transform(arr)
        tri = np.triu_indices(covs.shape[1])
        return covs[:, tri[0], tri[1]]


def make_normalizer(cfg: dict):
    rcfg = cfg.get("modeling", {}).get("riemannian", {})
    return OASCovarianceVectorizer(
        covariance_estimator=rcfg.get("covariance_estimator", "oas")
    )


def make_classifier(cfg: dict):
    """Return a future-ready pyriemann covariance-classifier pipeline.

    This is intentionally not registered in ``MODEL_FACTORIES`` yet because
    the current training data path is feature-parquet/tabular. Once an
    epoch-tensor training path is added, this pipeline keeps OAS covariance
    estimation inside each CV fold.
    """
    rcfg = cfg.get("modeling", {}).get("riemannian", {})
    estimator = rcfg.get("covariance_estimator", "oas")
    try:
        from pyriemann.classification import MDM
        from pyriemann.estimation import Covariances
    except Exception as exc:                            # noqa: BLE001
        raise RuntimeError(
            "Riemannian classifiers require the optional 'riemannian' extra. "
            "Install it with `pip install -e .[riemannian]`."
        ) from exc

    return Pipeline([
        ("covariance", Covariances(estimator=estimator)),
        ("classifier", MDM()),
    ])


class XDawnCovarianceTangentSpace(BaseEstimator, TransformerMixin):
    """xDAWN covariance followed by tangent-space projection."""

    def __init__(
        self,
        nfilter: int = 4,
        covariance_estimator: str = "oas",
        tangent_metric: str = "riemann",
    ):
        self.nfilter = nfilter
        self.covariance_estimator = covariance_estimator
        self.tangent_metric = tangent_metric

    def fit(self, X, y):
        XdawnCovariances, TangentSpace = _xdawn_classes()
        self.xdawn_ = XdawnCovariances(
            nfilter=self.nfilter,
            estimator=self.covariance_estimator,
            xdawn_estimator=self.covariance_estimator,
        )
        covs = self.xdawn_.fit_transform(np.asarray(X), y)
        self.tangent_ = TangentSpace(metric=self.tangent_metric)
        self.tangent_.fit(covs, y)
        return self

    def transform(self, X):
        covs = self.xdawn_.transform(np.asarray(X))
        return self.tangent_.transform(covs)


class BroadbandCovarianceTangentSpace(BaseEstimator, TransformerMixin):
    """Broadband covariance followed by tangent-space projection."""

    def __init__(self, covariance_estimator: str = "oas", tangent_metric: str = "riemann"):
        self.covariance_estimator = covariance_estimator
        self.tangent_metric = tangent_metric

    def fit(self, X, y=None):
        Covariances, TangentSpace = _covariance_classes()
        covs = Covariances(estimator=self.covariance_estimator).transform(np.asarray(X))
        self.tangent_ = TangentSpace(metric=self.tangent_metric)
        self.tangent_.fit(covs, y)
        return self

    def transform(self, X):
        Covariances, _TangentSpace = _covariance_classes()
        covs = Covariances(estimator=self.covariance_estimator).transform(np.asarray(X))
        return self.tangent_.transform(covs)


class FBCSPLogVariance(BaseEstimator, TransformerMixin):
    """Placeholder FBCSP-style log-variance features for mu/beta bands."""

    def __init__(self, bands: dict | None = None, eps: float = 1e-12):
        self.bands = bands
        self.eps = eps

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.asarray(X)
        if arr.ndim != 3:
            raise ValueError("FBCSPLogVariance expects (n_epochs, n_channels, n_times)")
        # Filtering is intentionally left to the eventual tensor-data training
        # path; this block captures the fold-local feature contract now.
        return np.log(np.var(arr, axis=2) + self.eps)


class RiemannianFeatureUnion(BaseEstimator, TransformerMixin):
    """Concatenate xDAWN TS, broadband TS, and FBCSP log-variance blocks."""

    def __init__(
        self,
        nfilter: int = 4,
        covariance_estimator: str = "oas",
        fbcsp_bands: dict | None = None,
    ):
        self.nfilter = nfilter
        self.covariance_estimator = covariance_estimator
        self.fbcsp_bands = fbcsp_bands

    def fit(self, X, y):
        self.xdawn_ = XDawnCovarianceTangentSpace(
            nfilter=self.nfilter,
            covariance_estimator=self.covariance_estimator,
        ).fit(X, y)
        self.broadband_ = BroadbandCovarianceTangentSpace(
            covariance_estimator=self.covariance_estimator,
        ).fit(X, y)
        self.fbcsp_ = FBCSPLogVariance(bands=self.fbcsp_bands).fit(X, y)
        return self

    def transform(self, X):
        return np.concatenate([
            self.xdawn_.transform(X),
            self.broadband_.transform(X),
            self.fbcsp_.transform(X),
        ], axis=1)


class BalancedShrinkageLDA(LinearDiscriminantAnalysis):
    """Shrinkage LDA with uniform class priors for minor imbalance."""

    def __init__(self):
        super().__init__(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])


def make_xdawn_covariance_pipeline(cfg: dict):
    """Future ERP/SCP Riemannian comparator ending in shrinkage LDA."""
    rcfg = cfg.get("modeling", {}).get("riemannian", {})
    estimator = rcfg.get("covariance_estimator", "oas")
    return Pipeline([
        ("features", RiemannianFeatureUnion(
            nfilter=int(rcfg.get("xdawn", {}).get("nfilter", 4)),
            covariance_estimator=estimator,
            fbcsp_bands=rcfg.get("fbcsp_bands"),
        )),
        ("classifier", BalancedShrinkageLDA()),
    ])


def make_riemannian(cfg: dict, **_kwargs):
    """Public model factory for ``--model riemannian``.

    Returns the xDAWN covariance + TangentSpace + FBCSP-log-variance feature
    union piped into a balanced shrinkage LDA. Ignores ``scale_pos_weight``
    and ``n_features`` keyword arguments forwarded by the generic training
    driver -- class imbalance is handled by the LDA's uniform priors and the
    pipeline accepts native ``(n_epochs, n_channels, n_times)`` tensors.
    """
    return make_xdawn_covariance_pipeline(cfg)


def param_grid(cfg: dict) -> dict:
    """Hyperparameter grid for the Riemannian pipeline.

    Keys follow the (features, classifier) pipeline step names so
    ``maybe_prefix_param_grid`` leaves them unchanged. The default below is
    intentionally small (xDAWN nfilter × covariance estimator) because the
    pipeline already does heavy lifting per fold and a wider sweep buys
    little; configs/riemannian.yaml can override it.
    """
    rcfg = cfg.get("modeling", {}).get("riemannian", {})
    grid = rcfg.get("param_grid")
    if grid:
        return grid
    return {
        "features__nfilter": [2, 4, 6],
        "features__covariance_estimator": ["oas", "lwf"],
    }


def _xdawn_classes():
    try:
        from pyriemann.estimation import XdawnCovariances
        from pyriemann.tangentspace import TangentSpace
    except Exception as exc:                            # noqa: BLE001
        raise RuntimeError(
            "xDAWN covariance features require the optional 'riemannian' extra. "
            "Install it with `pip install -e .[riemannian]`."
        ) from exc
    return XdawnCovariances, TangentSpace


def _covariance_classes():
    try:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
    except Exception as exc:                            # noqa: BLE001
        raise RuntimeError(
            "Riemannian covariance features require the optional 'riemannian' extra. "
            "Install it with `pip install -e .[riemannian]`."
        ) from exc
    return Covariances, TangentSpace
