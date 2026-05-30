"""Model-specific normalization hooks.

The current XGBoost path intentionally uses no scaling. Future models that do
depend on scale can provide a scikit-learn transformer here so GridSearchCV
fits normalization parameters inside each training fold only.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class IdentityNormalizer(BaseEstimator, TransformerMixin):
    """No-op transformer for models that do not require normalization."""

    def fit(self, X, y=None):  # noqa: D401
        return self

    def transform(self, X):
        return X


def make_normalizer(model_name: str, cfg: dict):
    """Return the normalization transformer for ``model_name``.

    ``None`` means the estimator should not be wrapped at all. This keeps the
    XGB path bit-for-bit closest to the existing behavior, and also lets
    ``riemannian`` return a fully-formed pipeline from its own factory
    (``make_xdawn_covariance_pipeline``) without being double-wrapped here.
    """
    if model_name == "cnn":
        from .cnn import make_normalizer as make_cnn_normalizer

        return make_cnn_normalizer(cfg)
    if model_name == "eegnet":
        from .eegnet import make_normalizer as make_eegnet_normalizer

        return make_eegnet_normalizer(cfg)
    # riemannian returns its own (features, classifier) pipeline from
    # make_riemannian() and does not want an outer (normalize, classifier)
    # wrap on top of it.
    return None


def maybe_wrap_estimator(estimator, model_name: str, cfg: dict):
    normalizer = make_normalizer(model_name, cfg)
    if normalizer is None:
        return estimator
    return Pipeline([
        ("normalize", normalizer),
        ("classifier", estimator),
    ])


def maybe_prefix_param_grid(param_grid: dict, estimator) -> dict:
    """Prefix classifier params for the (normalize, classifier) wrapper.

    Native pipelines from model factories (e.g. Riemannian's
    ``Pipeline([("features", ...), ("classifier", ...)])``) already use their
    own step names in the param grid, so we only apply the
    ``classifier__`` prefix when we recognise the wrapper layout this module
    builds.
    """
    if not isinstance(estimator, Pipeline):
        return param_grid
    if list(estimator.named_steps.keys()) != ["normalize", "classifier"]:
        return param_grid
    return {
        key if key.startswith("classifier__") else f"classifier__{key}": value
        for key, value in param_grid.items()
    }


def unwrap_classifier(estimator):
    """Return final classifier from a wrapped or bare estimator."""
    if isinstance(estimator, Pipeline):
        return estimator.named_steps["classifier"]
    return estimator
