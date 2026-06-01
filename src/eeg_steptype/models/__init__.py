"""Feature selection, classifier factories, and the per-participant train loop.

Submodules:

    feature_selection   correlation drop, SelectKBest, RFECV (iterated),
                        gain-based prune, SHAP-based prune
    xgb                 XGBoost factory + param grid (default model)
    svm                 SVM factory + param grid
    lstm                bidirectional-LSTM factory wrapped via scikeras
    logistic            tiny logistic-regression factory used by smoke tests
    cnn                 small EEGNet-inspired tensor CNN
    eegnet              compact EEGNet-style tensor CNN
    riemannian          xDAWN/covariance tensor comparator
    evaluate            confusion-matrix metrics + cohort rollup
    train               generic per-participant fit/eval driver

`MODEL_FACTORIES` is the registry that maps model name → factory functions
+ param-grid getter + capability flags (gain-prune / SHAP-prune support).
It is exposed lazily through ``__getattr__`` so that simply importing
``eeg_steptype.models`` does NOT pull in sklearn / xgboost / tensorflow.
The heavy modules only load when you actually look up a factory.
"""

from __future__ import annotations


def __getattr__(name):
    """Lazy attribute resolver for top-level names.

    Lets ``from eeg_steptype.models import MODEL_FACTORIES`` work without
    eagerly importing the training submodule (and its sklearn dependency)
    at package-import time. Anything else falls through to AttributeError.
    """
    if name == "MODEL_FACTORIES":
        from .train import MODEL_FACTORIES
        return MODEL_FACTORIES
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MODEL_FACTORIES"]
