"""Feature selection, classifier factories, and the per-participant train loop.

`MODEL_FACTORIES` is exposed lazily so `import eeg_steptype.models` does not
pull in sklearn / xgboost / tensorflow until you actually touch them.
"""

from __future__ import annotations


def __getattr__(name):
    if name == "MODEL_FACTORIES":
        from .train import MODEL_FACTORIES
        return MODEL_FACTORIES
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MODEL_FACTORIES"]
