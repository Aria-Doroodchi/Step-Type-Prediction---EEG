"""LSTM model factory.

Wraps a Keras Sequential in a `KerasClassifier` so it plugs into the same
GridSearchCV / per-participant loop as the classical models. Imports are
deferred so the rest of the package works without TensorFlow installed.
"""

from __future__ import annotations


def make_lstm(cfg: dict, *, n_features: int, n_timesteps: int = 1):
    """Build a `KerasClassifier` wrapping a Sequential bidirectional LSTM.

    The exact architecture matches the prior CNV_LSTM_3.py shape, parameterised
    by units / dropout drawn from cfg["modeling"]["lstm"].
    """
    from scikeras.wrappers import KerasClassifier
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input

    lstm_cfg = cfg["modeling"]["lstm"]

    def build_fn(units: int = 64, dropout: float = 0.2):
        m = Sequential([
            Input(shape=(n_timesteps, n_features)),
            Bidirectional(LSTM(units, return_sequences=False)),
            Dropout(dropout),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        m.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return m

    return KerasClassifier(
        model=build_fn,
        epochs=int(lstm_cfg.get("epochs", 50)),
        batch_size=int(lstm_cfg.get("batch_size", 32)),
        verbose=0,
    )


def param_grid(cfg: dict) -> dict:
    lstm_cfg = cfg["modeling"]["lstm"]
    return {
        "model__units":   list(lstm_cfg.get("units_grid",   [32, 64, 128])),
        "model__dropout": list(lstm_cfg.get("dropout_grid", [0.2, 0.4])),
    }
