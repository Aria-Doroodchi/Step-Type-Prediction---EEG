"""Simple CNN comparator for epoch tensors.

The model is intentionally small: temporal filters first, a depthwise spatial
filter across channels, then one separable temporal block. It runs on the same
``(n_epochs, n_channels, n_times)`` tensor cache as the Riemannian comparator.
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
        out = np.empty_like(arr, dtype=float)
        try:
            from braindecode.preprocessing import exponential_moving_standardize
        except Exception:                                   # noqa: BLE001
            exponential_moving_standardize = None

        for epoch_idx, epoch in enumerate(arr):
            if exponential_moving_standardize is not None:
                out[epoch_idx] = exponential_moving_standardize(
                    epoch,
                    factor_new=float(self.factor_new),
                    init_block_size=int(self.init_block_size),
                    eps=float(self.eps),
                )
            else:
                out[epoch_idx] = _standardize_epoch(
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


def _standardize_epoch(
    epoch: np.ndarray,
    *,
    factor_new: float,
    init_block_size: int,
    eps: float,
) -> np.ndarray:
    """Fallback per-channel exponential standardization.

    The input orientation is ``(channels, times)``. Braindecode is preferred
    when installed; this local implementation keeps the starter CNN runnable
    in lean environments.
    """
    arr = np.asarray(epoch, dtype=float)
    out = np.empty_like(arr, dtype=float)
    n_times = arr.shape[1]
    init = max(1, min(int(init_block_size), n_times))
    alpha = min(max(float(factor_new), 0.0), 1.0)

    for ch_idx, signal in enumerate(arr):
        mean = float(np.mean(signal[:init]))
        var = float(np.var(signal[:init]))
        for t_idx, value in enumerate(signal):
            if t_idx >= init:
                delta = value - mean
                mean = (1.0 - alpha) * mean + alpha * value
                var = (1.0 - alpha) * var + alpha * delta * delta
            out[ch_idx, t_idx] = (value - mean) / np.sqrt(var + eps)
    return out


def make_cnn(cfg: dict, *, input_shape: tuple[int, int], **_kwargs):
    """Return a SciKeras-wrapped EEGNet-lite binary classifier.

    ``input_shape`` is ``(n_channels, n_times)``. Imports stay inside this
    function so the package continues to import without TensorFlow installed.
    """
    from scikeras.wrappers import KerasClassifier
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    ccfg = cfg.get("modeling", {}).get("cnn", {})
    n_channels, n_times = int(input_shape[0]), int(input_shape[1])

    def _odd_kernel(value: int) -> int:
        value = max(3, min(int(value), n_times))
        return value if value % 2 else value - 1

    def build_fn(
        temporal_filters: int = 8,
        depth_multiplier: int = 2,
        separable_filters: int = 16,
        temporal_kernel: int = 65,
        separable_kernel: int = 17,
        pool_1: int = 4,
        pool_2: int = 8,
        dropout: float = 0.5,
        learning_rate: float = 1e-3,
        l2: float = 1e-4,
    ):
        temporal_kernel = _odd_kernel(temporal_kernel)
        separable_kernel = _odd_kernel(separable_kernel)
        pool_1 = max(1, min(int(pool_1), n_times))
        pool_2 = max(1, int(pool_2))

        inputs = tf.keras.Input(shape=(n_channels, n_times), name="epochs")
        x = layers.Reshape((n_channels, n_times, 1), name="add_image_axis")(inputs)

        x = layers.Conv2D(
            int(temporal_filters),
            kernel_size=(1, temporal_kernel),
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(float(l2)),
            name="temporal_conv",
        )(x)
        x = layers.BatchNormalization(name="temporal_bn")(x)

        x = layers.DepthwiseConv2D(
            kernel_size=(n_channels, 1),
            depth_multiplier=int(depth_multiplier),
            use_bias=False,
            depthwise_regularizer=regularizers.l2(float(l2)),
            name="spatial_depthwise",
        )(x)
        x = layers.BatchNormalization(name="spatial_bn")(x)
        x = layers.Activation("elu", name="spatial_elu")(x)
        x = layers.AveragePooling2D(pool_size=(1, pool_1), name="pool_1")(x)
        x = layers.Dropout(float(dropout), name="dropout_1")(x)

        x = layers.SeparableConv2D(
            int(separable_filters),
            kernel_size=(1, separable_kernel),
            padding="same",
            use_bias=False,
            depthwise_regularizer=regularizers.l2(float(l2)),
            pointwise_regularizer=regularizers.l2(float(l2)),
            name="separable_temporal",
        )(x)
        x = layers.BatchNormalization(name="separable_bn")(x)
        x = layers.Activation("elu", name="separable_elu")(x)
        x = layers.AveragePooling2D(pool_size=(1, pool_2), name="pool_2")(x)
        x = layers.Dropout(float(dropout), name="dropout_2")(x)

        x = layers.Flatten(name="flatten")(x)
        outputs = layers.Dense(1, activation="sigmoid", name="class_probability")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="eegnet_lite")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    return KerasClassifier(
        model=build_fn,
        epochs=int(ccfg.get("epochs", 30)),
        batch_size=int(ccfg.get("batch_size", 16)),
        verbose=int(ccfg.get("verbose", 0)),
        validation_split=float(ccfg.get("validation_split", 0.2)),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor=ccfg.get("early_stopping_monitor", "val_loss"),
                patience=int(ccfg.get("patience", 8)),
                restore_best_weights=True,
            )
        ],
    )


def param_grid(cfg: dict) -> dict:
    ccfg = cfg.get("modeling", {}).get("cnn", {})
    grid = ccfg.get("param_grid")
    if grid:
        return grid
    return {
        "model__temporal_filters": [8],
        "model__depth_multiplier": [2],
        "model__separable_filters": [16],
        "model__dropout": [0.5],
        "model__learning_rate": [1e-3],
    }
