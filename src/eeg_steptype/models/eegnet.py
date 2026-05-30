"""EEGNet-style classifier for epoch tensors.

This is a local Keras implementation of the compact EEGNet block from
Lawhern et al. (2018): temporal convolution, depthwise spatial filtering,
separable temporal convolution, then a dense classifier.
"""

from __future__ import annotations

from .cnn import ExponentialMovingStandardizer


def make_normalizer(cfg: dict):
    ecfg = cfg.get("modeling", {}).get("eegnet", {}).get("standardize", {})
    return ExponentialMovingStandardizer(
        factor_new=float(ecfg.get("factor_new", 0.001)),
        init_block_size=int(ecfg.get("init_block_size", 1000)),
        eps=float(ecfg.get("eps", 1e-4)),
    )


def make_eegnet(cfg: dict, *, input_shape: tuple[int, int], **_kwargs):
    """Return a SciKeras-wrapped binary EEGNet classifier.

    ``input_shape`` is ``(n_channels, n_times)``.
    """
    from scikeras.wrappers import KerasClassifier
    import tensorflow as tf
    from tensorflow.keras import constraints, layers

    ecfg = cfg.get("modeling", {}).get("eegnet", {})
    n_channels, n_times = int(input_shape[0]), int(input_shape[1])

    def _valid_kernel(value: int) -> int:
        return max(1, min(int(value), n_times))

    def build_fn(
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        kernel_length: int = 64,
        separable_kernel_length: int = 16,
        dropout_rate: float = 0.5,
        learning_rate: float = 1e-3,
        norm_rate: float = 0.25,
    ):
        kernel_length = _valid_kernel(kernel_length)
        separable_kernel_length = _valid_kernel(separable_kernel_length)

        inputs = tf.keras.Input(shape=(n_channels, n_times), name="epochs")
        x = layers.Reshape((n_channels, n_times, 1), name="add_image_axis")(inputs)

        x = layers.Conv2D(
            int(f1),
            kernel_size=(1, kernel_length),
            padding="same",
            use_bias=False,
            name="temporal_conv",
        )(x)
        x = layers.BatchNormalization(name="temporal_bn")(x)

        x = layers.DepthwiseConv2D(
            kernel_size=(n_channels, 1),
            depth_multiplier=int(depth_multiplier),
            use_bias=False,
            depthwise_constraint=constraints.max_norm(1.0),
            name="spatial_depthwise",
        )(x)
        x = layers.BatchNormalization(name="spatial_bn")(x)
        x = layers.Activation("elu", name="spatial_elu")(x)
        x = layers.AveragePooling2D(pool_size=(1, 4), name="pool_1")(x)
        x = layers.Dropout(float(dropout_rate), name="dropout_1")(x)

        x = layers.SeparableConv2D(
            int(f2),
            kernel_size=(1, separable_kernel_length),
            padding="same",
            use_bias=False,
            name="separable_conv",
        )(x)
        x = layers.BatchNormalization(name="separable_bn")(x)
        x = layers.Activation("elu", name="separable_elu")(x)
        x = layers.AveragePooling2D(pool_size=(1, 8), name="pool_2")(x)
        x = layers.Dropout(float(dropout_rate), name="dropout_2")(x)

        x = layers.Flatten(name="flatten")(x)
        outputs = layers.Dense(
            1,
            activation="sigmoid",
            kernel_constraint=constraints.max_norm(float(norm_rate)),
            name="class_probability",
        )(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="eegnet")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    return KerasClassifier(
        model=build_fn,
        epochs=int(ecfg.get("epochs", 50)),
        batch_size=int(ecfg.get("batch_size", 16)),
        verbose=int(ecfg.get("verbose", 0)),
        validation_split=float(ecfg.get("validation_split", 0.2)),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor=ecfg.get("early_stopping_monitor", "val_loss"),
                patience=int(ecfg.get("patience", 10)),
                restore_best_weights=True,
            )
        ],
    )


def param_grid(cfg: dict) -> dict:
    ecfg = cfg.get("modeling", {}).get("eegnet", {})
    grid = ecfg.get("param_grid")
    if grid:
        return grid
    return {
        "model__f1": [8],
        "model__depth_multiplier": [2],
        "model__f2": [16],
        "model__kernel_length": [64],
        "model__separable_kernel_length": [16],
        "model__dropout_rate": [0.5],
        "model__learning_rate": [1e-3],
        "model__norm_rate": [0.25],
    }
