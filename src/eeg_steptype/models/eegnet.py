"""EEGNet-style classifier for epoch tensors.

This is a local Keras implementation of the compact EEGNet block from
Lawhern et al. (2018): temporal convolution, depthwise spatial filtering,
separable temporal convolution, then a dense classifier.
"""

from __future__ import annotations

from .cnn import ExponentialMovingStandardizer, HybridTensorFeatureStandardizer


def make_normalizer(cfg: dict, n_features=None):
    ecfg = cfg.get("modeling", {}).get("eegnet", {}).get("standardize", {})
    hybrid = cfg.get("_neural_hybrid_input")
    if hybrid:
        return HybridTensorFeatureStandardizer(
            n_channels=int(hybrid["n_channels"]),
            n_times=int(hybrid["n_times"]),
            n_tabular_features=int(hybrid["n_tabular_features"]),
            factor_new=float(ecfg.get("factor_new", 0.001)),
            init_block_size=int(ecfg.get("init_block_size", 1000)),
            eps=float(ecfg.get("eps", 1e-4)),
        )
    return ExponentialMovingStandardizer(
        factor_new=float(ecfg.get("factor_new", 0.001)),
        init_block_size=int(ecfg.get("init_block_size", 1000)),
        eps=float(ecfg.get("eps", 1e-4)),
    )


def make_eegnet(cfg: dict, *, input_shape: tuple[int, int] | int, **_kwargs):
    """Return a SciKeras-wrapped binary EEGNet classifier.

    ``input_shape`` is ``(n_channels, n_times)`` for tensor-only runs, or the
    flattened hybrid feature count when ``cfg["_neural_hybrid_input"]`` is set.
    """
    from scikeras.wrappers import KerasClassifier
    import tensorflow as tf
    from tensorflow.keras import constraints, layers, regularizers

    ecfg = cfg.get("modeling", {}).get("eegnet", {})
    hybrid = cfg.get("_neural_hybrid_input")
    if hybrid:
        n_channels = int(hybrid["n_channels"])
        n_times = int(hybrid["n_times"])
        n_tabular = int(hybrid["n_tabular_features"])
        n_tensor_features = n_channels * n_times
        n_inputs = int(input_shape)
    else:
        n_channels, n_times = int(input_shape[0]), int(input_shape[1])
        n_tabular = 0
        n_tensor_features = n_channels * n_times
        n_inputs = n_tensor_features

    def _valid_kernel(value: int) -> int:
        return max(1, min(int(value), n_times))

    def build_fn(
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        kernel_length: int = 64,
        separable_kernel_length: int = 16,
        dropout_rate: float = 0.5,
        tabular_units: int = 32,
        fusion_units: int = 32,
        learning_rate: float = 1e-3,
        norm_rate: float = 0.25,
    ):
        kernel_length = _valid_kernel(kernel_length)
        separable_kernel_length = _valid_kernel(separable_kernel_length)

        inputs = tf.keras.Input(
            shape=(n_inputs,) if hybrid else (n_channels, n_times),
            name="hybrid_features" if hybrid else "epochs",
        )
        if hybrid:
            tensor_input = layers.Lambda(
                lambda z: z[:, :n_tensor_features],
                output_shape=(n_tensor_features,),
                name="tensor_slice",
            )(inputs)
            tensor_input = layers.Reshape(
                (n_channels, n_times),
                name="tensor_unflatten",
            )(tensor_input)
        else:
            tensor_input = inputs

        x = layers.Reshape((n_channels, n_times, 1), name="add_image_axis")(tensor_input)

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
        if hybrid and n_tabular > 0:
            tab = layers.Lambda(
                lambda z: z[:, n_tensor_features:],
                output_shape=(n_tabular,),
                name="tabular_slice",
            )(inputs)
            tab = layers.Dense(
                int(tabular_units),
                activation="elu",
                kernel_regularizer=regularizers.l2(float(1e-4)),
                name="tabular_dense",
            )(tab)
            tab = layers.Dropout(float(dropout_rate), name="tabular_dropout")(tab)
            x = layers.Concatenate(name="tensor_tabular_concat")([x, tab])
            x = layers.Dense(
                int(fusion_units),
                activation="elu",
                kernel_constraint=constraints.max_norm(float(norm_rate)),
                name="fusion_dense",
            )(x)
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
        "model__tabular_units": [32],
        "model__fusion_units": [32],
        "model__learning_rate": [1e-3],
        "model__norm_rate": [0.25],
    }
