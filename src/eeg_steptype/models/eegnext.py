"""EEGNeXt — a more sophisticated CNN built on the EEGNet-lite block.

Where ``eegnet.py`` / ``cnn.py`` use a single fixed temporal kernel followed by
one depthwise-spatial + one separable-temporal block, EEGNeXt adds three things
that are well-motivated for a slow, multi-rhythm signal like the CNV:

1. **Multi-scale temporal stem** — several temporal convolutions with *different*
   kernel lengths run in parallel and are concatenated, so delta/theta/alpha/beta
   timescales are captured in one layer instead of betting on a single kernel.
2. **Squeeze-and-Excitation channel attention** — a tiny gating branch reweights
   the feature maps by their global informativeness, letting the network suppress
   uninformative spatial-temporal filters per fold.
3. **Residual separable blocks** — depth is added through separable convolutions
   wrapped in skip connections, so the model can be deeper without the usual
   optimisation cost on small per-participant trial counts.

Like ``cnn`` / ``eegnet`` it is a **hybrid** model: the convolutional branch can
be fused with the XGB-style tabular + eLORETA source features (when
``modeling.eegnext.tabular_features.enabled``). TensorFlow imports stay inside
the factory so importing the package never requires TF.
"""
from __future__ import annotations

from .cnn import ExponentialMovingStandardizer, HybridTensorFeatureStandardizer


def make_normalizer(cfg: dict, n_features=None):
    """Per-channel exponential-moving standardizer (fold-local).

    Mirrors ``cnn.make_normalizer`` / ``eegnet.make_normalizer`` so the tensor
    (or hybrid tensor+tabular) input is standardized inside each CV fold.
    """
    ncfg = cfg.get("modeling", {}).get("eegnext", {}).get("standardize", {})
    hybrid = cfg.get("_neural_hybrid_input")
    if hybrid:
        return HybridTensorFeatureStandardizer(
            n_channels=int(hybrid["n_channels"]),
            n_times=int(hybrid["n_times"]),
            n_tabular_features=int(hybrid["n_tabular_features"]),
            factor_new=float(ncfg.get("factor_new", 0.001)),
            init_block_size=int(ncfg.get("init_block_size", 1000)),
            eps=float(ncfg.get("eps", 1e-4)),
        )
    return ExponentialMovingStandardizer(
        factor_new=float(ncfg.get("factor_new", 0.001)),
        init_block_size=int(ncfg.get("init_block_size", 1000)),
        eps=float(ncfg.get("eps", 1e-4)),
    )


def make_eegnext(cfg: dict, *, input_shape: tuple[int, int] | int, **_kwargs):
    """Return a SciKeras-wrapped EEGNeXt binary classifier.

    ``input_shape`` is ``(n_channels, n_times)`` for tensor-only runs, or the
    flattened hybrid feature count when ``cfg["_neural_hybrid_input"]`` is set.
    Imports stay inside this function so the package imports without TensorFlow.
    """
    from scikeras.wrappers import KerasClassifier
    import tensorflow as tf
    from tensorflow.keras import constraints, layers, regularizers

    ecfg = cfg.get("modeling", {}).get("eegnext", {})
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
        temporal_kernels: tuple[int, ...] = (16, 32, 64),
        depth_multiplier: int = 2,
        f2: int = 32,
        separable_kernel: int = 16,
        n_residual_blocks: int = 2,
        se_ratio: int = 4,
        pool_1: int = 4,
        pool_2: int = 8,
        dropout: float = 0.5,
        tabular_units: int = 32,
        fusion_units: int = 32,
        learning_rate: float = 1e-3,
        l2: float = 1e-4,
        norm_rate: float = 0.25,
    ):
        # temporal_kernels may arrive as a list/tuple from the grid; clamp each
        # to the available number of samples.
        kernels = tuple(_valid_kernel(k) for k in temporal_kernels) or (_valid_kernel(64),)
        separable_kernel_v = _valid_kernel(separable_kernel)
        pool_1_v = max(1, min(int(pool_1), n_times))
        pool_2_v = max(1, int(pool_2))
        reg = regularizers.l2(float(l2))

        def se_block(t, name: str):
            """Squeeze-and-Excitation gate over the feature-map (filter) axis."""
            channels = int(t.shape[-1])
            squeeze = layers.GlobalAveragePooling2D(name=f"{name}_squeeze")(t)
            squeeze = layers.Dense(
                max(1, channels // int(se_ratio)),
                activation="relu",
                name=f"{name}_reduce",
            )(squeeze)
            squeeze = layers.Dense(
                channels, activation="sigmoid", name=f"{name}_expand",
            )(squeeze)
            squeeze = layers.Reshape((1, 1, channels), name=f"{name}_reshape")(squeeze)
            return layers.Multiply(name=f"{name}_scale")([t, squeeze])

        def residual_separable(t, idx: int):
            """Separable-conv block with a projected skip connection."""
            shortcut = t
            y = layers.SeparableConv2D(
                int(f2), (1, separable_kernel_v), padding="same", use_bias=False,
                depthwise_regularizer=reg, pointwise_regularizer=reg,
                name=f"res{idx}_sep_a",
            )(t)
            y = layers.BatchNormalization(name=f"res{idx}_bn_a")(y)
            y = layers.Activation("elu", name=f"res{idx}_elu_a")(y)
            y = layers.SeparableConv2D(
                int(f2), (1, separable_kernel_v), padding="same", use_bias=False,
                depthwise_regularizer=reg, pointwise_regularizer=reg,
                name=f"res{idx}_sep_b",
            )(y)
            y = layers.BatchNormalization(name=f"res{idx}_bn_b")(y)
            if int(shortcut.shape[-1]) != int(f2):
                shortcut = layers.Conv2D(
                    int(f2), (1, 1), padding="same", use_bias=False,
                    kernel_regularizer=reg, name=f"res{idx}_proj",
                )(shortcut)
                shortcut = layers.BatchNormalization(name=f"res{idx}_proj_bn")(shortcut)
            y = layers.Add(name=f"res{idx}_add")([y, shortcut])
            y = layers.Activation("elu", name=f"res{idx}_elu_out")(y)
            return layers.Dropout(float(dropout), name=f"res{idx}_drop")(y)

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
                (n_channels, n_times), name="tensor_unflatten",
            )(tensor_input)
        else:
            tensor_input = inputs

        x = layers.Reshape((n_channels, n_times, 1), name="add_image_axis")(tensor_input)

        # --- Multi-scale temporal stem ------------------------------------
        branches = []
        for k in kernels:
            b = layers.Conv2D(
                int(f1), (1, int(k)), padding="same", use_bias=False,
                kernel_regularizer=reg, name=f"temporal_k{int(k)}",
            )(x)
            b = layers.BatchNormalization(name=f"temporal_k{int(k)}_bn")(b)
            branches.append(b)
        x = (
            layers.Concatenate(axis=-1, name="temporal_concat")(branches)
            if len(branches) > 1 else branches[0]
        )

        # --- Depthwise spatial filtering (EEGNet-style) -------------------
        x = layers.DepthwiseConv2D(
            (n_channels, 1), depth_multiplier=int(depth_multiplier), use_bias=False,
            depthwise_constraint=constraints.max_norm(1.0), name="spatial_depthwise",
        )(x)
        x = layers.BatchNormalization(name="spatial_bn")(x)
        x = layers.Activation("elu", name="spatial_elu")(x)
        x = layers.AveragePooling2D((1, pool_1_v), name="pool_1")(x)
        x = layers.Dropout(float(dropout), name="dropout_1")(x)

        # --- Channel attention -------------------------------------------
        x = se_block(x, name="se1")

        # --- Residual separable depth ------------------------------------
        for idx in range(int(n_residual_blocks)):
            x = residual_separable(x, idx)
        x = layers.AveragePooling2D((1, pool_2_v), name="pool_2")(x)
        x = layers.Dropout(float(dropout), name="dropout_2")(x)

        x = layers.Flatten(name="flatten")(x)

        # --- Hybrid tabular fusion ---------------------------------------
        if hybrid and n_tabular > 0:
            tab = layers.Lambda(
                lambda z: z[:, n_tensor_features:],
                output_shape=(n_tabular,),
                name="tabular_slice",
            )(inputs)
            tab = layers.Dense(
                int(tabular_units), activation="elu", kernel_regularizer=reg,
                name="tabular_dense",
            )(tab)
            tab = layers.Dropout(float(dropout), name="tabular_dropout")(tab)
            x = layers.Concatenate(name="tensor_tabular_concat")([x, tab])
            x = layers.Dense(
                int(fusion_units), activation="elu",
                kernel_constraint=constraints.max_norm(float(norm_rate)),
                name="fusion_dense",
            )(x)

        outputs = layers.Dense(
            1, activation="sigmoid",
            kernel_constraint=constraints.max_norm(float(norm_rate)),
            name="class_probability",
        )(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="eegnext")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    return KerasClassifier(
        model=build_fn,
        epochs=int(ecfg.get("epochs", 60)),
        batch_size=int(ecfg.get("batch_size", 16)),
        verbose=int(ecfg.get("verbose", 0)),
        validation_split=float(ecfg.get("validation_split", 0.2)),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor=ecfg.get("early_stopping_monitor", "val_loss"),
                patience=int(ecfg.get("patience", 12)),
                restore_best_weights=True,
            )
        ],
    )


def param_grid(cfg: dict) -> dict:
    ecfg = cfg.get("modeling", {}).get("eegnext", {})
    grid = ecfg.get("param_grid")
    if grid:
        return grid
    return {
        "model__f1": [8],
        "model__temporal_kernels": [[16, 32, 64]],
        "model__depth_multiplier": [2],
        "model__f2": [32],
        "model__separable_kernel": [16],
        "model__n_residual_blocks": [2],
        "model__se_ratio": [4],
        "model__dropout": [0.5],
        "model__tabular_units": [32],
        "model__fusion_units": [32],
        "model__learning_rate": [1e-3],
        "model__l2": [1e-4],
        "model__norm_rate": [0.25],
    }
