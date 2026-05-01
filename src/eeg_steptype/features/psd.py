"""Morlet TFR → band-averaged power, binned in time."""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd


def band_power(
    epochs: mne.BaseEpochs,
    bin_n: float,
    ch_names: list[str],
    *,
    freqs: np.ndarray,
    freq_bands: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Return wide DataFrame: one row per epoch; columns ``{channel}_{band}_bin_{b}``."""
    n_cycles = freqs / 2.0
    power = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        average=False,
    )
    df = power.to_data_frame()

    # Replace freq-bin column with band names.
    bands = list(freq_bands.keys())
    masks = [df["freq"].between(lo, hi) for lo, hi in freq_bands.values()]
    df["freq"] = np.select(masks, bands, default=df["freq"])

    df["bin"] = (df["time"] // bin_n).astype(int)
    df = df.drop(columns=[c for c in ("condition", "time") if c in df.columns])

    avg = (df.groupby(["freq", "epoch", "bin"]).mean(numeric_only=True).reset_index())

    long = avg.melt(
        id_vars=["freq", "epoch", "bin"],
        value_vars=ch_names,
        var_name="channel",
        value_name="power",
    )

    # First pivot: (channel, freq) → ch_band columns
    step1 = (long
             .pivot(index=["epoch", "bin"], columns=["channel", "freq"], values="power")
             .reset_index())
    step1.columns = [
        f"{ch}_{band}" if isinstance(ch, str) and ch not in ("epoch", "bin") else (ch or "")
        for ch, band in [(c if isinstance(c, tuple) else (c, "")) for c in step1.columns]
    ]
    # Re-name the leftmost two columns explicitly
    step1 = step1.rename(columns={"epoch_": "epoch", "bin_": "bin"})

    # Second pivot: bin → suffix
    long2 = step1.melt(
        id_vars=["epoch", "bin"],
        value_vars=[c for c in step1.columns if c not in ("epoch", "bin")],
        var_name="ch_band",
        value_name="power",
    )
    wide = (long2
            .pivot(index="epoch", columns=["ch_band", "bin"], values="power")
            .reset_index())
    wide.columns = [
        f"{cb}_bin_{b}" if isinstance(cb, str) and cb != "epoch" else "epoch"
        for cb, b in [(c if isinstance(c, tuple) else (c, "")) for c in wide.columns]
    ]
    return wide


def freq_array(cfg: dict) -> np.ndarray:
    fcfg = cfg["features"]["freqs"]
    return np.arange(float(fcfg["fmin"]), float(fcfg["fmax"]) + float(fcfg["fstep"]),
                     float(fcfg["fstep"]))


def freq_bands(cfg: dict) -> dict[str, tuple[float, float]]:
    return {k: tuple(v) for k, v in cfg["features"]["freq_bands"].items()}
