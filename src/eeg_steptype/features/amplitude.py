"""Binned mean amplitude per channel × bin → wide DataFrame.

One row per epoch; columns ``{channel}_bin_{b}``.
"""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd


def binned_mean_amplitude(
    epochs: mne.BaseEpochs,
    bin_n: float,
    ch_names: list[str],
) -> pd.DataFrame:
    df = epochs.to_data_frame()
    df["bin"] = (df["time"] // bin_n).astype(int)
    df = df.drop(columns=[c for c in ("Stim", "condition", "time") if c in df.columns])

    df = (df.groupby(["epoch", "bin"]).mean(numeric_only=True).reset_index())

    long = df.melt(
        id_vars=["epoch", "bin"],
        value_vars=ch_names,
        var_name="channel",
        value_name="amplitude",
    )
    wide = (long
            .pivot(index="epoch", columns=["channel", "bin"], values="amplitude")
            .reset_index())
    wide.columns = [
        f"{ch}_bin_{b}" if isinstance(ch, str) and ch != "epoch" else "epoch"
        for ch, b in [(c if isinstance(c, tuple) else (c, "")) for c in wide.columns]
    ]
    return wide
