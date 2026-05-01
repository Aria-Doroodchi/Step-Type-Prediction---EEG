"""Per-bin linear-trend slopes: a polyfit(deg=1) for each channel within each bin."""

from __future__ import annotations

import warnings

import mne
import numpy as np
import pandas as pd


def binned_slopes(
    epochs: mne.BaseEpochs,
    bin_n: float,
    ch_names: list[str],
) -> pd.DataFrame:
    df = epochs.to_data_frame()
    df["bin"] = (df["time"] // bin_n).astype(int)
    df = df.drop(columns=[c for c in ("Stim", "condition") if c in df.columns])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")
        slopes = (df
                  .groupby(["epoch", "bin"])
                  .apply(
                      lambda x: x[ch_names].apply(
                          lambda y: np.polyfit(x["time"], y, 1)[0]
                      ),
                      include_groups=False,
                  )
                  .reset_index())

    long = slopes.melt(
        id_vars=["epoch", "bin"],
        value_vars=ch_names,
        var_name="channel",
        value_name="slope",
    )
    wide = (long
            .pivot(index="epoch", columns=["channel", "bin"], values="slope")
            .reset_index())
    wide.columns = [
        f"slope_{ch}_bin_{b}" if isinstance(ch, str) and ch != "epoch" else "epoch"
        for ch, b in [(c if isinstance(c, tuple) else (c, "")) for c in wide.columns]
    ]
    return wide
