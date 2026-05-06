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
    data = epochs.get_data(picks=ch_names)
    bins = (epochs.times // bin_n).astype(int)
    unique_bins = np.unique(bins)

    out = {"epoch": epochs.selection}
    for b in unique_bins:
        mask = bins == b
        means = data[:, :, mask].mean(axis=2)
        for idx, ch in enumerate(ch_names):
            out[f"{ch}_bin_{b}"] = means[:, idx]
    return pd.DataFrame(out)
