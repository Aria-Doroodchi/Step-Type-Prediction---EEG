"""Per-bin linear-trend slopes: a polyfit(deg=1) for each channel within each bin."""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd


def binned_slopes(
    epochs: mne.BaseEpochs,
    bin_n: float,
    ch_names: list[str],
) -> pd.DataFrame:
    data = epochs.get_data(picks=ch_names)
    times = epochs.times
    bins = (times // bin_n).astype(int)
    unique_bins = np.unique(bins)

    out = {"epoch": epochs.selection}
    for b in unique_bins:
        mask = bins == b
        t = times[mask]
        denom = np.sum((t - t.mean()) ** 2)
        if denom == 0:
            slopes = np.zeros(data.shape[:2])
        else:
            weights = (t - t.mean()) / denom
            slopes = np.einsum("ect,t->ec", data[:, :, mask], weights)
        for idx, ch in enumerate(ch_names):
            out[f"slope_{ch}_bin_{b}"] = slopes[:, idx]
    return pd.DataFrame(out)
