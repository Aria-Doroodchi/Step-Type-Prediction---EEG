"""Binned amplitude features per channel x bin -> wide DataFrame.

One row per epoch. The legacy helper emits columns ``{channel}_bin_{b}``.
The configurable helper can emit several summary statistics and/or several
bin widths for binning-sensitivity experiments.
"""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd


SUPPORTED_STATS = ("mean", "std", "min", "max", "median")


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


def binned_amplitude_features(
    epochs: mne.BaseEpochs,
    *,
    bin_widths: list[float],
    stats: list[str],
    ch_names: list[str],
) -> pd.DataFrame:
    """Return amplitude features for one or more bin widths and statistics.

    When called with a single bin width and ``["mean"]``, the output preserves
    the historical column names. Any richer configuration uses explicit
    ``amp_w{width}_{channel}_{stat}_bin_{b}`` names so feature caches and model
    reports remain readable.
    """
    widths = [float(w) for w in bin_widths]
    if not widths:
        raise ValueError("features.amplitude.bin_widths must not be empty")

    requested_stats = [str(s).lower() for s in stats]
    if not requested_stats:
        raise ValueError("features.amplitude.stats must not be empty")
    unsupported = sorted(set(requested_stats) - set(SUPPORTED_STATS))
    if unsupported:
        raise ValueError(
            "Unsupported amplitude stat(s): "
            f"{unsupported}; choose from {list(SUPPORTED_STATS)}"
        )

    if len(widths) == 1 and requested_stats == ["mean"]:
        return binned_mean_amplitude(epochs, widths[0], ch_names)

    data = epochs.get_data(picks=ch_names)
    out = {"epoch": epochs.selection}
    for width in widths:
        bins = (epochs.times // width).astype(int)
        for b in np.unique(bins):
            mask = bins == b
            values = data[:, :, mask]
            for stat in requested_stats:
                summary = _summarize(values, stat)
                for idx, ch in enumerate(ch_names):
                    out[_column_name(width, ch, stat, int(b))] = summary[:, idx]
    return pd.DataFrame(out)


def _summarize(values: np.ndarray, stat: str) -> np.ndarray:
    if stat == "mean":
        return values.mean(axis=2)
    if stat == "std":
        return values.std(axis=2)
    if stat == "min":
        return values.min(axis=2)
    if stat == "max":
        return values.max(axis=2)
    if stat == "median":
        return np.median(values, axis=2)
    raise ValueError(f"Unsupported amplitude stat: {stat}")


def _column_name(width: float, channel: str, stat: str, bin_idx: int) -> str:
    return f"amp_w{_time_token(width)}_{channel}_{stat}_bin_{bin_idx}"


def _time_token(value: float) -> str:
    return str(float(value)).replace(".", "p").replace("-", "m")
