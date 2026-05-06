"""Interpretable late-CNV motor-channel amplitude benchmark features."""

from __future__ import annotations

import mne
import pandas as pd

from .amplitude import binned_mean_amplitude


DEFAULT_CNV_CHANNELS = ["Cz", "FCz", "CPz", "C1", "C2", "FC1", "FC2", "CP1", "CP2"]


def cnv_motor_amplitude_benchmark(
    epochs: mne.BaseEpochs,
    *,
    bin_n: float = 0.25,
    channels: list[str] | None = None,
) -> pd.DataFrame:
    """Mean amplitude in 250 ms bins over medial foot-motor channels."""
    requested = channels or DEFAULT_CNV_CHANNELS
    present = [ch for ch in requested if ch in epochs.ch_names]
    if not present:
        raise ValueError("None of the configured CNV benchmark channels are present")
    df = binned_mean_amplitude(epochs, bin_n, present)
    rename = {
        col: f"cnv_benchmark_{col}"
        for col in df.columns
        if col != "epoch"
    }
    return df.rename(columns=rename)
