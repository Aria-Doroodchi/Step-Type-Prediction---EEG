"""Assemble all feature blocks (amplitude, slopes, PSD, src) into one wide
DataFrame per (participant, condition), and cache to parquet.

Reading from parquet is *much* faster than recomputing from .fif, so model
runs become near-instant after the first preprocess+features pass.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from ..config import apply_participant_override
from ..io import (
    epochs_path,
    src_csv_path,
    features_path,
    write_parquet,
    read_parquet,
)
from ..logging_utils import get_logger
from .amplitude import binned_mean_amplitude
from .psd import band_power, freq_array, freq_bands
from .slopes import binned_slopes


log = get_logger(__name__)


def build_for_participant_condition(
    participant_id: str,
    condition: str,
    cfg: dict,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Return the wide feature DataFrame, building+caching if necessary."""
    out = features_path(cfg, participant_id, condition)
    if out.exists() and not force:
        log.info("[%s/%s] features cached; loading from %s",
                 participant_id, condition, out)
        return read_parquet(out)

    epo_path = epochs_path(cfg, participant_id, condition)
    if not epo_path.exists():
        raise FileNotFoundError(f"Missing epochs file: {epo_path}")

    epochs = mne.read_epochs(str(epo_path), preload=True)

    fcfg = cfg["features"]
    bin_n = float(fcfg["bin_n"])
    epochs = epochs.crop(tmin=float(fcfg["min_time"]), tmax=float(fcfg["max_time"]))

    ch_names = [c for c in epochs.ch_names if c != "Stim"]

    blocks: list[pd.DataFrame] = []
    requested = set(fcfg.get("blocks", ["amplitude", "slopes", "psd", "src"]))

    if "slopes" in requested:
        log.info("[%s/%s] slopes …", participant_id, condition)
        blocks.append(binned_slopes(epochs, bin_n, ch_names))

    if "amplitude" in requested:
        log.info("[%s/%s] amplitude …", participant_id, condition)
        blocks.append(binned_mean_amplitude(epochs, bin_n, ch_names))

    if "psd" in requested:
        log.info("[%s/%s] PSD (Morlet) …", participant_id, condition)
        blocks.append(band_power(
            epochs, bin_n, ch_names,
            freqs=freq_array(cfg),
            freq_bands=freq_bands(cfg),
        ))

    if "src" in requested:
        src_path = src_csv_path(cfg, participant_id, condition)
        if src_path.exists():
            log.info("[%s/%s] joining src csv: %s", participant_id, condition, src_path)
            blocks.append(pd.read_csv(src_path))
        else:
            log.warning("[%s/%s] src csv missing (%s); features will not include "
                        "source-space columns.", participant_id, condition, src_path)

    if not blocks:
        raise RuntimeError("No feature blocks requested.")

    df = blocks[0]
    for b in blocks[1:]:
        df = df.merge(b, on="epoch")

    df["condition"] = condition
    df["participant_id"] = participant_id

    write_parquet(df, out)
    log.info("[%s/%s] wrote features %s (shape=%s)",
             participant_id, condition, out, df.shape)
    return df


def build_for_participant(
    participant_id: str,
    cfg: dict,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Concatenate per-condition feature DataFrames for a participant."""
    cfg = apply_participant_override(cfg, participant_id)
    parts = []
    for cond in cfg["conditions"]:
        parts.append(build_for_participant_condition(
            participant_id, cond, cfg, force=force,
        ))
    return pd.concat(parts, ignore_index=True)


def correlation_drop(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Drop one of every pair of features with |corr| > threshold."""
    num = df.select_dtypes(include=[np.number])
    corr = num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    return df.drop(columns=to_drop)


def run(participant_id: str, cfg: dict, *, force: bool = False) -> None:
    """Stage entry point: build (and cache) features for one participant."""
    build_for_participant(participant_id, cfg, force=force)
