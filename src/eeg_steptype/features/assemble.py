"""Assemble all feature blocks (amplitude, slopes, PSD, src) into one wide
DataFrame per (participant, condition), and cache to parquet.

Reading from parquet is *much* faster than recomputing from .fif, so model
runs become near-instant after the first preprocess+features pass.
"""

from __future__ import annotations

import re

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
from ..resources import resolve_n_jobs
from .amplitude import binned_amplitude_features
from .cnv_benchmark import cnv_motor_amplitude_benchmark
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
    tmin = float(fcfg["min_time"])
    tmax = min(float(fcfg["max_time"]), float(epochs.tmax))
    epochs = epochs.crop(tmin=tmin, tmax=tmax)

    ch_names = [c for c in epochs.ch_names if c != "Stim"]

    blocks: list[pd.DataFrame] = []
    requested = set(fcfg.get("blocks", ["amplitude", "slopes", "psd", "src"]))

    if "slopes" in requested:
        log.info("[%s/%s] slopes …", participant_id, condition)
        blocks.append(binned_slopes(epochs, bin_n, ch_names))

    if "amplitude" in requested:
        acfg = fcfg.get("amplitude", {}) or {}
        amp_widths = acfg.get("bin_widths", [bin_n])
        amp_stats = acfg.get("stats", ["mean"])
        log.info(
            "[%s/%s] amplitude (widths=%s, stats=%s) …",
            participant_id, condition, amp_widths, amp_stats,
        )
        blocks.append(binned_amplitude_features(
            epochs,
            bin_widths=amp_widths,
            stats=amp_stats,
            ch_names=ch_names,
        ))

    if "psd" in requested:
        log.info("[%s/%s] PSD (Morlet) …", participant_id, condition)
        blocks.append(band_power(
            epochs, bin_n, ch_names,
            freqs=freq_array(cfg),
            freq_bands=freq_bands(cfg),
            n_jobs=resolve_n_jobs(cfg, default=-8),
        ))

    if "cnv_benchmark" in requested and fcfg.get("cnv_benchmark", {}).get("enabled", True):
        bcfg = fcfg.get("cnv_benchmark", {})
        log.info("[%s/%s] CNV benchmark amplitude …", participant_id, condition)
        blocks.append(cnv_motor_amplitude_benchmark(
            epochs,
            bin_n=float(bcfg.get("bin_n", 0.25)),
            channels=bcfg.get("channels"),
        ))

    if "src" in requested:
        src_path = src_csv_path(cfg, participant_id, condition)
        if src_path.exists():
            log.info("[%s/%s] joining src csv: %s", participant_id, condition, src_path)
            blocks.append(_filter_src_window(
                pd.read_csv(src_path),
                bin_n=bin_n,
                tmin=tmin,
                tmax=tmax,
            ))
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
    df["block_id"] = _block_ids(epochs)

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


def _block_ids(epochs: mne.Epochs) -> np.ndarray:
    """Return per-epoch block labels when recording metadata provides them."""
    metadata = epochs.metadata
    if metadata is not None and "block_id" in metadata.columns:
        return metadata["block_id"].to_numpy()
    if metadata is not None and "block" in metadata.columns:
        return metadata["block"].to_numpy()
    return np.full(len(epochs), "unknown", dtype=object)


def _filter_src_window(
    df: pd.DataFrame,
    *,
    bin_n: float,
    tmin: float,
    tmax: float,
) -> pd.DataFrame:
    """Keep only source-space bins that overlap the active prediction window."""
    keep = [
        col for col in df.columns
        if col == "epoch" or _src_bin_overlaps_window(col, bin_n, tmin, tmax)
    ]
    return df[keep]


def _src_bin_overlaps_window(column: str, bin_n: float, tmin: float, tmax: float) -> bool:
    m = re.search(r"_bin_(\d+)$", column)
    if not m:
        return True
    start = int(m.group(1)) * bin_n
    end = start + bin_n
    return start < tmax and end > tmin


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
