"""Assemble feature blocks into one wide DataFrame per (participant, condition).

Reuses the CNV window blocks (amplitude/slopes/psd) and src joining verbatim,
and adds the per-epoch ``sep`` block. The SEP block is **left-joined** onto the
window-block base and missing values filled, so analysis epochs that happen to
have no usable in-epoch e-stim are kept (with filled SEP columns) rather than
dropped by an inner join.
"""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd

from eeg_steptype.features.amplitude import binned_amplitude_features
from eeg_steptype.features.slopes import binned_slopes
from eeg_steptype.features.psd import band_power, freq_array, freq_bands
from eeg_steptype.features.assemble import _filter_src_window, _block_ids

from ..config import apply_participant_override
from ..io import (
    epochs_path,
    src_csv_path,
    sep_epochs_path,
    features_path,
    write_parquet,
    read_parquet,
)
from ..logging_utils import get_logger
from ..resources import resolve_n_jobs
from .sep import sep_features


log = get_logger(__name__)


def build_for_participant_condition(
    participant_id: str,
    condition: str,
    cfg: dict,
    *,
    force: bool = False,
) -> pd.DataFrame:
    out = features_path(cfg, participant_id, condition)
    if out.exists() and not force:
        log.info("[%s/%s] features cached; loading %s", participant_id, condition, out)
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

    requested = list(fcfg.get("blocks", ["amplitude", "slopes", "psd", "src", "sep"]))
    window_blocks: list[pd.DataFrame] = []

    if "slopes" in requested:
        log.info("[%s/%s] slopes …", participant_id, condition)
        window_blocks.append(binned_slopes(epochs, bin_n, ch_names))

    if "amplitude" in requested:
        acfg = fcfg.get("amplitude", {}) or {}
        window_blocks.append(binned_amplitude_features(
            epochs,
            bin_widths=acfg.get("bin_widths", [bin_n]),
            stats=acfg.get("stats", ["mean"]),
            ch_names=ch_names,
        ))

    if "psd" in requested:
        log.info("[%s/%s] PSD (Morlet) …", participant_id, condition)
        window_blocks.append(band_power(
            epochs, bin_n, ch_names,
            freqs=freq_array(cfg), freq_bands=freq_bands(cfg),
            n_jobs=resolve_n_jobs(cfg, default=-8),
        ))

    if "src" in requested:
        src_path = src_csv_path(cfg, participant_id, condition)
        if src_path.exists():
            log.info("[%s/%s] joining src csv: %s", participant_id, condition, src_path)
            window_blocks.append(_filter_src_window(
                pd.read_csv(src_path), bin_n=bin_n, tmin=tmin, tmax=tmax,
            ))
        else:
            log.warning("[%s/%s] src csv missing (%s); no source-space columns.",
                        participant_id, condition, src_path)

    if not window_blocks:
        raise RuntimeError("No window feature blocks requested.")

    # Window blocks share the CSD-epoch index space -> inner-merge; src is a
    # superset (no reject) -> left-merge keeps the survivors.
    df = window_blocks[0]
    for b in window_blocks[1:]:
        df = df.merge(b, on="epoch", how="left")

    # SEP block: left-join + fill so window epochs without an in-epoch e-stim
    # are kept (validity-safe; never broadcasts a per-condition mean).
    if "sep" in requested:
        df = _attach_sep(df, participant_id, condition, cfg)

    df["condition"] = condition
    df["participant_id"] = participant_id
    df["block_id"] = _block_ids(epochs)

    write_parquet(df, out)
    log.info("[%s/%s] wrote features %s (shape=%s)",
             participant_id, condition, out, df.shape)
    return df


def _attach_sep(df: pd.DataFrame, participant_id: str, condition: str,
                cfg: dict) -> pd.DataFrame:
    fill = float(cfg["features"]["sep"].get("missing_fill", 0.0))
    sep_path = sep_epochs_path(cfg, participant_id, condition)
    sep_df = None
    if sep_path.exists():
        sep_epochs = mne.read_epochs(str(sep_path), preload=True)
        sep_df = sep_features(sep_epochs, cfg)
    if sep_df is None or sep_df.empty or sep_df.shape[1] <= 1:
        cols = _canonical_sep_columns(cfg)
        log.warning("[%s/%s] no SEP epochs; filling %d SEP columns with %.3g.",
                    participant_id, condition, len(cols), fill)
        for c in cols:
            df[c] = fill
        df["sep_n_estim"] = 0
        return df

    sep_cols = [c for c in sep_df.columns if c != "epoch"]
    n_before = len(df)
    df = df.merge(sep_df, on="epoch", how="left")
    assert len(df) == n_before, "SEP left-join changed row count"
    n_missing = int(df[sep_cols[0]].isna().sum()) if sep_cols else 0
    df[sep_cols] = df[sep_cols].fillna(fill)
    if n_missing:
        log.info("[%s/%s] %d/%d window epochs had no in-epoch SEP -> filled.",
                 participant_id, condition, n_missing, n_before)
    return df


def _canonical_sep_columns(cfg: dict) -> list[str]:
    scfg = cfg["features"]["sep"]
    metrics = ["p50amp", "p50lat", "n90amp", "n90lat", "p2p", "rms"]
    cols: list[str] = []
    if bool(scfg.get("per_channel", True)):
        for ch in scfg.get("vertex_channels", ["Cz", "C1", "C2", "FCz", "CPz"]):
            cols += [f"sep_{ch}_{m}" for m in metrics]
    cols += [f"sep_vtx_{m}" for m in metrics]
    return cols


def build_for_participant(participant_id: str, cfg: dict, *,
                          force: bool = False) -> pd.DataFrame:
    cfg = apply_participant_override(cfg, participant_id)
    parts = [build_for_participant_condition(participant_id, c, cfg, force=force)
             for c in cfg["conditions"]]
    return pd.concat(parts, ignore_index=True)


def run(participant_id: str, cfg: dict, *, force: bool = False) -> None:
    build_for_participant(participant_id, cfg, force=force)
