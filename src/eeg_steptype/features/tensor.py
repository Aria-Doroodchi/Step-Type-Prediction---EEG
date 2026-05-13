"""Epoch-tensor cache for tensor-input models (Riemannian, future CNN).

The classical features stage (``assemble.py``) emits a wide parquet of binned
mean amplitudes, slopes, PSD band powers, and source activations -- one row per
epoch. That representation is fine for tree- and kernel-based classifiers but
throws away the temporal structure that Riemannian and convolutional models
need. This module caches the alternative: ``(n_epochs, n_channels, n_times)``
plus per-epoch metadata, written as a single compressed ``.npz`` next to the
flat feature parquet.

The cache mirrors the parquet path layout, so building both is cheap and the
two representations can coexist on disk without colliding.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from ..config import apply_participant_override
from ..io import ensure_dir, epoch_tensor_path, epochs_path
from ..logging_utils import get_logger


log = get_logger(__name__)


# ---------------------------------------------------------------------------
def build_tensor_for_participant_condition(
    participant_id: str,
    condition: str,
    cfg: dict,
    *,
    force: bool = False,
) -> dict:
    """Return (and cache) the epoch tensor + metadata for one (pid, cond).

    Returned dict shape::

        {
            "data":       np.ndarray (n_epochs, n_channels, n_times),
            "ch_names":   np.ndarray<str> (n_channels,),
            "block_ids":  np.ndarray<str> (n_epochs,),
            "selection":  np.ndarray<int> (n_epochs,),
            "sfreq":      float,
            "tmin":       float,
            "tmax":       float,
        }
    """
    out = epoch_tensor_path(cfg, participant_id, condition)
    if out.exists() and not force:
        log.info("[%s/%s] tensor cached; loading %s",
                 participant_id, condition, out)
        return _load_npz(out)

    epo_path = epochs_path(cfg, participant_id, condition)
    if not epo_path.exists():
        raise FileNotFoundError(f"Missing epochs file: {epo_path}")

    epochs = mne.read_epochs(str(epo_path), preload=True)

    fcfg = cfg["features"]
    tmin = float(fcfg["min_time"])
    tmax = min(float(fcfg["max_time"]), float(epochs.tmax))
    epochs = epochs.crop(tmin=tmin, tmax=tmax)

    ch_names = [c for c in epochs.ch_names if c != "Stim"]
    data = epochs.get_data(picks=ch_names)  # (n_epochs, n_channels, n_times)
    block_ids = _block_ids(epochs)

    payload = {
        "data": data.astype(np.float64, copy=False),
        "ch_names": np.array(ch_names, dtype=object),
        "block_ids": np.asarray(block_ids, dtype=object),
        "selection": np.asarray(epochs.selection, dtype=int),
        "sfreq": float(epochs.info["sfreq"]),
        "tmin": tmin,
        "tmax": tmax,
    }

    ensure_dir(out.parent)
    np.savez_compressed(
        out,
        data=payload["data"],
        ch_names=payload["ch_names"],
        block_ids=payload["block_ids"],
        selection=payload["selection"],
        sfreq=np.array(payload["sfreq"]),
        tmin=np.array(payload["tmin"]),
        tmax=np.array(payload["tmax"]),
    )
    log.info("[%s/%s] wrote tensor %s (shape=%s)",
             participant_id, condition, out, data.shape)
    return payload


def build_tensor_for_participant(
    participant_id: str,
    cfg: dict,
    *,
    force: bool = False,
) -> dict:
    """Concatenate per-condition tensors for one participant.

    Returned dict shape::

        {
            "data":       (n_epochs_total, n_channels, n_times),
            "labels":     np.ndarray<str> ("One" / "Two") (n_epochs_total,),
            "block_ids":  np.ndarray<str> (n_epochs_total,),
            "ch_names":   np.ndarray<str> (n_channels,),
            "selection":  np.ndarray<int> (n_epochs_total,),
            "sfreq":      float,
            "tmin":       float,
            "tmax":       float,
        }
    """
    cfg = apply_participant_override(cfg, participant_id)
    parts = []
    labels = []
    for cond in cfg["conditions"]:
        bundle = build_tensor_for_participant_condition(
            participant_id, cond, cfg, force=force,
        )
        parts.append(bundle)
        labels.extend([cond] * bundle["data"].shape[0])

    # Channels and sample rate are uniform across conditions for one participant.
    ch_names = parts[0]["ch_names"]
    sfreq = parts[0]["sfreq"]
    tmin = parts[0]["tmin"]
    tmax = parts[0]["tmax"]
    if not all(len(p["ch_names"]) == len(ch_names) for p in parts):
        raise RuntimeError(
            f"[{participant_id}] inconsistent channel counts across conditions"
        )

    data = np.concatenate([p["data"] for p in parts], axis=0)
    block_ids = np.concatenate([p["block_ids"] for p in parts], axis=0)
    selection = np.concatenate([p["selection"] for p in parts], axis=0)
    return {
        "data": data,
        "labels": np.array(labels, dtype=object),
        "ch_names": ch_names,
        "block_ids": block_ids,
        "selection": selection,
        "sfreq": sfreq,
        "tmin": tmin,
        "tmax": tmax,
    }


def run(participant_id: str, cfg: dict, *, force: bool = False) -> None:
    """Stage entry point: build (and cache) the tensor for one participant."""
    build_tensor_for_participant(participant_id, cfg, force=force)


# ---------------------------------------------------------------------------
def _block_ids(epochs: mne.BaseEpochs) -> np.ndarray:
    metadata = epochs.metadata
    if metadata is not None and "block_id" in metadata.columns:
        return metadata["block_id"].to_numpy()
    if metadata is not None and "block" in metadata.columns:
        return metadata["block"].to_numpy()
    return np.full(len(epochs), "unknown", dtype=object)


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as npz:
        return {
            "data": npz["data"],
            "ch_names": npz["ch_names"],
            "block_ids": npz["block_ids"],
            "selection": npz["selection"],
            "sfreq": float(npz["sfreq"]),
            "tmin": float(npz["tmin"]),
            "tmax": float(npz["tmax"]),
        }
