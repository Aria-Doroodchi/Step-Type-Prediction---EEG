"""Filesystem helpers and standard path layout.

Centralised here so no other module hard-codes "data/interim/..." string
fragments. If we ever rename or reshuffle the data folders, this is the only
file that has to change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# Standard path builders
# ---------------------------------------------------------------------------
def data_root(cfg: dict) -> Path:
    return Path(cfg["paths"]["data_dir"])


def raw_root(cfg: dict) -> Path:
    return Path(cfg["paths"]["raw_root"])


def outputs_root(cfg: dict) -> Path:
    return Path(cfg["paths"]["outputs_dir"])


def epochs_path(cfg: dict, participant_id: str, condition: str) -> Path:
    """Cleaned, condition-split epoch .fif file produced by 01_preprocess."""
    return data_root(cfg) / "interim" / "epochs" / f"{participant_id}_CNV_{condition}-epo.fif"


def source_epochs_path(cfg: dict, participant_id: str, condition: str) -> Path:
    """EEG-potential epochs for source localization before final CSD."""
    return data_root(cfg) / "interim" / "source_epochs" / f"{participant_id}_CNV_{condition}-epo.fif"


def src_csv_path(cfg: dict, participant_id: str, condition: str) -> Path:
    """Source-localized per-epoch label time-courses produced by 02_source_localize."""
    return data_root(cfg) / "src" / f"{participant_id}_{condition}_src.csv"


def features_path(cfg: dict, participant_id: str, condition: str) -> Path:
    """Wide feature matrix per (participant, condition) produced by 03_extract_features."""
    fcfg = cfg.get("features", {})
    suffix = _feature_window_suffix(fcfg)
    return data_root(cfg) / "features" / f"{participant_id}_{condition}_features{suffix}.parquet"


def epoch_tensor_path(cfg: dict, participant_id: str, condition: str) -> Path:
    """Cached (n_epochs, n_channels, n_times) tensor for tensor-input models.

    Used by the Riemannian (and future CNN) training path. Carries per-epoch
    block_ids and the original MNE selection so block-grouped CV and
    chronological ordering can be reconstructed without re-reading the
    .fif epoch file inside every CV fold.
    """
    fcfg = cfg.get("features", {})
    suffix = _feature_window_suffix(fcfg)
    return (
        data_root(cfg)
        / "features" / "tensor"
        / f"{participant_id}_{condition}_epochs{suffix}.npz"
    )


def qc_report_path(cfg: dict, participant_id: str) -> Path:
    return outputs_root(cfg) / "qc" / f"{participant_id}.html"


def run_dir(cfg: dict, run_id: str) -> Path:
    return outputs_root(cfg) / "runs" / run_id


def _feature_window_suffix(fcfg: dict) -> str:
    if "min_time" not in fcfg or "max_time" not in fcfg:
        return ""
    return f"_t{_time_token(fcfg['min_time'])}-{_time_token(fcfg['max_time'])}"


def _time_token(value) -> str:
    return str(float(value)).replace(".", "p").replace("-", "m")


# ---------------------------------------------------------------------------
# IO conveniences
# ---------------------------------------------------------------------------
def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def existing(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if Path(p).exists()]
