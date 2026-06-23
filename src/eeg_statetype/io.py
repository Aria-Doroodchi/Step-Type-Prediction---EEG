"""Filesystem layout for the 3-state module.

Sibling of ``eeg_steptype.io`` with a distinct namespace so state-task artifacts
never collide with the CNV pipeline's. All filenames carry ``_STATE_`` (vs the
CNV ``_CNV_``) and live under state-specific subdirectories. Reuses the CNV
io helpers (suffix builders, parquet/csv/dir conveniences) by import.
"""

from __future__ import annotations

from pathlib import Path

# Reuse the CNV io helpers verbatim — identical behaviour, no duplication.
from eeg_steptype.io import (  # noqa: F401
    data_root,
    raw_root,
    outputs_root,
    ensure_dir,
    write_parquet,
    read_parquet,
    write_csv,
    existing,
    timestamp_token,
    stamped_path,
    stamped_dir,
    _feature_window_suffix,
    _bin_width_suffix,
    _feature_cache_tag_suffix,
)


# ---------------------------------------------------------------------------
# State-task path builders. ``condition`` ∈ {standing, straight, diagonal}.
# ---------------------------------------------------------------------------
def epochs_path(cfg: dict, participant_id: str, condition: str) -> Path:
    """CSD analysis epochs (window features: amplitude/slopes/psd)."""
    return (
        data_root(cfg) / "interim" / "state_epochs"
        / f"{participant_id}_STATE_{condition}-epo.fif"
    )


def source_epochs_path(cfg: dict, participant_id: str, condition: str) -> Path:
    """Non-CSD avg-ref analysis epochs (source localization input)."""
    return (
        data_root(cfg) / "interim" / "state_source_epochs"
        / f"{participant_id}_STATE_{condition}-epo.fif"
    )


def sep_epochs_path(cfg: dict, participant_id: str, condition: str) -> Path:
    """Non-CSD avg-ref per-e-stim SEP epochs (SEP feature block input).

    Each SEP epoch carries metadata ``parent_epoch`` linking it to the analysis
    epoch (by original event index) it was attributed to, so the SEP block can
    average per analysis epoch without re-reading the raw recording.
    """
    return (
        data_root(cfg) / "interim" / "state_sep_epochs"
        / f"{participant_id}_STATE_{condition}-sep-epo.fif"
    )


def src_csv_path(cfg: dict, participant_id: str, condition: str) -> Path:
    """Source-localized per-epoch label time-courses."""
    slcfg = cfg.get("source_localization", {})
    suffix = _bin_width_suffix(slcfg.get("bin_n"))
    return data_root(cfg) / "src_state" / f"{participant_id}_{condition}_src{suffix}.csv"


def features_path(cfg: dict, participant_id: str, condition: str) -> Path:
    """Wide feature matrix per (participant, condition) — own cache namespace."""
    fcfg = cfg.get("features", {})
    suffix = (
        _feature_window_suffix(fcfg)
        + _bin_width_suffix(fcfg.get("bin_n"))
        + _feature_cache_tag_suffix(fcfg)
    )
    return (
        data_root(cfg) / "features_state"
        / f"{participant_id}_{condition}_features{suffix}.parquet"
    )


def qc_report_path(cfg: dict, participant_id: str) -> Path:
    return outputs_root(cfg) / "state_module" / "qc" / f"{participant_id}.html"


def run_dir(cfg: dict, run_id: str) -> Path:
    return outputs_root(cfg) / "state_module" / "runs" / run_id


def ledger_path(cfg: dict) -> Path:
    return outputs_root(cfg) / "state_module" / "LEDGER.md"
