"""Persistent diagnostics for the source-localization stage."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..io import ensure_dir, outputs_root


FILE_ERROR_COLUMNS = [
    "timestamp_utc",
    "participant_id",
    "condition",
    "stage",
    "path",
    "exception_type",
    "message",
]

VARIANCE_COLUMNS = [
    "timestamp_utc",
    "row_type",
    "participant_id",
    "condition",
    "epoch",
    "n_epochs",
    "variance_explained",
    "variance_explained_percent",
    "data_variance",
    "residual_variance",
    "mean_variance_explained",
    "std_variance_explained",
    "min_variance_explained",
    "max_variance_explained",
]


def source_diagnostics_dir(cfg: dict) -> Path:
    return ensure_dir(outputs_root(cfg) / "diagnostics" / "source_localization")


def file_errors_path(cfg: dict) -> Path:
    return source_diagnostics_dir(cfg) / "file_errors.csv"


def variance_explained_path(cfg: dict) -> Path:
    return source_diagnostics_dir(cfg) / "variance_explained.csv"


def log_file_error(
    cfg: dict,
    *,
    participant_id: str | None,
    condition: str | None,
    stage: str,
    path: str | Path | None,
    exception: BaseException | None = None,
    message: str | None = None,
) -> None:
    """Append a durable record of a file-related source-localization failure."""
    row = {
        "timestamp_utc": _utc_now(),
        "participant_id": participant_id or "",
        "condition": condition or "",
        "stage": stage,
        "path": str(path or ""),
        "exception_type": type(exception).__name__ if exception else "FileNotFoundError",
        "message": message if message is not None else str(exception or ""),
    }
    _append_rows(file_errors_path(cfg), [row], FILE_ERROR_COLUMNS)


def clear_variance_for_participant(cfg: dict, participant_id: str) -> None:
    """Remove stale variance diagnostics before a forced participant rerun."""
    path = variance_explained_path(cfg)
    if not path.exists():
        return
    df = pd.read_csv(path)
    if "participant_id" not in df.columns:
        return
    df = df[df["participant_id"].astype(str) != str(participant_id)]
    df.to_csv(path, index=False)


def append_variance_rows(cfg: dict, rows: Iterable[dict]) -> None:
    timestamp = _utc_now()
    rows = [dict(row) for row in rows]
    for row in rows:
        row["timestamp_utc"] = row.get("timestamp_utc") or timestamp
    _append_rows(variance_explained_path(cfg), rows, VARIANCE_COLUMNS)


def make_variance_summary_rows(
    epoch_rows: list[dict],
    *,
    participant_id: str,
) -> list[dict]:
    """Create per-condition and per-participant summary rows."""
    if not epoch_rows:
        return []

    timestamp = _utc_now()
    df = pd.DataFrame(epoch_rows)
    summaries: list[dict] = []
    for condition, group in df.groupby("condition", dropna=False):
        summaries.append(_summary_row(
            timestamp=timestamp,
            row_type="participant_condition_average",
            participant_id=participant_id,
            condition=str(condition),
            values=group["variance_explained"],
        ))
    summaries.append(_summary_row(
        timestamp=timestamp,
        row_type="participant_average",
        participant_id=participant_id,
        condition="all",
        values=df["variance_explained"],
    ))
    return summaries


def _summary_row(
    *,
    timestamp: str,
    row_type: str,
    participant_id: str,
    condition: str,
    values: pd.Series,
) -> dict:
    values = pd.to_numeric(values, errors="coerce").dropna()
    n_epochs = int(values.shape[0])
    mean = float(values.mean()) if n_epochs else float("nan")
    std = float(values.std(ddof=0)) if n_epochs else float("nan")
    min_value = float(values.min()) if n_epochs else float("nan")
    max_value = float(values.max()) if n_epochs else float("nan")
    return {
        "timestamp_utc": timestamp,
        "row_type": row_type,
        "participant_id": participant_id,
        "condition": condition,
        "epoch": "",
        "n_epochs": n_epochs,
        "variance_explained": "",
        "variance_explained_percent": "",
        "data_variance": "",
        "residual_variance": "",
        "mean_variance_explained": mean,
        "std_variance_explained": std,
        "min_variance_explained": min_value,
        "max_variance_explained": max_value,
    }


def _append_rows(path: Path, rows: Iterable[dict], columns: list[str]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    df = df[columns]
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
