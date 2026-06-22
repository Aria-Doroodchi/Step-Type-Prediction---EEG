"""Atomic, resumable result storage for long-running model workflows.

Each independently recoverable unit (for example ``P01/fold_0``) gets its own
directory. Expensive stages can checkpoint JSON or tabular artifacts as they
finish, and final metrics/predictions are committed atomically. A power loss can
therefore affect only files that had not yet reached ``os.replace``.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .io import ensure_dir


_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_component(value: str) -> str:
    component = _SAFE_COMPONENT.sub("_", str(value).strip()).strip("._")
    if not component:
        raise ValueError("Checkpoint identifiers must contain a safe character")
    return component


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """Write bytes durably, then atomically replace ``path``."""
    path = Path(path)
    ensure_dir(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | Path, value: Any) -> Path:
    text = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    return atomic_write_text(path, text)


def atomic_write_csv(path: str | Path, frame: pd.DataFrame) -> Path:
    return atomic_write_text(path, frame.to_csv(index=False))


class StepwiseResultStore:
    """Save progress and results one model unit at a time.

    ``unit_id`` should identify the smallest unit worth rerunning, such as
    ``P01/fold_0``. Every write is atomic. Units are considered resumable until
    ``complete_unit`` writes both their metrics and a final status record.
    """

    def __init__(self, root: str | Path):
        self.root = ensure_dir(Path(root))
        self.units_dir = ensure_dir(self.root / "units")
        self._aggregate_lock = threading.Lock()

    def unit_dir(self, unit_id: str) -> Path:
        parts = [part for part in str(unit_id).replace("\\", "/").split("/") if part]
        if not parts:
            raise ValueError("unit_id cannot be empty")
        return ensure_dir(self.units_dir.joinpath(*(_safe_component(p) for p in parts)))

    def status(self, unit_id: str) -> dict[str, Any] | None:
        path = self.unit_dir(unit_id) / "status.json"
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def is_complete(self, unit_id: str) -> bool:
        value = self.status(unit_id)
        return bool(value and value.get("status") == "complete")

    def start_unit(
        self,
        unit_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        previous = self.status(unit_id) or {}
        value = {
            **previous,
            **dict(metadata or {}),
            "unit_id": unit_id,
            "status": "running",
            "started_at": previous.get("started_at", _utc_now()),
            "updated_at": _utc_now(),
        }
        return atomic_write_json(self.unit_dir(unit_id) / "status.json", value)

    def save_stage(
        self,
        unit_id: str,
        stage: str,
        value: Any,
    ) -> Path:
        """Save a JSON-serializable intermediate stage snapshot."""
        stage_name = _safe_component(stage)
        path = self.unit_dir(unit_id) / "stages" / f"{stage_name}.json"
        written = atomic_write_json(path, value)
        self._record_latest_stage(unit_id, stage_name)
        return written

    def load_stage(self, unit_id: str, stage: str) -> Any:
        path = self.unit_dir(unit_id) / "stages" / f"{_safe_component(stage)}.json"
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def stage_exists(self, unit_id: str, stage: str) -> bool:
        path = self.unit_dir(unit_id) / "stages" / f"{_safe_component(stage)}.json"
        return path.exists()

    def save_table(
        self,
        unit_id: str,
        name: str,
        frame: pd.DataFrame,
    ) -> Path:
        """Save an intermediate or final table, such as predictions."""
        table_name = _safe_component(name)
        return atomic_write_csv(
            self.unit_dir(unit_id) / "tables" / f"{table_name}.csv",
            frame,
        )

    def complete_unit(
        self,
        unit_id: str,
        metrics: Mapping[str, Any],
        *,
        predictions: pd.DataFrame | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Commit final metrics and optional predictions for one unit."""
        unit = self.unit_dir(unit_id)
        metrics_value = {"unit_id": unit_id, **dict(metrics)}
        atomic_write_json(unit / "metrics.json", metrics_value)
        if predictions is not None:
            self.save_table(unit_id, "predictions", predictions)

        previous = self.status(unit_id) or {}
        status = {
            **previous,
            **dict(metadata or {}),
            "unit_id": unit_id,
            "status": "complete",
            "started_at": previous.get("started_at", _utc_now()),
            "completed_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        written = atomic_write_json(unit / "status.json", status)
        self.write_aggregate_metrics()
        return written

    def fail_unit(
        self,
        unit_id: str,
        error: BaseException | str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        previous = self.status(unit_id) or {}
        value = {
            **previous,
            **dict(metadata or {}),
            "unit_id": unit_id,
            "status": "failed",
            "error": repr(error) if isinstance(error, BaseException) else str(error),
            "updated_at": _utc_now(),
        }
        return atomic_write_json(self.unit_dir(unit_id) / "status.json", value)

    def completed_metrics(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.units_dir.rglob("metrics.json")):
            status_path = path.parent / "status.json"
            if not status_path.exists():
                continue
            with status_path.open(encoding="utf-8") as handle:
                status = json.load(handle)
            if status.get("status") != "complete":
                continue
            with path.open(encoding="utf-8") as handle:
                rows.append(json.load(handle))
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["unit_id"])

    def write_aggregate_metrics(
        self,
        filename: str = "metrics.partial.csv",
    ) -> Path:
        """Rebuild an atomic aggregate from every completed unit."""
        with self._aggregate_lock:
            return atomic_write_csv(self.root / filename, self.completed_metrics())

    def _record_latest_stage(self, unit_id: str, stage: str) -> None:
        previous = self.status(unit_id) or {
            "unit_id": unit_id,
            "status": "running",
            "started_at": _utc_now(),
        }
        previous["latest_stage"] = stage
        previous["updated_at"] = _utc_now()
        atomic_write_json(self.unit_dir(unit_id) / "status.json", previous)
