"""Config loading for the state module.

The state config is the CNV config plus state deltas, so all preprocessing and
modeling settings stay in sync with the CNV branch and only the dependent
variable / SEP block / standing epoching / multiclass objective change.

    load_config() merges:  configs/default.yaml  (CNV base)
                        <- configs/local.yaml    (per-machine paths)
                        <- configs/state/default.yaml  (state deltas)
                        <- [explicit overlays]

Per-participant raw-assembly lives in configs/state/overrides/Pxx.yaml and is
applied (raw_assembly only, by default) via apply_participant_override.
Reuses ``deep_merge`` / ``load_yaml`` from ``eeg_steptype.config``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Sequence

from eeg_steptype.config import deep_merge, load_yaml


_PKG_ROOT = Path(__file__).resolve().parents[2]   # repo root, not /src
CNV_DEFAULT_PATH    = _PKG_ROOT / "configs" / "default.yaml"
STATE_DEFAULT_PATH  = _PKG_ROOT / "configs" / "state" / "default.yaml"
LOCAL_CONFIG_PATH   = _PKG_ROOT / "configs" / "local.yaml"
OVERRIDES_DIR       = _PKG_ROOT / "configs" / "state" / "overrides"
PROJECT_ROOT        = _PKG_ROOT


def load_config(
    config_path: str | Path | Sequence[str | Path] | None = None,
    local_path: str | Path | None = None,
) -> dict:
    """Return the merged CNV-base + local + state + overlays config."""
    cfg_paths = _config_paths(config_path)
    local_path = Path(local_path) if local_path else LOCAL_CONFIG_PATH

    cfg = load_yaml(CNV_DEFAULT_PATH)
    cfg = deep_merge(cfg, load_yaml(local_path))
    cfg = deep_merge(cfg, load_yaml(STATE_DEFAULT_PATH))
    for cfg_path in cfg_paths:
        cfg = deep_merge(cfg, load_yaml(cfg_path))
    cfg = _resolve_paths(cfg)
    return cfg


def apply_participant_override(
    cfg: dict,
    participant_id: str,
    mode: str | None = None,
) -> dict:
    """Apply ``configs/state/overrides/{participant_id}.yaml`` on top of cfg.

    Modes mirror the CNV branch: ``raw_assembly_only`` (default) merges only the
    manual raw-file crops/concats; ``full`` merges everything; ``none`` ignores.
    """
    override = load_yaml(OVERRIDES_DIR / f"{participant_id}.yaml")
    if not override:
        return cfg
    mode = mode or cfg.get("participant_overrides", {}).get("mode", "raw_assembly_only")
    selected = _select_participant_override(override, mode)
    if not selected:
        return cfg
    merged = deep_merge(cfg, selected)
    merged.setdefault("_participant_overrides", {})[participant_id] = selected
    merged.setdefault("_participant_override_mode", mode)
    return merged


# ---------------------------------------------------------------------------
# Internal helpers (small copies from eeg_steptype.config to avoid private
# cross-package imports).
# ---------------------------------------------------------------------------
def _config_paths(config_path) -> list[Path]:
    if config_path is None:
        return []
    if isinstance(config_path, (str, Path)):
        return [Path(config_path)]
    return [Path(p) for p in config_path]


def _resolve_paths(cfg: dict) -> dict:
    paths = cfg.get("paths", {})
    for k, v in list(paths.items()):
        if not v:
            continue
        if k == "raw_root":
            paths[k] = str(Path(v))
            continue
        p = Path(v)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        paths[k] = str(p)
    cfg["paths"] = paths
    return cfg


def _select_participant_override(override: dict, mode: str) -> dict:
    if mode == "none":
        return {}
    if mode == "raw_assembly_only":
        if "raw_assembly" not in override:
            return {}
        return {"raw_assembly": copy.deepcopy(override["raw_assembly"])}
    if mode == "full":
        return override
    raise ValueError("participant override mode must be one of: raw_assembly_only, full, none")
