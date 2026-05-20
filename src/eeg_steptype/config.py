"""Configuration loading and merging.

The pipeline is driven entirely by YAML config:

    configs/default.yaml      ← committed; project defaults
    configs/local.yaml        ← gitignored; per-machine path overrides
    configs/overrides/Pxx.yaml ← gitignored or committed; per-participant tweaks

`load_config()` merges default <- local <- config override on every call.
`apply_participant_override(cfg, pid)` further merges in the per-participant
file when iterating over a specific participant. By default, only
``raw_assembly`` is applied so preprocessing parameters stay uniform across
the cohort; full per-participant tuning remains available as an explicit mode.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


# Resolved at import time so sub-modules can reach config files without
# carrying paths around.
_PKG_ROOT = Path(__file__).resolve().parents[2]   # repo root, not /src
DEFAULT_CONFIG_PATH    = _PKG_ROOT / "configs" / "default.yaml"
LOCAL_CONFIG_PATH      = _PKG_ROOT / "configs" / "local.yaml"
OVERRIDES_DIR          = _PKG_ROOT / "configs" / "overrides"
PROJECT_ROOT           = _PKG_ROOT


# ---------------------------------------------------------------------------
# Deep-merge utility
# ---------------------------------------------------------------------------
def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Recursive dict merge; right-hand values win on conflicts.

    Lists are *replaced* wholesale, not concatenated. Use a different key if
    you need additive semantics (e.g. ``bads_extra`` instead of ``bads``).
    """
    out = copy.deepcopy(dict(base))
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(
    config_path: str | Path | Sequence[str | Path] | None = None,
    local_path: str | Path | None = None,
) -> dict:
    """Return the merged default+local+config config as a plain dict.

    Resolves relative path entries under ``paths`` to absolute paths rooted
    at the repo root.
    """
    cfg_paths = _config_paths(config_path)
    local_path = Path(local_path)  if local_path  else LOCAL_CONFIG_PATH

    cfg = load_yaml(DEFAULT_CONFIG_PATH)
    cfg = deep_merge(cfg, load_yaml(local_path))
    for cfg_path in cfg_paths:
        cfg = deep_merge(cfg, load_yaml(cfg_path))
    cfg = _resolve_paths(cfg)
    return cfg


def apply_participant_override(
    cfg: dict,
    participant_id: str,
    mode: str | None = None,
) -> dict:
    """Apply ``configs/overrides/{participant_id}.yaml`` on top of ``cfg``.

    ``mode`` controls how much of the participant file is applied:

    - ``raw_assembly_only``: only manual raw-file crops/appends are merged.
    - ``full``: merge the whole participant override for fine-tuning.
    - ``none``: ignore participant overrides entirely.

    Returns a *new* dict; the input is not mutated.
    """
    override = load_yaml(OVERRIDES_DIR / f"{participant_id}.yaml")
    if not override:
        return cfg
    mode = mode or cfg.get("participant_overrides", {}).get(
        "mode", "raw_assembly_only"
    )
    selected = _select_participant_override(override, mode)
    if not selected:
        return cfg
    merged = deep_merge(cfg, selected)
    merged.setdefault("_participant_overrides", {})[participant_id] = selected
    merged.setdefault("_participant_override_mode", mode)
    return merged


def apply_prediction_window(cfg: dict, window_name: str | None) -> dict:
    """Return config with ``features.min_time/max_time`` set from a named window."""
    if not window_name:
        return cfg
    windows = cfg.get("prediction_windows", {})
    selected = None
    if windows.get("primary", {}).get("name") == window_name:
        selected = windows["primary"]
    else:
        selected = windows.get("secondary", {}).get(window_name)
    if selected is None:
        names = [windows.get("primary", {}).get("name")]
        names.extend((windows.get("secondary") or {}).keys())
        names = [n for n in names if n]
        raise ValueError(f"Unknown prediction window {window_name!r}; choose from {names}")
    out = copy.deepcopy(cfg)
    out.setdefault("features", {})["min_time"] = selected["min_time"]
    out.setdefault("features", {})["max_time"] = selected["max_time"]
    out.setdefault("_prediction_window", window_name)
    return out


def _config_paths(config_path: str | Path | Sequence[str | Path] | None) -> list[Path]:
    if config_path is None:
        return []
    if isinstance(config_path, (str, Path)):
        return [Path(config_path)]
    return [Path(p) for p in config_path]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _resolve_paths(cfg: dict) -> dict:
    """Resolve entries under ``paths`` to absolute paths.

    `raw_root` is left alone (it's typically outside the repo). The other
    paths are resolved relative to the project root if they aren't already
    absolute.
    """
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
    """Return the participant override subset for the requested mode."""
    if mode == "none":
        return {}
    if mode == "raw_assembly_only":
        if "raw_assembly" not in override:
            return {}
        return {"raw_assembly": copy.deepcopy(override["raw_assembly"])}
    if mode == "full":
        return override
    raise ValueError(
        "participant override mode must be one of: "
        "raw_assembly_only, full, none"
    )
