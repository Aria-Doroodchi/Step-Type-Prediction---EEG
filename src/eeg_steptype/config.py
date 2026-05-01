"""Configuration loading and merging.

The pipeline is driven entirely by YAML config:

    configs/default.yaml      ← committed; project defaults
    configs/local.yaml        ← gitignored; per-machine path overrides
    configs/overrides/Pxx.yaml ← gitignored or committed; per-participant tweaks

`load_config()` merges default ← local on every call.
`apply_participant_override(cfg, pid)` further merges in the per-participant
file when iterating over a specific participant.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

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
    config_path: str | Path | None = None,
    local_path: str | Path | None = None,
) -> dict:
    """Return the merged default+local config as a plain dict.

    Resolves relative path entries under ``paths`` to absolute paths rooted
    at the repo root.
    """
    cfg_path   = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    local_path = Path(local_path)  if local_path  else LOCAL_CONFIG_PATH

    cfg = load_yaml(cfg_path)
    cfg = deep_merge(cfg, load_yaml(local_path))
    cfg = _resolve_paths(cfg)
    return cfg


def apply_participant_override(cfg: dict, participant_id: str) -> dict:
    """Apply ``configs/overrides/{participant_id}.yaml`` on top of ``cfg``.

    Returns a *new* dict; the input is not mutated.
    """
    override = load_yaml(OVERRIDES_DIR / f"{participant_id}.yaml")
    if not override:
        return cfg
    merged = deep_merge(cfg, override)
    merged.setdefault("_participant_overrides", {})[participant_id] = override
    return merged


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
