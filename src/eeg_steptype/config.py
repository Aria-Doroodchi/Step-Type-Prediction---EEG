"""Configuration loading and merging.

The pipeline is driven entirely by layered YAML config. Layers are
deep-merged in this order (right-hand wins on conflicts):

    1. configs/preprocessing/<profile>.yaml
       Global preprocessing spec — filters, ICA, rejection, epoching.
       Selected by `preprocessing_profile:` in the cohort config (default
       "default"). Lives in its own file so you can A/B preprocessing
       variants for ML tuning without disturbing the rest of the config.

    2. configs/default.yaml  (or whatever `--config` points at)
       Cohort config: paths, participants, modeling, source localization.

    3. configs/local.yaml    (gitignored)
       Per-machine path overrides (mainly `paths.raw_root`).

    4. configs/overrides/Pxx.yaml
       Per-participant tweaks (raw_assembly cuts/concats, bads_extra,
       per-participant filter or ICA window). Applied last and only when
       processing that participant.

Public API:
    load_config()                       layers 1-3
    apply_participant_override(cfg, pid) layer 4 on top
    list_preprocessing_profiles()        names of YAMLs under configs/preprocessing/
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
PREPROCESSING_DIR      = _PKG_ROOT / "configs" / "preprocessing"
PROJECT_ROOT           = _PKG_ROOT


# ---------------------------------------------------------------------------
def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Recursive dict merge; right-hand values win on conflicts.

    Lists are *replaced* wholesale, not concatenated.
    """
    out = copy.deepcopy(dict(base))
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ---------------------------------------------------------------------------
def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(
    config_path: str | Path | None = None,
    local_path: str | Path | None = None,
    preprocessing_profile: str | None = None,
) -> dict:
    """Return the merged config as a plain dict.

    Layering (right-hand wins):
        preprocessing/<profile>.yaml  <-  default.yaml  <-  local.yaml

    The active preprocessing profile is chosen by, in priority order:
      (1) the explicit `preprocessing_profile` argument,
      (2) `preprocessing_profile:` in default.yaml or local.yaml,
      (3) "default" if neither is set.

    Resolves relative entries under `paths` to absolute paths.
    """
    cfg_path   = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    local_path = Path(local_path)  if local_path  else LOCAL_CONFIG_PATH

    cohort = load_yaml(cfg_path)
    local  = load_yaml(local_path)

    profile_name = (
        preprocessing_profile
        or local.get("preprocessing_profile")
        or cohort.get("preprocessing_profile")
        or "default"
    )
    profile = _load_preprocessing_profile(profile_name)

    cfg = deep_merge(profile, cohort)
    cfg = deep_merge(cfg, local)

    # Stamp the resolved profile name into the merged config so it ends up
    # in the run snapshot — invaluable for reproducing past results.
    cfg["preprocessing_profile"] = profile_name

    cfg = _resolve_paths(cfg)
    return cfg


def apply_participant_override(cfg: dict, participant_id: str) -> dict:
    """Apply configs/overrides/{participant_id}.yaml on top of cfg.

    Returns a new dict; the input is not mutated.
    """
    override = load_yaml(OVERRIDES_DIR / f"{participant_id}.yaml")
    if not override:
        return cfg
    merged = deep_merge(cfg, override)
    merged.setdefault("_participant_overrides", {})[participant_id] = override
    return merged


def list_preprocessing_profiles() -> list[str]:
    """Return the names of every YAML in configs/preprocessing/ (no extension)."""
    if not PREPROCESSING_DIR.exists():
        return []
    return sorted(p.stem for p in PREPROCESSING_DIR.glob("*.yaml"))


# ---------------------------------------------------------------------------
def _load_preprocessing_profile(name: str) -> dict:
    """Load configs/preprocessing/<name>.yaml. Hard-error on missing non-default."""
    path = PREPROCESSING_DIR / f"{name}.yaml"
    profile = load_yaml(path)
    if not profile and name != "default":
        raise FileNotFoundError(
            f"Preprocessing profile {name!r} not found at {path}. "
            f"Available: {list_preprocessing_profiles()}"
        )
    return profile


def _resolve_paths(cfg: dict) -> dict:
    """Resolve entries under `paths` to absolute paths."""
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
