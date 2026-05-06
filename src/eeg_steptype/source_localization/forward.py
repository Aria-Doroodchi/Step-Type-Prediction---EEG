"""Build (and optionally cache) the forward solution.

Forward depends only on sensor montage + head model + trans, none of which
change across epochs. Cache it per participant to avoid rebuilding.
"""

from __future__ import annotations

from pathlib import Path

import mne

from ..io import data_root, ensure_dir
from ..logging_utils import get_logger
from ..preflight import locate_source_assets
from ..resources import resolve_n_jobs


log = get_logger(__name__)


def fwd_cache_path(cfg: dict, participant_id: str) -> Path:
    return ensure_dir(data_root(cfg) / "src" / "fwd") / f"{participant_id}-fwd.fif"


def build_forward(
    info: mne.Info,
    cfg: dict,
    *,
    participant_id: str | None = None,
) -> mne.Forward:
    """Return a forward solution. If caching is enabled and a cached file
    exists for the participant, load it; otherwise build and save."""
    cache = (
        fwd_cache_path(cfg, participant_id)
        if (participant_id and cfg["source_localization"].get("cache_forward", True))
        else None
    )
    if cache is not None and cache.exists():
        log.info("Loading cached forward: %s", cache)
        return mne.read_forward_solution(str(cache))

    assets = locate_source_assets(cfg)
    src = mne.read_source_spaces(str(assets["source_localization.src_file"]))
    bem = _read_or_make_bem_solution(assets["source_localization.bem_file"])
    trans = str(assets["source_localization.trans_file"])

    log.info("Building forward solution …")
    fwd = mne.make_forward_solution(
        info,
        trans=trans,
        src=src,
        bem=bem,
        n_jobs=resolve_n_jobs(cfg, default=-8),
    )

    if cache is not None:
        mne.write_forward_solution(str(cache), fwd, overwrite=True)
        log.info("Cached forward to %s", cache)
    return fwd


def _read_or_make_bem_solution(path: Path):
    """Read a BEM solution, or build one from a BEM surface file."""
    try:
        return mne.read_bem_solution(str(path))
    except RuntimeError as exc:
        if "No BEM solution found" not in str(exc):
            raise
    surfaces = mne.read_bem_surfaces(str(path))
    log.info("Building BEM solution from surface file: %s", path)
    return mne.make_bem_solution(surfaces)
