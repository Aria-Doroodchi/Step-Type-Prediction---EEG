"""Build (and optionally cache) the forward solution.

Forward depends only on sensor montage + head model + trans, none of which
change across epochs. Cache it per participant to avoid rebuilding.
"""

from __future__ import annotations

from pathlib import Path

import mne

from ..io import data_root, ensure_dir
from ..logging_utils import get_logger


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

    sl = cfg["source_localization"]
    src = mne.read_source_spaces(sl["src_file"])
    bem = mne.read_bem_solution(sl["bem_file"])
    trans = sl["trans_file"]

    log.info("Building forward solution …")
    fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem, n_jobs=-1)

    if cache is not None:
        mne.write_forward_solution(str(cache), fwd, overwrite=True)
        log.info("Cached forward to %s", cache)
    return fwd
