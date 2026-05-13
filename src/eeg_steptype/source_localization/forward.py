"""Build (and cache) the forward solution shared across participants.

The forward solution depends only on the sensor montage + head model + trans,
none of which vary across participants once bad channels are interpolated
back to the canonical montage. We therefore build it ONCE per project,
cache it under a single shared filename, and reuse it for every participant
and every condition. Each per-participant call:

    1. Loads the shared cache from disk (or builds it on the first call).
    2. Subsets the forward to that participant's actual channel set if
       different (e.g. an uninterpolatable channel was dropped).

This guarantees the same leadfield matrix underlies every participant's
inverse operator, removes ~30× duplicate computation, and keeps the on-disk
footprint to a single ~150 MB file instead of one per participant.
"""

from __future__ import annotations

from pathlib import Path

import mne

from ..io import data_root, ensure_dir
from ..logging_utils import get_logger
from ..preflight import locate_source_assets
from ..resources import resolve_n_jobs


log = get_logger(__name__)


SHARED_FWD_FILENAME = "fsaverage-shared-fwd.fif"


def shared_fwd_cache_path(cfg: dict) -> Path:
    """Project-wide path for the canonical fsaverage forward solution."""
    return ensure_dir(data_root(cfg) / "src" / "fwd") / SHARED_FWD_FILENAME


def fwd_cache_path(cfg: dict, participant_id: str | None = None) -> Path:
    """Backwards-compatible alias. The cache is now project-wide, not per
    participant, so ``participant_id`` is ignored. Retained so downstream
    tools that imported the old name keep working."""
    return shared_fwd_cache_path(cfg)


def build_forward(
    info: mne.Info,
    cfg: dict,
    *,
    participant_id: str | None = None,
) -> mne.Forward:
    """Return the shared fsaverage forward, aligned to ``info``'s channels.

    Builds the forward on the FIRST call (or when caching is disabled),
    caches it under a single project-wide filename, and loads it from disk
    on every subsequent call. ``participant_id`` is used only for logging.
    """
    sl = cfg["source_localization"]
    use_cache = sl.get("cache_forward", True)
    cache = shared_fwd_cache_path(cfg) if use_cache else None
    tag = f"[{participant_id}] " if participant_id else ""

    if cache is not None and cache.exists():
        log.info("%sLoading shared forward: %s", tag, cache)
        fwd = mne.read_forward_solution(str(cache))
    else:
        fwd = _build_shared_forward(info, cfg, tag=tag)
        if cache is not None:
            mne.write_forward_solution(str(cache), fwd, overwrite=True)
            log.info("%sCached shared forward to %s", tag, cache)

    return _align_forward_to_info(fwd, info, participant_id=participant_id)


def _build_shared_forward(info: mne.Info, cfg: dict, *, tag: str) -> mne.Forward:
    """Build the forward solution from the fsaverage head model + ``info``."""
    assets = locate_source_assets(cfg)
    src = mne.read_source_spaces(str(assets["source_localization.src_file"]))
    bem = mne.read_bem_solution(str(assets["source_localization.bem_file"]))
    trans = str(assets["source_localization.trans_file"])

    log.info("%sBuilding shared forward solution from fsaverage head model …", tag)
    _validate_forward_info(info)
    return mne.make_forward_solution(
        info,
        trans=trans,
        src=src,
        bem=bem,
        meg=False,
        eeg=True,
        n_jobs=resolve_n_jobs(cfg, default=-8),
    )


def _align_forward_to_info(
    fwd: mne.Forward,
    info: mne.Info,
    *,
    participant_id: str | None,
) -> mne.Forward:
    """Subset the shared forward to the channels actually present in ``info``.

    With clean bad-channel interpolation, every participant has the canonical
    montage and no subsetting is needed (returns ``fwd`` unchanged). If a
    participant's channel set differs, we restrict the forward to the
    intersection and log it so the divergence is visible.
    """
    fwd_chs = set(fwd["info"]["ch_names"])
    info_chs_in_fwd = [ch for ch in info["ch_names"] if ch in fwd_chs]
    info_chs_missing = [ch for ch in info["ch_names"] if ch not in fwd_chs]
    tag = f"[{participant_id}] " if participant_id else ""

    if info_chs_missing:
        log.warning(
            "%s%d channel(s) in participant info are not in the shared forward "
            "and will be excluded from source modeling: %s",
            tag, len(info_chs_missing), info_chs_missing,
        )

    fwd_only = sorted(fwd_chs - set(info["ch_names"]))
    if fwd_only:
        log.info(
            "%sSubsetting shared forward from %d to %d channels "
            "(dropping channels not present in this participant: %s)",
            tag, len(fwd_chs), len(info_chs_in_fwd), fwd_only,
        )
        return mne.pick_channels_forward(
            fwd, include=info_chs_in_fwd, ordered=False, verbose=False
        )

    return fwd


def _validate_forward_info(info: mne.Info) -> None:
    """Fail clearly before MNE's lower-level forward setup."""
    eeg_picks = mne.pick_types(info, eeg=True, meg=False, exclude=[])
    if len(eeg_picks) == 0:
        types = sorted(set(info.get_channel_types()))
        raise RuntimeError(
            "No EEG channels found in source-localization epochs. "
            f"Channel types present: {types}. Source localization must use "
            "data/interim/source_epochs, not final CSD epochs."
        )
