"""Raw assembly: load one or more .bdf files, optionally cropped, and concat.

Default behaviour: load ``{raw_root}/{pid}/{pid}_CNV.bdf``.

Override schema (in ``configs/overrides/Pxx.yaml``):

    raw_assembly:
      files:
        - "Pxx/file_a.bdf"                                     # full file
        - {path: "Pxx/file_b.bdf", tmin: 0,   tmax: 525}       # crop before concat
        - {path: "Pxx/file_b.bdf", tmin: 580, tmax: 1091}      # second window

Each entry is loaded, optionally cropped, then ``mne.concatenate_raws`` joins
them in order. This faithfully preserves the manual cuts/appends from the
original per-participant scripts (e.g. P02 concat, P08 cut+concat, P14 crop,
P19 crop, P37 cut+concat).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne

from .montage import PICK_CHANNELS, CHANNEL_MAPPING, MONTAGE
from ..logging_utils import get_logger


log = get_logger(__name__)


# ---------------------------------------------------------------------------
def default_raw_files(participant_id: str) -> list[dict]:
    """Return the default single-file recipe for a participant."""
    return [{"path": f"{participant_id}/{participant_id}_CNV.bdf"}]


def load_raw(cfg: dict, participant_id: str) -> mne.io.BaseRaw:
    """Load (and assemble, if needed) the raw recording for one participant.

    Honours ``raw_assembly`` if provided in the participant override; otherwise
    loads the default single file. Applies channel picks, name mapping, and
    montage in all cases.
    """
    raw_root = Path(cfg["paths"]["raw_root"])
    assembly = cfg.get("raw_assembly") or {"files": default_raw_files(participant_id)}
    file_specs = _normalize_specs(assembly["files"])

    raws = []
    for spec in file_specs:
        path = raw_root / spec["path"]
        log.info("Loading %s%s", path,
                 _crop_repr(spec.get("tmin"), spec.get("tmax")))
        raw = mne.io.read_raw_bdf(str(path), preload=True)
        if spec.get("tmin") is not None or spec.get("tmax") is not None:
            raw.crop(tmin=spec.get("tmin") or 0.0, tmax=spec.get("tmax"))
        raws.append(raw)

    raw = raws[0] if len(raws) == 1 else mne.concatenate_raws(raws)

    # Channel picks, rename, montage. Override-supported.
    picks = list(cfg.get("channels_pick", PICK_CHANNELS))
    raw.pick(picks)

    mapping = dict(CHANNEL_MAPPING)
    mapping.update(cfg.get("montage_mapping_override", {}))   # e.g. P37
    raw.rename_channels(mapping)

    montage = mne.channels.make_standard_montage(cfg.get("montage_name", MONTAGE))
    raw.set_montage(montage)

    return raw


# ---------------------------------------------------------------------------
def _normalize_specs(items: list[Any]) -> list[dict]:
    """Accept either bare path strings or {path, tmin, tmax} dicts."""
    out = []
    for it in items:
        if isinstance(it, str):
            out.append({"path": it})
        elif isinstance(it, dict) and "path" in it:
            out.append({k: it.get(k) for k in ("path", "tmin", "tmax")})
        else:
            raise ValueError(f"Bad raw_assembly entry: {it!r}")
    return out


def _crop_repr(tmin, tmax) -> str:
    if tmin is None and tmax is None:
        return ""
    return f" [{tmin}–{tmax}s]"
