"""Raw assembly for the state module.

Loads ``{raw_root}/{pid}/{pid}_{Stim,Standing}.bdf`` with optional per-participant
crop/concat from ``configs/state/overrides/Pxx.yaml`` (keyed by source file), then
applies channel picks, the A/B→10-20 rename, and the biosemi64 montage. Robust to
the two naming schemes in the cohort: most files store A1–B32 names; a few (e.g.
P09) already store 10-20 names. The trigger channel ``Status`` is always present
and is renamed to ``Stim`` to match the CNV/stim event logic.
"""

from __future__ import annotations

from pathlib import Path

import mne

from eeg_steptype.preprocessing.montage import PICK_CHANNELS, CHANNEL_MAPPING, MONTAGE
from eeg_steptype.preprocessing.load import _normalize_specs, _crop_repr
from ..logging_utils import get_logger


log = get_logger(__name__)

_EEG64 = [c for c in PICK_CHANNELS if c != "Status"]            # A1..B32
_STIM_SRC_NAMES = ("Status", "Stim")


def default_raw_files(participant_id: str, source_file: str) -> list[dict]:
    return [{"path": f"{participant_id}/{participant_id}_{source_file}.bdf"}]


def _assembly_specs(cfg: dict, participant_id: str, source_file: str) -> list[dict]:
    assembly = (cfg.get("raw_assembly") or {}).get(source_file)
    if assembly and assembly.get("files"):
        return _normalize_specs(assembly["files"])
    return _normalize_specs(default_raw_files(participant_id, source_file))


def source_file_present(cfg: dict, participant_id: str, source_file: str) -> bool:
    """True if every raw file the assembly needs exists on disk."""
    raw_root = Path(cfg["paths"]["raw_root"])
    return all((raw_root / s["path"]).exists()
               for s in _assembly_specs(cfg, participant_id, source_file))


def load_state_raw(cfg: dict, participant_id: str, source_file: str) -> mne.io.BaseRaw:
    """Load + assemble one source recording with picks/rename/montage applied."""
    raw_root = Path(cfg["paths"]["raw_root"])
    specs = _assembly_specs(cfg, participant_id, source_file)

    raws = []
    for spec in specs:
        path = raw_root / spec["path"]
        log.info("[%s/%s] loading %s%s", participant_id, source_file, path,
                 _crop_repr(spec.get("tmin"), spec.get("tmax")))
        raw = mne.io.read_raw_bdf(str(path), preload=True, verbose="ERROR")
        if spec.get("tmin") is not None or spec.get("tmax") is not None:
            raw.crop(tmin=spec.get("tmin") or 0.0, tmax=spec.get("tmax"))
        raws.append(raw)
    raw = raws[0] if len(raws) == 1 else mne.concatenate_raws(raws)

    raw = _pick_rename_montage(raw, cfg, participant_id)
    return raw


def _pick_rename_montage(raw: mne.io.BaseRaw, cfg: dict, pid: str) -> mne.io.BaseRaw:
    montage_name = cfg.get("montage_name", MONTAGE)
    montage = mne.channels.make_standard_montage(montage_name)
    montage_names = set(montage.ch_names)

    stim_src = next((c for c in _STIM_SRC_NAMES if c in raw.ch_names), None)
    ab_present = [c for c in _EEG64 if c in raw.ch_names]
    renamed_present = [c for c in raw.ch_names if c in montage_names]

    if len(ab_present) >= 32:
        keep = ab_present + ([stim_src] if stim_src else [])
        raw.pick(keep)
        mapping = {k: v for k, v in CHANNEL_MAPPING.items() if k in raw.ch_names}
        mapping.update(cfg.get("montage_mapping_override", {}))
        raw.rename_channels(mapping)
    elif len(renamed_present) >= 32:
        keep = renamed_present + ([stim_src] if stim_src else [])
        raw.pick(keep)
        if stim_src == "Status":
            raw.rename_channels({"Status": "Stim"})
        if cfg.get("montage_mapping_override"):
            raw.rename_channels(cfg["montage_mapping_override"])
    else:
        raise ValueError(
            f"[{pid}] cannot identify 64 EEG channels (have {raw.ch_names[:6]}...)"
        )

    raw.set_montage(montage, on_missing="ignore")
    return raw
