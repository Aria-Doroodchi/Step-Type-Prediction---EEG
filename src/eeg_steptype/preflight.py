"""Preflight checks for files that synthetic tests cannot exercise."""

from __future__ import annotations

from pathlib import Path

import mne

from .config import PROJECT_ROOT
from .io import source_epochs_path


def resolve_project_path(path: str | Path) -> Path:
    """Resolve repo-relative paths used in config files."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def source_asset_paths(cfg: dict) -> dict[str, Path]:
    """Return the actual paths the source stage will use, including fallbacks."""
    return locate_source_assets(cfg)


def locate_source_assets(cfg: dict) -> dict[str, Path]:
    sl = cfg.get("source_localization", {})
    return {
        "source_localization.src_file": _first_existing([
            sl.get("src_file", ""),
            "fsaverage/bem/fsaverage-ico-5-src.fif",
            "fsaverage/bem/fsaverage-ico-4-src.fif",
        ]),
        "source_localization.bem_file": _first_existing([
            sl.get("bem_file", ""),
            "fsaverage/bem/fsaverage-bem-sol.fif",
            "fsaverage/bem/fsaverage-inner_skull-bem-sol.fif",
            *_mne_fsaverage_candidates("fsaverage-inner_skull-bem.fif"),
        ]),
        "source_localization.trans_file": _first_existing([
            sl.get("trans_file", ""),
            "fsaverage/bem/fsaverage-trans.fif",
            "fsaverage/fsaverage-trans.fif",
            *_mne_fsaverage_candidates("fsaverage-trans.fif"),
        ]),
    }


def validate_source_assets(cfg: dict) -> None:
    """Raise a clear error if required fsaverage source assets are missing."""
    missing = {
        key: path
        for key, path in source_asset_paths(cfg).items()
        if not path.exists()
    }
    subjects_dir = resolve_project_path(
        cfg.get("source_localization", {}).get("subjects_dir", ".")
    )
    label_dir = subjects_dir / "fsaverage" / "label"
    if not label_dir.exists():
        missing["source_localization.subjects_dir labels"] = label_dir
    if not missing:
        return
    lines = ["Missing source-localization asset files:"]
    lines.extend(f"  - {key}: {path}" for key, path in missing.items())
    lines.append(
        "Copy these files into the repo root or set absolute paths under "
        "source_localization in configs/local.yaml."
    )
    raise FileNotFoundError("\n".join(lines))


def run_preflight(cfg: dict) -> None:
    """Run all external-file checks needed before a real full workflow."""
    validate_source_assets(cfg)


def validate_source_epochs_file(path: str | Path, *, participant_id: str, condition: str) -> None:
    """Ensure an epoch file can be used by MNE forward modeling."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing source-compatible EEG epochs for {participant_id}/{condition}: {path}\n"
            "Run preprocessing with --force to create data/interim/source_epochs."
        )
    epochs = mne.read_epochs(str(path), preload=False, verbose=False)
    eeg_picks = mne.pick_types(epochs.info, eeg=True, meg=False, exclude=[])
    if len(eeg_picks) == 0:
        types = sorted(set(epochs.get_channel_types()))
        raise RuntimeError(
            f"No EEG channels found in source epochs for {participant_id}/{condition}: {path}\n"
            f"Channel types present: {types}\n"
            "Source localization requires EEG-potential epochs, not final CSD epochs. "
            "Run preprocessing with --force so the pipeline writes "
            "data/interim/source_epochs before applying CSD."
        )


def validate_source_epochs(cfg: dict, participant_id: str, condition: str) -> None:
    validate_source_epochs_file(
        source_epochs_path(cfg, participant_id, condition),
        participant_id=participant_id,
        condition=condition,
    )


def _first_existing(candidates: list[str | Path]) -> Path:
    resolved = [resolve_project_path(c) for c in candidates if c]
    if resolved and Path(candidates[0]).is_absolute() and not resolved[0].exists():
        return resolved[0]
    for path in resolved:
        if path.exists():
            return path
    return resolved[0] if resolved else PROJECT_ROOT


def _mne_fsaverage_candidates(filename: str) -> list[Path]:
    try:
        import mne
    except Exception:
        return []
    base = Path(mne.__file__).resolve().parent / "data" / "fsaverage"
    return [base / filename]
