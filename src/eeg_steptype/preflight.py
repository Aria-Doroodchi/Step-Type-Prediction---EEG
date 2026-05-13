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


def source_asset_paths(cfg: dict) -> dict[str, Path | str]:
    """Return the actual paths the source stage will use, including fallbacks."""
    return locate_source_assets(cfg)


def locate_source_assets(cfg: dict) -> dict[str, Path | str]:
    sl = cfg.get("source_localization", {})
    return {
        "source_localization.src_file": _first_existing([
            sl.get("src_file", ""),
            "fsaverage/bem/fsaverage-ico-5-src.fif",
            "fsaverage/bem/fsaverage-ico-4-src.fif",
            *_mne_fsaverage_candidates("bem/fsaverage-ico-5-src.fif"),
        ]),
        "source_localization.bem_file": _first_existing([
            sl.get("bem_file", ""),
            "fsaverage/bem/fsaverage-5120-5120-5120-bem-sol.fif",
            *_mne_fsaverage_candidates("bem/fsaverage-5120-5120-5120-bem-sol.fif"),
        ]),
        "source_localization.trans_file": _resolve_trans(sl.get("trans_file", "fsaverage")),
    }


def _resolve_trans(value: str | Path) -> Path | str:
    if value == "fsaverage":
        return "fsaverage"
    return _first_existing([
        value,
        "fsaverage/bem/fsaverage-trans.fif",
        "fsaverage/fsaverage-trans.fif",
        *_mne_fsaverage_candidates("bem/fsaverage-trans.fif"),
        *_mne_fsaverage_candidates("fsaverage-trans.fif"),
    ])


def validate_source_assets(cfg: dict) -> None:
    """Raise a clear error if required fsaverage source assets are missing."""
    assets = source_asset_paths(cfg)
    missing = {
        key: path
        for key, path in assets.items()
        if path != "fsaverage" and not Path(path).exists()
    }
    subjects_dir = locate_subjects_dir(cfg)
    label_dir = subjects_dir / "fsaverage" / "label"
    if not label_dir.exists():
        missing["source_localization.subjects_dir labels"] = label_dir
    if not missing:
        _validate_bem_model_for_eeg(Path(assets["source_localization.bem_file"]))
        return
    lines = ["Missing source-localization asset files:"]
    lines.extend(f"  - {key}: {path}" for key, path in missing.items())
    lines.append(
        "Copy these files into the repo root or set absolute paths under "
        "source_localization in configs/local.yaml."
    )
    raise FileNotFoundError("\n".join(lines))


def _validate_bem_model_for_eeg(path: Path) -> None:
    """Reject one-layer/homogeneous BEMs before MNE forward modeling does."""
    try:
        bem = mne.read_bem_solution(str(path), verbose=False)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Configured BEM is not a BEM solution: {path}\n"
            "For EEG, use the fsaverage 3-layer solution named "
            "fsaverage-5120-5120-5120-bem-sol.fif."
        ) from exc
    n_layers = len(bem.get("surfs", []))
    if n_layers < 3:
        raise RuntimeError(
            f"Configured BEM has {n_layers} layer(s): {path}\n"
            "MNE requires a 3-layer BEM for EEG forward calculations "
            "(inner skull, outer skull, and scalp). Use "
            "fsaverage/bem/fsaverage-5120-5120-5120-bem-sol.fif or run "
            "mne.datasets.fetch_fsaverage() to install the fsaverage BEM."
        )


def run_preflight(cfg: dict) -> None:
    """Run all external-file checks needed before a real full workflow."""
    validate_source_assets(cfg)


def locate_subjects_dir(cfg: dict) -> Path:
    """Return a subjects_dir containing fsaverage labels, with MNE fallbacks."""
    sl = cfg.get("source_localization", {})
    configured = sl.get("subjects_dir", ".")
    candidates = [
        resolve_project_path(configured),
        *[root for root in _mne_data_roots()],
        *[root / "MNE-fsaverage-data" for root in _mne_data_roots()],
    ]
    for path in candidates:
        if (path / "fsaverage" / "label").exists():
            return path
    return candidates[0]


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


def _mne_fsaverage_candidates(relative: str) -> list[Path]:
    return [
        p / "fsaverage" / relative
        for p in _mne_data_roots()
    ]


def _mne_data_roots() -> list[Path]:
    roots: list[Path] = []
    configured = mne.get_config("MNE_DATA", default=None)
    if configured:
        roots.append(Path(configured))
    roots.extend([
        Path.home() / "mne_data" / "MNE-fsaverage-data",
        Path.home() / "mne_data",
        Path(mne.__file__).resolve().parent / "data",
    ])
    out = []
    for root in roots:
        if root not in out:
            out.append(root)
    return out
