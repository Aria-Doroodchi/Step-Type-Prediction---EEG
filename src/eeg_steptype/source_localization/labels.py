"""Parcellation labels (Brodmann-area surrogates from `aparc.a2009s`)."""

from __future__ import annotations

import mne

from ..preflight import locate_subjects_dir


def load_labels(cfg: dict) -> tuple[list, list[str]]:
    """Return (labels, label_names) using the configured parcellation."""
    sl = cfg["source_localization"]
    subjects_dir = locate_subjects_dir(cfg)
    labels = mne.read_labels_from_annot(
        "fsaverage",
        parc=sl.get("parcellation", "aparc.a2009s"),
        subjects_dir=subjects_dir,
    )
    return labels, [lab.name for lab in labels]


def extract_label_courses(stc: mne.SourceEstimate, labels, src) -> list:
    return mne.extract_label_time_course(stc, labels, src=src, mode="mean")
