"""BioSemi 64-channel constants used during raw load.

These are the same for every participant (they're a property of the recording
hardware, not the subject). If a single participant needs a deviation
— e.g. P37 needs B17 and B22 swapped on the lab's recording — set
``montage_mapping_override`` in that participant's overrides YAML.
"""

from __future__ import annotations


# Channels to keep from the raw .bdf (32 from the 'A' bank, 32 from 'B', plus Status).
PICK_CHANNELS: list[str] = [
    *(f"A{i}" for i in range(1, 33)),
    *(f"B{i}" for i in range(1, 33)),
    "Status",
]


# Standard BioSemi A/B → 10-20 names. Kept as a single dict so a participant
# override can replace individual entries (e.g. {"B17": "C2", "B22": "CP6"}).
CHANNEL_MAPPING: dict[str, str] = {
    "A1": "Fp1", "A2": "AF7", "A3": "AF3", "A4": "F1",  "A5": "F3",  "A6": "F5",
    "A7": "F7",  "A8": "FT7", "A9": "FC5", "A10": "FC3", "A11": "FC1",
    "A12": "C1", "A13": "C3", "A14": "C5", "A15": "T7", "A16": "TP7",
    "A17": "CP5", "A18": "CP3", "A19": "CP1", "A20": "P1", "A21": "P3",
    "A22": "P5", "A23": "P7", "A24": "P9", "A25": "PO7", "A26": "PO3",
    "A27": "O1", "A28": "Iz", "A29": "Oz", "A30": "POz", "A31": "Pz",
    "A32": "CPz", "B1": "Fpz", "B2": "Fp2", "B3": "AF8", "B4": "AF4",
    "B5": "AFz", "B6": "Fz", "B7": "F2", "B8": "F4", "B9": "F6",
    "B10": "F8", "B11": "FT8", "B12": "FC6", "B13": "FC4", "B14": "FC2",
    "B15": "FCz", "B16": "Cz", "B17": "C2", "B18": "C4", "B19": "C6",
    "B20": "T8", "B21": "TP8", "B22": "CP6", "B23": "CP4", "B24": "CP2",
    "B25": "P2", "B26": "P4", "B27": "P6", "B28": "P8", "B29": "P10",
    "B30": "PO8", "B31": "PO4", "B32": "O2",
    "Status": "Stim",
}

MONTAGE = "biosemi64"
