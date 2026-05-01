"""Static smoke test: every package module imports cleanly.

This catches syntax errors and broken cross-module references in <1 second
without touching real data. Always run this BEFORE the slower
`test_smoke_pipeline.py` — if imports break, nothing else can pass.
"""

import importlib

import pytest


PACKAGE_MODULES = [
    "eeg_steptype",
    "eeg_steptype.config",
    "eeg_steptype.io",
    "eeg_steptype.logging_utils",
    # preprocessing
    "eeg_steptype.preprocessing",
    "eeg_steptype.preprocessing.montage",
    "eeg_steptype.preprocessing.load",
    "eeg_steptype.preprocessing.bads",
    "eeg_steptype.preprocessing.filter",
    "eeg_steptype.preprocessing.reference",
    "eeg_steptype.preprocessing.ica",
    "eeg_steptype.preprocessing.events",
    "eeg_steptype.preprocessing.epoching",
    "eeg_steptype.preprocessing.reject",
    "eeg_steptype.preprocessing.pipeline",
    # source_localization
    "eeg_steptype.source_localization",
    "eeg_steptype.source_localization.forward",
    "eeg_steptype.source_localization.inverse",
    "eeg_steptype.source_localization.labels",
    "eeg_steptype.source_localization.pipeline",
    # features
    "eeg_steptype.features",
    "eeg_steptype.features.amplitude",
    "eeg_steptype.features.slopes",
    "eeg_steptype.features.psd",
    "eeg_steptype.features.assemble",
    # models
    "eeg_steptype.models",
    "eeg_steptype.models.feature_selection",
    "eeg_steptype.models.xgb",
    "eeg_steptype.models.svm",
    "eeg_steptype.models.logistic",
    "eeg_steptype.models.evaluate",
    "eeg_steptype.models.train",
    # viz
    "eeg_steptype.viz",
    "eeg_steptype.viz.results",
]


@pytest.mark.parametrize("module_name", PACKAGE_MODULES)
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)


def test_config_load(smoke_config_path):
    from eeg_steptype.config import load_config
    cfg = load_config(smoke_config_path)
    assert "participants" in cfg
    assert cfg["participants"] == ["P25"]
    assert cfg["modeling"]["k_best"] == 50


def test_overrides_apply():
    from eeg_steptype.config import load_config, apply_participant_override
    cfg = load_config()
    # P02 has a multi-file raw_assembly in its override.
    p02 = apply_participant_override(cfg, "P02")
    assert "raw_assembly" in p02
    files = [f if isinstance(f, str) else f["path"] for f in p02["raw_assembly"]["files"]]
    assert any("P02_CNV_2.bdf" in f for f in files), \
        "P02 override should include the second file"


def test_p37_montage_swap():
    """P37 has B17/B22 swapped — make sure the override carries that mapping."""
    from eeg_steptype.config import load_config, apply_participant_override
    cfg = load_config()
    p37 = apply_participant_override(cfg, "P37")
    mp = p37.get("montage_mapping_override", {})
    assert mp.get("B17") == "CP6"
    assert mp.get("B22") == "C2"


def test_p08_raw_crops():
    """P08 has two crop windows from one file — the schema must preserve both."""
    from eeg_steptype.config import load_config, apply_participant_override
    cfg = load_config()
    p08 = apply_participant_override(cfg, "P08")
    files = p08["raw_assembly"]["files"]
    assert len(files) == 2
    crops = [(f["tmin"], f["tmax"]) for f in files]
    assert crops == [(72.0, 135.0), (215.0, 1100.0)]
