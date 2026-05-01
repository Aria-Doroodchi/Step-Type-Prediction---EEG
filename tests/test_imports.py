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
    "eeg_steptype.source_localization",
    "eeg_steptype.source_localization.forward",
    "eeg_steptype.source_localization.inverse",
    "eeg_steptype.source_localization.labels",
    "eeg_steptype.source_localization.pipeline",
    "eeg_steptype.features",
    "eeg_steptype.features.amplitude",
    "eeg_steptype.features.slopes",
    "eeg_steptype.features.psd",
    "eeg_steptype.features.assemble",
    "eeg_steptype.models",
    "eeg_steptype.models.feature_selection",
    "eeg_steptype.models.xgb",
    "eeg_steptype.models.svm",
    "eeg_steptype.models.logistic",
    "eeg_steptype.models.evaluate",
    "eeg_steptype.models.train",
    "eeg_steptype.viz",
    "eeg_steptype.viz.results",
]


@pytest.mark.parametrize("module_name", PACKAGE_MODULES)
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)


# ---------- config + override layering ----------
def test_config_load(smoke_config_path):
    from eeg_steptype.config import load_config
    cfg = load_config(smoke_config_path)
    assert cfg["participants"] == ["P25"]
    assert cfg["modeling"]["k_best"] == 50


def test_overrides_apply():
    from eeg_steptype.config import load_config, apply_participant_override
    cfg = load_config()
    p02 = apply_participant_override(cfg, "P02")
    files = [f if isinstance(f, str) else f["path"] for f in p02["raw_assembly"]["files"]]
    assert any("P02_CNV_2.bdf" in f for f in files)


def test_p37_montage_swap():
    from eeg_steptype.config import load_config, apply_participant_override
    cfg = load_config()
    p37 = apply_participant_override(cfg, "P37")
    assert p37["montage_mapping_override"]["B17"] == "CP6"
    assert p37["montage_mapping_override"]["B22"] == "C2"


def test_p08_raw_crops():
    from eeg_steptype.config import load_config, apply_participant_override
    cfg = load_config()
    p08 = apply_participant_override(cfg, "P08")
    crops = [(f["tmin"], f["tmax"]) for f in p08["raw_assembly"]["files"]]
    assert crops == [(72.0, 135.0), (215.0, 1100.0)]


# ---------- preprocessing-profile mechanism ----------
def test_preprocessing_profile_default():
    from eeg_steptype.config import load_config
    cfg = load_config()
    assert cfg["preprocessing_profile"] == "default"
    assert cfg["preprocessing"]["filter"]["bandpass"] == [0.1, 40]
    assert cfg["preprocessing"]["ica"]["n_components"] == 20
    assert cfg["preprocessing"]["reject"]["method"] == "autoreject"


def test_preprocessing_profile_smoke(smoke_config_path):
    from eeg_steptype.config import load_config
    cfg = load_config(smoke_config_path)
    assert cfg["preprocessing_profile"] == "smoke"
    assert cfg["preprocessing"]["filter"]["bandpass"] == [0.5, 30]
    assert cfg["preprocessing"]["ica"]["n_components"] == 12
    assert cfg["preprocessing"]["reject"]["method"] == "threshold"


def test_preprocessing_profile_arg_override():
    from eeg_steptype.config import load_config
    cfg = load_config(preprocessing_profile="smoke")
    assert cfg["preprocessing_profile"] == "smoke"
    assert cfg["preprocessing"]["ica"]["n_components"] == 12


def test_preprocessing_profile_missing_raises():
    from eeg_steptype.config import load_config
    with pytest.raises(FileNotFoundError):
        load_config(preprocessing_profile="does_not_exist")


def test_list_preprocessing_profiles():
    from eeg_steptype.config import list_preprocessing_profiles
    profs = list_preprocessing_profiles()
    assert "default" in profs
    assert "smoke" in profs


def test_per_participant_override_beats_profile():
    """P25 has its own filter override; it should win over the default profile."""
    from eeg_steptype.config import load_config, apply_participant_override
    cfg = load_config()
    assert cfg["preprocessing"]["filter"]["bandpass"] == [0.1, 40]   # profile default
    p25 = apply_participant_override(cfg, "P25")
    assert p25["preprocessing"]["filter"]["bandpass"] == [0.15, 36]  # P25 wins
