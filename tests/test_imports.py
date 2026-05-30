"""Static smoke test: every package module imports cleanly.

This catches syntax errors and broken cross-module references in <1 second
without touching real data. Always run this BEFORE the slower
`test_smoke_pipeline.py` — if imports break, nothing else can pass.
"""

import importlib
import sys
import types

import pytest


PACKAGE_MODULES = [
    "eeg_steptype",
    "eeg_steptype.config",
    "eeg_steptype.io",
    "eeg_steptype.logging_utils",
    "eeg_steptype.preflight",
    "eeg_steptype.resources",
    # preprocessing
    "eeg_steptype.preprocessing",
    "eeg_steptype.preprocessing.montage",
    "eeg_steptype.preprocessing.load",
    "eeg_steptype.preprocessing.bads",
    "eeg_steptype.preprocessing.asr",
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
    "eeg_steptype.source_localization.diagnostics",
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
    "eeg_steptype.models.normalization",
    "eeg_steptype.models.riemannian",
    "eeg_steptype.models.cnn",
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
    assert cfg["conditions"] == ["One", "Two"]
    assert "participants" in cfg
    assert cfg["participants"] == ["P25"]
    assert cfg["resources"]["n_jobs"] == 2
    assert cfg["features"]["min_time"] == 1.0
    assert cfg["features"]["max_time"] == 2.0
    assert cfg["prediction_windows"]["primary"]["name"] == "late_cnv"
    assert cfg["prediction_windows"]["secondary"]["full_cnv"]["cropped_training"]["stride"] == 0.25
    assert cfg["prediction_windows"]["secondary"]["sliding_auc"]["enabled"] is True
    assert cfg["features"]["cnv_benchmark"]["enabled"] is False
    assert cfg["features"]["cnv_benchmark"]["bin_n"] == 0.25
    assert cfg["channel_selection"]["mode"] == "full"
    assert "Cz" in cfg["channel_selection"]["roi"]["channels"]
    assert "CP2" in cfg["channel_selection"]["roi"]["channels"]
    assert cfg["modeling"]["k_best"] == 50
    assert cfg["modeling"]["cv"]["mode"] == "repeated_stratified"
    assert cfg["modeling"]["cv"]["n_splits"] == 2
    assert cfg["modeling"]["cv"]["n_repeats"] == 1
    assert cfg["modeling"]["cv"]["inner_splits"] == 2
    assert cfg["modeling"]["cv"]["chronological_check"] is False
    assert cfg["modeling"]["search"]["method"] == "auto"
    assert cfg["modeling"]["search"]["n_jobs"] is None
    assert cfg["modeling"]["search"]["n_iter"] == 4
    assert cfg["modeling"]["search"]["halving"]["resource"] == "n_estimators"
    assert cfg["modeling"]["riemannian"]["covariance_estimator"] == "oas"
    assert cfg["modeling"]["riemannian"]["estimator"] == "xdawn_covariances"
    assert cfg["modeling"]["riemannian"]["xdawn"]["nfilter"] == 4
    assert cfg["modeling"]["riemannian"]["fbcsp_bands"]["Mu"] == [8.0, 13.0]
    assert cfg["modeling"]["cnn"]["standardize"]["factor_new"] == 0.001
    assert cfg["preprocessing"]["bads"]["reject_by_annotation"] == "omit"
    assert "notch" not in cfg["preprocessing"]["filter"]
    assert cfg["preprocessing"]["line_noise"]["freq"] == 60
    assert cfg["preprocessing"]["line_noise"]["nremove"] == 1
    assert cfg["preprocessing"]["asr"]["k"] == 20
    assert cfg["preprocessing"]["ica"]["method"] == "picard"
    assert cfg["preprocessing"]["ica"]["extended"] is True
    assert cfg["preprocessing"]["ica"]["n_components"] == "rank_minus_one"
    assert cfg["preprocessing"]["ica"]["iclabel_artifact_prob_threshold"] == 0.8
    assert cfg["preprocessing"]["reject"]["method"] == "autoreject"
    assert cfg["preprocessing"]["reject"]["n_interpolate"] == [1, 4, 8, 16]
    assert cfg["preprocessing"]["reject"]["consensus"] == [0.2, 0.4, 0.6, 0.8]
    assert cfg["preprocessing"]["reject"]["random_state"] == 42
    assert cfg["preprocessing"]["reference"]["csd"]["lambda2"] == 1.0e-5
    assert cfg["preprocessing"]["reference"]["csd"]["stiffness"] == 4


def test_overrides_apply():
    from eeg_steptype.config import load_config, apply_participant_override
    cfg = load_config()
    # P02 has a multi-file raw_assembly, which is the default allowed override.
    p02 = apply_participant_override(cfg, "P02")
    assert "raw_assembly" in p02
    files = [f if isinstance(f, str) else f["path"] for f in p02["raw_assembly"]["files"]]
    assert any("P02_CNV_2.bdf" in f for f in files), \
        "P02 override should include the second file"
    assert "bads_extra" not in p02


def test_prediction_window_override_sets_feature_window():
    from eeg_steptype.config import apply_prediction_window, load_config

    cfg = load_config()
    full = apply_prediction_window(cfg, "full_cnv")

    assert full["features"]["min_time"] == 0.0
    assert full["features"]["max_time"] == 2.0
    assert full["_prediction_window"] == "full_cnv"


def test_source_asset_preflight_reports_missing_files(tmp_path):
    from eeg_steptype.preflight import validate_source_assets

    cfg = {
        "source_localization": {
            "src_file": tmp_path / "missing-src.fif",
            "bem_file": tmp_path / "missing-bem.fif",
            "trans_file": tmp_path / "missing-trans.fif",
        }
    }

    with pytest.raises(FileNotFoundError) as exc:
        validate_source_assets(cfg)

    msg = str(exc.value)
    assert "source_localization.src_file" in msg
    assert "configs/local.yaml" in msg


def test_source_file_error_diagnostic_is_persistent(tmp_path):
    from eeg_steptype.source_localization.diagnostics import file_errors_path, log_file_error

    cfg = {"paths": {"outputs_dir": str(tmp_path / "outputs")}}
    missing = tmp_path / "fsaverage" / "bem" / "missing-src.fif"

    log_file_error(
        cfg,
        participant_id="P00",
        condition="One",
        stage="validate_source_assets",
        path=missing,
        message="missing fsaverage source space",
    )

    path = file_errors_path(cfg)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "P00" in text
    assert "missing fsaverage source space" in text


def test_source_asset_preflight_rejects_one_layer_bem(monkeypatch, tmp_path):
    from eeg_steptype import preflight

    bem_path = tmp_path / "one-layer-bem-sol.fif"
    bem_path.write_bytes(b"not a real fif; read is monkeypatched")
    monkeypatch.setattr(
        preflight.mne,
        "read_bem_solution",
        lambda *args, **kwargs: {"surfs": [object()]},
    )

    with pytest.raises(RuntimeError) as exc:
        preflight._validate_bem_model_for_eeg(bem_path)

    assert "3-layer BEM" in str(exc.value)


def test_default_source_asset_preflight_accepts_resolved_fsaverage():
    from eeg_steptype.config import load_config
    from eeg_steptype.preflight import locate_source_assets, run_preflight

    cfg = load_config()
    assets = locate_source_assets(cfg)

    assert assets["source_localization.trans_file"] == "fsaverage"
    assert "5120-5120-5120-bem-sol" in str(assets["source_localization.bem_file"])
    run_preflight(cfg)


def test_source_epoch_preflight_rejects_non_eeg_epochs(tmp_path):
    import mne
    import numpy as np
    from eeg_steptype.preflight import validate_source_epochs_file

    info = mne.create_info(["Cz"], sfreq=100, ch_types=["csd"])
    epochs = mne.EpochsArray(
        np.zeros((1, 1, 10)),
        info,
        events=np.array([[0, 0, 96]]),
        event_id={"96": 96},
        verbose=False,
    )
    path = tmp_path / "bad-csd-epo.fif"
    epochs.save(str(path), overwrite=True)

    with pytest.raises(RuntimeError) as exc:
        validate_source_epochs_file(path, participant_id="P00", condition="One")

    assert "No EEG channels found" in str(exc.value)
    assert "not final CSD epochs" in str(exc.value)


def test_forward_info_preflight_rejects_non_eeg_info():
    import mne
    from eeg_steptype.source_localization.forward import _validate_forward_info

    info = mne.create_info(["Cz"], sfreq=100, ch_types=["csd"])

    with pytest.raises(RuntimeError) as exc:
        _validate_forward_info(info)

    assert "No EEG channels found" in str(exc.value)


def test_source_modeling_adds_average_reference_projector():
    import mne
    import numpy as np
    from eeg_steptype.source_localization.inverse import ensure_average_reference_projection

    info = mne.create_info(["Cz", "Fz", "Pz", "Oz"], sfreq=100, ch_types="eeg")
    info.set_montage("standard_1020")
    epochs = mne.EpochsArray(
        np.zeros((2, 4, 10)),
        info,
        events=np.array([[0, 0, 96], [20, 0, 96]]),
        event_id={"96": 96},
        verbose=False,
    )

    out = ensure_average_reference_projection(epochs)

    assert out is epochs
    assert any(proj["desc"] == "Average EEG reference" for proj in epochs.info["projs"])


def test_feature_path_is_window_aware():
    from eeg_steptype.config import load_config
    from eeg_steptype.io import features_path

    cfg = load_config()
    path = features_path(cfg, "P01", "One")

    assert path.name == "P01_One_features_t1p0-2p0.parquet"


def test_feature_path_includes_optional_cache_tag():
    from eeg_steptype.config import load_config
    from eeg_steptype.io import features_path

    cfg = load_config()
    cfg["features"]["cache_tag"] = "bin_stats_0125"
    path = features_path(cfg, "P01", "One")

    assert path.name == "P01_One_features_t1p0-2p0_bin_stats_0125.parquet"


def test_config_loader_accepts_multiple_overlays(tmp_path):
    from eeg_steptype.config import load_config

    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("features:\n  cache_tag: first\n", encoding="utf-8")
    second.write_text("features:\n  cache_tag: second\n", encoding="utf-8")

    cfg = load_config([first, second])

    assert cfg["features"]["cache_tag"] == "second"


def test_full_override_mode_keeps_fine_tuning_available():
    """Full mode preserves non-raw participant tuning as an opt-in path."""
    from eeg_steptype.config import load_config, apply_participant_override
    cfg = load_config()
    p37 = apply_participant_override(cfg, "P37", mode="full")
    mp = p37.get("montage_mapping_override", {})
    assert mp.get("B17") == "CP6"
    assert mp.get("B22") == "C2"


def test_default_override_mode_keeps_preprocessing_uniform():
    from eeg_steptype.config import load_config, apply_participant_override
    cfg = load_config()
    p25 = apply_participant_override(cfg, "P25")
    assert p25["preprocessing"]["filter"]["bandpass"] == [0.1, 40]
    assert "bads_extra" not in p25


def test_p08_raw_crops():
    """P08 has two crop windows from one file — the schema must preserve both."""
    from eeg_steptype.config import load_config, apply_participant_override
    cfg = load_config()
    p08 = apply_participant_override(cfg, "P08")
    files = p08["raw_assembly"]["files"]
    assert len(files) == 2
    crops = [(f["tmin"], f["tmax"]) for f in files]
    assert crops == [(72.0, 135.0), (215.0, 1100.0)]


def test_pyprep_bad_channel_config_is_forwarded(monkeypatch):
    from eeg_steptype.preprocessing import bads

    calls = {}

    class DummyRaw:
        ch_names = ["Cz", "Stim"]

        def copy(self):
            return self

        def drop_channels(self, channels):
            calls["dropped"] = channels
            return self

    class DummyNoisyChannels:
        def __init__(self, raw, **kwargs):
            calls["init"] = kwargs

        def find_all_bads(self, **kwargs):
            calls["find_all_bads"] = kwargs

        def get_bads(self):
            return ["Cz"]

    fake_module = types.SimpleNamespace(NoisyChannels=DummyNoisyChannels)
    monkeypatch.setitem(sys.modules, "pyprep.find_noisy_channels", fake_module)
    cfg = {
        "preprocessing": {
            "bads": {
                "random_state": 7,
                "ransac": False,
                "channel_wise": True,
                "reject_by_annotation": "omit",
            }
        }
    }

    detected = bads._pyprep_detect(DummyRaw(), cfg["preprocessing"]["bads"])

    assert detected == ["Cz"]
    assert calls["dropped"] == ["Stim"]
    assert calls["init"] == {
        "random_state": 7,
        "ransac": False,
        "reject_by_annotation": "omit",
    }
    assert calls["find_all_bads"] == {
        "channel_wise": True,
        "reject_by_annotation": "omit",
    }


def test_car_can_be_undone_on_preloaded_raw():
    import numpy as np
    import mne
    from eeg_steptype.preprocessing.reference import apply_car, undo_car

    raw = mne.io.RawArray(
        np.arange(12, dtype=float).reshape(3, 4),
        mne.create_info(["Cz", "Fz", "Pz"], sfreq=100, ch_types="eeg"),
        verbose=False,
    )
    original = raw.get_data().copy()

    raw, state = apply_car(raw)
    assert not np.allclose(raw.get_data(), original)
    raw = undo_car(raw, state)

    assert np.allclose(raw.get_data(), original)


def test_csd_config_is_forwarded(monkeypatch):
    import mne
    import numpy as np
    from eeg_steptype.preprocessing import reference

    calls = {}
    raw = mne.io.RawArray(
        np.zeros((3, 10)),
        mne.create_info(["Cz", "Fz", "Pz"], sfreq=100, ch_types="eeg"),
        verbose=False,
    )

    def fake_csd(inst, **kwargs):
        calls.update(kwargs)
        return inst

    monkeypatch.setattr(reference, "compute_current_source_density", fake_csd)
    cfg = {"preprocessing": {"reference": {"csd": {"lambda2": 2e-5, "stiffness": 5}}}}

    assert reference.apply_csd(raw, cfg) is raw
    assert calls == {"lambda2": 2e-5, "stiffness": 5, "copy": False}


def test_zapline_config_is_forwarded(monkeypatch):
    import mne
    import numpy as np
    from eeg_steptype.preprocessing import filter as filt

    calls = {}
    original = np.arange(20, dtype=float).reshape(2, 10)
    raw = mne.io.RawArray(
        original.copy(),
        mne.create_info(["Cz", "Fz"], sfreq=200, ch_types="eeg"),
        verbose=False,
    )

    def fake_dss_line(data, *, fline, sfreq, nremove):
        calls["shape"] = data.shape
        calls["fline"] = fline
        calls["sfreq"] = sfreq
        calls["nremove"] = nremove
        return data + 1, data * 0

    monkeypatch.setattr(filt, "_dss_line", fake_dss_line)
    cfg = {"preprocessing": {"line_noise": {"freq": 60, "nremove": 2}}}

    filt.apply_line_noise_removal(raw, cfg)

    assert calls == {
        "shape": (10, 2, 1),
        "fline": 60.0,
        "sfreq": 200.0,
        "nremove": 2,
    }
    assert np.allclose(raw.get_data(), original + 1)


def test_asr_config_is_forwarded(monkeypatch):
    import mne
    import numpy as np
    from eeg_steptype.preprocessing import asr

    calls = {}
    original = np.arange(20, dtype=float).reshape(2, 10)
    raw = mne.io.RawArray(
        original.copy(),
        mne.create_info(["Cz", "Fz"], sfreq=200, ch_types="eeg"),
        verbose=False,
    )

    class DummyASR:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def fit(self, data):
            calls["fit_shape"] = data.shape

        def transform(self, data):
            calls["transform_shape"] = data.shape
            return data + 2

    monkeypatch.setattr(asr, "_make_asr", lambda **kwargs: DummyASR(**kwargs))
    cfg = {"preprocessing": {"asr": {"k": 20, "method": "euclid", "estimator": "scm"}}}

    asr.apply_asr(raw, cfg)

    assert calls == {
        "init": {"sfreq": 200.0, "cutoff": 20.0, "method": "euclid", "estimator": "scm"},
        "fit_shape": (2, 10),
        "transform_shape": (2, 10),
    }
    assert np.allclose(raw.get_data(), original + 2)


def test_ica_rank_minus_one_resolution(monkeypatch):
    import mne
    import numpy as np
    from eeg_steptype.preprocessing import ica

    raw = mne.io.RawArray(
        np.zeros((4, 20)),
        mne.create_info(["Cz", "Fz", "Pz", "Oz"], sfreq=100, ch_types="eeg"),
        verbose=False,
    )
    monkeypatch.setattr(ica.mne, "compute_rank", lambda *args, **kwargs: {"eeg": 4})
    cfg = {"preprocessing": {"ica": {"n_components": "rank_minus_one"}}}

    assert ica._resolve_n_components(raw, cfg) == 3


def test_preprocessing_dual_filter_ica_order(monkeypatch, tmp_path):
    from eeg_steptype.preprocessing import pipeline
    from eeg_steptype.preprocessing.bads import BadChannelSummary

    calls = []

    class DummyRaw:
        def __init__(self):
            self.info = {"bads": []}

        def interpolate_bads(self, reset_bads=True):
            calls.append("interpolate_bads")
            if reset_bads:
                self.info["bads"] = []
            return self

    class DummyEpochs:
        def __len__(self):
            return 2

        def save(self, path, overwrite=True):
            calls.append("save_epochs")
            return None

    raw = DummyRaw()
    summary = BadChannelSummary(
        pyprep=["Cz"],
        override=[],
        final=["Cz"],
        interpolated=[],
    )

    monkeypatch.setattr(pipeline, "load_raw", lambda cfg, pid: calls.append("load_raw") or raw)
    monkeypatch.setattr(
        pipeline._filter,
        "apply_line_noise_removal",
        lambda raw, cfg: calls.append("line_noise") or raw,
    )
    monkeypatch.setattr(
        pipeline._bads,
        "detect_bads",
        lambda raw, cfg, return_summary=False: calls.append("detect_bads") or (["Cz"], summary),
    )

    def fake_apply_bads(raw, bads):
        calls.append("apply_bads")
        raw.info["bads"] = list(bads)
        return raw

    monkeypatch.setattr(pipeline._bads, "apply_bads", fake_apply_bads)
    monkeypatch.setattr(pipeline._asr, "apply_asr", lambda raw, cfg: calls.append("asr") or raw)
    monkeypatch.setattr(
        pipeline._ref,
        "apply_car",
        lambda raw: calls.append("apply_car") or (raw, "car_state"),
    )
    monkeypatch.setattr(
        pipeline._filter,
        "make_analysis_copy",
        lambda raw, cfg: calls.append("make_analysis_copy") or raw,
    )
    monkeypatch.setattr(
        pipeline._filter,
        "make_ica_training_copy",
        lambda raw, cfg: calls.append("make_ica_training_copy") or raw,
    )
    monkeypatch.setattr(pipeline._ica, "fit_ica", lambda raw, cfg: calls.append("fit_ica") or "ica")
    monkeypatch.setattr(
        pipeline._ica,
        "auto_exclude",
        lambda ica, raw, cfg: calls.append("iclabel") or [0],
    )
    monkeypatch.setattr(
        pipeline._ica,
        "apply_ica",
        lambda ica, raw, excluded: calls.append("apply_ica_to_analysis") or raw,
    )
    monkeypatch.setattr(
        pipeline._ref,
        "undo_car",
        lambda raw, state: calls.append("undo_car") or raw,
    )
    monkeypatch.setattr(pipeline._ref, "apply_csd", lambda raw, cfg: calls.append("apply_csd") or raw)
    monkeypatch.setattr(
        pipeline,
        "find_step_events",
        lambda raw, cfg: calls.append("find_events") or {"One": [1], "Two": []},
    )
    monkeypatch.setattr(
        pipeline,
        "build_epochs",
        lambda raw, events, cfg, cond: calls.append("epoch") or DummyEpochs(),
    )
    monkeypatch.setattr(
        pipeline,
        "reject_epochs",
        lambda epochs, cfg, cond: calls.append("autoreject") or epochs,
    )

    cfg = {
        "participants": ["P01"],
        "conditions": ["One", "Two"],
        "paths": {"data_dir": str(tmp_path / "data"), "outputs_dir": str(tmp_path / "outputs")},
        "preprocessing": {"qc_report": False},
        "participant_overrides": {"mode": "none"},
    }

    pipeline.run("P01", cfg, force=True)

    assert calls[:17] == [
        "load_raw",
        "line_noise",
        "detect_bads",
        "apply_bads",
        "interpolate_bads",
        "asr",
        "apply_car",
        "make_analysis_copy",
        "make_ica_training_copy",
        "fit_ica",
        "iclabel",
        "apply_ica_to_analysis",
        "undo_car",
        "find_events",
        "epoch",
        "save_epochs",
        "apply_csd",
    ]
    assert calls[17:] == ["epoch", "autoreject", "save_epochs"]


def test_autoreject_config_is_forwarded(monkeypatch):
    import numpy as np
    from eeg_steptype.preprocessing import reject

    calls = {}

    class DummyLog:
        bad_epochs = np.array([False, True, False])
        labels = np.array([[0, 2], [1, 1], [0, 0]])

    class DummyAutoReject:
        def __init__(self, **kwargs):
            calls["init"] = kwargs
            self.n_interpolate_ = {"eeg": 4}
            self.consensus_ = {"eeg": 0.4}

        def fit_transform(self, epochs, return_log=False):
            calls["return_log"] = return_log
            return epochs, DummyLog()

    monkeypatch.setitem(
        __import__("sys").modules,
        "autoreject",
        type("FakeAutorejectModule", (), {"AutoReject": DummyAutoReject}),
    )
    epochs = [object(), object(), object()]
    rcfg = {
        "n_interpolate": [1, 4],
        "consensus": [0.2, 0.4],
        "cv": 10,
        "random_state": 7,
        "n_jobs": 2,
    }

    assert reject._autoreject(epochs, rcfg, "One") is epochs
    assert calls["init"] == {
        "n_interpolate": [1, 4],
        "consensus": [0.2, 0.4],
        "cv": 3,
        "random_state": 7,
        "n_jobs": 2,
        "verbose": False,
    }
    assert calls["return_log"] is True


def test_xgb_normalization_is_noop():
    from eeg_steptype.models.normalization import maybe_wrap_estimator

    sentinel = object()
    assert maybe_wrap_estimator(sentinel, "xgb", {}) is sentinel


def test_future_model_param_grid_is_prefixed():
    from sklearn.dummy import DummyClassifier
    from eeg_steptype.models.normalization import maybe_prefix_param_grid, maybe_wrap_estimator

    cfg = {"modeling": {"cnn": {"standardize": {}}}}
    estimator = maybe_wrap_estimator(DummyClassifier(), "cnn", cfg)
    grid = maybe_prefix_param_grid({"strategy": ["most_frequent"]}, estimator)

    assert list(grid) == ["classifier__strategy"]


def test_riemannian_normalizer_uses_covariance_estimator_key():
    from eeg_steptype.models.riemannian import make_normalizer

    norm = make_normalizer({"modeling": {"riemannian": {"covariance_estimator": "oas"}}})

    assert norm.covariance_estimator == "oas"


def test_xdawn_riemannian_pipeline_uses_config():
    from eeg_steptype.models.riemannian import make_xdawn_covariance_pipeline

    pipe = make_xdawn_covariance_pipeline({
        "modeling": {
            "riemannian": {
                "covariance_estimator": "oas",
                "xdawn": {"nfilter": 3},
                "fbcsp_bands": {"Mu": [8.0, 13.0]},
            }
        }
    })

    features = pipe.named_steps["features"]
    assert features.nfilter == 3
    assert features.covariance_estimator == "oas"
    assert features.fbcsp_bands == {"Mu": [8.0, 13.0]}
    assert pipe.named_steps["classifier"].priors == [0.5, 0.5]


def test_sklearn_classifiers_use_balanced_class_weights():
    from eeg_steptype.models.logistic import make_logistic
    from eeg_steptype.models.svm import make_svm

    cfg = {"modeling": {"random_state": 1, "svm": {"param_grid": {}}}}

    assert make_logistic(cfg).class_weight == "balanced"
    assert make_svm(cfg).class_weight == "balanced"


def test_outer_cv_defaults_to_repeated_stratified():
    import pandas as pd
    from eeg_steptype.models.train import _outer_splits

    X = pd.DataFrame({"epoch": range(12), "x": range(12)})
    y = pd.Series([0, 1] * 6)
    cfg = {"modeling": {"random_state": 1, "cv": {
        "mode": "repeated_stratified",
        "n_splits": 3,
        "n_repeats": 2,
        "chronological_check": False,
    }}}

    splits = _outer_splits(X, y, None, cfg)

    assert len(splits) == 6
    assert {s["cv_mode"] for s in splits} == {"repeated_stratified"}
    assert {s["repeat"] for s in splits} == {0, 1}


def test_chronological_check_is_reported_as_separate_cv_mode():
    import pandas as pd
    from eeg_steptype.models.train import _outer_splits

    X = pd.DataFrame({"epoch": [0, 0, 1, 1, 2, 2, 3, 3], "x": range(8)})
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    cfg = {"modeling": {"random_state": 1, "cv": {
        "mode": "repeated_stratified",
        "n_splits": 2,
        "n_repeats": 1,
        "chronological_check": True,
    }}}

    splits = _outer_splits(X, y, None, cfg)

    assert [s["cv_mode"] for s in splits].count("repeated_stratified") == 2
    assert [s["cv_mode"] for s in splits].count("chronological") == 2


def test_xgb_auto_search_uses_halving_random_search():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import HalvingRandomSearchCV
    from eeg_steptype.models.train import _make_search_cv

    cfg = {"modeling": {
        "random_state": 1,
        "search": {
            "method": "auto",
            "n_iter": 5,
            "halving": {
                "resource": "n_estimators",
                "min_resources": 2,
                "max_resources": 8,
                "factor": 2,
            },
        },
        "xgb": {"n_estimators": 8},
    }}
    cv = StratifiedKFold(n_splits=2)

    search = _make_search_cv(
        estimator=RandomForestClassifier(random_state=1),
        param_grid={"max_depth": [1, 2, 3]},
        cfg=cfg,
        model_name="xgb",
        cv=cv,
    )

    assert isinstance(search, HalvingRandomSearchCV)
    assert search.resource == "n_estimators"
    assert search.n_candidates == 3


def test_auto_search_keeps_grid_for_non_xgb_models():
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from eeg_steptype.models.train import _make_search_cv

    cfg = {"modeling": {"random_state": 1, "search": {"method": "auto"}}}
    cv = StratifiedKFold(n_splits=2)

    search = _make_search_cv(
        estimator=DummyClassifier(),
        param_grid={"strategy": ["most_frequent"]},
        cfg=cfg,
        model_name="logistic",
        cv=cv,
    )

    assert isinstance(search, GridSearchCV)


def test_negative_resource_n_jobs_reserves_processors(monkeypatch):
    from eeg_steptype.resources import resolve_n_jobs

    monkeypatch.setattr("os.cpu_count", lambda: 32)

    assert resolve_n_jobs({"resources": {"n_jobs": -8}}) == 24
    assert resolve_n_jobs({"resources": {"n_jobs": -8}}, 4) == 4
    assert resolve_n_jobs({"resources": {"n_jobs": -64}}) == 1


def test_xgb_auto_search_avoids_nested_parallelism_by_default():
    from eeg_steptype.models.train import _search_n_jobs

    cfg = {"resources": {"n_jobs": -8}, "modeling": {"search": {"method": "auto", "n_jobs": None}}}

    assert _search_n_jobs(cfg, "xgb") == 1


def test_roi_channel_selection_filters_electrode_features():
    import pandas as pd
    from eeg_steptype.models.train import _apply_channel_selection

    cfg = {
        "channel_selection": {
            "mode": "roi",
            "roi": {"channels": ["Cz", "FCz"]},
        }
    }
    df = pd.DataFrame({
        "participant_id": ["P01"],
        "condition": ["One"],
        "epoch": [0],
        "Cz_bin_0": [1.0],
        "P8_bin_0": [2.0],
        "amp_w0p125_Cz_std_bin_8": [2.5],
        "amp_w0p125_P8_std_bin_8": [2.6],
        "slope_FCz_bin_0": [3.0],
        "slope_P8_bin_0": [4.0],
        "Cz_Theta_bin_0": [5.0],
        "P8_Theta_bin_0": [6.0],
        "cnv_benchmark_FCz_bin_8": [6.5],
        "cnv_benchmark_P8_bin_8": [6.6],
        "G_precentral-lh_bin_0": [7.0],
    })

    out = _apply_channel_selection(df, cfg, "xgb")

    assert list(out.columns) == [
        "participant_id",
        "condition",
        "epoch",
        "Cz_bin_0",
        "amp_w0p125_Cz_std_bin_8",
        "slope_FCz_bin_0",
        "Cz_Theta_bin_0",
        "cnv_benchmark_FCz_bin_8",
        "G_precentral-lh_bin_0",
    ]


def test_source_feature_window_filter_keeps_overlapping_bins():
    import pandas as pd
    from eeg_steptype.features.assemble import _filter_src_window

    df = pd.DataFrame({
        "epoch": [0],
        "G_precentral-lh_bin_7": [1.0],   # 0.875-1.0, no overlap
        "G_precentral-lh_bin_8": [2.0],   # 1.0-1.125
        "G_precentral-lh_bin_15": [3.0],  # 1.875-2.0
        "G_precentral-lh_bin_16": [4.0],  # 2.0-2.125, no overlap
    })

    out = _filter_src_window(df, bin_n=0.125, tmin=1.0, tmax=2.0)

    assert list(out.columns) == [
        "epoch",
        "G_precentral-lh_bin_8",
        "G_precentral-lh_bin_15",
    ]


def test_cnv_benchmark_feature_block_names_motor_bins():
    import mne
    import numpy as np
    from eeg_steptype.features.cnv_benchmark import cnv_motor_amplitude_benchmark

    ch_names = ["Cz", "FCz", "Pz"]
    info = mne.create_info(ch_names, sfreq=4.0, ch_types="eeg")
    epochs = mne.EpochsArray(
        np.ones((2, len(ch_names), 4)),
        info,
        events=np.column_stack([
            np.arange(2),
            np.zeros(2, dtype=int),
            np.ones(2, dtype=int),
        ]),
        tmin=1.0,
        verbose=False,
    )

    df = cnv_motor_amplitude_benchmark(
        epochs,
        bin_n=0.25,
        channels=["Cz", "FCz"],
    )

    assert "epoch" in df.columns
    assert "cnv_benchmark_Cz_bin_4" in df.columns
    assert "cnv_benchmark_FCz_bin_7" in df.columns
    assert all("Pz" not in col for col in df.columns)


def test_rich_amplitude_feature_names_include_width_and_stat():
    import mne
    import numpy as np
    from eeg_steptype.features.amplitude import binned_amplitude_features

    ch_names = ["Cz", "Pz"]
    info = mne.create_info(ch_names, sfreq=8.0, ch_types="eeg")
    epochs = mne.EpochsArray(
        np.arange(2 * len(ch_names) * 8, dtype=float).reshape(2, len(ch_names), 8),
        info,
        events=np.column_stack([
            np.arange(2),
            np.zeros(2, dtype=int),
            np.ones(2, dtype=int),
        ]),
        tmin=1.0,
        verbose=False,
    )

    df = binned_amplitude_features(
        epochs,
        bin_widths=[0.125, 0.25],
        stats=["mean", "std", "min", "max", "median"],
        ch_names=["Cz"],
    )

    assert "epoch" in df.columns
    assert "amp_w0p125_Cz_mean_bin_8" in df.columns
    assert "amp_w0p125_Cz_std_bin_8" in df.columns
    assert "amp_w0p25_Cz_median_bin_4" in df.columns
    assert all("Pz" not in col for col in df.columns)


def test_full_channel_selection_is_forced_for_future_tensor_models():
    import pandas as pd
    from eeg_steptype.models.train import _apply_channel_selection

    cfg = {
        "channel_selection": {
            "mode": "roi",
            "roi": {"channels": ["Cz"]},
        }
    }
    df = pd.DataFrame({"condition": ["One"], "Cz_bin_0": [1], "P8_bin_0": [2]})

    out = _apply_channel_selection(df, cfg, "cnn")

    assert list(out.columns) == ["condition", "Cz_bin_0", "P8_bin_0"]
