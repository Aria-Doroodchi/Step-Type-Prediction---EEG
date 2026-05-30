"""Occlusion diagnostics for tensor-input neural EEG models.

This complements ``07_feature_informativeness.py``. That script explains
engineered tabular features for XGB; this one probes CNN/EEGNet-style models
that consume ``(epochs, channels, times)`` tensors.

The diagnostic is interpretability-oriented, not a held-out performance
estimate: for each participant it fits the requested model on all available
epochs, then measures how much the fitted model's AUC changes when masking
one channel or one time window at a time.

Example:
    python scripts/08_tensor_model_diagnostics.py ^
        --run outputs/runs/eegnet_p25_starter ^
        --participants P25
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402

from eeg_steptype.config import apply_prediction_window, load_config  # noqa: E402
from eeg_steptype.features.tensor import build_tensor_for_participant  # noqa: E402
from eeg_steptype.io import ensure_dir, outputs_root  # noqa: E402
from eeg_steptype.logging_utils import get_logger, setup_logging  # noqa: E402
from eeg_steptype.models.train import (  # noqa: E402
    MODEL_FACTORIES,
    _make_search_estimator,
    _scale_pos_weight,
)


SPEED_TIERS = {
    "cnn": "configs/cnn.yaml",
    "eegnet": "configs/eegnet.yaml",
}

TENSOR_MODELS = {"cnn", "eegnet"}

log = get_logger(__name__)


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run) if args.run else None

    cfg = _load_diagnostic_config(args, project_root, run_dir)
    cfg = apply_prediction_window(cfg, args.prediction_window)
    if args.participant_override_mode:
        cfg.setdefault("participant_overrides", {})["mode"] = args.participant_override_mode
    if args.epochs is not None:
        cfg.setdefault("modeling", {}).setdefault(_infer_model(args, cfg, run_dir), {})[
            "epochs"
        ] = int(args.epochs)

    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    model_name = _infer_model(args, cfg, run_dir)
    if model_name not in TENSOR_MODELS:
        raise SystemExit(
            f"Tensor diagnostics currently support {sorted(TENSOR_MODELS)}, got {model_name!r}."
        )

    participants = _participants(args, cfg, run_dir)
    run_label = run_dir.name if run_dir else f"{model_name}_manual"
    out_dir = ensure_dir(
        Path(args.output_dir)
        if args.output_dir
        else outputs_root(cfg) / "diagnostics" / f"{model_name}_tensor_diagnostics" / run_label
    )
    figures_dir = ensure_dir(out_dir / "figures")

    run_metrics = _read_run_metrics(run_dir)
    channel_rows: list[dict] = []
    time_rows: list[dict] = []
    summary_rows: list[dict] = []
    for pid in participants:
        log.info("[%s/%s] tensor occlusion diagnostics", pid, model_name)
        participant = _diagnose_participant(
            participant_id=pid,
            cfg=cfg,
            model_name=model_name,
            run_metrics=run_metrics,
            time_bin_s=float(args.time_bin_s),
        )
        channel_rows.extend(participant["channel_rows"])
        time_rows.extend(participant["time_rows"])
        summary_rows.append(participant["summary"])

    channel_df = pd.DataFrame(channel_rows)
    time_df = pd.DataFrame(time_rows)
    summary_df = pd.DataFrame(summary_rows)
    channel_df.to_csv(out_dir / "channel_occlusion.csv", index=False)
    time_df.to_csv(out_dir / "time_occlusion.csv", index=False)
    summary_df.to_csv(out_dir / "participant_summary.csv", index=False)

    figure_paths = _write_plots(
        channel_df=channel_df,
        time_df=time_df,
        figures_dir=figures_dir,
        model_name=model_name,
    )
    report_path = out_dir / "TENSOR_MODEL_DIAGNOSTICS.md"
    report_path.write_text(
        _render_report(
            cfg=cfg,
            model_name=model_name,
            run_label=run_label,
            participants=participants,
            channel_df=channel_df,
            time_df=time_df,
            summary_df=summary_df,
            figure_paths=figure_paths,
            report_path=report_path,
            time_bin_s=float(args.time_bin_s),
        ),
        encoding="utf-8",
    )
    log.info("Wrote tensor diagnostic report: %s", report_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create occlusion diagnostics for tensor-input CNN/EEGNet runs."
    )
    p.add_argument("--run", default=None, help="Path to outputs/runs/<run_id>.")
    p.add_argument(
        "--config",
        nargs="+",
        default=None,
        help="One or more config overlays. Ignored when --run provides config.yaml unless explicit.",
    )
    p.add_argument("--speed-tier", choices=list(SPEED_TIERS), default=None)
    p.add_argument("--model", choices=sorted(TENSOR_MODELS), default=None)
    p.add_argument("--participants", nargs="+", default=None)
    p.add_argument("--prediction-window", default=None)
    p.add_argument("--participant-override-mode", choices=["raw_assembly_only", "full", "none"], default=None)
    p.add_argument("--time-bin-s", type=float, default=0.25)
    p.add_argument("--epochs", type=int, default=None, help="Override diagnostic refit epochs.")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def _load_diagnostic_config(args: argparse.Namespace, project_root: Path, run_dir: Path | None) -> dict:
    if args.config:
        return load_config(args.config)
    if run_dir is not None:
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            return loaded
    if args.speed_tier:
        return load_config(str(project_root / SPEED_TIERS[args.speed_tier]))
    return load_config()


def _read_run_metrics(run_dir: Path | None) -> pd.DataFrame:
    if run_dir is None:
        return pd.DataFrame()
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return pd.DataFrame()
    return pd.read_csv(metrics_path)


def _infer_model(args: argparse.Namespace, cfg: dict, run_dir: Path | None) -> str:
    if args.model:
        return args.model
    metrics = _read_run_metrics(run_dir)
    if not metrics.empty and "model" in metrics.columns and metrics["model"].notna().any():
        return str(metrics["model"].dropna().iloc[0])
    default = cfg.get("modeling", {}).get("default_model")
    if default:
        return str(default)
    if args.speed_tier in TENSOR_MODELS:
        return str(args.speed_tier)
    if run_dir is not None:
        name = run_dir.name.lower()
        for model in sorted(TENSOR_MODELS, key=len, reverse=True):
            if model in name:
                return model
    return "eegnet"


def _participants(args: argparse.Namespace, cfg: dict, run_dir: Path | None) -> list[str]:
    if args.participants:
        return list(args.participants)
    metrics = _read_run_metrics(run_dir)
    if not metrics.empty and "participant_id" in metrics.columns:
        return sorted(str(x) for x in metrics["participant_id"].dropna().unique())
    return list(cfg["participants"])


def _diagnose_participant(
    *,
    participant_id: str,
    cfg: dict,
    model_name: str,
    run_metrics: pd.DataFrame,
    time_bin_s: float,
) -> dict:
    bundle = build_tensor_for_participant(participant_id, cfg, force=False)
    X = np.asarray(bundle["data"], dtype=np.float32)
    y = pd.Series(bundle["labels"]).map({"One": 0, "Two": 1}).astype(int)
    ch_names = [str(ch) for ch in bundle["ch_names"]]
    sfreq = float(bundle["sfreq"])
    tmin = float(bundle["tmin"])

    estimator = _make_estimator(
        cfg=cfg,
        model_name=model_name,
        X=X,
        y=y,
        params=_best_params_for_participant(run_metrics, participant_id, model_name),
    )
    estimator.fit(X, y)
    baseline_proba = _positive_proba(estimator, X)
    baseline_auc = _auc(y, baseline_proba)
    baseline_acc = float(accuracy_score(y, baseline_proba >= 0.5))

    channel_rows = []
    for channel_idx, channel in enumerate(ch_names):
        X_masked = X.copy()
        X_masked[:, channel_idx, :] = 0.0
        proba = _positive_proba(estimator, X_masked)
        auc = _auc(y, proba)
        acc = float(accuracy_score(y, proba >= 0.5))
        channel_rows.append({
            "participant": participant_id,
            "model": model_name,
            "channel": channel,
            "channel_index": channel_idx,
            "baseline_auc": baseline_auc,
            "occluded_auc": auc,
            "delta_auc": baseline_auc - auc,
            "baseline_accuracy": baseline_acc,
            "occluded_accuracy": acc,
            "delta_accuracy": baseline_acc - acc,
        })

    samples_per_bin = max(1, int(round(time_bin_s * sfreq)))
    time_rows = []
    for start in range(0, X.shape[2], samples_per_bin):
        stop = min(start + samples_per_bin, X.shape[2])
        X_masked = X.copy()
        X_masked[:, :, start:stop] = 0.0
        proba = _positive_proba(estimator, X_masked)
        auc = _auc(y, proba)
        acc = float(accuracy_score(y, proba >= 0.5))
        time_rows.append({
            "participant": participant_id,
            "model": model_name,
            "sample_start": start,
            "sample_stop": stop,
            "time_start_s": tmin + (start / sfreq),
            "time_stop_s": tmin + (stop / sfreq),
            "baseline_auc": baseline_auc,
            "occluded_auc": auc,
            "delta_auc": baseline_auc - auc,
            "baseline_accuracy": baseline_acc,
            "occluded_accuracy": acc,
            "delta_accuracy": baseline_acc - acc,
        })

    summary = {
        "participant": participant_id,
        "model": model_name,
        "n_epochs": int(X.shape[0]),
        "n_channels": int(X.shape[1]),
        "n_times": int(X.shape[2]),
        "sfreq": sfreq,
        "tmin": tmin,
        "tmax": float(bundle["tmax"]),
        "baseline_auc": baseline_auc,
        "baseline_accuracy": baseline_acc,
        "top_channel": _top_value(channel_rows, "channel"),
        "top_channel_delta_auc": _top_delta(channel_rows),
        "top_time_window": _top_time_window(time_rows),
        "top_time_delta_auc": _top_delta(time_rows),
    }
    return {"channel_rows": channel_rows, "time_rows": time_rows, "summary": summary}


def _make_estimator(
    *,
    cfg: dict,
    model_name: str,
    X: np.ndarray,
    y: pd.Series,
    params: dict,
):
    factory = MODEL_FACTORIES[model_name]
    estimator, _grid = _make_search_estimator(
        factory,
        cfg,
        model_name,
        scale_pos_weight=_scale_pos_weight(y),
        n_features=X.shape[1:],
    )
    if params:
        supported = set(estimator.get_params(deep=True))
        params = {key: value for key, value in params.items() if key in supported}
        if params:
            estimator.set_params(**params)
    return estimator


def _best_params_for_participant(metrics: pd.DataFrame, participant_id: str, model_name: str) -> dict:
    if metrics.empty or "best_params" not in metrics.columns:
        return {}
    part = metrics.copy()
    if "participant_id" in part.columns:
        part = part[part["participant_id"].astype(str) == str(participant_id)]
    if "model" in part.columns:
        part = part[part["model"].astype(str) == str(model_name)]
    parsed = []
    for value in part["best_params"].dropna():
        try:
            params = ast.literal_eval(str(value))
        except (SyntaxError, ValueError):
            continue
        if isinstance(params, dict):
            parsed.append(tuple(sorted(params.items())))
    if not parsed:
        return {}
    common = Counter(parsed).most_common(1)[0][0]
    return dict(common)


def _positive_proba(estimator, X: np.ndarray) -> np.ndarray:
    proba = estimator.predict_proba(X)
    arr = np.asarray(proba)
    if arr.ndim == 2 and arr.shape[1] > 1:
        return arr[:, 1]
    return arr.reshape(-1)


def _auc(y: pd.Series, proba: np.ndarray) -> float:
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, proba))


def _top_delta(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return float(max(rows, key=lambda row: row["delta_auc"])["delta_auc"])


def _top_value(rows: list[dict], key: str) -> str:
    if not rows:
        return ""
    return str(max(rows, key=lambda row: row["delta_auc"])[key])


def _top_time_window(rows: list[dict]) -> str:
    if not rows:
        return ""
    row = max(rows, key=lambda item: item["delta_auc"])
    return f"{row['time_start_s']:.3f}-{row['time_stop_s']:.3f}s"


def _write_plots(
    *,
    channel_df: pd.DataFrame,
    time_df: pd.DataFrame,
    figures_dir: Path,
    model_name: str,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if not channel_df.empty:
        paths["channel"] = figures_dir / "channel_occlusion_delta_auc.png"
        _plot_channel_occlusion(channel_df, paths["channel"], model_name=model_name)
    if not time_df.empty:
        paths["time"] = figures_dir / "time_occlusion_delta_auc.png"
        _plot_time_occlusion(time_df, paths["time"], model_name=model_name)
    return paths


def _plot_channel_occlusion(df: pd.DataFrame, path: Path, *, model_name: str) -> None:
    values = (
        df.groupby("channel")["delta_auc"]
        .mean()
        .sort_values(ascending=True)
        .tail(30)
    )
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(values))))
    ax.barh(values.index, values.values, color="#416f8f")
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Mean AUC drop when occluded")
    ax.set_title(f"{model_name}: channel occlusion")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_time_occlusion(df: pd.DataFrame, path: Path, *, model_name: str) -> None:
    grouped = (
        df.groupby(["time_start_s", "time_stop_s"], as_index=False)["delta_auc"]
        .mean()
        .sort_values("time_start_s")
    )
    centers = (grouped["time_start_s"] + grouped["time_stop_s"]) / 2.0
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(centers, grouped["delta_auc"], marker="o", color="#416f8f")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean AUC drop when occluded")
    ax.set_title(f"{model_name}: temporal occlusion")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def _render_report(
    *,
    cfg: dict,
    model_name: str,
    run_label: str,
    participants: list[str],
    channel_df: pd.DataFrame,
    time_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    figure_paths: dict[str, Path],
    report_path: Path,
    time_bin_s: float,
) -> str:
    rel_figs = {
        name: path.relative_to(report_path.parent).as_posix()
        for name, path in figure_paths.items()
    }
    top_channels = (
        channel_df.groupby("channel", as_index=False)["delta_auc"].mean()
        .sort_values("delta_auc", ascending=False)
        .head(15)
        if not channel_df.empty else pd.DataFrame()
    )
    top_times = (
        time_df.assign(
            time_window=lambda df: df.apply(
                lambda row: f"{row['time_start_s']:.3f}-{row['time_stop_s']:.3f}s",
                axis=1,
            )
        )
        .groupby("time_window", as_index=False)["delta_auc"].mean()
        .sort_values("delta_auc", ascending=False)
        .head(15)
        if not time_df.empty else pd.DataFrame()
    )

    lines = [
        f"# {model_name.upper()} Tensor Model Diagnostics",
        "",
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by `scripts/08_tensor_model_diagnostics.py`._",
        "",
        "## Inputs",
        "",
        f"- Run: `{run_label}`",
        f"- Participants: {', '.join(participants)}",
        f"- Feature window: {cfg.get('features', {}).get('min_time')} to {cfg.get('features', {}).get('max_time')} s",
        f"- Time occlusion bin: {time_bin_s:.3f} s",
        "",
        "## Caveat",
        "",
        "This is an interpretability probe, not a held-out performance estimate. "
        "The model is refit on all available epochs for each participant, then "
        "channels/time windows are masked to see which inputs the fitted model "
        "appears to rely on.",
        "",
        "## Summary",
        "",
        _format_md_table(summary_df, {
            "baseline_auc": "{:.4f}",
            "baseline_accuracy": "{:.4f}",
            "top_channel_delta_auc": "{:+.4f}",
            "top_time_delta_auc": "{:+.4f}",
            "sfreq": "{:.2f}",
            "tmin": "{:.3f}",
            "tmax": "{:.3f}",
        }),
        "",
    ]
    if "channel" in rel_figs:
        lines.extend(["## Channel Occlusion", "", f"![Channel occlusion]({rel_figs['channel']})", ""])
    lines.extend([
        "### Top Channels",
        "",
        _format_md_table(top_channels, {"delta_auc": "{:+.4f}"}),
        "",
    ])
    if "time" in rel_figs:
        lines.extend(["## Time Occlusion", "", f"![Time occlusion]({rel_figs['time']})", ""])
    lines.extend([
        "### Top Time Windows",
        "",
        _format_md_table(top_times, {"delta_auc": "{:+.4f}"}),
        "",
        "## Output Files",
        "",
        "- `channel_occlusion.csv`: one row per participant-channel mask.",
        "- `time_occlusion.csv`: one row per participant-time-window mask.",
        "- `participant_summary.csv`: baseline and top occlusion summaries.",
        "",
    ])
    return "\n".join(lines)


def _format_md_table(df: pd.DataFrame, formats: dict[str, str]) -> str:
    if df is None or df.empty:
        return "_(no data)_"
    out = df.copy()
    for col in out.columns:
        fmt = formats.get(col)
        if fmt:
            out[col] = out[col].apply(lambda value, f=fmt: "n/a" if pd.isna(value) else f.format(value))
        else:
            out[col] = out[col].apply(lambda value: "n/a" if pd.isna(value) else str(value))
        out[col] = out[col].str.replace("|", "\\|", regex=False)
    cols = list(out.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
