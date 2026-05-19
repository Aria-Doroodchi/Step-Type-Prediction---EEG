"""Diagnose per-participant feature informativeness for XGB winner/loser splits.

This script is intentionally diagnostic rather than another training tier. It
reads cached feature parquets, computes a per-feature univariate signal score
for each participant, fits a lightweight fixed XGB model to get gain shares,
and writes a human-readable Markdown report plus plots.

Example:
    python scripts/07_feature_informativeness.py ^
        --config configs/features_rich.yaml ^
        --prediction-window full_cnv ^
        --screening-md outputs/screening/SCREENING_RESULTS_RICHFEATS.md
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from eeg_steptype.config import apply_prediction_window, load_config  # noqa: E402
from eeg_steptype.features.assemble import build_for_participant  # noqa: E402
from eeg_steptype.io import ensure_dir, features_path, outputs_root  # noqa: E402
from eeg_steptype.logging_utils import get_logger, setup_logging  # noqa: E402
from eeg_steptype.models.train import _apply_channel_selection  # noqa: E402


SPEED_TIERS = {
    "lightning": "configs/lightning.yaml",
    "express": "configs/express.yaml",
    "quick": "configs/quick.yaml",
    "riemannian": "configs/riemannian.yaml",
}


log = get_logger(__name__)


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = _resolve_config_path(args.config, args.speed_tier, project_root)
    cfg = load_config(cfg_path)
    cfg = apply_prediction_window(cfg, args.prediction_window)
    if args.participant_override_mode:
        cfg.setdefault("participant_overrides", {})["mode"] = args.participant_override_mode
    if args.n_jobs is not None:
        cfg.setdefault("resources", {})["n_jobs"] = int(args.n_jobs)

    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    out_dir = ensure_dir(
        Path(args.output_dir)
        if args.output_dir
        else outputs_root(cfg) / "diagnostics" / "xgb_feature_informativeness"
    )
    figures_dir = ensure_dir(out_dir / "figures")

    rank_matrix = _read_screening_auc_matrix(Path(args.screening_md)) if args.screening_md else pd.DataFrame()
    groups = _participant_groups(args, cfg, rank_matrix)
    participants = groups["participant"].tolist()
    _check_feature_cache(participants, cfg, build_missing=args.build_missing)

    all_rows: list[pd.DataFrame] = []
    for pid in participants:
        log.info("[%s] diagnosing feature informativeness", pid)
        df = build_for_participant(pid, cfg, force=False)
        df = _apply_channel_selection(df, cfg, "xgb", channel_mode=args.channel_mode)
        rows = _diagnose_participant(pid, df, cfg, args)
        rows = rows.merge(groups, on="participant", how="left")
        all_rows.append(rows)

    feature_df = pd.concat(all_rows, ignore_index=True)
    family_df = _family_summary(feature_df, top_k=args.family_top_k)
    cohort_df = _cohort_feature_summary(feature_df, top_n=args.top_n)
    top_df = (
        feature_df.sort_values(
            ["participant", "univariate_info_auc", "abs_cohens_d"],
            ascending=[True, False, False],
        )
        .groupby("participant", as_index=False)
        .head(args.top_n)
        .reset_index(drop=True)
    )

    feature_df.to_csv(out_dir / "feature_informativeness.csv", index=False)
    family_df.to_csv(out_dir / "family_summary_by_participant.csv", index=False)
    cohort_df.to_csv(out_dir / "cohort_feature_summary.csv", index=False)
    top_df.to_csv(out_dir / "top_features_by_participant.csv", index=False)
    groups.to_csv(out_dir / "participant_groups.csv", index=False)

    figure_paths = _write_plots(
        feature_df=feature_df,
        family_df=family_df,
        top_df=top_df,
        groups=groups,
        figures_dir=figures_dir,
        top_n=args.plot_top_n,
    )
    report_path = out_dir / "FEATURE_INFORMATIVENESS_REPORT.md"
    report_path.write_text(
        _render_report(
            cfg=cfg,
            args=args,
            groups=groups,
            feature_df=feature_df,
            family_df=family_df,
            cohort_df=cohort_df,
            top_df=top_df,
            figure_paths=figure_paths,
            report_path=report_path,
        ),
        encoding="utf-8",
    )
    log.info("Wrote diagnosis report: %s", report_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create a human-readable feature-informativeness diagnosis for "
            "XGB winner/loser participants."
        )
    )
    p.add_argument("--config", default=None, help="Config overlay, e.g. configs/features_rich.yaml.")
    p.add_argument(
        "--speed-tier",
        choices=list(SPEED_TIERS.keys()),
        default=None,
        help="Use a speed-tier config when --config is not supplied.",
    )
    p.add_argument(
        "--prediction-window",
        default=None,
        help="Named prediction window from config, e.g. late_cnv or full_cnv.",
    )
    p.add_argument(
        "--screening-md",
        default="outputs/screening/SCREENING_RESULTS_RICHFEATS.md",
        help="Screening report with a per-participant AUC matrix.",
    )
    p.add_argument("--participants", nargs="+", default=None)
    p.add_argument("--winners", nargs="+", default=None, help="Override XGB winner participant IDs.")
    p.add_argument("--losers", nargs="+", default=None, help="Override XGB loser participant IDs.")
    p.add_argument("--channel-mode", choices=["full", "roi"], default=None)
    p.add_argument(
        "--build-missing",
        action="store_true",
        help="Build missing feature parquets instead of failing fast.",
    )
    p.add_argument(
        "--participant-override-mode",
        choices=["raw_assembly_only", "full", "none"],
        default=None,
    )
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--top-n", type=int, default=20, help="Top features per participant in CSV/report.")
    p.add_argument("--plot-top-n", type=int, default=10, help="Top features per participant in plots.")
    p.add_argument(
        "--family-top-k",
        type=int,
        default=5,
        help="Average the top K feature AUCs when summarizing each family.",
    )
    p.add_argument(
        "--xgb-top-k",
        type=int,
        default=400,
        help="Fit the lightweight XGB on the top K univariate features per participant.",
    )
    p.add_argument("--xgb-estimators", type=int, default=300)
    p.add_argument("--xgb-max-depth", type=int, default=3)
    p.add_argument("--random-state", type=int, default=None)
    return p.parse_args()


def _resolve_config_path(explicit: str | None, speed_tier: str | None, project_root: Path) -> str | None:
    if explicit:
        return explicit
    if speed_tier:
        return str(project_root / SPEED_TIERS[speed_tier])
    return None


def _participant_groups(args: argparse.Namespace, cfg: dict, matrix: pd.DataFrame) -> pd.DataFrame:
    if args.winners or args.losers:
        winners = args.winners or []
        losers = args.losers or []
        records = [{"participant": p, "xgb_group": "winner"} for p in winners]
        records.extend({"participant": p, "xgb_group": "loser"} for p in losers)
        out = pd.DataFrame(records)
        if out.empty:
            raise SystemExit("No participants were supplied.")
        return out.drop_duplicates("participant").reset_index(drop=True)

    if args.participants:
        requested = list(args.participants)
    elif not matrix.empty:
        requested = matrix["participant"].tolist()
    else:
        requested = list(cfg["participants"])

    if matrix.empty or "xgb" not in matrix.columns:
        return pd.DataFrame({"participant": requested, "xgb_group": "unknown"})

    rows = []
    model_cols = [c for c in matrix.columns if c != "participant"]
    for _, row in matrix[matrix["participant"].isin(requested)].iterrows():
        scores = row[model_cols].dropna().astype(float)
        if scores.empty:
            group = "unknown"
            rank = np.nan
        else:
            ordered = scores.sort_values(ascending=False)
            rank = int(list(ordered.index).index("xgb") + 1) if "xgb" in ordered.index else np.nan
            group = "winner" if rank == 1 else "loser"
        rows.append(
            {
                "participant": str(row["participant"]),
                "xgb_group": group,
                "xgb_auc": row.get("xgb", np.nan),
                "xgb_rank": rank,
            }
        )
    out = pd.DataFrame(rows)
    missing = [p for p in requested if p not in set(out["participant"])]
    if missing:
        out = pd.concat(
            [out, pd.DataFrame({"participant": missing, "xgb_group": "unknown"})],
            ignore_index=True,
        )
    return out.reset_index(drop=True)


def _read_screening_auc_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.warning("Screening markdown not found: %s", path)
        return pd.DataFrame()
    text = path.read_text(encoding="utf-8")
    marker = "### Per-participant AUC matrix"
    if marker not in text:
        log.warning("No per-participant AUC matrix found in %s", path)
        return pd.DataFrame()
    section = text.split(marker, 1)[1]
    lines = []
    for line in section.splitlines():
        if line.startswith("## "):
            break
        if line.strip().startswith("|"):
            lines.append(line.strip())
    if len(lines) < 3:
        return pd.DataFrame()
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == len(header):
            rows.append(cells)
    df = pd.DataFrame(rows, columns=header)
    for col in df.columns:
        if col != "participant":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _check_feature_cache(participants: list[str], cfg: dict, *, build_missing: bool) -> None:
    if build_missing:
        return
    missing = []
    for pid in participants:
        for condition in cfg["conditions"]:
            path = features_path(cfg, pid, condition)
            if not path.exists():
                missing.append(path)
    if not missing:
        return
    shown = "\n".join(f"  - {p}" for p in missing[:12])
    more = f"\n  ... and {len(missing) - 12} more" if len(missing) > 12 else ""
    raise SystemExit(
        "Missing cached feature parquet(s). Re-run after feature engineering finishes, "
        "or pass --build-missing to let this script build them.\n"
        f"{shown}{more}"
    )


def _diagnose_participant(
    participant: str,
    df: pd.DataFrame,
    cfg: dict,
    args: argparse.Namespace,
) -> pd.DataFrame:
    y = df["condition"].map({"One": 0, "Two": 1}).astype(int)
    X = df.drop(columns=["condition", "participant_id", "block_id"], errors="ignore")
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    rows = []
    bands = set((cfg.get("features", {}).get("freq_bands") or {}).keys())
    bin_n = float(cfg.get("features", {}).get("bin_n", 0.125))

    for feature in X.columns:
        values = pd.to_numeric(X[feature], errors="coerce")
        mask = values.notna() & y.notna()
        if mask.sum() < 4 or y[mask].nunique() < 2 or values[mask].nunique(dropna=True) <= 1:
            continue
        yv = y[mask].to_numpy()
        xv = values[mask].to_numpy(dtype=float)
        try:
            auc = float(roc_auc_score(yv, xv))
        except ValueError:
            continue
        desc = _describe_feature(feature, bands=bands, bin_n=bin_n)
        mean_one = float(np.mean(xv[yv == 0]))
        mean_two = float(np.mean(xv[yv == 1]))
        rows.append(
            {
                "participant": participant,
                "feature": feature,
                **desc,
                "n_valid": int(mask.sum()),
                "auc_raw": auc,
                "univariate_info_auc": max(auc, 1.0 - auc),
                "direction": "Two higher" if auc >= 0.5 else "One higher",
                "mean_One": mean_one,
                "mean_Two": mean_two,
                "mean_diff_Two_minus_One": mean_two - mean_one,
                "cohens_d": _cohens_d(xv[yv == 0], xv[yv == 1]),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_cohens_d"] = out["cohens_d"].abs()
    out["xgb_gain"] = 0.0
    out["xgb_gain_share"] = 0.0
    out = _add_xgb_gain(out, X, y, cfg, args)
    return out.sort_values(["univariate_info_auc", "xgb_gain_share"], ascending=False)


def _add_xgb_gain(
    info: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    args: argparse.Namespace,
) -> pd.DataFrame:
    top_features = info.nlargest(min(args.xgb_top_k, len(info)), "univariate_info_auc")[
        "feature"
    ].tolist()
    if not top_features:
        return info
    X_fit = X[top_features].copy()
    medians = X_fit.median(axis=0, numeric_only=True)
    X_fit = X_fit.fillna(medians).fillna(0.0)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    scale_pos_weight = (neg / pos) if pos else 1.0
    random_state = (
        int(args.random_state)
        if args.random_state is not None
        else int(cfg.get("modeling", {}).get("random_state", 1))
    )
    model = XGBClassifier(
        n_estimators=int(args.xgb_estimators),
        max_depth=int(args.xgb_max_depth),
        learning_rate=0.05,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        reg_alpha=0.1,
        gamma=0.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        importance_type="gain",
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(X_fit, y)
    gains = np.asarray(model.feature_importances_, dtype=float)
    total_gain = float(gains.sum())
    gain_map = dict(zip(top_features, gains))
    info["xgb_gain"] = info["feature"].map(gain_map).fillna(0.0)
    info["xgb_gain_share"] = info["xgb_gain"] / total_gain if total_gain > 0 else 0.0
    return info


def _describe_feature(feature: str, *, bands: set[str], bin_n: float) -> dict:
    if feature == "epoch":
        return {"family": "metadata", "channel": "", "band": "", "bin": np.nan, "time_start_s": np.nan}

    if feature.startswith("cnv_benchmark_"):
        inner = feature.removeprefix("cnv_benchmark_")
        desc = _describe_feature(inner, bands=bands, bin_n=bin_n)
        desc["family"] = "cnv_benchmark"
        return desc

    m = re.match(r"^slope_(?P<channel>.+?)_bin_(?P<bin>-?\d+)$", feature)
    if m:
        b = int(m.group("bin"))
        return {
            "family": "slope",
            "channel": m.group("channel"),
            "band": "",
            "bin": b,
            "time_start_s": b * bin_n,
        }

    for band in sorted(bands, key=len, reverse=True):
        suffix = f"_{band}_bin_"
        if suffix in feature:
            channel, bin_text = feature.rsplit(suffix, 1)
            if re.match(r"^-?\d+$", bin_text):
                b = int(bin_text)
                return {
                    "family": "psd",
                    "channel": channel,
                    "band": band,
                    "bin": b,
                    "time_start_s": b * bin_n,
                }

    m = re.match(r"^(?P<name>.+?)_bin_(?P<bin>-?\d+)$", feature)
    if m:
        name = m.group("name")
        b = int(m.group("bin"))
        family = "source" if _looks_like_source_label(name) else "amplitude"
        return {
            "family": family,
            "channel": "" if family == "source" else name,
            "band": "",
            "bin": b,
            "time_start_s": b * bin_n,
        }

    return {"family": "other", "channel": "", "band": "", "bin": np.nan, "time_start_s": np.nan}


def _looks_like_source_label(name: str) -> bool:
    return (
        "-" in name
        or name.endswith(("_lh", "_rh", "-lh", "-rh"))
        or name.startswith(("G_", "S_", "Pole_", "Lat_", "Med_"))
    )


def _cohens_d(one: np.ndarray, two: np.ndarray) -> float:
    if len(one) < 2 or len(two) < 2:
        return 0.0
    n1, n2 = len(one), len(two)
    v1, v2 = float(np.var(one, ddof=1)), float(np.var(two, ddof=1))
    pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / max(n1 + n2 - 2, 1))
    if pooled == 0 or not np.isfinite(pooled):
        return 0.0
    return float((np.mean(two) - np.mean(one)) / pooled)


def _family_summary(feature_df: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    rows = []
    for (pid, group, family), part in feature_df.groupby(["participant", "xgb_group", "family"]):
        top = part.nlargest(min(top_k, len(part)), "univariate_info_auc")
        rows.append(
            {
                "participant": pid,
                "xgb_group": group,
                "family": family,
                "n_features": int(len(part)),
                "max_univariate_auc": float(part["univariate_info_auc"].max()),
                f"top{top_k}_mean_univariate_auc": float(top["univariate_info_auc"].mean()),
                "max_abs_cohens_d": float(part["abs_cohens_d"].max()),
                "xgb_gain_share": float(part["xgb_gain_share"].sum()),
                "best_feature": str(part.sort_values("univariate_info_auc", ascending=False).iloc[0]["feature"]),
            }
        )
    return pd.DataFrame(rows)


def _cohort_feature_summary(feature_df: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    top_flags = feature_df.copy()
    top_flags["is_participant_top_n"] = (
        top_flags.sort_values(["participant", "univariate_info_auc"], ascending=[True, False])
        .groupby("participant")
        .cumcount()
        < top_n
    )
    rows = []
    for (feature, family), part in top_flags.groupby(["feature", "family"]):
        winner = part[part["xgb_group"] == "winner"]
        loser = part[part["xgb_group"] == "loser"]
        rows.append(
            {
                "feature": feature,
                "family": family,
                "n_participants": int(part["participant"].nunique()),
                "mean_univariate_auc": float(part["univariate_info_auc"].mean()),
                "max_univariate_auc": float(part["univariate_info_auc"].max()),
                "winner_mean_univariate_auc": float(winner["univariate_info_auc"].mean())
                if not winner.empty
                else np.nan,
                "loser_mean_univariate_auc": float(loser["univariate_info_auc"].mean())
                if not loser.empty
                else np.nan,
                "winner_minus_loser_auc": (
                    float(winner["univariate_info_auc"].mean() - loser["univariate_info_auc"].mean())
                    if not winner.empty and not loser.empty
                    else np.nan
                ),
                f"participant_top{top_n}_count": int(part["is_participant_top_n"].sum()),
                "mean_xgb_gain_share": float(part["xgb_gain_share"].mean()),
                "max_xgb_gain_share": float(part["xgb_gain_share"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_univariate_auc", f"participant_top{top_n}_count"],
        ascending=[False, False],
    )


def _write_plots(
    *,
    feature_df: pd.DataFrame,
    family_df: pd.DataFrame,
    top_df: pd.DataFrame,
    groups: pd.DataFrame,
    figures_dir: Path,
    top_n: int,
) -> dict[str, Path]:
    paths = {}
    paths["family_heatmap"] = figures_dir / "family_signal_heatmap.png"
    _plot_family_heatmap(family_df, groups, paths["family_heatmap"])

    paths["gain_share"] = figures_dir / "xgb_gain_family_share.png"
    _plot_gain_share(family_df, groups, paths["gain_share"])

    paths["winner_loser_delta"] = figures_dir / "winner_loser_family_delta.png"
    _plot_winner_loser_delta(family_df, paths["winner_loser_delta"])

    paths["top_feature_heatmap"] = figures_dir / "top_feature_heatmap.png"
    _plot_top_feature_heatmap(feature_df, groups, paths["top_feature_heatmap"])

    paths["participant_top_features"] = figures_dir / "top_features_by_participant.png"
    _plot_participant_top_features(top_df, groups, paths["participant_top_features"], top_n=top_n)
    return paths


def _participant_order(groups: pd.DataFrame) -> list[str]:
    order_key = {"winner": 0, "loser": 1, "unknown": 2}
    ordered = groups.assign(_order=groups["xgb_group"].map(order_key).fillna(2))
    return ordered.sort_values(["_order", "participant"])["participant"].tolist()


def _plot_family_heatmap(family_df: pd.DataFrame, groups: pd.DataFrame, path: Path) -> None:
    value_col = next(c for c in family_df.columns if c.startswith("top") and c.endswith("_mean_univariate_auc"))
    pivot = family_df.pivot_table(index="participant", columns="family", values=value_col, aggfunc="mean")
    pivot = pivot.reindex(_participant_order(groups))
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    _heatmap(
        pivot,
        path,
        title="Feature-family signal by participant",
        cbar_label="Mean AUC of top family features",
        vmin=0.5,
        vmax=max(0.75, float(np.nanmax(pivot.to_numpy())) if pivot.size else 0.75),
    )


def _plot_gain_share(family_df: pd.DataFrame, groups: pd.DataFrame, path: Path) -> None:
    pivot = family_df.pivot_table(index="participant", columns="family", values="xgb_gain_share", aggfunc="sum")
    pivot = pivot.fillna(0.0).reindex(_participant_order(groups))
    pivot = pivot.loc[:, pivot.sum().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(11, max(4, 0.45 * len(pivot))))
    left = np.zeros(len(pivot))
    cmap = plt.get_cmap("tab20")
    for i, col in enumerate(pivot.columns):
        values = pivot[col].to_numpy()
        ax.barh(np.arange(len(pivot)), values, left=left, label=col, color=cmap(i % 20))
        left += values
    ax.set_yticks(np.arange(len(pivot)))
    ax.set_yticklabels(pivot.index)
    ax.invert_yaxis()
    ax.set_xlabel("XGB gain share")
    ax.set_title("Where the lightweight XGB puts its gain")
    ax.set_xlim(0, max(1.0, float(left.max()) if len(left) else 1.0))
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_winner_loser_delta(family_df: pd.DataFrame, path: Path) -> None:
    value_col = next(c for c in family_df.columns if c.startswith("top") and c.endswith("_mean_univariate_auc"))
    grouped = family_df.pivot_table(index="family", columns="xgb_group", values=value_col, aggfunc="mean")
    if "winner" not in grouped:
        grouped["winner"] = np.nan
    if "loser" not in grouped:
        grouped["loser"] = np.nan
    grouped["winner_minus_loser"] = grouped["winner"] - grouped["loser"]
    grouped = grouped.sort_values("winner_minus_loser", ascending=True)
    colors = np.where(grouped["winner_minus_loser"] >= 0, "#3b7c6e", "#b65f49")
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.45 * len(grouped))))
    ax.barh(grouped.index, grouped["winner_minus_loser"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Winner mean AUC - loser mean AUC")
    ax.set_title("Feature families that separate XGB winners from losers")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_top_feature_heatmap(feature_df: pd.DataFrame, groups: pd.DataFrame, path: Path) -> None:
    top_features = (
        feature_df.groupby("feature")["univariate_info_auc"]
        .mean()
        .sort_values(ascending=False)
        .head(30)
        .index
    )
    pivot = feature_df[feature_df["feature"].isin(top_features)].pivot_table(
        index="participant", columns="feature", values="univariate_info_auc", aggfunc="mean"
    )
    pivot = pivot.reindex(_participant_order(groups))
    pivot = pivot.reindex(top_features, axis=1)
    pivot.columns = [_short_label(c, 28) for c in pivot.columns]
    _heatmap(
        pivot,
        path,
        title="Cohort-top features across participants",
        cbar_label="Univariate feature AUC",
        vmin=0.5,
        vmax=max(0.75, float(np.nanmax(pivot.to_numpy())) if pivot.size else 0.75),
        x_rotation=65,
    )


def _plot_participant_top_features(
    top_df: pd.DataFrame,
    groups: pd.DataFrame,
    path: Path,
    *,
    top_n: int,
) -> None:
    participants = _participant_order(groups)
    ncols = 2
    nrows = math.ceil(len(participants) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, max(4, nrows * 3.1)), squeeze=False)
    group_map = groups.set_index("participant")["xgb_group"].to_dict()
    for ax, pid in zip(axes.ravel(), participants):
        part = top_df[top_df["participant"] == pid].nlargest(top_n, "univariate_info_auc")
        labels = [_short_label(x, 34) for x in part["feature"]]
        values = part["univariate_info_auc"].to_numpy()
        y = np.arange(len(part))
        ax.barh(y, values, color="#416f8f")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0.5, max(0.75, float(values.max()) + 0.02 if len(values) else 0.75))
        ax.set_title(f"{pid} ({group_map.get(pid, 'unknown')})", fontsize=10)
        ax.set_xlabel("Feature AUC")
        _style_axes(ax)
    for ax in axes.ravel()[len(participants) :]:
        ax.axis("off")
    fig.suptitle("Top univariate features per participant", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _heatmap(
    pivot: pd.DataFrame,
    path: Path,
    *,
    title: str,
    cbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
    x_rotation: int = 35,
) -> None:
    fig, ax = plt.subplots(
        figsize=(max(8, 0.42 * len(pivot.columns)), max(4, 0.42 * len(pivot.index)))
    )
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=x_rotation, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label)
    ax.set_facecolor("#f2f2f2")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def _short_label(value: str, max_len: int) -> str:
    value = str(value)
    if len(value) <= max_len:
        return value
    return value[: max_len - 1] + "..."


def _render_report(
    *,
    cfg: dict,
    args: argparse.Namespace,
    groups: pd.DataFrame,
    feature_df: pd.DataFrame,
    family_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    top_df: pd.DataFrame,
    figure_paths: dict[str, Path],
    report_path: Path,
) -> str:
    value_col = next(c for c in family_df.columns if c.startswith("top") and c.endswith("_mean_univariate_auc"))
    group_counts = groups["xgb_group"].value_counts().to_dict()
    family_group = family_df.pivot_table(index="family", columns="xgb_group", values=value_col, aggfunc="mean")
    if "winner" not in family_group:
        family_group["winner"] = np.nan
    if "loser" not in family_group:
        family_group["loser"] = np.nan
    family_group["winner_minus_loser"] = family_group["winner"] - family_group["loser"]
    family_group = family_group.reset_index().sort_values("winner_minus_loser", ascending=False)

    top_feature_cols = [
        "feature",
        "family",
        "mean_univariate_auc",
        "winner_mean_univariate_auc",
        "loser_mean_univariate_auc",
        "winner_minus_loser_auc",
        "mean_xgb_gain_share",
    ]
    cohort_top = cohort_df[top_feature_cols].head(20).copy()

    per_participant = top_df.groupby("participant").head(5)[
        ["participant", "xgb_group", "feature", "family", "univariate_info_auc", "direction", "xgb_gain_share"]
    ]

    rel_figs = {
        name: path.relative_to(report_path.parent).as_posix()
        for name, path in figure_paths.items()
    }
    lines = [
        "# XGB Feature Informativeness Diagnosis",
        "",
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by `scripts/07_feature_informativeness.py`._",
        "",
        "## Inputs",
        "",
        f"- Participants: {', '.join(groups['participant'])}",
        f"- XGB winner/loser counts: {group_counts}",
        f"- Feature window: {cfg.get('features', {}).get('min_time')} to {cfg.get('features', {}).get('max_time')} s",
        f"- Feature blocks: {', '.join(cfg.get('features', {}).get('blocks', []))}",
        f"- Channel mode: {args.channel_mode or cfg.get('channel_selection', {}).get('mode', 'full')}",
        "",
        "## How To Read This",
        "",
        textwrap.dedent(
            """
            `univariate_info_auc` is the direction-free ROC-AUC for a single feature.
            A value near 0.50 means the feature is not informative by itself; values
            above roughly 0.65 are worth inspecting. `direction` tells whether larger
            values point toward condition `One` or `Two`. `xgb_gain_share` comes from
            a lightweight fixed XGB fit on each participant's strongest univariate
            features; it is an interpretability probe, not a nested-CV performance
            estimate.
            """
        ).strip(),
        "",
        "## Figures",
        "",
        f"![Feature-family signal heatmap]({rel_figs['family_heatmap']})",
        "",
        f"![XGB gain family share]({rel_figs['gain_share']})",
        "",
        f"![Winner loser family delta]({rel_figs['winner_loser_delta']})",
        "",
        f"![Top feature heatmap]({rel_figs['top_feature_heatmap']})",
        "",
        f"![Top features by participant]({rel_figs['participant_top_features']})",
        "",
        "## Winner vs Loser Family Summary",
        "",
        _format_md_table(
            family_group,
            {
                "winner": "{:.4f}",
                "loser": "{:.4f}",
                "winner_minus_loser": "{:+.4f}",
            },
        ),
        "",
        "## Cohort-Strongest Features",
        "",
        _format_md_table(
            cohort_top,
            {
                "mean_univariate_auc": "{:.4f}",
                "winner_mean_univariate_auc": "{:.4f}",
                "loser_mean_univariate_auc": "{:.4f}",
                "winner_minus_loser_auc": "{:+.4f}",
                "mean_xgb_gain_share": "{:.4f}",
            },
        ),
        "",
        "## Top Features Per Participant",
        "",
        _format_md_table(
            per_participant,
            {
                "univariate_info_auc": "{:.4f}",
                "xgb_gain_share": "{:.4f}",
            },
        ),
        "",
        "## Output Files",
        "",
        "- `feature_informativeness.csv`: one row per participant-feature.",
        "- `family_summary_by_participant.csv`: feature-family summaries per participant.",
        "- `cohort_feature_summary.csv`: cross-participant feature summary.",
        "- `top_features_by_participant.csv`: top feature shortlist per participant.",
        "- `participant_groups.csv`: winner/loser labels used in this report.",
        "",
    ]
    return "\n".join(lines)


def _format_md_table(df: pd.DataFrame, formats: dict[str, str]) -> str:
    if df.empty:
        return "_(no data)_"
    out = df.copy()
    for col in out.columns:
        fmt = formats.get(col)
        if fmt:
            out[col] = out[col].apply(lambda v, f=fmt: "n/a" if pd.isna(v) else f.format(v))
        else:
            out[col] = out[col].apply(lambda v: "n/a" if pd.isna(v) else str(v))
        out[col] = out[col].str.replace("|", "\\|", regex=False)
    cols = list(out.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
