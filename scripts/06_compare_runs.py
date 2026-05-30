"""Aggregate model screening runs into a SCREENING_RESULTS.md report.

Reads each ``outputs/runs/<run_id>/`` directory supplied on the CLI,
computes the five diagnostic cells described in SCRIPT_GUIDES.md, and
writes a clean markdown report.

The five diagnostics are:

    1. Mean test AUC ± 95% CI per (model, tier)
    2. Tier-response slope per classical model (Express AUC − Lightning AUC)
    3. Across-fold AUC variance per model on the primary (Express) tier
    4. Inner-vs-outer gap (inner_best_score − overall_accuracy)
    5. Per-participant model ranking distribution

Usage::

    python scripts/06_compare_runs.py \\
        --runs outputs/runs/screen_xgb_express outputs/runs/screen_xgb_lightning ... \\
        --output outputs/screening/SCREENING_RESULTS.md

If ``--runs`` contains glob patterns they are expanded. Missing or
malformed runs are skipped with a warning so a partial sweep still
produces a report.
"""

from __future__ import annotations

import argparse
import glob
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs", nargs="+", required=True,
        help="Run directories or glob patterns under outputs/runs/.",
    )
    p.add_argument(
        "--output", default="outputs/screening/SCREENING_RESULTS.md",
        help="Output markdown path (default: outputs/screening/SCREENING_RESULTS.md).",
    )
    p.add_argument(
        "--no-root-copy", action="store_true",
        help="Skip writing a SCREENING_RESULTS.md copy at the repo root.",
    )
    p.add_argument(
        "--default-tier", default=None,
        choices=["lightning", "express", "quick", "riemannian", "cnn", "eegnet"],
        help=(
            "Tier to assign when neither the run directory name nor the "
            "config snapshot reveals one. Useful when run-ids were named "
            "after a features overlay rather than the speed tier."
        ),
    )
    args = p.parse_args()

    run_dirs = _expand_run_dirs(args.runs)
    print(f"Found {len(run_dirs)} run directories:", file=sys.stderr)
    for d in run_dirs:
        print(f"  {d}", file=sys.stderr)

    runs = []
    for d in run_dirs:
        try:
            runs.append(_load_run(d, default_tier=args.default_tier))
        except Exception as exc:                              # noqa: BLE001
            print(f"  WARNING: skipping {d}: {exc}", file=sys.stderr)

    if not runs:
        print("No runs could be loaded; nothing to aggregate.", file=sys.stderr)
        sys.exit(1)

    md = build_report(runs)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out}")

    if not args.no_root_copy:
        root = Path(__file__).resolve().parents[1] / "SCREENING_RESULTS.md"
        root.write_text(md, encoding="utf-8")
        print(f"Wrote {root}")


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _expand_run_dirs(patterns: list[str]) -> list[Path]:
    """Expand explicit paths and glob patterns into a deduplicated dir list."""
    out: list[Path] = []
    for pat in patterns:
        p = Path(pat)
        if p.exists() and p.is_dir():
            out.append(p)
            continue
        for match in glob.glob(pat):
            mp = Path(match)
            if mp.is_dir():
                out.append(mp)
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        unique.append(p)
    return unique


def _load_run(run_dir: Path, *, default_tier: str | None = None) -> dict:
    metrics = run_dir / "metrics.csv"
    if not metrics.exists():
        raise FileNotFoundError(f"missing metrics.csv")
    df = pd.read_csv(metrics)
    if df.empty:
        raise ValueError("metrics.csv is empty")

    cfg: dict = {}
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:                                      # noqa: BLE001
            cfg = {}

    model = _infer_model(df, cfg, run_dir)
    tier = _infer_tier(cfg, run_dir, default_tier=default_tier)
    return {
        "run_dir": run_dir,
        "metrics": df,
        "config": cfg,
        "model": model,
        "tier": tier,
    }


def _infer_model(df: pd.DataFrame, cfg: dict, run_dir: Path) -> str:
    if "model" in df.columns and df["model"].notna().any():
        return str(df["model"].dropna().iloc[0])
    default = cfg.get("modeling", {}).get("default_model")
    if default:
        return str(default)
    name = run_dir.name.lower()
    for m in ("riemannian", "eegnet", "cnn", "logistic", "xgb", "svm", "lstm"):
        if m in name:
            return m
    return "unknown"


def _infer_tier(cfg: dict, run_dir: Path, *, default_tier: str | None = None) -> str:
    """Recover the speed tier this run was launched with.

    Four sources of evidence, in priority order:

    1. ``cfg["_speed_tier"]`` — canonical, stamped by ``run.py`` /
       ``scripts/04_train.py`` when ``--speed-tier`` is passed.
    2. A tier token in the run-directory name (e.g. ``screen_xgb_express``).
    3. Modeling-settings signature match against the known tier configs --
       handles runs whose ID was named after a *features* overlay rather
       than the modeling tier (e.g. ``screen_xgb_richfeats``).
    4. ``default_tier`` CLI override -- the explicit escape hatch.
    """
    # 1. Explicit stamp (most reliable; canonical for new runs).
    if isinstance(cfg, dict):
        stamped = cfg.get("_speed_tier")
        if stamped and isinstance(stamped, str) and stamped.lower() != "unknown":
            return stamped.lower()

    # 2. Run-directory name token.
    name = run_dir.name.lower()
    for t in ("riemannian", "eegnet", "cnn", "lightning", "express", "quick"):
        if t in name:
            return t

    # 3. Modeling-settings signature.
    sig = _tier_from_modeling_signature(cfg)
    if sig:
        return sig

    # 4. Caller-supplied default.
    if default_tier:
        return default_tier

    return "unknown"


def _tier_from_modeling_signature(cfg: dict) -> str | None:
    """Match cfg["modeling"] against the known tier overlays.

    The tier YAMLs in ``configs/`` set distinctive combinations of
    cv.n_splits/n_repeats, rfecv.enabled, shap_prune.enabled, gain_prune.refit,
    and search.method. This function tries to identify which tier the
    snapshot most likely came from.
    """
    if not isinstance(cfg, dict):
        return None
    m = cfg.get("modeling") or {}
    cv = m.get("cv") or {}
    rfecv = m.get("rfecv") or {}
    shap = m.get("shap_prune") or {}
    gain = m.get("gain_prune") or {}
    search = m.get("search") or {}

    try:
        n_splits  = int(cv.get("n_splits", -1)) if cv.get("n_splits") is not None else -1
        n_repeats = int(cv.get("n_repeats", -1)) if cv.get("n_repeats") is not None else -1
    except (TypeError, ValueError):
        n_splits, n_repeats = -1, -1

    rfecv_enabled = bool(rfecv.get("enabled", True))
    shap_enabled = bool(shap.get("enabled", shap.get("quantile", m.get("shap_prune_quantile", 0)) > 0))
    gain_refit = bool(gain.get("refit", True))
    search_method = str(search.get("method", "auto"))
    default_model = m.get("default_model")

    # Tensor-model tiers pin default_model.
    if default_model in {"riemannian", "cnn", "eegnet"}:
        return str(default_model)

    # Lightning: aggressive trim. 3x1 CV, no RFECV, no SHAP, grid search.
    if (n_splits == 3 and n_repeats == 1
            and not rfecv_enabled and not shap_enabled
            and search_method == "grid"):
        return "lightning"

    # Express: 5x2 CV, RFECV on (1 iter), gain refit on, SHAP off.
    if (n_splits == 5 and n_repeats == 2
            and rfecv_enabled and not shap_enabled and gain_refit):
        return "express"

    # Quick: 5x5 CV with RFECV (2 iters) and SHAP enabled.
    if (n_splits == 5 and n_repeats == 5
            and rfecv_enabled and shap_enabled):
        return "quick"

    # Default research-grade config: 5x20 CV with full FS pipeline.
    if (n_splits == 5 and n_repeats == 20
            and rfecv_enabled and shap_enabled):
        return "default"

    return None


# ---------------------------------------------------------------------------
# Diagnostic computations
# ---------------------------------------------------------------------------
def _auc_series(df: pd.DataFrame) -> pd.Series:
    if "auc" not in df.columns:
        return pd.Series(dtype=float)
    return df["auc"].dropna()


def diagnostic_1_auc(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        auc = _auc_series(r["metrics"])
        if len(auc) == 0:
            rows.append({
                "model": r["model"], "tier": r["tier"],
                "n_folds": 0, "mean_auc": float("nan"), "ci95": float("nan"),
                "sd": float("nan"),
            })
            continue
        mean = float(auc.mean())
        sd = float(auc.std(ddof=1)) if len(auc) > 1 else 0.0
        ci = 1.96 * sd / np.sqrt(len(auc)) if len(auc) > 1 else 0.0
        rows.append({
            "model": r["model"], "tier": r["tier"],
            "n_folds": int(len(auc)), "mean_auc": mean, "ci95": ci, "sd": sd,
        })
    return pd.DataFrame(rows).sort_values(["model", "tier"]).reset_index(drop=True)


def diagnostic_2_slope(runs: list[dict]) -> pd.DataFrame:
    by_model_tier: dict[tuple[str, str], pd.Series] = {}
    models: set[str] = set()
    for r in runs:
        by_model_tier[(r["model"], r["tier"])] = _auc_series(r["metrics"])
        models.add(r["model"])

    rows = []
    for model in sorted(models):
        express = by_model_tier.get((model, "express"))
        lightning = by_model_tier.get((model, "lightning"))
        if express is None or lightning is None or len(express) == 0 or len(lightning) == 0:
            rows.append({
                "model": model,
                "express_auc": float(express.mean()) if express is not None and len(express) else float("nan"),
                "lightning_auc": float(lightning.mean()) if lightning is not None and len(lightning) else float("nan"),
                "slope (Express − Lightning)": float("nan"),
                "interpretation": "n/a (single-tier model)"
                if model in {"riemannian", "cnn", "eegnet"} else "missing tier run",
            })
            continue
        ex_mean = float(express.mean())
        li_mean = float(lightning.mean())
        slope = ex_mean - li_mean
        if slope >= 0.04:
            interp = "high headroom — rewards tuning"
        elif slope >= 0.015:
            interp = "moderate headroom"
        elif slope >= 0.0:
            interp = "near ceiling — flat response"
        else:
            interp = "negative (noisy)"
        rows.append({
            "model": model,
            "express_auc": ex_mean,
            "lightning_auc": li_mean,
            "slope (Express − Lightning)": slope,
            "interpretation": interp,
        })
    return pd.DataFrame(rows)


def diagnostic_3_variance(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        # Only consider primary-tier results: express for tabular models, or
        # each tensor model's single tier.
        if r["tier"] not in ("express", "riemannian", "cnn", "eegnet"):
            continue
        df = r["metrics"]
        if "auc" not in df.columns or "participant_id" not in df.columns:
            continue
        clean = df.dropna(subset=["auc"])
        per_part = clean.groupby("participant_id")["auc"].agg(["mean", "std", "count"])
        if per_part.empty:
            continue
        rows.append({
            "model": r["model"],
            "tier": r["tier"],
            "mean_within_participant_sd": float(per_part["std"].dropna().mean())
                if per_part["std"].notna().any() else float("nan"),
            "max_within_participant_sd": float(per_part["std"].dropna().max())
                if per_part["std"].notna().any() else float("nan"),
            "n_participants": int(len(per_part)),
        })
    return pd.DataFrame(rows).sort_values(["model"]).reset_index(drop=True)


def diagnostic_4_inner_outer_gap(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        if r["tier"] not in ("express", "riemannian", "cnn", "eegnet"):
            continue
        df = r["metrics"]
        if "inner_best_score" not in df.columns or "overall_accuracy" not in df.columns:
            continue
        gap = (df["inner_best_score"] - df["overall_accuracy"]).dropna()
        if len(gap) == 0:
            continue
        mean_gap = float(gap.mean())
        if abs(mean_gap) < 0.02:
            interp = "well-calibrated"
        elif mean_gap >= 0.05:
            interp = "overfits inner CV"
        elif mean_gap >= 0.02:
            interp = "mild optimism"
        else:
            interp = "underconfident inner CV"
        rows.append({
            "model": r["model"],
            "tier": r["tier"],
            "mean_gap": mean_gap,
            "median_gap": float(gap.median()),
            "max_gap": float(gap.max()),
            "interpretation": interp,
        })
    return pd.DataFrame(rows).sort_values(["model"]).reset_index(drop=True)


def diagnostic_5_ranking(runs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_part: dict[str, dict[str, float]] = defaultdict(dict)
    for r in runs:
        if r["tier"] not in ("express", "riemannian", "cnn", "eegnet"):
            continue
        df = r["metrics"]
        if "auc" not in df.columns or "participant_id" not in df.columns:
            continue
        clean = df.dropna(subset=["auc"])
        if clean.empty:
            continue
        by_pid = clean.groupby("participant_id")["auc"].mean()
        for pid, val in by_pid.items():
            by_part[str(pid)][r["model"]] = float(val)

    rank_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for pid, scores in by_part.items():
        sorted_models = sorted(scores.keys(), key=lambda m: -scores[m])
        for rank, m in enumerate(sorted_models, start=1):
            rank_counts[m][rank] += 1

    rank_rows = []
    n_models = max((len(s) for s in by_part.values()), default=0)
    for m in sorted(rank_counts.keys()):
        row = {"model": m}
        for r in range(1, n_models + 1):
            row[f"rank_{r}"] = rank_counts[m].get(r, 0)
        rank_rows.append(row)

    matrix_rows = []
    all_models = sorted({m for v in by_part.values() for m in v.keys()})
    for pid in sorted(by_part.keys()):
        row = {"participant": pid}
        for m in all_models:
            row[m] = by_part[pid].get(m, float("nan"))
        matrix_rows.append(row)

    return pd.DataFrame(rank_rows), pd.DataFrame(matrix_rows)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def build_report(runs: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Model Screening Results")
    lines.append("")
    lines.append(
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by `scripts/06_compare_runs.py`_"
    )
    lines.append("")

    # ---- Overview -----------------------------------------------------------
    participants: set[str] = set()
    for r in runs:
        df = r["metrics"]
        if "participant_id" in df.columns:
            participants.update(str(x) for x in df["participant_id"].dropna().unique())
    models = sorted({r["model"] for r in runs})
    tiers = sorted({r["tier"] for r in runs})

    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Models compared:** {', '.join(models)}")
    lines.append(f"- **Tiers:** {', '.join(tiers)}")
    lines.append(f"- **Participants:** {', '.join(sorted(participants))} ({len(participants)} total)")
    lines.append(f"- **Total runs aggregated:** {len(runs)}")
    lines.append("")
    lines.append("All five diagnostics are computed on the **Express** tier "
                 "(primary tier; for tensor models: their single model-specific tier). "
                 "Diagnostic 2 (tier-response slope) additionally consumes the **Lightning** tier "
                 "runs for the three classical models.")
    lines.append("")

    # ---- Diagnostic 1 -------------------------------------------------------
    lines.append("## Diagnostic 1 — Mean test AUC ± 95% CI")
    lines.append("")
    lines.append(
        "Per-fold ROC-AUC averaged across all CV folds × participants, with a "
        "Wald 95% CI. Higher mean is better; tighter CI means more consistent "
        "estimates."
    )
    lines.append("")
    d1 = diagnostic_1_auc(runs)
    lines.append(_format_md_table(d1, {
        "mean_auc": "{:.4f}",
        "ci95":     "± {:.4f}",
        "sd":       "{:.4f}",
    }))
    lines.append("")

    # ---- Diagnostic 2 -------------------------------------------------------
    lines.append("## Diagnostic 2 — Tier-response slope")
    lines.append("")
    lines.append(
        "Mean Express AUC minus mean Lightning AUC for the same model. "
        "Positive slope means the model rewards heavier optimization budget "
        "(more CV repeats, RFECV pass, gain-prune refit). A near-zero slope "
        "suggests the model is already near its ceiling; switching model "
        "family will pay off more than further tuning. Single-tier tensor "
        "models are not slope-comparable here."
    )
    lines.append("")
    d2 = diagnostic_2_slope(runs)
    lines.append(_format_md_table(d2, {
        "express_auc":   "{:.4f}",
        "lightning_auc": "{:.4f}",
        "slope (Express − Lightning)": "{:+.4f}",
    }))
    lines.append("")

    # ---- Diagnostic 3 -------------------------------------------------------
    lines.append("## Diagnostic 3 — Across-fold AUC variance")
    lines.append("")
    lines.append(
        "Standard deviation of per-fold AUC within each participant, averaged "
        "across participants. Lower means the model gives more stable "
        "fold-to-fold predictions. A high `max_within_participant_sd` flags "
        "a participant whose AUC swings wildly between folds — usually a "
        "data-quality issue or a fundamentally hard subgroup."
    )
    lines.append("")
    d3 = diagnostic_3_variance(runs)
    lines.append(_format_md_table(d3, {
        "mean_within_participant_sd": "{:.4f}",
        "max_within_participant_sd":  "{:.4f}",
    }))
    lines.append("")

    # ---- Diagnostic 4 -------------------------------------------------------
    lines.append("## Diagnostic 4 — Inner-vs-outer gap")
    lines.append("")
    lines.append(
        "Mean of (`inner_best_score` − `overall_accuracy`) across folds. "
        "Small absolute gap means the hyperparameter search generalizes — "
        "the inner CV's optimism matches the outer held-out score. A large "
        "positive gap (≥ 0.05) means the model overfits the inner search "
        "and further tuning will mostly chase noise. Negative gap means the "
        "inner CV was unusually pessimistic relative to the outer fold "
        "(can happen when inner folds are small)."
    )
    lines.append("")
    d4 = diagnostic_4_inner_outer_gap(runs)
    lines.append(_format_md_table(d4, {
        "mean_gap":   "{:+.4f}",
        "median_gap": "{:+.4f}",
        "max_gap":    "{:+.4f}",
    }))
    lines.append("")

    # ---- Diagnostic 5 -------------------------------------------------------
    lines.append("## Diagnostic 5 — Per-participant model ranking")
    lines.append("")
    lines.append(
        "For each participant, the four models are ranked by mean AUC on the "
        "primary tier. The table below counts how often each model finished "
        "in each rank. Concentrated rankings (one model usually #1) = "
        "homogeneous signal across the cohort. Scattered rankings (each model "
        "wins for some subset) = heterogeneous signal — per-participant model "
        "selection or an ensemble may help more than tuning one model."
    )
    lines.append("")
    rank_df, matrix_df = diagnostic_5_ranking(runs)
    lines.append(_format_md_table(rank_df, {}))
    lines.append("")

    lines.append("### Per-participant AUC matrix")
    lines.append("")
    matrix_fmt: dict[str, str] = {
        col: "{:.4f}" for col in matrix_df.columns if col != "participant"
    }
    lines.append(_format_md_table(matrix_df, matrix_fmt))
    lines.append("")

    # ---- Interpretation -----------------------------------------------------
    lines.append("## How to read this report")
    lines.append("")
    lines.append(
        "A model has **high potential** (worth further tuning) when its Diagnostic-2 "
        "slope is large AND its Diagnostic-4 gap is small. That combination "
        "means optimization buys you real generalizing gains. A model with a "
        "big slope but a big gap is overfitting its hyperparameter search; "
        "additional tuning will mostly improve inner CV without moving outer "
        "performance. A model with a small slope is already near its ceiling — "
        "switch family rather than tune. A model with high Diagnostic-3 "
        "variance is unstable, and any apparent improvement from tuning could "
        "be noise."
    )
    lines.append("")

    # ---- Source runs --------------------------------------------------------
    lines.append("## Source runs aggregated")
    lines.append("")
    src_rows = []
    for r in runs:
        df = r["metrics"]
        src_rows.append({
            "run_id": r["run_dir"].name,
            "model": r["model"],
            "tier": r["tier"],
            "n_rows": int(len(df)),
            "n_participants": int(df["participant_id"].nunique())
                if "participant_id" in df.columns else 0,
        })
    src_df = pd.DataFrame(src_rows).sort_values(["model", "tier"]).reset_index(drop=True)
    lines.append(_format_md_table(src_df, {}))
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------
def _safe_fmt(value, fmt: str) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, (int, float, np.number)):
        return fmt.format(value)
    return str(value)


def _format_md_table(df: pd.DataFrame, formats: dict[str, str]) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table."""
    if df is None or df.empty:
        return "_(no data)_"
    df = df.copy()
    for col in df.columns:
        fmt = formats.get(col)
        if fmt is not None:
            df[col] = df[col].apply(lambda v, f=fmt: _safe_fmt(v, f))
        else:
            df[col] = df[col].apply(lambda v: "n/a" if pd.isna(v) else str(v))

    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    rows = ["| " + " | ".join(str(row[c]) for c in cols) + " |"
            for _, row in df.iterrows()]
    return "\n".join([header, sep] + rows)


if __name__ == "__main__":
    main()
