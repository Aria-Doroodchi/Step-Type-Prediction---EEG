"""Stage 4 — feature parquets → trained model + metrics CSV.

Usage:
    python scripts/04_train.py --model xgb
    python scripts/04_train.py --model lstm  --config configs/default.yaml
    python scripts/04_train.py --model logistic --config configs/smoke.yaml
    python scripts/04_train.py --model xgb --speed-tier express
    python scripts/04_train.py --model xgb --speed-tier quick --parallel-participants 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_steptype.config import apply_prediction_window, load_config
from eeg_steptype.logging_utils import setup_logging, get_logger
from eeg_steptype.models.train import run as run_train, MODEL_FACTORIES


SPEED_TIERS = {
    "lightning":  "configs/lightning.yaml",
    "express":    "configs/express.yaml",
    "quick":      "configs/quick.yaml",
    "riemannian": "configs/riemannian.yaml",
}


def _resolve_config_path(
    explicit: str | None,
    speed_tier: str | None,
    project_root: Path,
) -> str | None:
    """Pick the config path: explicit --config wins, else the speed-tier overlay."""
    if explicit:
        return explicit
    if speed_tier:
        return str(project_root / SPEED_TIERS[speed_tier])
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument(
        "--speed-tier",
        choices=list(SPEED_TIERS.keys()),
        default=None,
        help=(
            "Use a pre-built speed-tier overlay (configs/<tier>.yaml). "
            "lightning ~1-3 min/participant, express ~4-8, quick ~10-15. "
            "Ignored if --config is explicitly provided."
        ),
    )
    p.add_argument(
        "--parallel-participants",
        type=int,
        default=None,
        help=(
            "Number of participants to train concurrently with joblib. "
            "Overrides modeling.parallel.participants from the config. "
            "Use 1 for sequential execution."
        ),
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help=(
            "Override resources.n_jobs from the config. Project convention: "
            "negative values reserve that many logical CPUs (e.g. -8 = "
            "use all cores except 8). Inside parallel-participants workers "
            "the inner threads are still pinned to 1."
        ),
    )
    p.add_argument(
        "--model",
        choices=sorted(MODEL_FACTORIES.keys()),
        default=None,
        help=(
            "Model name. Falls back to modeling.default_model from the "
            "config (e.g. 'riemannian' when --speed-tier riemannian is "
            "active), then to 'xgb'."
        ),
    )
    p.add_argument("--run-id", default=None)
    p.add_argument(
        "--prediction-window",
        default=None,
        help="Named prediction window from config, e.g. late_cnv or full_cnv.",
    )
    p.add_argument(
        "--channel-mode",
        choices=["full", "roi"],
        default=None,
        help="Training-time channel feature subset. Use twice for comparison: full and roi.",
    )
    p.add_argument(
        "--cv-mode",
        choices=["repeated_stratified", "grouped", "chronological"],
        default=None,
        help="Outer CV strategy. Defaults to modeling.cv.mode from the config.",
    )
    p.add_argument(
        "--participant-override-mode",
        choices=["raw_assembly_only", "full", "none"],
        default=None,
        help="Accepted for workflow consistency; training uses already-built features.",
    )
    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg_path = _resolve_config_path(args.config, args.speed_tier, project_root)
    cfg = load_config(cfg_path)
    cfg = apply_prediction_window(cfg, args.prediction_window)
    if args.participant_override_mode:
        cfg.setdefault("participant_overrides", {})["mode"] = args.participant_override_mode
    if args.parallel_participants is not None:
        cfg.setdefault("modeling", {}).setdefault("parallel", {})[
            "participants"
        ] = int(args.parallel_participants)
    if args.n_jobs is not None:
        cfg.setdefault("resources", {})["n_jobs"] = int(args.n_jobs)

    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    log = get_logger("scripts.04_train")
    if args.speed_tier and not args.config:
        log.info("Using speed tier: %s (%s)", args.speed_tier, cfg_path)
    elif args.config and args.speed_tier:
        log.warning(
            "Both --config and --speed-tier were provided; --config %r wins.",
            args.config,
        )

    # Resolve effective model: explicit CLI flag wins, else config tier default,
    # else 'xgb'.
    effective_model = (
        args.model
        or cfg.get("modeling", {}).get("default_model")
        or "xgb"
    )

    log.info(
        "Training model=%s channel_mode=%s on %d participants (parallel=%s)",
        effective_model,
        args.channel_mode or cfg.get("channel_selection", {}).get("mode", "full"),
        len(cfg["participants"]),
        cfg.get("modeling", {}).get("parallel", {}).get("participants", 1),
    )

    run_train(
        cfg,
        model=effective_model,
        run_id=args.run_id,
        channel_mode=args.channel_mode,
        cv_mode=args.cv_mode,
    )


if __name__ == "__main__":
    main()
