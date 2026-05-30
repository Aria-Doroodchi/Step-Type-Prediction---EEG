"""Single-process pipeline driver. Runs stages in sequence.

Usage:
    python run.py --config configs/default.yaml
    python run.py --stages preprocess src features train --model xgb
    python run.py --config configs/smoke.yaml --model logistic
    python run.py --speed-tier express --stages train --model xgb
    python run.py --speed-tier quick --parallel-participants 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from eeg_steptype.config import apply_prediction_window, load_config
from eeg_steptype.logging_utils import setup_logging, get_logger
from eeg_steptype.preprocessing import pipeline as preprocess
from eeg_steptype.source_localization import pipeline as src_loc
from eeg_steptype.features import assemble as features
from eeg_steptype.models.train import run as run_train


STAGES = ["preprocess", "src", "features", "train"]

SPEED_TIERS = {
    "lightning":  "configs/lightning.yaml",
    "express":    "configs/express.yaml",
    "quick":      "configs/quick.yaml",
    "riemannian": "configs/riemannian.yaml",
    "cnn":        "configs/cnn.yaml",
    "eegnet":     "configs/eegnet.yaml",
}


def _resolve_config_path(
    explicit: list[str] | None,
    speed_tier: str | None,
    project_root: Path,
) -> list[str] | str | None:
    """Pick the config path: explicit --config wins, else the speed-tier overlay."""
    if explicit:
        return explicit
    if speed_tier:
        return str(project_root / SPEED_TIERS[speed_tier])
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        nargs="+",
        default=None,
        help=(
            "One or more YAML overlays. They are merged left-to-right on top "
            "of default.yaml and local.yaml."
        ),
    )
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
            "Overrides modeling.parallel.participants from the config."
        ),
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help=(
            "Override resources.n_jobs from the config. Project convention: "
            "negative values reserve that many logical CPUs (e.g. -8 = "
            "use all cores except 8). Honored by preprocess/src/features "
            "stages and by sequential training. Inside parallel-participants "
            "workers the inner threads are still pinned to 1."
        ),
    )
    p.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    p.add_argument("--participants", nargs="*")
    p.add_argument(
        "--model",
        default=None,
        help=(
            "Model name. Falls back to modeling.default_model from the "
            "config (e.g. 'riemannian' when --speed-tier riemannian is "
            "active), then to 'xgb'."
        ),
    )
    p.add_argument("--run-id", default=None)
    p.add_argument("--channel-mode", choices=["full", "roi"], default=None)
    p.add_argument("--cv-mode", choices=["repeated_stratified", "grouped", "chronological"], default=None)
    p.add_argument("--prediction-window", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--participant-override-mode",
        choices=["raw_assembly_only", "full", "none"],
        default=None,
        help="How much of configs/overrides/Pxx.yaml to apply.",
    )
    args = p.parse_args()

    project_root = Path(__file__).resolve().parent
    cfg_path = _resolve_config_path(args.config, args.speed_tier, project_root)
    cfg = load_config(cfg_path)
    cfg = apply_prediction_window(cfg, args.prediction_window)
    # Stamp the speed tier into the config so the run-id snapshot
    # unambiguously identifies which tier was active, regardless of the
    # run-id name. Used by scripts/06_compare_runs.py for tier inference.
    if args.speed_tier and not args.config:
        cfg["_speed_tier"] = args.speed_tier
    if args.participant_override_mode:
        cfg.setdefault("participant_overrides", {})["mode"] = args.participant_override_mode
    if args.parallel_participants is not None:
        cfg.setdefault("modeling", {}).setdefault("parallel", {})[
            "participants"
        ] = int(args.parallel_participants)
    if args.n_jobs is not None:
        cfg.setdefault("resources", {})["n_jobs"] = int(args.n_jobs)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    log = get_logger("run")
    if args.speed_tier and not args.config:
        log.info("Using speed tier: %s (%s)", args.speed_tier, cfg_path)
    elif args.config and args.speed_tier:
        log.warning(
            "Both --config and --speed-tier were provided; --config %r wins.",
            args.config,
        )

    pids = args.participants or cfg["participants"]
    # When --participants is given, narrow the training cohort to match so the
    # train stage doesn't try to read features for participants we skipped in
    # the earlier stages.
    if args.participants:
        cfg["participants"] = list(args.participants)

    # Resolve the effective model: explicit CLI flag wins, else config tier
    # default (set by configs/riemannian.yaml), else 'xgb'.
    effective_model = (
        args.model
        or cfg.get("modeling", {}).get("default_model")
        or "xgb"
    )

    for stage in args.stages:
        log.info("\n" + "=" * 70 + f"\nStage: {stage}\n" + "=" * 70)
        if stage == "preprocess":
            for pid in pids:
                preprocess.run(pid, cfg, force=args.force)
        elif stage == "src":
            for pid in pids:
                src_loc.run(pid, cfg, force=args.force)
        elif stage == "features":
            for pid in pids:
                features.run(pid, cfg, force=args.force)
        elif stage == "train":
            run_train(
                cfg,
                model=effective_model,
                run_id=args.run_id,
                channel_mode=args.channel_mode,
                cv_mode=args.cv_mode,
            )


if __name__ == "__main__":
    main()
