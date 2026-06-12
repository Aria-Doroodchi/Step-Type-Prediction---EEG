"""Pooled and partially-pooled training workflows for the tabular models.

**Why this exists.** The per-participant models in :mod:`train` are each fit on
~80 epochs against tens of thousands of features (``p >> n``). That regime is
the root cause of the large inner-vs-outer overfitting gap (the inner search
looks ~0.2 AUC better than the held-out fold). *Sharing data across subjects is
the single strongest lever against that gap*, so this module adds two
cross-subject workflows that reuse the **exact** in-fold feature-selection funnel
and nested hyperparameter search from :mod:`train` (so the only thing that
changes between the three workflows is how data is shared, not the model):

* ``per_participant`` -- baseline; one model per subject (lives in
  :func:`train.train_one_participant`, included here only for comparison).
* ``full``    -- one model trained on **all other** subjects and evaluated on a
  fully held-out subject (leave-one-subject-out transfer). Largest training set,
  honest cross-subject test, but no subject-specific adaptation.
* ``partial`` -- one model trained on **all other** subjects **plus the target
  subject's own training split**, evaluated on the target's held-out split
  (global prior + local adaptation). The middle of the pooling spectrum.

The three differ only in how much cross-subject data is added to the target
subject's own training data, which isolates the effect of pooling on the gap.

Both pooled modes use **subject-grouped inner CV** (via the ``groups`` argument
threaded into :func:`train._fit_score_split`) so the hyperparameter search never
peeks across the train/test subject boundary -- otherwise the inner score would
be optimistic and the whole point (an honest gap) would be lost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..features.assemble import build_for_participant
from ..logging_utils import get_logger
from . import train as T

log = get_logger(__name__)

POOLING_MODES = ("per_participant", "full", "partial")
_ID_COLUMNS = ["condition", "participant_id", "block_id", "epoch"]


# ---------------------------------------------------------------------------
def build_pooled_frame(
    cfg: dict,
    participants: list[str],
    model_name: str = "xgb",
    *,
    channel_mode: str | None = None,
) -> pd.DataFrame:
    """Concatenate per-participant feature frames into one pooled frame.

    Keeps only feature columns present for *every* subject (identical in
    practice -- the feature schema is shared -- but intersected defensively).
    The per-epoch ``epoch`` index is preserved as a column here and dropped in
    :func:`_prep_xyg`, because its integer value repeats across subjects and
    would otherwise act as a spurious shared feature once pooled.
    """
    frames: list[pd.DataFrame] = []
    for pid in participants:
        df = build_for_participant(pid, cfg)
        df = T._apply_channel_selection(df, cfg, model_name, channel_mode=channel_mode)
        frames.append(df)

    shared = set(frames[0].columns)
    for df in frames[1:]:
        shared &= set(df.columns)
    keep = [c for c in frames[0].columns if c in shared]
    dropped = [c for c in frames[0].columns if c not in shared]
    if dropped:
        log.warning("[pooling] %d column(s) not shared by all subjects were dropped",
                    len(dropped))
    pooled = pd.concat([df[keep] for df in frames], ignore_index=True)
    pooled = pooled.dropna(axis=1, how="any")
    log.info("[pooling] pooled frame: %d epochs x %d columns from %d subjects",
             pooled.shape[0], pooled.shape[1], len(participants))
    return pooled


def _prep_xyg(pooled: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Split the pooled frame into (X features, y labels, subject groups)."""
    groups = pooled["participant_id"].to_numpy()
    X = pooled.drop(columns=[c for c in _ID_COLUMNS if c in pooled.columns])
    X = X.reset_index(drop=True)
    y = pooled["condition"].map({"One": 0, "Two": 1}).astype(int).reset_index(drop=True)
    return X, y, groups


def _subject_inner_splits(cfg: dict, s_idx: np.ndarray, y: pd.Series) -> list[tuple]:
    """Within-subject stratified folds for the target subject (partial mode)."""
    ys = y.iloc[s_idx]
    n_splits = T._bounded_splits(ys, int(T._cv_config(cfg).get("n_splits", 5)))
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(cfg["modeling"].get("random_state", 1)),
    )
    return list(skf.split(s_idx, ys))


# ---------------------------------------------------------------------------
def train_pooled(
    cfg: dict,
    model_name: str = "xgb",
    *,
    mode: str = "full",
    channel_mode: str | None = None,
    pooled_frame: pd.DataFrame | None = None,
) -> list[dict]:
    """Run a pooled (``full``) or partially-pooled (``partial``) workflow.

    Returns one metrics row per outer evaluation, with the same schema as
    :func:`train.train_one_participant` plus ``pooling_mode`` and
    ``held_out_participant``:

    * ``per_participant`` -- one row per (subject, within-subject fold); train on
      the subject's own training split only (the baseline, reproduced from the
      pooled frame so it shares features/folds with ``partial``).
    * ``partial`` -- same folds/test sets as ``per_participant`` but the other
      subjects' epochs join the training split (paired comparison).
    * ``full`` -- one row per held-out subject (leave-one-subject-out transfer).

    ``pooled_frame`` lets a caller (e.g. the comparison script) build the frame
    once and reuse it across all three modes for a perfectly matched comparison.
    """
    if mode not in POOLING_MODES:
        raise ValueError(f"pooling mode must be one of {POOLING_MODES}; got {mode!r}")
    factory = T.MODEL_FACTORIES[model_name]
    if factory.get("data_representation") != "tabular":
        raise ValueError(
            "pooling workflows support tabular models only (xgb / svm / logistic)"
        )

    participants = list(cfg["participants"])
    pooled = pooled_frame if pooled_frame is not None else build_pooled_frame(
        cfg, participants, model_name, channel_mode=channel_mode
    )
    X, y, groups = _prep_xyg(pooled)
    subjects = list(pd.unique(groups))
    window = cfg.get("_prediction_window", "full_cnv")

    rows: list[dict] = []
    for s in subjects:
        is_s = groups == s
        s_idx = np.where(is_s)[0]
        others = np.where(~is_s)[0]
        if len(others) == 0:
            log.warning("[pooling/%s] only one subject present; skipping %s", mode, s)
            continue

        if mode == "full":
            # Leave-one-subject-out: train on everyone else, test the whole subject.
            splits = [(others, s_idx, 0)]
        elif mode == "partial":
            # Target's own train split joins the pooled training set.
            splits = [
                (np.concatenate([s_idx[tr_pos], others]), s_idx[te_pos], fold)
                for fold, (tr_pos, te_pos) in enumerate(_subject_inner_splits(cfg, s_idx, y))
            ]
        else:  # per_participant baseline: subject's own train split only
            splits = [
                (s_idx[tr_pos], s_idx[te_pos], fold)
                for fold, (tr_pos, te_pos) in enumerate(_subject_inner_splits(cfg, s_idx, y))
            ]

        # Grouped inner CV only matters when >1 subject is in the training set.
        inner_groups = None if mode == "per_participant" else groups

        for train_idx, test_idx, fold in splits:
            if y.iloc[test_idx].nunique() < 2 or y.iloc[train_idx].nunique() < 2:
                log.warning("[pooling/%s] subject %s fold %d is single-class; skipping",
                            mode, s, fold)
                continue
            row = T._fit_score_split(
                s, cfg, model_name, factory, X, y, train_idx, test_idx, groups=inner_groups,
            )
            row["pooling_mode"] = mode
            row["held_out_participant"] = s
            row["cv_mode"] = f"pooled_{mode}"
            row["repeat"] = 0
            row["fold"] = fold
            row["n_train"] = int(len(train_idx))
            row["n_train_subjects"] = int(pd.unique(groups[train_idx]).size)
            row["prediction_window"] = window
            row["window_min_time"] = float(cfg["features"]["min_time"])
            row["window_max_time"] = float(cfg["features"]["max_time"])
            rows.append(row)
            log.info("[pooling/%s] %s fold %d: testAUC=%.3f innerCV=%.3f (train=%d epochs/%d subj)",
                     mode, s, fold, row["auc"], row["inner_best_score"],
                     row["n_train"], row["n_train_subjects"])
    return rows


def overfit_gap(rows: list[dict] | pd.DataFrame) -> dict:
    """Summarise inner-vs-outer overfitting for a set of metrics rows."""
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    if df.empty:
        return {"n": 0, "test_auc": float("nan"), "inner_cv": float("nan"), "gap": float("nan")}
    test_auc = float(df["auc"].mean())
    inner_cv = float(df["inner_best_score"].mean())
    return {
        "n": int(len(df)),
        "n_subjects": int(df["held_out_participant"].nunique()) if "held_out_participant" in df else 0,
        "test_auc": test_auc,
        "test_auc_sd": float(df["auc"].std(ddof=1)) if len(df) > 1 else 0.0,
        "inner_cv": inner_cv,
        "gap": inner_cv - test_auc,
    }
