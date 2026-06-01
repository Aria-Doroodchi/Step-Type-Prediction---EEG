"""Per-participant source-localization pipeline.

For each (participant, condition):
    1. Load cleaned epochs.
    2. Build forward (cached) + noise cov + inverse op  ← ONCE per condition.
    3. Loop over epochs:
         - average single-epoch evoked
         - apply inverse → SourceEstimate
         - extract label time courses
         - bin in time, transpose to a one-row-per-epoch dataframe.
    4. Write `data/src/{pid}_{cond}_src.csv`.
"""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd

from ..config import apply_participant_override
from ..io import source_epochs_path, src_csv_path, write_csv
from ..logging_utils import get_logger
from ..preflight import validate_source_assets, validate_source_epochs
from .diagnostics import (
    append_variance_rows,
    clear_variance_for_participant,
    log_file_error,
    make_variance_summary_rows,
)
from .forward import build_forward
from .inverse import (
    apply_to_evoked,
    build_inverse,
    compute_noise_cov,
    ensure_average_reference_projection,
)
from .labels import load_labels, extract_label_courses


log = get_logger(__name__)


def run(participant_id: str, cfg: dict, *, force: bool = False) -> None:
    cfg = apply_participant_override(cfg, participant_id)
    if force:
        clear_variance_for_participant(cfg, participant_id)
    try:
        validate_source_assets(cfg)
    except FileNotFoundError as exc:
        log_file_error(
            cfg,
            participant_id=participant_id,
            condition=None,
            stage="validate_source_assets",
            path="multiple",
            exception=exc,
        )
        raise
    sl = cfg["source_localization"]
    bin_n = float(sl.get("bin_n", 0.125))
    min_t = float(sl.get("min_time", 0.0))
    response_code = str(cfg["events"]["response"])

    try:
        labels, ba_names = load_labels(cfg)
    except FileNotFoundError as exc:
        log_file_error(
            cfg,
            participant_id=participant_id,
            condition=None,
            stage="load_labels",
            path="fsaverage labels",
            exception=exc,
        )
        raise
    participant_variance_rows = []

    for cond in cfg["conditions"]:
        out_path = src_csv_path(cfg, participant_id, cond)
        if out_path.exists() and not force:
            log.info("[%s/%s] src csv exists; skipping.", participant_id, cond)
            continue

        epo_path = source_epochs_path(cfg, participant_id, cond)
        if not epo_path.exists():
            log.warning("[%s/%s] source epochs file missing: %s", participant_id, cond, epo_path)
            log_file_error(
                cfg,
                participant_id=participant_id,
                condition=cond,
                stage="load_source_epochs",
                path=epo_path,
                message=f"Source epochs file missing: {epo_path}",
            )
            continue

        try:
            validate_source_epochs(cfg, participant_id, cond)
            epochs = mne.read_epochs(str(epo_path), preload=True)
            epochs = ensure_average_reference_projection(epochs)

            # Build the heavy machinery ONCE for this (participant, condition).
            fwd = build_forward(epochs.info, cfg, participant_id=participant_id)
            noise_cov = compute_noise_cov(epochs)
            inv_op = build_inverse(epochs.info, fwd, noise_cov)
        except FileNotFoundError as exc:
            log_file_error(
                cfg,
                participant_id=participant_id,
                condition=cond,
                stage="prepare_source_localization",
                path=epo_path,
                exception=exc,
            )
            raise

        rows = []
        condition_variance_rows = []
        epoch_nums = epochs.selection.tolist()
        for idx, num in enumerate(epoch_nums):
            sub_evoked = epochs[[idx]][response_code].average()
            stc, residual = apply_to_evoked(
                sub_evoked,
                inv_op,
                cfg,
                return_residual=True,
            )
            condition_variance_rows.append(_variance_diagnostic_row(
                participant_id=participant_id,
                condition=cond,
                epoch=num,
                evoked=sub_evoked,
                residual=residual,
            ))
            bm_activity = extract_label_courses(stc, labels, fwd["src"])

            df = (
                pd.DataFrame(bm_activity, index=ba_names, columns=sub_evoked.times)
                .T.reset_index().rename(columns={"index": "time"})
            )
            df = df[df["time"] >= min_t]
            df["bin"] = (df["time"] // bin_n).astype(int)
            binned = df.groupby("bin").mean(numeric_only=True).reset_index()

            wide_parts = []
            for col in ba_names:
                tmp = binned[["bin", col]].set_index("bin").T
                tmp.columns = [f"{col}_bin_{b}" for b in tmp.columns]
                tmp.reset_index(drop=True, inplace=True)
                wide_parts.append(tmp)

            row = pd.concat(wide_parts, axis=1)
            row["epoch"] = num
            rows.append(row)

        bm_df = pd.concat(rows, axis=0, ignore_index=True)
        write_csv(bm_df, out_path)
        append_variance_rows(cfg, condition_variance_rows)
        participant_variance_rows.extend(condition_variance_rows)
        log.info("[%s/%s] wrote %s (%d epochs, %d cols)",
                 participant_id, cond, out_path, len(bm_df), bm_df.shape[1])

    summary_rows = make_variance_summary_rows(
        participant_variance_rows,
        participant_id=participant_id,
    )
    append_variance_rows(cfg, summary_rows)


def _variance_diagnostic_row(
    *,
    participant_id: str,
    condition: str,
    epoch: int,
    evoked: mne.Evoked,
    residual: mne.Evoked,
) -> dict:
    data_variance = float(np.var(evoked.data))
    residual_variance = float(np.var(residual.data))
    variance_explained = (
        float("nan")
        if data_variance <= 0.0
        else 1.0 - (residual_variance / data_variance)
    )
    return {
        "row_type": "epoch",
        "participant_id": participant_id,
        "condition": condition,
        "epoch": epoch,
        "n_epochs": "",
        "variance_explained": variance_explained,
        "variance_explained_percent": variance_explained * 100.0,
        "data_variance": data_variance,
        "residual_variance": residual_variance,
        "mean_variance_explained": "",
        "std_variance_explained": "",
        "min_variance_explained": "",
        "max_variance_explained": "",
    }
