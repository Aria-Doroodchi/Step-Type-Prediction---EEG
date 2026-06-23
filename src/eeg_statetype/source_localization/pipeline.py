"""Per-participant source localization for the state module.

Reuses the CNV source primitives (shared fsaverage forward, eLORETA inverse,
label time-courses) on the state non-CSD avg-ref source epochs, writing
``data/src_state/{pid}_{cond}_src.csv``. Per-epoch eLORETA is the cost driver
(budget ≈ the CNV cost × the larger state epoch count); time-logged per
condition.
"""

from __future__ import annotations

import time

import mne
import numpy as np
import pandas as pd

from eeg_steptype.preflight import validate_source_assets, validate_source_epochs_file
from eeg_steptype.source_localization.forward import build_forward
from eeg_steptype.source_localization.inverse import (
    apply_to_evoked,
    build_inverse,
    compute_noise_cov,
    ensure_average_reference_projection,
)
from eeg_steptype.source_localization.labels import load_labels, extract_label_courses

from ..config import apply_participant_override
from ..io import source_epochs_path, src_csv_path, write_csv
from ..logging_utils import get_logger


log = get_logger(__name__)


def run(participant_id: str, cfg: dict, *, force: bool = False) -> None:
    cfg = apply_participant_override(cfg, participant_id)
    validate_source_assets(cfg)

    sl = cfg["source_localization"]
    bin_n = float(sl.get("bin_n", 0.0625))
    min_t = float(sl.get("min_time", 0.0))
    response_code = str(cfg["events"]["response"])
    labels, ba_names = load_labels(cfg)

    for cond in cfg["conditions"]:
        out_path = src_csv_path(cfg, participant_id, cond)
        if out_path.exists() and not force:
            log.info("[%s/%s] src csv exists; skipping.", participant_id, cond)
            continue

        epo_path = source_epochs_path(cfg, participant_id, cond)
        if not epo_path.exists():
            log.warning("[%s/%s] source epochs missing: %s", participant_id, cond, epo_path)
            continue
        validate_source_epochs_file(epo_path, participant_id=participant_id, condition=cond)

        epochs = mne.read_epochs(str(epo_path), preload=True)
        epochs = ensure_average_reference_projection(epochs)
        fwd = build_forward(epochs.info, cfg, participant_id=participant_id)
        noise_cov = compute_noise_cov(epochs)
        inv_op = build_inverse(epochs.info, fwd, noise_cov)

        t0 = time.perf_counter()
        rows = []
        var_expl = []
        epoch_nums = epochs.selection.tolist()
        for idx, num in enumerate(epoch_nums):
            sub_evoked = epochs[[idx]][response_code].average()
            stc, residual = apply_to_evoked(sub_evoked, inv_op, cfg, return_residual=True)
            dv = float(np.var(sub_evoked.data))
            rv = float(np.var(residual.data))
            var_expl.append(1.0 - rv / dv if dv > 0 else np.nan)
            bm_activity = extract_label_courses(stc, labels, fwd["src"])
            df = (pd.DataFrame(bm_activity, index=ba_names, columns=sub_evoked.times)
                  .T.reset_index().rename(columns={"index": "time"}))
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
        dt = time.perf_counter() - t0
        log.info("[%s/%s] wrote %s (%d epochs, %d cols) in %.1fs (%.2fs/epoch; "
                 "mean var-expl %.1f%%)", participant_id, cond, out_path.name,
                 len(bm_df), bm_df.shape[1], dt, dt / max(1, len(epoch_nums)),
                 100.0 * float(np.nanmean(var_expl)) if var_expl else float("nan"))
