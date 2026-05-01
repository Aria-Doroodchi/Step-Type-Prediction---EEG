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
from ..io import epochs_path, src_csv_path, write_csv
from ..logging_utils import get_logger
from .forward import build_forward
from .inverse import compute_noise_cov, build_inverse, apply_to_evoked
from .labels import load_labels, extract_label_courses


log = get_logger(__name__)


def run(participant_id: str, cfg: dict, *, force: bool = False) -> None:
    cfg = apply_participant_override(cfg, participant_id)
    sl = cfg["source_localization"]
    bin_n = float(sl.get("bin_n", 0.125))
    min_t = float(sl.get("min_time", 0.0))
    response_code = str(cfg["events"]["response"])

    labels, ba_names = load_labels(cfg)

    for cond in cfg["conditions"]:
        out_path = src_csv_path(cfg, participant_id, cond)
        if out_path.exists() and not force:
            log.info("[%s/%s] src csv exists; skipping.", participant_id, cond)
            continue

        epo_path = epochs_path(cfg, participant_id, cond)
        if not epo_path.exists():
            log.warning("[%s/%s] epochs file missing: %s", participant_id, cond, epo_path)
            continue

        epochs = mne.read_epochs(str(epo_path), preload=True)

        # Build the heavy machinery ONCE for this (participant, condition).
        fwd = build_forward(epochs.info, cfg, participant_id=participant_id)
        noise_cov = compute_noise_cov(epochs)
        inv_op = build_inverse(epochs.info, fwd, noise_cov)

        rows = []
        epoch_nums = epochs.selection.tolist()
        for idx, num in enumerate(epoch_nums):
            sub_evoked = epochs[[idx]][response_code].average()
            stc = apply_to_evoked(sub_evoked, inv_op, cfg)
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
        log.info("[%s/%s] wrote %s (%d epochs, %d cols)",
                 participant_id, cond, out_path, len(bm_df), bm_df.shape[1])
