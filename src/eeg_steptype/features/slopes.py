"""Per-bin linear-trend slopes: a polyfit(deg=1) for each channel within each bin."""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd

from ..logging_utils import get_logger


log = get_logger(__name__)


def binned_slopes(
    epochs: mne.BaseEpochs,
    bin_n: float,
    ch_names: list[str],
) -> pd.DataFrame:
    """Return wide DataFrame: one row per epoch; columns ``slope_{channel}_bin_{b}``.

    Bins that contain fewer than two time samples are skipped: a linear slope
    is undefined for a single point and would otherwise yield a constant-zero
    column for every channel/epoch, which downstream feature selection has to
    discard. Concretely, MNE's ``epochs.crop(tmin, tmax)`` is inclusive on both
    ends so the sample at ``t == tmax`` lands alone in its own trailing bin;
    that bin is dropped here.
    """
    data = epochs.get_data(picks=ch_names)
    times = epochs.times
    bins = (times // bin_n).astype(int)
    unique_bins, counts = np.unique(bins, return_counts=True)
    valid_bins = unique_bins[counts >= 2]
    dropped = unique_bins[counts < 2]
    if len(dropped):
        log.debug(
            "[slopes] dropping %d single-sample bin(s): %s",
            len(dropped), dropped.tolist(),
        )

    out = {"epoch": epochs.selection}
    for b in valid_bins:
        mask = bins == b
        t = times[mask]
        denom = np.sum((t - t.mean()) ** 2)
        if denom == 0:
            # Defensive guard: with >=2 distinct time samples this should not
            # trigger, but if all samples happen to share the same timestamp
            # (e.g. numerical edge cases) skip rather than emit constant zeros.
            log.debug("[slopes] bin %d has zero time-variance; skipping.", int(b))
            continue
        weights = (t - t.mean()) / denom
        slopes = np.einsum("ect,t->ec", data[:, :, mask], weights)
        for idx, ch in enumerate(ch_names):
            out[f"slope_{ch}_bin_{b}"] = slopes[:, idx]
    return pd.DataFrame(out)
