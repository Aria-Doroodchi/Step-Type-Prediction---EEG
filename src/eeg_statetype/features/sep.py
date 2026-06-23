"""Per-epoch foot-SEP feature block (the state module's novel block).

For each analysis epoch, the SEP epochs of *its own* in-epoch e-stims (≈4) are
averaged into one cleaner per-epoch SEP, then vertex P50/N90/P2P/RMS components
are read on a common 1024 Hz ms grid. This is **leakage-safe**: every epoch's
SEP comes only from that epoch's own e-stims — never a per-condition mean
broadcast (validity trap #1).

Components are read at fixed physiological windows ≥15 ms (after the t=0 stim
artifact), so there is no data-driven window selection:
  * P50  — peak (max) amplitude + latency in 40–50 ms
  * N90  — trough (min) amplitude + latency in 75–90 ms
  * P2P  — peak-to-peak over 15–130 ms
  * RMS  — root-mean-square over 15–130 ms
emitted per vertex channel ([Cz, C1, C2, FCz, CPz]) and for the vertex mean.

Output: one row per analysis epoch, keyed ``epoch`` (= the SEP epochs'
``parent_epoch`` metadata, i.e. original-event-index space — aligns with the
window blocks' ``epoch`` = ``epochs.selection``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import mne

from ..logging_utils import get_logger


log = get_logger(__name__)


def _grid_ms(scfg: dict) -> np.ndarray:
    sf = float(scfg.get("grid_sfreq", 1024.0))
    tmin, tmax = float(scfg["tmin"]), float(scfg["tmax"])
    return np.arange(tmin * 1000.0, tmax * 1000.0 + 1e-6, 1000.0 / sf)


def _read_components(wave_uv: np.ndarray, grid_ms: np.ndarray, scfg: dict) -> dict:
    """Read fixed-window SEP components from one baselined waveform (µV)."""
    p50 = scfg.get("p50_ms", [40, 50])
    n90 = scfg.get("n90_ms", [75, 90])
    full = scfg.get("window_ms", [15, 130])

    def w(lo, hi):
        return (grid_ms >= lo) & (grid_ms <= hi)

    out: dict[str, float] = {}
    mp = w(*p50)
    if mp.any():
        seg = wave_uv[mp]
        j = int(np.nanargmax(seg))
        out["p50amp"] = float(seg[j])
        out["p50lat"] = float(grid_ms[mp][j])
    mn = w(*n90)
    if mn.any():
        seg = wave_uv[mn]
        j = int(np.nanargmin(seg))
        out["n90amp"] = float(seg[j])
        out["n90lat"] = float(grid_ms[mn][j])
    mf = w(*full)
    if mf.any():
        seg = wave_uv[mf]
        out["p2p"] = float(np.nanmax(seg) - np.nanmin(seg))
        out["rms"] = float(np.sqrt(np.nanmean(seg ** 2)))
    return out


def sep_features(sep_epochs: mne.BaseEpochs, cfg: dict) -> pd.DataFrame:
    """Return per-epoch SEP features keyed on ``epoch`` (parent analysis epoch).

    Empty input (no SEP epochs) returns a frame with just the ``epoch`` column
    so the caller's left-join + fill still works.
    """
    scfg = cfg["features"]["sep"]
    vertex = [c for c in scfg.get("vertex_channels", ["Cz", "C1", "C2", "FCz", "CPz"])
              if c in sep_epochs.ch_names]
    per_channel = bool(scfg.get("per_channel", True))
    if not vertex:
        log.warning("SEP block: none of the vertex channels present; "
                    "available=%s", sep_epochs.ch_names[:8])
        return pd.DataFrame({"epoch": []})

    grid_ms = _grid_ms(scfg)
    base = scfg.get("baseline", [-0.05, 0.0])
    base_lo, base_hi = float(base[0]) * 1000.0, float(base[1]) * 1000.0
    bmask = (grid_ms >= base_lo) & (grid_ms <= base_hi)

    data = sep_epochs.get_data(picks=vertex) * 1e6        # (n_ep, n_vtx, n_t) µV
    t_ms = sep_epochs.times * 1000.0
    parent = np.asarray(sep_epochs.metadata["parent_epoch"].to_numpy(), dtype=int)

    rows = []
    for pe in np.unique(parent):
        grp = data[parent == pe]                          # (k, n_vtx, n_t)
        avg = grp.mean(axis=0)                            # (n_vtx, n_t)
        row: dict[str, float] = {"epoch": int(pe)}

        chan_waves = []
        for ci, ch in enumerate(vertex):
            wave = np.interp(grid_ms, t_ms, avg[ci])
            if bmask.any():
                wave = wave - np.nanmean(wave[bmask])
            chan_waves.append(wave)
            if per_channel:
                for k, v in _read_components(wave, grid_ms, scfg).items():
                    row[f"sep_{ch}_{k}"] = v

        vtx = np.mean(chan_waves, axis=0)                 # vertex-mean waveform
        if bmask.any():
            vtx = vtx - np.nanmean(vtx[bmask])
        for k, v in _read_components(vtx, grid_ms, scfg).items():
            row[f"sep_vtx_{k}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    log.info("SEP block: %d epochs × %d features (vertex=%s, per_channel=%s)",
             len(df), df.shape[1] - 1, vertex, per_channel)
    return df
