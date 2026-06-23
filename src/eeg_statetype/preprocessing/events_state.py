"""3-way event building + foot-sole e-stim attribution for the state module.

Two source recordings, three conditions:

* **straight / diagonal** (``Pxx_Stim.bdf``): a single ordered walk over the
  trigger stream. A ``256``/``512`` cue opens a prompt; the first following
  ``96`` (prev==0) is the analysis-epoch onset (locked exactly like the CNV
  window and ``Pxx_Stim_2s.py``); the subsequent ``1024`` e-stims (prev==0, up
  to 4) are attributed to that prompt. Verified order (LEDGER 2026-06-22):
  ``256 → 96(+0.55s) → 1024×4(+1.6…+3.2s)`` — i.e. the e-stims trail the onset,
  so **per-prompt** attribution (not strict time-containment) keeps ≈4 e-stims
  per stepping epoch while remaining leakage-safe (each epoch's SEP comes only
  from its own trial's e-stims).

* **standing** (``Pxx_Standing.bdf``): no cue structure. Tiled (default) or
  random non-overlapping 2 s analysis windows are cut from the continuous
  recording; e-stims are attributed by **time-containment** (corrected sample
  inside the window). ISI ≈ 0.52 s ⇒ ≈4 e-stims per window.

Every ``1024`` sample is shifted to the true stim by the uniform 273 ms offset
(``round(0.273 * sfreq)`` per *file* — sfreq is read from the recording, since
e.g. P06/P29 have a 1024 Hz Stim but a 2048 Hz Standing).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import mne

from ..logging_utils import get_logger


log = get_logger(__name__)

# Synthetic analysis-onset code (the t=0 marker for every analysis epoch). The
# real condition is carried in epoch metadata, not the event code.
ONSET_CODE = 96

_COND_TO_HANDSET = {"standing": "Control", "straight": "One", "diagonal": "Two"}


def _codes(cfg: dict) -> tuple[int, int, int, int]:
    se = cfg["state_events"]
    return int(se["one"]), int(se["two"]), int(se["response"]), int(se["stim"])


def estim_offset_samples(cfg: dict, sfreq: float, pid: str | None = None,
                         condition: str | None = None) -> int:
    """Trigger→true-stim offset in SAMPLES for this recording.

    Default: uniform 273 ms (Task-1 calibration) ⇒ round(0.273 * sfreq).
    Fallback (``apply_uniform_offset: false``): the researcher's hand-set
    per-participant/condition delta from ``configs/stim.yaml`` (already in
    samples; note these under-correct the 2048 Hz files — see LEDGER).
    """
    se = cfg["state_events"]
    if se.get("apply_uniform_offset", True) or pid is None or condition is None:
        return int(round(float(se["trigger_offset_s"]) * sfreq))
    deltas = _handset_deltas()
    rec = deltas.get(pid, {})
    val = rec.get(_COND_TO_HANDSET.get(condition))
    if val is None:
        log.warning("[%s/%s] no hand-set delta; using uniform %.0f ms offset.",
                    pid, condition, float(se["trigger_offset_s"]) * 1000)
        return int(round(float(se["trigger_offset_s"]) * sfreq))
    return int(val)


@lru_cache(maxsize=1)
def _handset_deltas() -> dict:
    import yaml
    p = Path(__file__).resolve().parents[3] / "configs" / "stim.yaml"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return (data.get("stim_events", {}) or {}).get("hand_set_deltas", {}) or {}


def find_state_events(raw: mne.io.BaseRaw, cfg: dict, source_file: str,
                      pid: str | None = None) -> dict[str, dict]:
    """Return {condition: {analysis, estim, parent}} for one source recording.

    ``analysis`` is an (n, 3) event array (code ONSET_CODE) for the 2 s epochs;
    ``estim`` is an (m, 3) array of **offset-corrected** 1024 events for the SEP
    epochs; ``parent`` is an (m,) int array mapping each e-stim to its analysis
    epoch's original index (0..n-1).
    """
    all_events = mne.find_events(
        raw, stim_channel="Stim", initial_event=True, min_duration=0.002,
        shortest_event=0.002, consecutive=True, output="onset", verbose=False,
    )
    sfreq = float(raw.info["sfreq"])
    if source_file == "Stim":
        return _stim_events(all_events, cfg, sfreq, pid)
    if source_file == "Standing":
        return _standing_events(all_events, cfg, sfreq, raw, pid)
    raise ValueError(f"source_file must be 'Stim' or 'Standing', got {source_file!r}")


def _empty() -> dict:
    return {"analysis": [], "estim": [], "parent": []}


def _finalize(d: dict) -> dict:
    return {
        "analysis": np.array(d["analysis"], dtype=int) if d["analysis"]
        else np.empty((0, 3), int),
        "estim": np.array(d["estim"], dtype=int) if d["estim"]
        else np.empty((0, 3), int),
        "parent": np.array(d["parent"], dtype=int) if d["parent"]
        else np.empty((0,), int),
    }


def _stim_events(all_events: np.ndarray, cfg: dict, sfreq: float,
                 pid: str | None) -> dict[str, dict]:
    one, two, resp, stim = _codes(cfg)
    max_stims = int(cfg["state_events"].get("max_stims_per_prompt", 4))
    out = {"straight": _empty(), "diagonal": _empty()}
    code_to_cond = {one: "straight", two: "diagonal"}
    offsets = {c: estim_offset_samples(cfg, sfreq, pid, c) for c in out}

    cond: str | None = None
    got_onset = False
    stim_count = 0
    idx = {"straight": -1, "diagonal": -1}

    for row in all_events:
        s, prev, c = int(row[0]), int(row[1]), int(row[2])
        if c in code_to_cond:
            cond = code_to_cond[c]
            got_onset = False
            stim_count = 0
        elif cond is not None and c == resp and prev == 0 and not got_onset:
            idx[cond] += 1
            out[cond]["analysis"].append([s, 0, ONSET_CODE])
            got_onset = True
            stim_count = 0
        elif (cond is not None and got_onset and c == stim and prev == 0
              and stim_count < max_stims):
            stim_count += 1
            out[cond]["estim"].append([s + offsets[cond], 0, stim])
            out[cond]["parent"].append(idx[cond])

    res = {k: _finalize(v) for k, v in out.items()}
    for cond in res:
        log.info("[%s] %s: %d analysis epochs, %d e-stims (%.2f/epoch)",
                 pid or "?", cond, len(res[cond]["analysis"]),
                 len(res[cond]["estim"]),
                 len(res[cond]["estim"]) / max(1, len(res[cond]["analysis"])))
    return res


def _standing_events(all_events: np.ndarray, cfg: dict, sfreq: float,
                     raw: mne.io.BaseRaw, pid: str | None) -> dict[str, dict]:
    _one, _two, _resp, stim = _codes(cfg)
    offset = estim_offset_samples(cfg, sfreq, pid, "standing")
    scfg = cfg.get("standing", {})
    ep = cfg["preprocessing"]["epoch"]
    tmin, tmax = float(ep["tmin"]), float(ep["tmax"])
    sep_tmax = float(cfg["features"]["sep"]["tmax"])

    estim = all_events[(all_events[:, 2] == stim) & (all_events[:, 1] == 0)]
    estim_corr = estim[:, 0].astype(int) + offset

    first = int(raw.first_samp)
    last = first + int(raw.n_times) - 1
    edge = float(scfg.get("drop_stim_edge_s", 0.30))
    # Bounds so [onset+tmin, onset+tmax] and each contained e-stim's SEP window
    # [s+sep_tmin, s+sep_tmax] stay inside the recording.
    start = first + int(round((-tmin + edge) * sfreq))
    stop = last - int(round((tmax + sep_tmax + edge) * sfreq))
    step = int(round((tmax - tmin) * sfreq))                 # non-overlapping
    if step <= 0 or stop <= start:
        log.warning("[%s] standing: recording too short for any 2 s window.", pid)
        return {"standing": _finalize(_empty())}

    onsets = np.arange(start, stop, step, dtype=int)
    method = str(scfg.get("method", "tiled")).lower()
    if method == "random":
        rng = np.random.default_rng(int(scfg.get("random_state", 42)))
        # candidate grid at 0.25 s resolution, sampled without replacement,
        # then enforced non-overlapping by sorting + greedy spacing.
        grid = np.arange(start, stop, int(round(0.25 * sfreq)), dtype=int)
        rng.shuffle(grid)
        picked: list[int] = []
        for cand in grid:
            if all(abs(cand - p) >= step for p in picked):
                picked.append(int(cand))
            if len(picked) >= len(onsets):
                break
        onsets = np.array(sorted(picked), dtype=int)
    elif method != "tiled":
        raise ValueError("standing.method must be 'tiled' or 'random'")

    # Optional cap on standing windows BEFORE downstream src/SEP cost (standing is
    # balanced down to the stepping count at train time anyway). Off by default.
    max_windows = scfg.get("max_windows")
    if max_windows and len(onsets) > int(max_windows):
        rng = np.random.default_rng(int(scfg.get("random_state", 42)))
        keep = np.sort(rng.choice(len(onsets), size=int(max_windows), replace=False))
        log.info("[%s] standing: capping %d -> %d windows", pid, len(onsets), int(max_windows))
        onsets = onsets[keep]

    d = _empty()
    win_lo = (onsets + int(round(tmin * sfreq)))
    win_hi = (onsets + int(round(tmax * sfreq)))
    for k, onset in enumerate(onsets):
        d["analysis"].append([int(onset), 0, ONSET_CODE])
    # attribute each corrected e-stim to the window containing it (if any)
    for s in estim_corr:
        hit = np.where((s >= win_lo) & (s <= win_hi))[0]
        if len(hit):
            d["estim"].append([int(s), 0, stim])
            d["parent"].append(int(hit[0]))

    res = {"standing": _finalize(d)}
    log.info("[%s] standing: %d windows (%s), %d e-stims (%.2f/window)",
             pid or "?", len(res["standing"]["analysis"]), method,
             len(res["standing"]["estim"]),
             len(res["standing"]["estim"]) / max(1, len(res["standing"]["analysis"])))
    return res
