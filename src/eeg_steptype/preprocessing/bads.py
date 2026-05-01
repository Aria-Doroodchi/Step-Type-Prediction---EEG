"""Automated bad-channel detection.

Strategy: run PyPREP's NoisyChannels detector on the raw recording, then
union with any explicit ``bads_extra`` from the participant override. The
override is the escape hatch for channels the detector misses.

Falls back gracefully to the override-only path if PyPREP isn't installed.
"""

from __future__ import annotations

from typing import Iterable

import mne

from ..logging_utils import get_logger


log = get_logger(__name__)


def detect_bads(raw: mne.io.BaseRaw, cfg: dict) -> list[str]:
    """Return the union of auto-detected bads and ``bads_extra`` overrides.

    ``cfg`` may contain:
        preprocessing.bads.method       "auto" (default) | "manual"
        preprocessing.bads.pyprep_random_state
        bads_extra:                     list of channel names (override)
    """
    method = cfg.get("preprocessing", {}).get("bads", {}).get("method", "auto")
    extra: list[str] = list(cfg.get("bads_extra") or [])

    auto: list[str] = []
    if method == "auto":
        auto = _pyprep_detect(
            raw,
            random_state=cfg.get("preprocessing", {}).get("bads", {}).get(
                "pyprep_random_state", 42
            ),
        )

    bads = sorted(set(auto) | set(extra))
    log.info("Bads: pyprep=%s, override=%s, final=%s", auto, extra, bads)
    return bads


def apply_bads(raw: mne.io.BaseRaw, bads: Iterable[str]) -> mne.io.BaseRaw:
    raw.info["bads"] = list(bads)
    if bads:
        raw.interpolate_bads(reset_bads=True)
    return raw


# ---------------------------------------------------------------------------
def _pyprep_detect(raw: mne.io.BaseRaw, random_state: int = 42) -> list[str]:
    try:
        from pyprep.find_noisy_channels import NoisyChannels
    except Exception as exc:                            # noqa: BLE001
        log.warning("PyPREP unavailable (%s); skipping auto bad detection.", exc)
        return []

    try:
        # Drop the Stim channel for the noisy-channel scan.
        scan = raw.copy().drop_channels(
            [c for c in raw.ch_names if c.lower() == "stim"]
        )
        nc = NoisyChannels(scan, random_state=random_state)
        nc.find_all_bads()
        return nc.get_bads()
    except Exception as exc:                            # noqa: BLE001
        log.warning("PyPREP failed: %s; falling back to no auto bads.", exc)
        return []
