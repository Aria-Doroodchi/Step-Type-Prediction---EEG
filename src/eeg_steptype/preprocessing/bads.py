"""Automated bad-channel detection.

Strategy: run PyPREP's NoisyChannels detector on the raw recording, then
union with any explicit ``bads_extra`` when full participant override mode is
enabled. The override is the escape hatch for channels the detector misses.

Falls back gracefully to the override-only path if PyPREP isn't installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mne

from ..logging_utils import get_logger


log = get_logger(__name__)


@dataclass(frozen=True)
class BadChannelSummary:
    pyprep: list[str]
    override: list[str]
    final: list[str]
    interpolated: list[str]


def detect_bads(
    raw: mne.io.BaseRaw,
    cfg: dict,
    *,
    return_summary: bool = False,
) -> list[str] | tuple[list[str], BadChannelSummary]:
    """Return the union of auto-detected bads and ``bads_extra`` overrides.

    ``cfg`` may contain:
        preprocessing.bads.method       "auto" (default) | "manual"
        preprocessing.bads.random_state
        preprocessing.bads.ransac
        preprocessing.bads.channel_wise
        preprocessing.bads.reject_by_annotation
        bads_extra:                     list of channel names (override)
    """
    bads_cfg = cfg.get("preprocessing", {}).get("bads", {})
    method = bads_cfg.get("method", "auto")
    extra: list[str] = list(cfg.get("bads_extra") or [])

    auto: list[str] = []
    if method == "auto":
        auto = _pyprep_detect(raw, bads_cfg)

    bads = sorted(set(auto) | set(extra))
    log.info("Bads: pyprep=%s, override=%s, final=%s", auto, extra, bads)
    summary = BadChannelSummary(
        pyprep=sorted(auto),
        override=sorted(extra),
        final=bads,
        interpolated=[],
    )
    if return_summary:
        return bads, summary
    return bads


def apply_bads(raw: mne.io.BaseRaw, bads: Iterable[str]) -> mne.io.BaseRaw:
    """Mark channels as bad; interpolation is performed by the pipeline."""
    raw.info["bads"] = list(bads)
    return raw


# ---------------------------------------------------------------------------
def _pyprep_detect(raw: mne.io.BaseRaw, bads_cfg: dict) -> list[str]:
    try:
        from pyprep.find_noisy_channels import NoisyChannels
    except Exception as exc:                            # noqa: BLE001
        log.warning("PyPREP unavailable (%s); skipping auto bad detection.", exc)
        return []

    try:
        random_state = bads_cfg.get(
            "random_state", bads_cfg.get("pyprep_random_state", 42)
        )
        ransac = bool(bads_cfg.get("ransac", True))
        channel_wise = bool(bads_cfg.get("channel_wise", False))
        reject_by_annotation = bads_cfg.get("reject_by_annotation")

        # Drop the Stim channel for the noisy-channel scan.
        scan = raw.copy().drop_channels(
            [c for c in raw.ch_names if c.lower() == "stim"]
        )
        nc = NoisyChannels(
            scan,
            random_state=random_state,
            ransac=ransac,
            reject_by_annotation=reject_by_annotation,
        )
        nc.find_all_bads(
            channel_wise=channel_wise,
            reject_by_annotation=reject_by_annotation,
        )
        return nc.get_bads()
    except Exception as exc:                            # noqa: BLE001
        log.warning("PyPREP failed: %s; falling back to no auto bads.", exc)
        return []
