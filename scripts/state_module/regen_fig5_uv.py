# -*- coding: utf-8 -*-
"""Regenerate the per-condition window ERP (report Figure 5) in average-reference
µV from the non-CSD analysis epochs already on disk (state_source_epochs/).

Standalone so it does not need a full training run's metrics — it only needs cfg
+ the participant list. Writes outputs/state_module/figs/condition_window_erp.png,
which scripts/state_module/build_report.py then copies into the report as
fig_condition_erp.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eeg_statetype.config import load_config
from eeg_statetype.io import source_epochs_path
from eeg_statetype.viz.results import condition_erp, figs_dir


def main() -> None:
    cfg = load_config(["configs/state/default.yaml"])
    all_pids = list(cfg["participants"])
    # keep only participants that actually have avg-ref analysis epochs on disk
    pids = [p for p in all_pids
            if any(source_epochs_path(cfg, p, c).exists()
                   for c in ("standing", "straight", "diagonal"))]
    out = figs_dir(cfg) / "condition_window_erp.png"
    print(f"participants with source epochs: {len(pids)}/{len(all_pids)}")
    condition_erp(cfg, pids, out)
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
