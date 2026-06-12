"""Task 2 stage 1: preprocess + epoch every participant, cache per-cell evokeds.

Writes outputs/stim_module/evokeds/<pid>.npz so the stats/plot stage never
re-reads raw BDFs. Resumable: skips a participant whose npz already exists
(pass --force to recompute).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from stim_preprocess import process_participant, GRID_MS

OUT = Path("outputs/stim_module/evokeds")
OUT.mkdir(parents=True, exist_ok=True)

COHORT = ("P01 P02 P03 P05 P06 P07 P08 P10 P11 P12 P13 P14 P15 P16 P18 P19 P21 "
          "P23 P24 P25 P26 P27 P28 P29 P30 P31 P33 P35 P37 P39").split()


def save(rec):
    pid = rec["pid"]
    arrs = {"grid_ms": rec["grid_ms"]}
    counts = {}
    for cell, d in rec["cells"].items():
        if d["data"] is not None:
            arrs[f"data__{cell}"] = d["data"].astype(np.float32)
        counts[cell] = d["n"]
    np.savez_compressed(
        OUT / f"{pid}.npz",
        ch_names=np.array(rec["ch_names"] or [], dtype=object),
        counts=np.array([counts], dtype=object),
        meta=np.array([rec["meta"]], dtype=object),
        **arrs,
    )


def main():
    force = "--force" in sys.argv
    pids = [a for a in sys.argv[1:] if a.startswith("P")] or COHORT
    for pid in pids:
        fp = OUT / f"{pid}.npz"
        if fp.exists() and not force:
            print(f"{pid}: cached, skip")
            continue
        try:
            rec = process_participant(pid)
            if not rec["cells"]:
                print(f"{pid}: NO DATA (skipped)")
                continue
            save(rec)
            cs = {k: v["n"] for k, v in rec["cells"].items()}
            print(f"{pid}: {cs}  meta={rec['meta']}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"{pid}: ERROR {e!r}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
