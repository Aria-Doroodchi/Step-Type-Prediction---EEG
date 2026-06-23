"""Generate configs/state/overrides/Pxx.yaml raw-assembly files.

Recipes were extracted (and verified) from the researcher's precedent scripts
(`Pxx_Stim_2s.py` for the Stim/stepping assembly, `Pxx_Stim.py` for the
Control/Standing assembly). Only the irreducibly-manual raw crop/concat is kept
(per the brief). Bad-channel detection, ICA exclusion, referencing, filtering,
and rejection are AUTOMATED in the state pipeline — the precedent's hand-set
lists are intentionally NOT transcribed (the precedent scripts remain the
provenance of record). Only participants whose assembly differs from the
default single file get an override.
"""

from __future__ import annotations

from pathlib import Path

OVERRIDES_DIR = Path("configs/state/overrides")

# segment = [path_relative_to_raw_root, tmin_or_None, tmax_or_None]
STIM = {
    "P02": [["P02/P02_Stim.bdf", 0, 539], ["P02/P02_Stim.bdf", 553, 1089]],
    "P03": [["P03/P03_Stim.bdf", None, None], ["P03/P03_Stim_2.bdf", 6, 543]],
    "P05": [["P05/P05_Stim_1.bdf", None, None]],          # real data; Stim.bdf is a 40s fragment
    "P07": [["P07/P07_Stim.bdf", None, None], ["P07/P07_Stim_2.bdf", None, None]],
    "P10": [["P10/P10_Stim.bdf", 316, None]],
    "P11": [["P11/P11_Stim.bdf", 10, 543], ["P11/P11_Stim.bdf", 558, 1090]],
    "P13": [["P13/P13_Stim.bdf", 12, 543], ["P13/P13_Stim.bdf", 560, 1094]],
    "P29": [["P29/P29_Stim.bdf", 20, 1098]],
    "P37": [["P37/P37_Stim.bdf", 0, 545], ["P37/P37_Stim.bdf", 562, 1100]],
    "P39": [["P39/P39_Stim.bdf", 0, 544], ["P39/P39_Stim_2.bdf", 0, 544]],
}
STANDING = {
    "P06": [["P06/P06_Standing.bdf", 0, 80], ["P06/P06_Standing.bdf", 100, 155],
            ["P06/P06_Standing.bdf", 170, None]],
    "P07": [["P07/P07_Standing.bdf", 20, 190]],
    "P08": [["P08/P08_Standing.bdf", 0, 80]],
    "P09": [["P09/P09_Standing.bdf", 10, 174.9]],
    "P11": [["P11/P11_Standing.bdf", 10, 164]],
    "P13": [["P13/P13_Standing.bdf", 4, 125]],
    "P17": [["P17/P17_Standing.bdf", 173, 284]],
    "P27": [["P27/P27_Standing.bdf", 0, 165]],
    "P29": [["P29/P29_Standing.bdf", 0, 38], ["P29/P29_Standing.bdf", 122, 273]],
}


def _seg_yaml(seg) -> str:
    path, tmin, tmax = seg
    parts = [f'path: "{path}"']
    if tmin is not None:
        parts.append(f"tmin: {tmin}")
    if tmax is not None:
        parts.append(f"tmax: {tmax}")
    return "      - {" + ", ".join(parts) + "}"


def main():
    OVERRIDES_DIR.mkdir(parents=True, exist_ok=True)
    pids = sorted(set(STIM) | set(STANDING))
    for pid in pids:
        lines = [
            f"# {pid} — state raw assembly (manual crop/concat).",
            "# Extracted from the precedent Pxx_Stim_2s.py (Stim) / Pxx_Stim.py (Control).",
            "# Everything else (bads, ICA, reference, filter, reject) is AUTOMATED.",
            "raw_assembly:",
        ]
        if pid in STIM:
            lines.append("  Stim:")
            lines.append("    files:")
            lines += [_seg_yaml(s) for s in STIM[pid]]
        if pid in STANDING:
            lines.append("  Standing:")
            lines.append("    files:")
            lines += [_seg_yaml(s) for s in STANDING[pid]]
        text = "\n".join(lines) + "\n"
        (OVERRIDES_DIR / f"{pid}.yaml").write_text(text, encoding="utf-8")
        print(f"wrote {pid}.yaml ({'Stim ' if pid in STIM else ''}"
              f"{'Standing' if pid in STANDING else ''})")
    print(f"\n{len(pids)} override files written to {OVERRIDES_DIR}")


if __name__ == "__main__":
    main()
