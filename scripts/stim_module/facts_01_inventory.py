"""Task 1 — fact-finding pass 1: inventory of hand-set deltas, sfreq, event counts.

Reads every participant's Pxx_Stim*.py to harvest the hand-set trigger deltas,
opens each Pxx_Stim.bdf header to read sampling frequency, and runs the event
parsing (256/512/1024) to count One-stim vs Two-stim attributions.

Writes a tidy CSV to outputs/stim_module/ and prints a summary.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mne

mne.set_log_level("ERROR")

RAW_ROOT = Path(
    "C:/Users/Ali D/OneDrive - The University of Western Ontario/MSc/Thesis/Data/Participants"
)
OUT = Path("outputs/stim_module")
OUT.mkdir(parents=True, exist_ok=True)

DELTA_RE = re.compile(r"(\w+)_trig_diff\s*=\s*(-?\d+)")


def parse_deltas(py_path: Path) -> dict:
    out = {}
    if not py_path.exists():
        return out
    txt = py_path.read_text(encoding="utf-8", errors="ignore")
    for m in DELTA_RE.finditer(txt):
        out[m.group(1)] = int(m.group(2))
    return out


def find_stim_py(pdir: Path, pid: str) -> Path | None:
    # Prefer the canonical Pxx_Stim.py; fall back to _2s / _2 variants.
    for name in (f"{pid}_Stim.py", f"{pid}_Stim_2s.py", f"{pid}_Stim_2.py"):
        p = pdir / name
        if p.exists():
            return p
    cands = sorted(pdir.glob(f"{pid}_Stim*.py"))
    return cands[0] if cands else None


def pair_stims(events: np.ndarray) -> dict:
    """Replicate the per-participant script's parsing.

    For each 256/512 prompt, the following 1024 events (with prev-value col==0)
    are bound to that condition, up to 4 stims. Returns counts and the bound
    event arrays.
    """
    one, two = [], []
    cond = None
    count = 0
    for row in events:
        c = row[2]
        if c == 256:
            cond, count = "One", 0
        elif c == 512:
            cond, count = "Two", 0
        elif c == 1024 and row[1] == 0:
            count += 1
            if cond == "One":
                one.append(row)
            elif cond == "Two":
                two.append(row)
            if count >= 4:
                cond = None
    return {
        "one": np.array(one) if one else np.empty((0, 3), int),
        "two": np.array(two) if two else np.empty((0, 3), int),
    }


def main():
    pids = [p.name for p in sorted(RAW_ROOT.glob("P*")) if re.fullmatch(r"P\d+", p.name)]
    rows = []
    for pid in pids:
        pdir = RAW_ROOT / pid
        bdf = pdir / f"{pid}_Stim.bdf"
        rec = {"pid": pid, "has_bdf": bdf.exists()}
        # deltas
        py = find_stim_py(pdir, pid)
        rec["stim_py"] = py.name if py else None
        d = parse_deltas(py) if py else {}
        rec["d_Control"] = d.get("Control")
        rec["d_One"] = d.get("One")
        rec["d_Two"] = d.get("Two")
        if not bdf.exists():
            rows.append(rec)
            continue
        try:
            raw = mne.io.read_raw_bdf(bdf, preload=False, verbose="ERROR")
            rec["sfreq"] = float(raw.info["sfreq"])
            rec["n_ch"] = len(raw.ch_names)
            rec["dur_s"] = round(raw.n_times / raw.info["sfreq"], 1)
            # events
            try:
                ev = mne.find_events(
                    raw, stim_channel="Status", initial_event=True,
                    min_duration=0.002, shortest_event=0.002, consecutive=True,
                    output="onset", verbose="ERROR",
                )
            except Exception:
                # channel may already be named Stim in some files
                ev = mne.find_events(
                    raw, initial_event=True, min_duration=0.002,
                    shortest_event=0.002, consecutive=True, output="onset",
                    verbose="ERROR",
                )
            codes = ev[:, 2]
            rec["n_256"] = int(np.sum(codes == 256))
            rec["n_512"] = int(np.sum(codes == 512))
            rec["n_1024"] = int(np.sum(codes == 1024))
            paired = pair_stims(ev)
            rec["n_one_stim"] = len(paired["one"])
            rec["n_two_stim"] = len(paired["two"])
            rec["n_paired"] = rec["n_one_stim"] + rec["n_two_stim"]
            rec["n_1024_unpaired"] = rec["n_1024"] - rec["n_paired"]
            # unique trigger codes present (small ones)
            uniq = sorted(set(int(c) for c in codes))
            rec["uniq_codes"] = ",".join(str(c) for c in uniq if c < 5000)
        except Exception as e:
            rec["error"] = repr(e)[:200]
        rows.append(rec)
        print(f"{pid}: sfreq={rec.get('sfreq')} 256={rec.get('n_256')} "
              f"512={rec.get('n_512')} 1024={rec.get('n_1024')} "
              f"one_stim={rec.get('n_one_stim')} two_stim={rec.get('n_two_stim')} "
              f"unpaired={rec.get('n_1024_unpaired')} "
              f"d=({rec.get('d_Control')},{rec.get('d_One')},{rec.get('d_Two')})")
        sys.stdout.flush()

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "facts_inventory.csv", index=False)
    print("\n=== SUMMARY ===")
    print("n participants:", len(df))
    if "sfreq" in df:
        print("sfreq values:", df["sfreq"].dropna().unique())
    for col in ("d_Control", "d_One", "d_Two"):
        s = df[col].dropna()
        if len(s):
            print(f"{col}: n={len(s)} mean={s.mean():.1f} sd={s.std():.1f} "
                  f"min={s.min()} max={s.max()}")
    print("\nWrote", OUT / "facts_inventory.csv")


if __name__ == "__main__":
    main()
