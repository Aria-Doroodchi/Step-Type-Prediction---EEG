"""Phase 0 — fact verification for the 3-state classification module.

Confirms, directly from the raw .bdf trigger channel, the claims in the brief:
  * standing = Pxx_Standing.bdf carries ONLY 1024 e-stims (no 256/512 cues);
  * straight/diagonal = Pxx_Stim.bdf 256/512 prompts -> 4x 1024 + a 96 response;
  * pair_stims reproduces ~40 prompts/condition with 4 stims each;
  * sfreq per recording (1024 vs the 2048 Hz outliers);
  * a count of candidate non-overlapping 2 s standing windows.

Read-only. Does NOT preload full data (find_events only needs the Status
channel), so it is fast. Mirrors scripts/stim_module/facts_01_inventory.py.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import mne

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stim_module"))
from stim_common import pair_stims, RAW_ROOT, CODE_ONE, CODE_TWO, CODE_STIM  # noqa: E402

mne.set_log_level("ERROR")

RESP = 96
STAND_WIN_S = 2.0

# A representative slice: P01 (no Standing), P02/P11 (2048 Hz), P03 (half),
# P07 (long standing), P26 (no Stim), plus a couple of regular ones.
SMOKE = ["P01", "P02", "P03", "P05", "P06", "P07", "P11", "P26", "P29"]


def _events(path: Path):
    raw = mne.io.read_raw_bdf(str(path), preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    ev = mne.find_events(raw, stim_channel="Status", initial_event=True,
                         min_duration=0.002, shortest_event=0.002,
                         consecutive=True, output="onset", verbose="ERROR")
    dur = raw.n_times / sfreq
    return ev, sfreq, dur


def main(pids):
    for pid in pids:
        print(f"\n===== {pid} =====")
        stim_p = RAW_ROOT / pid / f"{pid}_Stim.bdf"
        stand_p = RAW_ROOT / pid / f"{pid}_Standing.bdf"

        # ---- Stim (straight=One/256, diagonal=Two/512) ----
        if stim_p.exists():
            ev, sf, dur = _events(stim_p)
            codes = Counter(ev[:, 2].tolist())
            one, two, info = pair_stims(ev)
            # cue->response pairing (straight/diagonal stepping window)
            n_one_resp = _pair_resp(ev, CODE_ONE)
            n_two_resp = _pair_resp(ev, CODE_TWO)
            stims_per = Counter(info["order"].tolist()) if len(info["order"]) else {}
            print(f"  Stim.bdf      sfreq={sf:.0f}  dur={dur:.0f}s")
            print(f"    codes: 256={codes.get(CODE_ONE,0)} 512={codes.get(CODE_TWO,0)} "
                  f"96={codes.get(RESP,0)} 1024={codes.get(CODE_STIM,0)}")
            print(f"    pair_stims: One-stims={len(one)} Two-stims={len(two)} "
                  f"(per-prompt order hist={dict(sorted(stims_per.items()))})")
            print(f"    cue->96 pairs: straight(256->96)={n_one_resp} "
                  f"diagonal(512->96)={n_two_resp}")
        else:
            print("  Stim.bdf      MISSING")

        # ---- Standing (Control) ----
        if stand_p.exists():
            ev, sf, dur = _events(stand_p)
            codes = Counter(ev[:, 2].tolist())
            stim_ev = ev[(ev[:, 2] == CODE_STIM) & (ev[:, 1] == 0)]
            isi = np.diff(stim_ev[:, 0]) / sf if len(stim_ev) > 1 else np.array([])
            n_win = int(dur // STAND_WIN_S)
            print(f"  Standing.bdf  sfreq={sf:.0f}  dur={dur:.0f}s")
            print(f"    codes: 256={codes.get(CODE_ONE,0)} 512={codes.get(CODE_TWO,0)} "
                  f"96={codes.get(RESP,0)} 1024={codes.get(CODE_STIM,0)}")
            if len(isi):
                print(f"    1024 ISI: median={np.median(isi):.3f}s "
                      f"[{np.percentile(isi,5):.3f},{np.percentile(isi,95):.3f}]")
            print(f"    candidate non-overlapping {STAND_WIN_S:.0f}s windows ~= {n_win}")
        else:
            print("  Standing.bdf  MISSING")


def _pair_resp(events, condition_code, response_code=RESP):
    """Count response (96, prev==0) events that follow a condition trigger."""
    out, active = 0, False
    for row in events:
        if row[2] == condition_code:
            active = True
        elif row[2] == response_code and row[1] == 0 and active:
            out += 1
            active = False
    return out


if __name__ == "__main__":
    main(sys.argv[1:] or SMOKE)
