#!/usr/bin/env python
"""Compare the harness result rows of two runs key by key: cell / pooled
scores, chosen blend weights and the saved probabilities (sprint 2026-10-01
gates: a numerically equivalent change must reproduce committed rows).

    python ~/codabench/analysis/compare_results.py <old dirs> <new dirs> <study> [--tol 1e-8]

<old dirs> / <new dirs>: comma-separated log dirs holding
results_<study>.jsonl and probs/ (e.g. the two lanes of a release_ablations
run). PASS iff every new row has an old row with the same key, equal cell
and pooled scores (|d| <= 1e-12), equal blend weights, and max |dP| <= tol.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import xsess_lib as L  # noqa: E402


def rows(dirs, study):
    out = {}
    for d in dirs.split(","):
        d = Path(d).expanduser()
        for r in L.read_results(d / f"results_{study}.jsonl"):
            if r.get("study") == study:     # read_results also reads the siblings
                out[r["key"]] = (r, d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("study")
    ap.add_argument("--tol", type=float, default=1e-8)
    a = ap.parse_args()
    O, N = rows(a.old, a.study), rows(a.new, a.study)
    ok = bool(N)
    print(f"| key | old cell | new cell | d cell | d pooled | w old/new | max abs dP |\n"
          f"|---|---|---|---|---|---|---|")
    worst = 0.0
    for k, (rn, dn) in N.items():
        if k not in O:
            print(f"| `{k}` | missing in old | {rn['cell']:.6f} | | | | |")
            ok = False
            continue
        ro, do = O[k]
        dc, dpo = abs(rn["cell"] - ro["cell"]), abs(rn["pooled"] - ro["pooled"])
        w = [(ro.get(f), rn.get(f)) for f in ("blend_w", "blend_calib_w") if f in rn or f in ro]
        w_ok = all(x == y for x, y in w)
        dP = float("nan")
        if rn.get("probs") and ro.get("probs"):
            Po = np.load(do / "probs" / ro["probs"])["P"]
            Pn = np.load(dn / "probs" / rn["probs"])["P"]
            dP = float(np.abs(Po - Pn).max()) if Po.shape == Pn.shape else float("inf")
            worst = max(worst, dP)
        good = dc <= 1e-12 and dpo <= 1e-12 and w_ok and not dP > a.tol
        ok &= good
        print(f"| `{k}` | {ro['cell']:.6f} | {rn['cell']:.6f} | {dc:.1e} | {dpo:.1e} | "
              f"{'; '.join(f'{x}/{y}' for x, y in w) or '-'} | {dP:.1e}"
              f"{'' if good else ' **DIFF**'} |")
    print(f"\n{len(N)} new rows, {sum(k in O for k in N)} matched; worst |dP| {worst:.2e}")
    print(f"COMPARE {a.study}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
