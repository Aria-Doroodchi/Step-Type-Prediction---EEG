#!/usr/bin/env python
"""Phase 2 decision table: alignment conditions side by side.

    python ~/codabench/analysis/sealed_decide.py --tag p1 --tag p2

Per (study, model, mode): cell-averaged balanced accuracy (mean over seeds)
for every alignment condition; gain_b = best oracle - none; recovery of the
router (d) and train-only (c) conditions = (cond - none) / gain_b; router
accuracy. Decision rules (weekend prompt, Phase 2): alignment is in the recipe
if gain_b >= 0.02 on >= 2 datasets; adopt the router if its accuracy >= 90 %
and it recovers >= 70 % of gain_b, else train-only.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import xsess_lib as L  # noqa: E402

COLS = ["none", "oracle:euclid", "oracle:riemann", "trainonly:euclid", "trainonly:riemann",
        "router-psd:euclid", "router-psd:riemann", "routerb-psd:riemann", "batch:riemann"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", required=True)
    args = ap.parse_args()
    rows = []
    for t in args.tag:
        rows += L.read_results(Path.home() / f"codabench/logs/sealed_{t}/results.jsonl")
    rows = list({r["key"]: r for r in rows}.values())   # same config in two phases: once
    cell = defaultdict(list)
    racc = defaultdict(list)
    for r in rows:
        cls = "all" if r["classes"] is None else ",".join(map(str, r["classes"]))
        k = (r["study"], cls, r["spec"], r["mode"])
        cell[k + (r["align"],)].append(r["cell"])
        if "router_acc_window" in r:
            racc[(r["study"], cls, r["align"])].append(r["router_acc_window"])
    keys = sorted({k[:4] for k in cell})
    short = {c: c.replace("oracle", "orc").replace("trainonly", "tro").replace("router-psd", "rt")
             .replace("routerb-psd", "rtb").replace(":euclid", ":E").replace(":riemann", ":R")
             for c in COLS}
    print("| study | model | mode | " + " | ".join(short[c] for c in COLS)
          + " | gain_b | rec(rt best) | rec(tro best) |")
    print("|" + "---|" * (len(COLS) + 6))
    for k in keys:
        if (k + ("none",)) not in cell:
            continue
        m = {c: np.mean(cell[k + (c,)]) for c in COLS if (k + (c,)) in cell}
        n = {c: len(cell[k + (c,)]) for c in m}
        base = m["none"]
        orc = max([m[c] for c in m if c.startswith("oracle")], default=np.nan)
        gain = orc - base
        rt = max([m[c] for c in m if c.startswith("router-psd")], default=np.nan)
        tro = max([m[c] for c in m if c.startswith("trainonly")], default=np.nan)
        rec = lambda v: f"{(v - base) / gain:.0%}" if gain > 0.005 and not np.isnan(v) else "n/a"
        cells = [f"{m[c]:.3f}" + (f" (n{n[c]})" if n[c] > 1 else "") if c in m else "" for c in COLS]
        print(f"| {k[0]}{'' if k[1] == 'all' else ' [' + k[1] + ']'} | {k[2]} | {k[3]} | "
              + " | ".join(cells) + f" | {gain:+.3f} | {rec(rt)} | {rec(tro)} |")
    print("\nRouter accuracy (per window, before fallback):")
    for (study, cls, align), v in sorted(racc.items()):
        print(f"- {study} {cls} {align}: {np.mean(v):.3f}")


if __name__ == "__main__":
    main()
