#!/usr/bin/env python
"""Phase 3 decision table: pooling vs personalisation variants side by side.

    python ~/codabench/analysis/sealed_decide_p3.py --tag p3 --tag p3b [--id router-id]

Per (study, classes, family, align): cell-averaged balanced accuracy (seed
mean) of pooled / calib / blend / blend_calib / persubject with the chosen id
source (router-id by default; oracle-id for comparison). The weekend rule:
the recipe takes the variant with the best cell score that wins on >= 2 of 3
datasets with router ids; ties (< 2 points) go to the simpler variant.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import xsess_lib as L  # noqa: E402

VARIANTS = ["pooled", "calib", "blend", "blend_calib", "persubject"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--id", default="router-id", choices=["router-id", "oracle-id"])
    args = ap.parse_args()
    rows = []
    for t in args.tag:
        rows += L.read_results(Path.home() / f"codabench/logs/sealed_{t}/results.jsonl")
    rows = list({r["key"]: r for r in rows}.values())
    cell = defaultdict(list)
    for r in rows:
        fam, _, var = r["spec"].partition("/")
        if var not in VARIANTS or r["mode"] not in ("none", args.id):
            continue
        cls = "all" if r["classes"] is None else ",".join(map(str, r["classes"]))
        cell[(r["study"], cls, fam, r["align"], var)].append(r["cell"])
    keys = sorted({k[:4] for k in cell})
    print(f"Personal variants with {args.id} (pooled needs no id); seed-mean cell score, "
          "best in bold.\n")
    print("| study | family | align | " + " | ".join(VARIANTS) + " |")
    print("|---|---|---|" + "---|" * len(VARIANTS))
    for k in keys:
        vals = {v: np.mean(cell[k + (v,)]) for v in VARIANTS if (k + (v,)) in cell}
        n = {v: len(cell[k + (v,)]) for v in vals}
        best = max(vals.values())
        cells = []
        for v in VARIANTS:
            if v not in vals:
                cells.append("")
                continue
            s = f"{vals[v]:.3f}" + (f" (n{n[v]})" if n[v] > 1 else "")
            cells.append(f"**{s}**" if vals[v] == best else s)
        study = k[0] + ("" if k[1] == "all" else f" [{k[1]}]")
        print(f"| {study} | {k[2]} | {k[3]} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
