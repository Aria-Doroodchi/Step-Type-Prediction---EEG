#!/usr/bin/env python
"""Markdown summary of a sealed_<tag>/results.jsonl.

    python ~/codabench/analysis/sealed_summarize.py --tag p1 [--tag p2 ...]

One table per study (and class subset): mean +/- SD over seeds of the
cell-averaged and pooled balanced accuracy, per (model, mode, align), plus
router accuracy / fallback where present and the mean best epoch.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import xsess_lib as L  # noqa: E402


def fmt(v):
    v = np.asarray(v, float)
    return f"{v.mean():.3f}" if len(v) == 1 else f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", required=True)
    args = ap.parse_args()
    rows = []
    for t in args.tag:
        rows += L.read_results(Path.home() / f"codabench/logs/sealed_{t}/results.jsonl")
    groups = defaultdict(list)
    for r in rows:
        cls = "all" if r["classes"] is None else ",".join(map(str, r["classes"]))
        groups[(r["study"], cls)].append(r)
    print(f"# Results: {', '.join(args.tag)}\n")
    print("Cell = balanced accuracy averaged over (subject, test session) cells; "
          "pooled = over all test windows. Mean ± SD over seeds (n). "
          "Test = each subject's last session.\n")
    for (study, cls), rs in groups.items():
        shape = rs[0]["shape"]
        print(f"## {study} (classes {cls}), X={tuple(shape)}, "
              f"train={rs[0]['n_train']}, test={rs[0]['n_test']}\n")
        print("| model | mode | align | n | cell | pooled | router acc | fallback | epochs | s/run |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        by = defaultdict(list)
        for r in rs:
            by[(r["spec"], r["mode"], r["align"])].append(r)
        for (spec, mode, align), g in sorted(by.items(), key=lambda kv: -np.mean([r["cell"] for r in kv[1]])):
            ra = [r["router_acc"] for r in g if "router_acc" in r]
            fb = [r["router_fallback"] for r in g if "router_fallback" in r]
            ep = [np.mean(r["epochs"]) for r in g if r.get("epochs")]
            print(f"| {spec} | {mode} | {align} | {len(g)} | {fmt([r['cell'] for r in g])} | "
                  f"{fmt([r['pooled'] for r in g])} | {fmt(ra) if ra else ''} | "
                  f"{fmt(fb) if fb else ''} | {np.mean(ep):.0f} | "
                  f"{np.mean([r['seconds'] for r in g]):.0f} |" if ep else
                  f"| {spec} | {mode} | {align} | {len(g)} | {fmt([r['cell'] for r in g])} | "
                  f"{fmt([r['pooled'] for r in g])} | {fmt(ra) if ra else ''} | "
                  f"{fmt(fb) if fb else ''} |  | "
                  f"{np.mean([r['seconds'] for r in g]):.0f} |")
        print()


if __name__ == "__main__":
    main()
