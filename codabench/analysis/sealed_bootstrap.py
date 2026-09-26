#!/usr/bin/env python
"""Paired subject-level bootstrap for the recipe's key claims.

    python ~/codabench/analysis/sealed_bootstrap.py > logs/sealed_bootstrap.md

Each proxy has one test session per subject, so the cell score is the mean of
the per-subject scores stored in every result row. For a comparison A - B on
the same study: per-subject differences, their mean, a 95 % percentile
bootstrap CI over subjects (10,000 resamples) and how many subjects improve.
With 4-9 subjects the CIs are wide; that is the point of reporting them.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import xsess_lib as L  # noqa: E402

TAGS = ["p1", "p2", "p3", "p3b", "p4"]
RIEM = "riemann:xd=1,fb=1"
# (label, study, classes, (spec, mode, align) A, (spec, mode, align) B)
CLAIMS = []
for st in ("tangermann2012", "scherer2015", "zhou2016"):
    CLAIMS += [
        ("per-subject vs pooled (no alignment)", st, None,
         (RIEM, "persubject", "none"), (RIEM, "pooled", "none")),
        ("blend_calib (router ids) vs pooled, clean router alignment", st, None,
         (RIEM + "/blend_calib", "router-id", "router-psd:riemann"),
         (RIEM + "/pooled", "none", "router-psd:riemann")),
        ("blend_calib (router ids) vs per-subject no alignment (Phase 1 best)", st, None,
         (RIEM + "/blend_calib", "router-id", "router-psd:riemann"),
         (RIEM, "persubject", "none")),
        ("online-64 vs clean router, blend_calib (router ids)", st, None,
         (RIEM + "/blend_calib", "router-id", "online-64:riemann"),
         (RIEM + "/blend_calib", "router-id", "router-psd:riemann")),
    ]
CLAIMS += [
    ("xDAWN: all blocks vs no xDAWN, per-subject none", "scherer2015", [0, 1, 3],
     (RIEM, "persubject", "none"), ("riemann:xd=0,fb=1", "persubject", "none")),
    ("blend_calib (router ids) vs pooled, clean", "scherer2015", [0, 1, 3],
     (RIEM + "/blend_calib", "router-id", "router-psd:riemann"),
     (RIEM + "/pooled", "none", "router-psd:riemann")),
    ("online-64 vs clean router, blend_calib (router ids)", "scherer2015", [0, 1, 3],
     (RIEM + "/blend_calib", "router-id", "online-64:riemann"),
     (RIEM + "/blend_calib", "router-id", "router-psd:riemann")),
]


def main():
    rows = []
    for t in TAGS:
        rows += L.read_results(Path.home() / f"codabench/logs/sealed_{t}/results.jsonl")
    idx = {}
    for r in rows:
        idx[(r["study"], tuple(r["classes"]) if r["classes"] else None,
             r["spec"], r["mode"], r["align"])] = r
    rng = np.random.default_rng(0)
    print("| study | claim | A | B | A − B (95 % CI over subjects) | subjects A > B |")
    print("|---|---|---|---|---|---|")
    for label, st, cls, a, b in CLAIMS:
        ka, kb = (st, tuple(cls) if cls else None) + a, (st, tuple(cls) if cls else None) + b
        if ka not in idx or kb not in idx:
            print(f"| {st} | {label} | missing | | | |")
            continue
        pa, pb = idx[ka]["per_subject"], idx[kb]["per_subject"]
        subs = sorted(set(pa) & set(pb), key=int)
        d = np.array([pa[s] - pb[s] for s in subs])
        boots = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        name = st + ("" if not cls else f" [{','.join(map(str, cls))}]")
        print(f"| {name} | {label} | {idx[ka]['cell']:.3f} | {idx[kb]['cell']:.3f} | "
              f"{d.mean():+.3f} ({lo:+.3f}, {hi:+.3f}) | {(d > 0).sum()}/{len(d)} |")


if __name__ == "__main__":
    main()
