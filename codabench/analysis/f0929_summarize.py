#!/usr/bin/env python
"""Sprint 2026-09-29 tables: the feature-block screen (Phase 2) and the
blend_calib confirmation (Phase 3), with the brief's pre-registered rules.

    python ~/codabench/analysis/f0929_summarize.py [--phase screen|confirm|both]

Screen (brief § 5 Phase 2): the screen score of a spec on a study is the mean
of its pooled and persubject cells under the clean router alignment
(router-psd:riemann); per subject, the mean of the two per-subject scores.
Every comparison is paired against the recipe union riemann:xd=1,fb=1 on the
same subjects: mean difference, 95 % percentile bootstrap CI over subjects
(10,000 resamples, seed 0, as sealed_bootstrap.py) and subjects improved.
Pass = Scherer 3-class delta >= +1.0 point and the mean delta over
{Tangermann, Scherer 5-class, Zhou} >= -0.5 point.

Confirm (Phase 3): blend_calib (router ids, clean router alignment) of each
advancing spec vs the baseline's blend_calib (sealed_personal.py rows, tag
f0929p). ADOPT = Scherer 3-class delta >= +2.0 with CI > 0, mean delta over the
other three >= 0 and none < -2.0; PROMISING = Scherer 3-class delta >= +1.0 and
>= 0 on >= 2 of the other three; else NO GAIN.
"""

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import xsess_lib as L  # noqa: E402

LOGS = Path.home() / "codabench/logs"
BASE = "riemann:xd=1,fb=1"
AL = "router-psd:riemann"
S3 = ("scherer2015", (0, 1, 3))
OTHERS = [("tangermann2012", None), ("scherer2015", None), ("zhou2016", None)]
NAMES = {S3: "Scherer 3-cl.", ("tangermann2012", None): "Tangermann",
         ("scherer2015", None): "Scherer 5-cl.", ("zhou2016", None): "Zhou"}
FAMILY = {"tseg": "T", "acm": "T", "fb8": "T", "fbd": "T", "bpt": "T", "sl=1": "T",
          "tcut": "T (EDA)", "fbfrom": "control",
          "fblv": "S", "fbrlv": "S", "reg": "S", "csp": "S", "icoh": "S",
          "ref=": "control"}


def load_rows(tags):
    idx = {}
    for t in tags:
        for r in L.read_results(LOGS / f"sealed_{t}" / "results.jsonl"):
            cls = tuple(r["classes"]) if r.get("classes") else None
            idx[(r["study"], cls, r["spec"], r["mode"], r["align"])] = r   # last wins
    return idx


def label(spec):
    s = spec[len(BASE):].lstrip(",") if spec.startswith(BASE) else spec
    swap = "blocks=xdawn+broad+logvar,x="
    if s.startswith(swap):
        s = s[len(swap):] + " (replaces FB)"
    return s.replace("x=", "") or "baseline"


def family(spec):
    lab = label(spec)
    for k, v in FAMILY.items():
        if lab.startswith(k) or k in lab:
            return v
    return "-"


def ge(x, t):
    """x >= t, float-safe (the rules test the unrounded value to 1e-6)."""
    return round(float(x), 6) >= t


def boot(d, rng):
    boots = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), int((d > 0).sum()), len(d)


def per_subject(idx, study, cls, spec, keys):
    """{subject: mean over ``keys`` (mode, align) of its per-subject score},
    or None when a key is missing."""
    rows = [idx.get((study, cls, spec, m, a)) for m, a in keys]
    if any(r is None for r in rows):
        return None
    subs = set.intersection(*[set(r["per_subject"]) for r in rows])
    return {s: float(np.mean([r["per_subject"][s] for r in rows])) for s in subs}


def delta(idx, study, cls, spec, base, keys):
    """Paired A - B over subjects; the bootstrap is seeded per comparison (crc32
    of the comparison), so a CI does not move when other rows are added."""
    a = per_subject(idx, study, cls, spec, keys)
    b = per_subject(idx, study, cls, base, keys)
    if a is None or b is None:
        return None
    subs = sorted(set(a) & set(b), key=int)
    rng = np.random.default_rng(zlib.crc32(f"{study}|{cls}|{spec}|{base}|{keys}".encode()))
    return boot(np.array([a[s] - b[s] for s in subs]), rng)


def fmt(res, pts=True):
    if res is None:
        return "missing"
    m, lo, hi, k, n = res
    f = 100 if pts else 1
    return f"{m * f:+.2f} ({lo * f:+.2f}, {hi * f:+.2f}) {k}/{n}"


def cell(idx, study, cls, spec, mode, align):
    r = idx.get((study, cls, spec, mode, align))
    return "—" if r is None else f"{r['cell']:.3f}"


def screen(idx, reverse=False):
    keys_r = [("pooled", AL), ("persubject", AL)]
    keys_n = [("pooled", "none"), ("persubject", "none")]
    specs = sorted({k[2] for k in idx if k[:2] == S3 and k[2].startswith("riemann")
                    and "/" not in k[2]},            # not sealed_personal's family/variant rows
                   key=lambda s: (family(s), label(s)))
    print("## " + ("Reverse-time split (--split first): Scherer 3-class (WORD / SUB / HAND), "
                   "train on the later session, test on the first\n" if reverse else
                   "Phase 2 screen — Scherer 3-class (WORD / SUB / HAND), last session\n"))
    print("Cell-averaged balanced accuracy. Δ in points vs the recipe union "
          f"(`{BASE}`), paired over the 9 subjects: mean (95 % bootstrap CI) "
          "subjects improved. Screen = mean of pooled and persubject under the "
          "clean router alignment.\n")
    print("| block | fam. | pooled none | persubj. none | pooled router | persubj. router "
          "| Δ screen (router) | Δ none (info) |")
    print("|---|---|---|---|---|---|---|---|")
    d3 = {}
    for s in specs:
        d3[s] = delta(idx, *S3, s, BASE, keys_r)
        dn = delta(idx, *S3, s, BASE, keys_n)
        print(f"| {label(s)} | {family(s)} | {cell(idx, *S3, s, 'pooled', 'none')} | "
              f"{cell(idx, *S3, s, 'persubject', 'none')} | {cell(idx, *S3, s, 'pooled', AL)} | "
              f"{cell(idx, *S3, s, 'persubject', AL)} | {fmt(d3[s]) if s != BASE else '—'} | "
              f"{fmt(dn) if s != BASE else '—'} |")
    if reverse:
        return []
    print("\n## Phase 2 screen — replication on the other proxies (router, screen score)\n")
    print("| block | " + " | ".join(NAMES[o] for o in OTHERS) + " | mean Δ others | "
          "Scherer 3-cl. Δ | pass |")
    print("|---|" + "---|" * len(OTHERS) + "---|---|---|")
    passing = []
    for s in specs:
        if s == BASE:
            continue
        ds = [delta(idx, st, c, s, BASE, keys_r) for st, c in OTHERS]
        have = [x[0] for x in ds if x is not None]
        mo = float(np.mean(have)) if len(have) == len(OTHERS) else None
        m3 = d3[s][0] if d3[s] else None
        ok = m3 is not None and mo is not None and ge(m3, 0.010) and ge(mo, -0.005)
        if ok and family(s) != "control":
            passing.append((m3, s))
        print(f"| {label(s)} | " + " | ".join(fmt(x) for x in ds) +
              f" | {'missing' if mo is None else f'{mo * 100:+.2f}'} | "
              f"{'missing' if m3 is None else f'{m3 * 100:+.2f}'} | "
              f"{'**PASS**' if ok else ('incomplete' if mo is None or m3 is None else 'no')} |")
    passing.sort(reverse=True)
    print("\nPassing (ranked by Scherer 3-class Δ; the brief advances at most 3, "
          "plus the best-T ∪ best-S union if both families pass; a tie goes to fewer "
          "features, applied by hand: rows carry no feature count): "
          + (", ".join(f"`{label(s)}` ({m * 100:+.2f})" for m, s in passing) or "none"))
    return passing


def confirm(idx):
    fams = sorted({k[2].rsplit("/", 1)[0] for k in idx
                   if k[2].endswith("/blend_calib") and k[3] == "router-id" and k[4] == AL})
    if BASE not in fams:
        print("\n## Phase 3 confirm: no baseline blend_calib rows yet\n")
        return
    print("\n## Phase 3 confirm — blend_calib (router ids), clean router alignment\n")
    print("| block | " + " | ".join(NAMES[o] for o in [S3] + OTHERS) + " | verdict |")
    print("|---|" + "---|" * (1 + len(OTHERS)) + "---|")
    keys = [("router-id", AL)]
    for f in fams:
        cells = []
        for st, c in [S3] + OTHERS:
            r = idx.get((st, c, f + "/blend_calib", "router-id", AL))
            cells.append("—" if r is None else f"{r['cell']:.3f}")
        if f == BASE:
            print(f"| baseline | " + " | ".join(cells) + " | — |")
            continue
        ds = [delta(idx, st, c, f + "/blend_calib", BASE + "/blend_calib", keys)
              for st, c in [S3] + OTHERS]
        d3, do = ds[0], ds[1:]
        if d3 is None or any(x is None for x in do):
            verdict = "incomplete"
        elif (ge(d3[0], 0.020) and round(d3[1], 6) > 0 and ge(np.mean([x[0] for x in do]), 0)
              and ge(min(x[0] for x in do), -0.020)):
            verdict = "**ADOPT**"
        elif ge(d3[0], 0.010) and sum(ge(x[0], 0) for x in do) >= 2:
            verdict = "PROMISING"
        else:
            verdict = "NO GAIN"
        print(f"| {label(f)} | " + " | ".join(f"{c_} Δ {fmt(x)}" for c_, x in zip(cells, ds))
              + f" | {verdict} |")
    for f in fams:     # rule-dependent, information only
        r = idx.get((*S3, f + "/blend_calib", "router-id", "online-64:riemann"))
        if r is not None:
            print(f"\n(online-64, rule-dependent, information only) {label(f)} Scherer 3-cl. "
                  f"blend_calib = {r['cell']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="both", choices=["screen", "confirm", "both"])
    ap.add_argument("--reverse", action="store_true",
                    help="the rows are the reverse-time split (--split first): Scherer 3-cl. only")
    ap.add_argument("--tags", nargs="+", default=["f0929", "f0929s5", "f0929p", "f0929p5"])
    args = ap.parse_args()
    idx = load_rows(args.tags)
    if args.phase in ("screen", "both"):
        screen(idx, reverse=args.reverse)
    if args.phase in ("confirm", "both"):
        confirm(idx)


if __name__ == "__main__":
    main()
