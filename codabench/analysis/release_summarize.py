#!/usr/bin/env python
"""RESULTS table and release-day DECISIONS for scripts/release_ablations.sh.

    python ~/codabench/analysis/release_summarize.py --tag rel --study graz2026 \
        [--split replica:3 --test_subjects 0,1,... --spec riemann:xd=1,fb=1 \
         --align router-psd:riemann --wcv last --cap 0.5]

Reads the rows of sealed_run.py / sealed_personal.py in logs/sealed_<tag>/ and
its lane folders logs/sealed_<tag>_<A-Z>/, keeps the rows of --study (all
classes), --split, --test_subjects and --cap, one row per config key (the
last), and prints:

- RESULTS: the recipe variants (blend_calib with router ids: the deployed
  recipe needs no subject id at test time) and every row, with the cell metric
  (balanced accuracy over subject x session x context cells) and, when the
  cache has a context column, the cell metric within each context, recomputed
  from the saved test probabilities;
- DECISIONS: the release-day rules pre-registered on 2026-09-28 (sprint brief
  prompts/2026-09-28_release_readiness.md; SEALED_RECIPE.md section 3). Each
  rule compares one ablation row with the recipe row "base" (both blend_calib,
  router ids). Points = cell metric x 100 on the replica test sessions. The
  paired subject bootstrap is analysis/sealed_bootstrap.py's: per-subject
  differences (each subject's cells averaged), 95 % percentile CI of their
  mean over 10,000 resamples of subjects (seed 0 per comparison).
    channels  a non-EEG set replaces eeg only if >= 2 points better AND the CI
              excludes 0 (the best such set if several pass);
    context   router-psdctx replaces router-psd only if >= 2 points better AND
              the CI excludes 0;
    xDAWN     dropped if xd=0 is >= 1 point better (a tie keeps it);
    wcv       loso unless it is >= 2 points worse than last (it uses more data
              and is what SEALED_RECIPE describes);
    pool      all unless test-only is >= 2 points better AND the CI excludes 0;
    online-64 rule-dependent: reported in its own line, never adopted here.
    blocks    (sprint 2026-09-29) step xb: a recipe with x=<blocks> drops them
              only if the row without them is >= 1 point better; step xa
              (--xb <blocks>): the blocks are added only if the row with them
              is >= 1 point better AND no context is > 1 point worse.
A missing row (a step not run, or failed) prints a loud "MISSING: <step> ..."
line and keeps the recipe's setting (for wcv that is loso: nothing measured
says it is worse); a summary MISSING line after the result lists every such
step (2026-09-29). Deltas print with two decimals (the rules test the
unrounded value). The last lines give the resulting settings
as train_sealed.sh environment variables, with BLEND_W=auto for wcv loso and
BLEND_W=harness for wcv last (RELEASE_DAY rule 4).
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import xsess_lib as L  # noqa: E402

LOGS = Path.home() / "codabench/logs"
LABEL = "release-day rules (pre-registered 2026-09-28)"
N_BOOT = 10000
CHAN_ALTS = ("eeg+eog", "eeg+emg", "all")


def load_rows(tag, study, split, cap, test_subjects=None):
    """Deduplicated rows of the study (last row per key) from the tag's folder
    and its lane folders, for this split / router cap / --test_subjects (as
    sealed_run stores it: sorted, comma-joined; None = the split's default);
    each row gets "_dir" (for its probabilities)."""
    dirs = [LOGS / f"sealed_{tag}"] + sorted(p for p in LOGS.glob(f"sealed_{tag}_[A-Z]")
                                             if p.is_dir())
    rows = {}
    for d in dirs:
        for r in L.read_results(d / "results.jsonl"):
            if r.get("study") != study or r.get("classes") is not None:
                continue
            r["_dir"] = d
            rows[r["key"]] = r
    rows = list(rows.values())
    splits = sorted({r.get("split", "last") for r in rows})
    if split is None:
        if len(splits) > 1:
            sys.exit(f"rows with several splits {splits}: pass --split")
        split = splits[0] if splits else "last"
    rows = [r for r in rows if r.get("split", "last") == split and r.get("router_cap") == cap
            and r.get("test_subjects") == test_subjects]
    return rows, dirs, split


def ctx_cells(rows, study):
    """Per-context cell metric of each row (from its probs .npz), or {} when
    the cache has no context column."""
    d = L.load_study(study, mmap=True)
    if not d["has_ctx"]:
        return [], {}
    y, subj, sess, ctx = d["y"], d["subj"], d["sess"], d["ctx"]
    out = {}
    for r in rows:
        p = r["_dir"] / "probs" / r.get("probs", "")
        if not p.is_file():
            continue
        z = np.load(p)
        te, yhat = z["te_idx"], z["P"].argmax(1)
        out[r["key"]] = [L.score(y[te][m], yhat[m], subj[te][m], sess[te][m], ctx[te][m])["cell"]
                         if m.any() else float("nan")
                         for m in (ctx[te] == c for c in range(len(d["ctx_names"])))]
    return d["ctx_names"], out


def paired(a, b):
    """(mean per-subject difference a - b, CI low, CI high, subjects a > b, n)."""
    pa, pb = a["per_subject"], b["per_subject"]
    subs = sorted(set(pa) & set(pb), key=int)
    dd = np.array([pa[s] - pb[s] for s in subs])
    rng = np.random.default_rng(0)
    boots = dd[rng.integers(0, len(dd), (N_BOOT, len(dd)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(dd.mean()), float(lo), float(hi), int((dd > 0).sum()), len(dd)


def fmt_cmp(a, b):
    """'+x.xx pts (CI +x.xx, +x.xx; k/n subjects up)' for a - b (two decimals:
    the rules test the unrounded value, so +1.96 must not print as +2.0)."""
    dm, lo, hi, up, n = paired(a, b)
    return (f"{100 * (a['cell'] - b['cell']):+.2f} pts (95 % CI {100 * lo:+.2f}, "
            f"{100 * hi:+.2f}; {up}/{n} subjects up)"), a["cell"] - b["cell"], lo, hi


def missing(step):
    """The loud DECISIONS line for an ablation step with no row."""
    return f"MISSING: {step} has no row (not run or failed): rule not applied"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--study", required=True)
    ap.add_argument("--split", default=None, help="default: the rows' only split")
    ap.add_argument("--spec", default="riemann:xd=1,fb=1")
    ap.add_argument("--align", default="router-psd:riemann")
    ap.add_argument("--wcv", default="last", help="the recipe row's blend-weight CV")
    ap.add_argument("--cap", default="0.5", help="router cap of the rows; none = uncapped")
    ap.add_argument("--test_subjects", default="",
                    help="comma list the rows were run with (default: the split's own)")
    ap.add_argument("--xb", default="",
                    help="extra blocks step xb added (e.g. bpt4) when --spec has no x=")
    args = ap.parse_args()
    cap = None if args.cap == "none" else float(args.cap)
    ts = ",".join(map(str, sorted({int(s) for s in args.test_subjects.split(",") if s}))) or None
    rows, dirs, split = load_rows(args.tag, args.study, args.split, cap, ts)
    if not rows:
        sys.exit(f"no rows for study={args.study} split={split} router_cap={cap} "
                 f"test_subjects={ts} in {[str(d) for d in dirs]}")
    kind = args.align.split(":")[1]
    nx = args.spec.replace("xd=1", "xd=0") if "xd=1" in args.spec else None
    # steps xb / xa (release_ablations.sh, same string rules as the shell):
    # xb = the recipe without its x=<blocks>; xa = with the --xb blocks added
    has_x = ",x=" in args.spec
    xb_alt = re.sub(r",x=[^,]*", "", args.spec) if has_x else None
    xa_alt = None
    if args.xb:
        xa_alt = (re.sub(r"(,x=[^,]*)", lambda m: m.group(1) + "+" + args.xb, args.spec)
                  if has_x else f"{args.spec},x={args.xb}")
    wcv_alt = "loso" if args.wcv == "last" else "last"

    def find(spec=args.spec, align=args.align, chans="eeg", pool="all", wcv=args.wcv,
             variant="blend_calib", mode="router-id"):
        # (a sealed_personal row from before the wcv field is wcv=last)
        hits = [r for r in rows if r["spec"] == f"{spec}/{variant}" and r["mode"] == mode
                and r["align"] == align and r.get("chans", "eeg") == chans
                and r.get("pool", "all") == pool and r.get("wcv", "last") == wcv]
        return hits[-1] if hits else None

    # the recipe variants, one per ablation step (blend_calib, router ids)
    variants = [("base (recipe)", find())]
    variants += [(f"chans {c}", find(chans=c)) for c in CHAN_ALTS]
    variants += [(f"align router-psdctx:{kind}", find(align=f"router-psdctx:{kind}"))]
    variants += [("no xDAWN (" + (nx or "n/a") + ")", find(spec=nx) if nx else None)]
    if xb_alt is not None:
        variants += [(f"without blocks ({xb_alt})", find(spec=xb_alt))]
    if xa_alt is not None:
        variants += [(f"with blocks ({xa_alt})", find(spec=xa_alt))]
    variants += [(f"wcv {wcv_alt}", find(wcv=wcv_alt)), ("pool test", find(pool="test"))]
    online = find(align=f"online-64:{kind}")
    base = variants[0][1]
    ctx_names, cc = ctx_cells(rows, args.study)

    r0 = base or rows[0]
    n_cells = sorted({r["n_cells"] for r in rows})
    print(f"# Release ablations: {args.tag} / {args.study}\n")
    print(f"Split {split}" + (f" (test subjects {ts})" if ts else "")
          + f", router cap {cap}, X={tuple(r0['shape'])} (after the channel pick), "
          f"train={r0['n_train']} test={r0['n_test']} windows, test cells per row "
          f"{n_cells}{' (OK: one value)' if len(n_cells) == 1 else ' (DIFFERENT: check)'}, "
          f"contexts {ctx_names or 'none'}. Cell = balanced accuracy averaged over "
          f"subject x session x context cells of the test sessions; pooled = over all "
          f"test windows. Recipe: {args.spec}, {args.align}, chans eeg, pool all, "
          f"wcv {args.wcv}. Rows read from {', '.join(d.name for d in dirs)}.\n")

    ctx_hdr = "".join(f" {c} |" for c in ctx_names)
    ctx_sep = "---|" * len(ctx_names)

    def ctx_vals(r):
        v = cc.get(r["key"])
        return "".join(f" {x:.3f} |" for x in v) if v else " |" * len(ctx_names)

    print("## Recipe variants (blend_calib, router ids)\n")
    print(f"| variant | cell | vs recipe |{ctx_hdr} pooled | w | router acc | s |")
    print(f"|---|---|---|{ctx_sep}---|---|---|---|")
    for label, r in variants + [("online-64 (RULE-DEPENDENT)", online)]:
        if r is None:
            print(f"| {label} | not run |  |" + " |" * len(ctx_names) + "  |  |  |  |")
            continue
        vs = "" if r is base or base is None else f"{100 * (r['cell'] - base['cell']):+.2f}"
        print(f"| {label} | {r['cell']:.4f} | {vs} |{ctx_vals(r)} {r['pooled']:.4f} | "
              f"{r.get('blend_calib_w', '')} | {r.get('router_psd_acc', float('nan')):.3f} | "
              f"{r['seconds']:.0f} |")

    print("\n## All rows\n")
    print(f"| spec | mode | align | chans | pool | wcv | cell |{ctx_hdr} pooled | n_cells | "
          f"router acc | fallback | s |")
    print(f"|---|---|---|---|---|---|---|{ctx_sep}---|---|---|---|---|")
    order = sorted(rows, key=lambda r: (r["spec"], r["mode"], r["align"], r.get("chans", "eeg"),
                                        r.get("pool", "all"), str(r.get("wcv"))))
    for r in order:
        ra = r.get("router_acc", r.get("router_psd_acc"))
        fb = r.get("router_fallback", r.get("router_psd_fallback"))
        print(f"| {r['spec']} | {r['mode']} | {r['align']} | {r.get('chans', 'eeg')} | "
              f"{r.get('pool', 'all')} | {r.get('wcv') or ''} | {r['cell']:.4f} |{ctx_vals(r)} "
              f"{r['pooled']:.4f} | {r['n_cells']} | {'' if ra is None else f'{ra:.3f}'} | "
              f"{'' if fb is None else f'{fb:.3f}'} | {r['seconds']:.0f} |")

    # ---- DECISIONS ---------------------------------------------------------
    print(f"\n## DECISIONS: {LABEL}\n")
    print("```")
    if base is None:
        print("recipe row (base) missing: no decision possible")
        print("```")
        return
    print(f"recipe: {args.spec} / {args.align} / chans eeg / pool all / wcv {args.wcv}: "
          f"cell {base['cell']:.4f} ({base['n_cells']} cells)")
    ok = lambda d, t: round(d, 6) >= t          # noqa: E731  (>= t points, float-safe)
    lost = []   # release_ablations.sh steps whose rule could not be applied

    def miss(step):
        lost.append(step)
        print(missing(step))

    # channels
    chans, passing = "eeg", []
    for c in CHAN_ALTS:
        r = find(chans=c)
        if r is None:
            miss(f"ch_{c.replace('+', '_')}")
            print(f"channels  eeg vs {c}: no row -> keep eeg")
            continue
        txt, dlt, lo, hi = fmt_cmp(r, base)
        good = ok(dlt, 0.02) and lo > 0
        print(f"channels  {c} vs eeg: {txt}: {'passes' if good else 'fails'} "
              f"(>= +2.0 and CI > 0)")
        if good:
            passing.append((dlt, c))
    if passing:
        chans = max(passing)[1]
    print(f"  -> chans = {chans}")
    # context alignment
    align = args.align
    r = find(align=f"router-psdctx:{kind}")
    if r is None and not ctx_names:
        print("context   router-psdctx: not applicable (no context column) -> keep router-psd")
    elif r is None:
        miss("al_ctx")
        print("context   router-psdctx: no row -> keep router-psd")
    else:
        txt, dlt, lo, hi = fmt_cmp(r, base)
        adopt = ok(dlt, 0.02) and lo > 0
        print(f"context   router-psdctx vs router-psd: {txt} -> "
              f"{'ADOPT router-psdctx' if adopt else 'keep router-psd'} (>= +2.0 and CI > 0)")
        align = f"router-psdctx:{kind}" if adopt else align
    # xDAWN
    spec = args.spec
    r = find(spec=nx) if nx else None
    if nx is None:
        print(f"xDAWN     not applicable ({args.spec} has no xd=1) -> keep {args.spec}")
    elif r is None:
        miss("xd0")
        print("xDAWN     no-xDAWN row missing -> keep xDAWN")
    else:
        txt, dlt, lo, hi = fmt_cmp(r, base)
        drop = ok(dlt, 0.01)
        print(f"xDAWN     xd=0 vs xd=1: {txt} -> "
              f"{'DROP xDAWN' if drop else 'keep xDAWN'} (drop if xd=0 >= +1.0)")
        spec = nx if drop else spec
    # extra feature blocks (sprint 2026-09-29): xb (drop the recipe's), xa (add)
    def blocks_cmp(r):
        txt, dlt, lo, hi = fmt_cmp(r, base)              # alternative - recipe
        dc = [a - b for a, b in zip(cc.get(r["key"], []), cc.get(base["key"], []))]
        ctx_txt = (" (per context " + ", ".join(f"{100 * v:+.2f}" for v in dc) + ")"
                   if dc else "")
        return txt + ctx_txt, dlt, dc
    if xb_alt is None and xa_alt is None:
        print("blocks    not applicable (recipe has no x=, no --xb) -> none")
    drop = False
    if xb_alt is not None:
        r = find(spec=xb_alt)
        if r is None:
            miss("xb")
            print(f"blocks    row {xb_alt} missing -> keep the recipe's blocks")
        else:
            txt, dlt, dc = blocks_cmp(r)
            drop = ok(dlt, 0.01)
            print(f"blocks    without vs with the recipe's x=: {txt} -> "
                  f"{'DROP the blocks' if drop else 'keep the blocks'} "
                  "(drop only if >= +1.0 without them)")
    add = False
    if xa_alt is not None:
        r = find(spec=xa_alt)
        if r is None:
            miss("xa")
            print(f"blocks    row {xa_alt} missing -> do not add {args.xb}")
        else:
            txt, dlt, dc = blocks_cmp(r)
            add = ok(dlt, 0.01) and all(round(v, 6) >= -0.01 for v in dc)
            print(f"blocks    adding {args.xb} vs the recipe: {txt} -> "
                  f"{'ADD ' + args.xb if add else 'do not add ' + args.xb} "
                  "(add only if >= +1.0 and no context < -1.0)")
    if drop and add:     # both single changes won: take the larger gain only
        print("blocks    both rules fired: dropping the recipe's blocks and adding "
              f"{args.xb} were each tested alone; keep the recipe's blocks, add "
              f"{args.xb} (the tested xa row) and re-check the combination in "
              "train_sealed.sh's harness steps")
        drop = False
    if drop:
        spec = re.sub(r",x=[^,]*", "", spec)
    if add:
        spec = (re.sub(r"(,x=[^,]*)", lambda m: m.group(1) + "+" + args.xb, spec)
                if ",x=" in spec else f"{spec},x={args.xb}")
    # blend-weight CV (rule 4: loso unless measured >= 2 points worse; with no
    # loso-vs-last pair measured nothing says it is worse, so loso, i.e. the
    # recipe's BLEND_W=auto; RELEASE_DAY section 9, V1 2026-09-29)
    wcv = args.wcv
    r = find(wcv=wcv_alt)
    if r is None:
        wcv = "loso"
        miss(f"wcv_{wcv_alt}")
        print(f"wcv       wcv={wcv_alt} row missing -> wcv = loso (rule 4's default: loso "
              f"unless measured <= -2.0 vs last)")
    else:
        lo_r, la_r = (r, base) if wcv_alt == "loso" else (base, r)
        txt, dlt, lo, hi = fmt_cmp(lo_r, la_r)
        wcv = "last" if round(dlt, 6) <= -0.02 else "loso"
        print(f"wcv       loso vs last: {txt} (w: loso {lo_r.get('blend_calib_w')}, last "
              f"{la_r.get('blend_calib_w')}) -> wcv = {wcv} (loso unless <= -2.0)")
    # training pool
    pool = "all"
    r = find(pool="test")
    if r is None:
        miss("pool_test")
        print("pool      pool=test row missing -> keep all")
    else:
        txt, dlt, lo, hi = fmt_cmp(r, base)
        pool = "test" if ok(dlt, 0.02) and lo > 0 else "all"
        print(f"pool      test-only vs all: {txt} -> pool = {pool} (test only if >= +2.0 "
              f"and CI > 0)")
    # rule-dependent
    if online is None:
        miss("online")
        print("online-64 (rule-dependent) no row")
    else:
        txt, dlt, lo, hi = fmt_cmp(online, base)
        print(f"online-64 (RULE-DEPENDENT, report only) vs clean: {txt}")
    print(f"\nresult: spec {spec} / align {align} / chans {chans} / pool {pool} / wcv {wcv}")
    if lost:
        print(f"MISSING: {len(lost)} step(s) without a row ({' '.join(lost)}): their rules were "
              f"NOT applied (defaults kept); rerun them (release_ablations.sh STEPS=\""
              f"{' '.join(lost)}\") before deciding")
    # rule 4: loso deploys as the solver's own weight (BLEND_W=auto), last as
    # the harness's weight baked in (BLEND_W=harness)
    bw = "auto" if wcv == "loso" else "harness"
    env = (f"RECIPE_SPEC={spec} RECIPE_ALIGN={align} CHANS={chans} WCV={wcv} BLEND_W={bw} "
           f"SPLIT={split}" + (f" TEST_SUBJECTS={ts}" if ts else ""))
    print(f"train_sealed.sh: {env} bash ~/codabench/scripts/train_sealed.sh ...")
    print("(each rule compared one change with the recipe; train_sealed.sh's harness steps "
          "score the combined settings)")
    if "psdctx" in align:
        print("NOTE: router-psdctx deploys as Riemann-Sealed align=\"subject_context\" "
              "(train_sealed.sh bakes it from RECIPE_ALIGN); it has no online mode, so "
              "it cannot be combined with ADAPT=online")
    if pool != "all":
        print("NOTE: Riemann-Sealed trains on every labelled window: pool=test needs a "
              "solver option (or a filtered train loader) before it can be trained")
    print("```")


if __name__ == "__main__":
    main()
