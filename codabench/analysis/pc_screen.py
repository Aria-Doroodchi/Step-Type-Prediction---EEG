#!/usr/bin/env python
"""Sprint 2026-10-01 Phase 7 (brief addendum, pre-registered 20:05): a
personal LDA with a shared covariance, screened against blend_calib.

The recipe's personal part ("calib") is a shrinkage LDA refitted on one
subject's windows (pooled feature extractor). With ~120-700 windows against
3-9 k features its Ledoit-Wolf shrinkage is near 1. Variant calibpc<g>: the
subject's class means with the covariance
    Sigma_k = g * Sigma_pooled + (1 - g) * Sigma_k^LW
(Sigma_pooled = the Ledoit-Wolf within-class covariance of every training
subject's re-centred windows, the pooled LDA's own estimate), i.e.
regularised discriminant analysis with subject-to-subject transfer. Each
variant is blended with the pooled model like blend_calib, its weight chosen
by sealed_personal's training CV (choose_w's folds and rule, --wcv last).

Everything else is sealed_personal's code path (prepare_data, the log-PSD
router, aligned_data, riemann_all, select, the cell metric), so this script's
own blend_calib row must equal sealed_personal's committed row: printed as a
sanity check. Rows -> ~/codabench/logs/sealed_<tag>/results_<study>.jsonl.

    python ~/codabench/analysis/pc_screen.py --tag s1001pc --study scherer2015 \
        --classes 0,1,3 --family riemann:xd=1,fb=1,x=bpt4
    python ~/codabench/analysis/pc_screen.py --summary --tag s1001pc   # the rule
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import xsess_lib as L  # noqa: E402
import sealed_personal as SP  # noqa: E402
from sealed_run import add_data_args, aligned_data, config_key, ctx_groups, prepare_data  # noqa: E402

GAMMAS = {"calibpc1": 1.0, "calibpc5": 0.5}
W_GRID = SP.W_GRID


def pc_lda(F, y, K, Sig_pool, gamma):
    """Fitted sklearn LDA (lsqr, uniform priors) whose covariance is
    gamma * Sig_pool + (1 - gamma) * the subject's own Ledoit-Wolf estimate."""
    from scipy import linalg
    from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis, _class_cov,
                                               _class_means)

    class _PC(LinearDiscriminantAnalysis):
        def _solve_lstsq(self, X, y, shrinkage, covariance_estimator):
            self.means_ = _class_means(X, y)
            own = (_class_cov(X, y, self.priors_, shrinkage, covariance_estimator)
                   if gamma < 1 else 0.0)
            S = gamma * Sig_pool + (1 - gamma) * own
            self.coef_ = linalg.solve(S, self.means_.T, assume_a="pos").T
            self.intercept_ = (-0.5 * np.diag(self.means_ @ self.coef_.T)
                               + np.log(self.priors_))
    m = _PC(solver="lsqr", shrinkage="auto", priors=np.full(K, 1.0 / K)).fit(F, y)
    m.__class__ = LinearDiscriminantAnalysis
    return m


def pc_stacks(meta, spec, X, y, subj, fit_idx, pred_idx, own_rows_only):
    """riemann_all (calib) + the calibpc stacks on the same pooled extractor."""
    from sklearn.discriminant_analysis import _class_cov
    K = int(meta["n_classes"])
    subjects = np.unique(subj[fit_idx])
    pooled = L.make_model(spec, meta).fit(X[fit_idx], y[fit_idx])
    P_pool = pooled.predict_proba(X[pred_idx])
    F_fit, F_pred = pooled.features(X[fit_idx]), pooled.features(X[pred_idx])
    yf = y[fit_idx]
    Sig_pool = _class_cov(F_fit, yf, np.full(K, 1.0 / K), "auto", None)
    names = ["calib"] + list(GAMMAS)
    st = {n: np.repeat(P_pool[None], len(subjects), 0) for n in names}
    for k, s in enumerate(subjects):
        mf = subj[fit_idx] == s
        rows = (subj[pred_idx] == s) if own_rows_only else np.ones(len(pred_idx), bool)
        if not rows.any() or len(np.unique(yf[mf])) < K:
            continue
        st["calib"][k, rows] = SP._lda(K).fit(F_fit[mf], yf[mf]).predict_proba(F_pred[rows])
        for n, g in GAMMAS.items():
            st[n][k, rows] = pc_lda(F_fit[mf], yf[mf], K, Sig_pool, g).predict_proba(
                F_pred[rows])
    return subjects, P_pool, st


def choose_ws(d, meta, spec, Xa, tr_idx, wcv):
    """sealed_personal.choose_w for every personal stack here: the same folds,
    cell metric and rule (ties -> the more pooled w)."""
    y, subj, sess, ctx = d["y"], d["subj"], d["sess"], d["ctx"]
    folds = SP.blend_loso_folds(d, tr_idx) if wcv == "loso" else SP.blend_cv_folds(d, tr_idx)
    names = ["calib"] + list(GAMMAS)
    scores = {n: np.zeros(len(W_GRID)) for n in names}
    for fit_idx, val_idx in folds:
        subjects, P_pool, st = pc_stacks(meta, spec, Xa, y, subj, fit_idx, val_idx, True)
        for n in names:
            P_p = SP.select(st[n], subjects, subj[val_idx], P_pool)
            for k, w in enumerate(W_GRID):
                Pb = w * P_pool + (1 - w) * P_p
                scores[n][k] += L.score(y[val_idx], Pb.argmax(1), subj[val_idx],
                                        sess[val_idx], ctx[val_idx])["cell"]
    out = {}
    for n, sc in scores.items():
        sc = sc / len(folds)
        k = max(range(len(W_GRID)), key=lambda i: (round(sc[i], 6), W_GRID[i]))
        out[n] = (W_GRID[k], [round(float(v), 4) for v in sc])
    return out


def run(args):
    classes = None if args.classes is None else [int(c) for c in args.classes.split(",")]
    out = Path.home() / f"codabench/logs/sealed_{args.tag}"
    (out / "probs").mkdir(parents=True, exist_ok=True)
    res_path = out / f"results_{args.study}.jsonl"
    d = L.load_study(args.study, classes)
    args.mmap = False
    sp, drop, opts, si = prepare_data(d, args)
    opts.update(wcv=args.wcv, wvariant="calib")
    meta = dict(d["meta"], n_classes=int(d["y"].max() + 1))
    tr, te = sp["train"], sp["test"]
    tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
    y, subj, sess, ctx = d["y"], d["subj"], d["sess"], d["ctx"]
    t0 = time.time()
    grp, n_ctx = ctx_groups(d, args.align)
    router = L.SubjectRouter("psd", meta["sfreq"], cap=args.router_cap).fit(
        d["X"][tr], grp[tr], sess[tr])
    Pr = router.predict_proba(d["X"][te])
    cache = {"router_psd": (router, Pr)}
    rid = router.subjects[Pr.argmax(1)]
    r_fb = Pr.max(1) < router.thr
    Xa, info = aligned_data(d, tr, te, args.align, cache, args.router_cap)
    L.log(f"[data] {args.study} classes={classes or 'all'} train={len(tr_idx)} "
          f"test={len(te_idx)} family={args.family} align={args.align}")
    subjects, P_pool, st = pc_stacks(meta, args.family, Xa, y, subj, tr_idx, te_idx, False)
    ws = choose_ws(d, meta, args.family, Xa, tr_idx, args.wcv)
    for n, (w, cv) in ws.items():
        L.log(f"  blend_{n}: w_pooled={w} (training CV {cv})")
    rows = []
    for n in st:
        w, cv = ws[n]
        Pb = w * P_pool[None] + (1 - w) * st[n]
        for mode, ids, fb in (("oracle-id", subj[te_idx], None), ("router-id", rid, r_fb)):
            rows.append((f"blend_{n}", mode, SP.select(Pb, subjects, ids, P_pool, fb),
                         dict(blend_w=w, blend_cv=cv)))
    for v, mode, Pv, extra in rows:
        key = config_key(args.study, classes, f"{args.family}/{v}", mode, args.align, 33, **opts)
        sc = L.score(y[te_idx], Pv.argmax(1), subj[te_idx], sess[te_idx], ctx[te_idx])
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        np.savez_compressed(out / "probs" / f"{h}.npz", P=Pv, te_idx=te_idx, key=key)
        L.append_result(res_path, dict(
            key=key, study=args.study, classes=classes, spec=f"{args.family}/{v}", mode=mode,
            align=args.align, seed=33, probs=f"{h}.npz", cell=sc["cell"], pooled=sc["pooled"],
            per_subject=sc["per_subject"], n_cells=sc["n_cells"], n_train=len(tr_idx),
            n_test=len(te_idx), seconds=round(time.time() - t0, 1), **opts, **extra))
        L.log(f"[done] {key} cell={sc['cell']:.4f}")


def summary(args):
    """The pre-registered rule (brief Phase 7) on this tag's rows; sanity check
    of blend_calib against the committed sealed_personal rows."""
    tag_dir = Path.home() / f"codabench/logs/sealed_{args.tag}"
    rng = np.random.default_rng(0)
    print("| study | variant | w | cell | blend_calib cell | Δ (95 % CI over subjects) | "
          "subjects up / down |\n|---|---|---|---|---|---|---|")
    deltas = {}
    # read_results reads every results_*.jsonl of the folder: key by study too
    rows = {}
    for r in L.read_results(tag_dir / "results.jsonl"):
        if r["mode"] == "router-id":
            rows[(r["study"], tuple(r["classes"] or []), r["spec"].split("/")[1])] = r
    if True:
        for (study, cls, v), r in sorted(rows.items()):
            if v == "blend_calib":
                continue
            b = rows[(study, cls, "blend_calib")]
            ks = sorted(r["per_subject"], key=int)
            dlt = np.array([r["per_subject"][k] - b["per_subject"][k] for k in ks])
            bs = rng.choice(dlt, (10000, len(dlt))).mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            name = r["study"] + ("" if not cls else f" {'-'.join(map(str, cls))}")
            deltas[(name, v)] = (dlt.mean(), lo, hi)
            print(f"| {name} | {v} | {r.get('blend_w')} | {r['cell']:.4f} | {b['cell']:.4f} | "
                  f"{100 * dlt.mean():+.2f} ({100 * lo:+.2f}, {100 * hi:+.2f}) | "
                  f"{int((dlt > 1e-9).sum())} / {int((dlt < -1e-9).sum())} |")
    print()
    for v in GAMMAS:
        v = "blend_" + v
        main = deltas.get(("scherer2015 0-1-3", v))
        others = [deltas[k][0] for k in deltas if k[1] == v and k[0] != "scherer2015 0-1-3"]
        if main is None or len(others) < 3:
            print(f"{v}: INCOMPLETE ({len(others)} other proxies)")
            continue
        ok = main[0] >= 0.015 and main[1] > 0 and np.mean(others) >= 0
        print(f"{v}: Scherer 3-cl {100 * main[0]:+.2f} (CI {100 * main[1]:+.2f}, "
              f"{100 * main[2]:+.2f}); others mean {100 * np.mean(others):+.2f} -> "
              f"{'PROMISING' if ok else 'NO GAIN'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--study")
    ap.add_argument("--classes", default=None)
    ap.add_argument("--family", default="riemann:xd=1,fb=1,x=bpt4")
    ap.add_argument("--align", default="router-psd:riemann")
    ap.add_argument("--wcv", default="last", choices=["last", "loso"])
    add_data_args(ap)
    args = ap.parse_args()
    if args.summary:
        summary(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
