#!/usr/bin/env python
"""Phase 3: pooling vs personalisation in the cross-session proxy setting.

    python ~/codabench/analysis/sealed_personal.py --tag p3 --study tangermann2012 \
        --family riemann:xd=1,fb=1 --align router-psd:riemann [--seeds 33 34 35]

Variants (all on the same aligned data; the alignment is the Phase 2 winner):
  pooled      one model on every subject's training sessions
  calib       pooled + per-subject calibration. Riemann: the pooled feature
              extractor (tangent spaces, xDAWN, filter bank) with the LDA refit
              on the subject's own training windows. EEGNet: the pooled network
              with only the classifier layer fine-tuned on the subject's
              training windows (15 epochs, lr 1e-3, BatchNorm/dropout frozen)
  blend       w * P_pooled + (1 - w) * P_persubject, w in {0, .25, .5, .75, 1}
              chosen on TRAINING data only (--wcv):
                last (default): the val session where subjects have >= 2
                  training sessions (each subject's LAST training session,
                  Zhou), else two chronological halves of each subject's
                  training session (fit one, score the other, both ways);
                loso: leave-one-session-out, one fold per chronological
                  session index k (val = session k of the subjects with >= 2
                  training sessions, fit = every other training window); w
                  maximises the mean fold cell score. The solver's
                  blend_w="auto" uses the same definition. Riemann only
              --wref strict (opt-in, sprint 2026-10-01): each fold's
                  whitening references from its fit rows only (router
                  alignments; a pair with < ctx_min fit rows takes its
                  subject's), as the solver's wcv_ref="strict"; default all
                  (the whole training set's references, ~2 points of CV bias)
  blend_calib as blend, with the calib model as the personal part (own weight)
  persubject  one model per subject on its own training sessions (Riemann only
              here; EEGNet per-subject comes from Phases 1-2)
Personalised variants need a subject id at test time. Every subject's
personal model predicts EVERY test window; a row then takes the model of its
TRUE subject ("oracle-id") or of the subject the log-PSD router picked
("router-id": a wrong route applies the wrong subject's model; a window the
router rejects as an outlier gets the pooled prediction). Under
--align router-psdctx:<kind> the router predicts (subject, context) pairs and
router-id takes the pair's subject.
Split / channel options (--split, --test_subjects, --chans, --pool,
--router_cap, --mmap) as in sealed_run.py; scores average over subject x
session x context cells. --router_cap 0.5 caps the personal-model router's
fallback threshold like the solver (default: uncapped, as committed).
--wvariant calib (opt-in; default both): only pooled, calib and blend_calib
rows, without the per-subject full Riemann models that only persubject /
blend need (in choose_w's folds too); blend_calib's weight and rows are the
same as under both, ~1.5-2x faster.
Rows -> ~/codabench/logs/sealed_<tag>/results_<study>.jsonl, spec =
family/variant, mode = none | oracle-id | router-id.
"""

import argparse
import copy
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import xsess_lib as L  # noqa: E402
from sealed_run import (add_data_args, aligned_data, config_key, ctx_groups,  # noqa: E402
                        data_line, prepare_data)

W_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


class _LDASpec:
    """``.fit(F, y)`` -> fitted shrinkage LDA with uniform priors over K
    classes: a plain sklearn LinearDiscriminantAnalysis from the solvers'
    ``fit_shrinkage_lda`` (Cholesky solve above 4000 features, as the
    submission; sklearn's own fit below)."""

    def __init__(self, K):
        self.priors = np.full(K, 1.0 / K)

    def fit(self, F, y):
        import riemann_steptype as RS
        return RS.fit_shrinkage_lda(F, y, self.priors)


def _lda(K):
    return _LDASpec(K)


def select(stack, subjects, ids, P_pool, fallback=None):
    """Row i <- stack[subject index of ids[i], i]; fallback rows <- pooled."""
    out = np.array(P_pool)
    pos = {s: k for k, s in enumerate(subjects)}
    for i, s in enumerate(ids):
        if (fallback is None or not fallback[i]) and s in pos:
            out[i] = stack[pos[s], i]
    return out


# ---------------------------------------------------------------------------
# Riemann: shared extractor + per-subject LDA; per-subject models
# ---------------------------------------------------------------------------
def riemann_all(meta, spec, X, y, subj, fit_idx, pred_idx, own_rows_only=False,
                persubject=True):
    """pooled (n, K) and per-subject stacks (S, n, K) for calib / persubject.
    own_rows_only: each subject's models predict only its own rows (CV use).
    persubject=False: skip the per-subject full models (only "calib" is
    returned; --wvariant calib)."""
    K = int(meta["n_classes"])
    subjects = np.unique(subj[fit_idx])
    pooled = L.make_model(spec, meta).fit(X[fit_idx], y[fit_idx])
    P_pool = pooled.predict_proba(X[pred_idx])
    F_fit, F_pred = pooled.features(X[fit_idx]), pooled.features(X[pred_idx])
    calib = np.repeat(P_pool[None], len(subjects), 0)
    ps = np.repeat(P_pool[None], len(subjects), 0) if persubject else None
    for k, s in enumerate(subjects):
        mf = subj[fit_idx] == s
        rows = (subj[pred_idx] == s) if own_rows_only else np.ones(len(pred_idx), bool)
        if not rows.any() or len(np.unique(y[fit_idx][mf])) < K:
            continue
        calib[k, rows] = _lda(K).fit(F_fit[mf], y[fit_idx][mf]).predict_proba(F_pred[rows])
        if persubject:
            m = L.make_model(spec, meta).fit(X[fit_idx][mf], y[fit_idx][mf])
            ps[k, rows] = m.predict_proba(X[pred_idx][rows])
    stacks = {"calib": calib}
    if persubject:
        stacks["persubject"] = ps
    return subjects, P_pool, stacks


def _halves_folds(d, tr_idx):
    """Two chronological halves of each subject's training windows (fit one,
    score the other, both ways)."""
    subj = d["subj"]
    first = np.zeros(len(tr_idx), bool)
    for s in np.unique(subj[tr_idx]):      # rows are in recording order
        i = np.where(subj[tr_idx] == s)[0]
        first[i[: len(i) // 2]] = True
    return [(tr_idx[first], tr_idx[~first]), (tr_idx[~first], tr_idx[first])]


def blend_cv_folds(d, tr_idx):
    """(fit_idx, val_idx) pairs inside the training windows only (--wcv last):
    each subject's last training session when every subject has >= 2
    training sessions, else chronological halves."""
    subj, sess = d["subj"], d["sess"]
    subjects = np.unique(subj[tr_idx])
    if min(len(np.unique(sess[tr_idx][subj[tr_idx] == s])) for s in subjects) >= 2:
        last = np.zeros(subj.max() + 1, dtype=np.int64)
        for s in subjects:
            last[s] = sess[tr_idx][subj[tr_idx] == s].max()
        v = sess[tr_idx] == last[subj[tr_idx]]
        return [(tr_idx[~v], tr_idx[v])]
    return _halves_folds(d, tr_idx)


def blend_loso_folds(d, tr_idx):
    """(fit_idx, val_idx) pairs inside the training windows (--wcv loso; the
    solver's blend_w="auto" uses the same definition). For each chronological
    session index k: val = training windows of session k of the subjects with
    >= 2 training sessions; fit = every other training window. Folds with an
    empty val are skipped. No subject with >= 2 training sessions -> the
    chronological-halves folds."""
    subj, sess = d["subj"][tr_idx], d["sess"][tr_idx]
    multi = [s for s in np.unique(subj) if len(np.unique(sess[subj == s])) >= 2]
    if not multi:
        return _halves_folds(d, tr_idx)
    folds = []
    for k in np.unique(sess):
        v = (sess == k) & np.isin(subj, multi)
        if v.any():
            folds.append((tr_idx[~v], tr_idx[v]))
    return folds


def strict_fold_X(d, align, covs, tr_idx, fit_idx, out):
    """--wref strict: the training rows whitened with references from
    ``fit_idx`` only, written into ``out[tr_idx]`` (full-size, row-indexed as
    d["X"]; choose_w passes Xa, whose training rows it no longer needs), as
    the solver's ``_fold_refs`` + float32 whitening: per group (ctx_groups:
    subject, or (subject, context) pair) its fit rows' mean covariance; a pair
    with < CTX_MIN_WINDOWS (the solver's default ctx_min) fit rows takes its
    subject's fit-row reference, a subject without fit rows the global one
    (computed only then)."""
    from riemann_sealed import CTX_MIN_WINDOWS
    kind = align.split(":")[1]
    grp, n_ctx = ctx_groups(d, align)
    subj, X = d["subj"], d["X"]
    fit = np.zeros(len(subj), bool)
    fit[fit_idx] = True
    Wg, Wsub = [], {}

    def glob_W():
        if not Wg:
            Wg.append(L.inv_sqrtm(L.mean_cov(covs[fit], kind)))
        return Wg[0]

    def subj_W(k):
        if k not in Wsub:
            m = fit & (subj == k)
            Wsub[k] = L.inv_sqrtm(L.mean_cov(covs[m], kind)) if m.any() else glob_W()
        return Wsub[k]
    g_tr = grp[tr_idx]
    for g in np.unique(g_tr):
        m = fit & (grp == g)
        if n_ctx == 1:
            W = subj_W(g)
        else:
            W = (L.inv_sqrtm(L.mean_cov(covs[m], kind)) if m.sum() >= CTX_MIN_WINDOWS
                 else subj_W(g // n_ctx))
        loc = np.where(g_tr == g)[0]
        for i in range(0, len(loc), 1024):
            c = loc[i:i + 1024]
            out[tr_idx[c]] = L.apply_W(X[tr_idx[c]], W)
    return out


def choose_w(d, meta, spec, Xa, tr_idx, wcv="last", variant="both", wref="all",
             align=None, covs=None):
    """Pooled weight for blending pooled with each personal stack
    ("persubject" -> blend, "calib" -> blend_calib), chosen on training CV
    (``wcv`` = last | loso); cells include the context. ``variant`` = both
    (default) | calib: only blend_calib's weight, without the per-subject
    full models that only blend needs (--wvariant calib; its weight and fold
    scores are the same as under both). Returns {name: (w, mean fold scores
    per W_GRID, per-fold scores)} plus "folds": one descriptor per fold.
    ``wref`` = all (default: Xa's whole-training-set references) | strict
    (``strict_fold_X`` per fold, overwriting Xa's training rows; needs
    ``align`` = router-psd:<kind> or router-psdctx:<kind> and the full-size
    window ``covs``)."""
    y, subj, sess, ctx = d["y"], d["subj"], d["sess"], d["ctx"]
    if wref not in ("all", "strict"):
        raise ValueError(f"wref {wref!r}: expected all | strict")
    if wref == "strict" and (align is None or align.split(":")[0] not in
                             ("router-psd", "router-psdctx")):
        raise ValueError(f"wref='strict' needs align router-psd:<kind> | "
                         f"router-psdctx:<kind>, got {align!r}")
    if wcv not in ("last", "loso"):
        raise ValueError(f"wcv {wcv!r}: expected last | loso")
    if variant not in ("both", "calib"):
        raise ValueError(f"variant {variant!r}: expected both | calib")
    folds = blend_loso_folds(d, tr_idx) if wcv == "loso" else blend_cv_folds(d, tr_idx)
    n_sess = [len(np.unique(sess[tr_idx][subj[tr_idx] == s])) for s in np.unique(subj[tr_idx])]
    kind = (("loso" if max(n_sess) >= 2 else "halves") if wcv == "loso"
            else ("last-session" if min(n_sess) >= 2 else "halves"))
    names = ("persubject", "calib") if variant == "both" else ("calib",)
    scores = {name: np.zeros(len(W_GRID)) for name in names}
    per_fold = {name: [] for name in scores}
    desc = []
    for i, (fit_idx, val_idx) in enumerate(folds):
        if wref == "strict":     # this fold's references, into Xa's training rows
            strict_fold_X(d, align, covs, tr_idx, fit_idx, Xa)
        subjects, P_pool, st = riemann_all(meta, spec, Xa, y, subj, fit_idx, val_idx,
                                           own_rows_only=True,
                                           persubject=variant == "both")
        for name in scores:
            P_p = select(st[name], subjects, subj[val_idx], P_pool)
            fold_sc = []
            for k, w in enumerate(W_GRID):
                Pb = w * P_pool + (1 - w) * P_p
                c = L.score(y[val_idx], Pb.argmax(1), subj[val_idx], sess[val_idx],
                            ctx[val_idx])["cell"]
                scores[name][k] += c
                fold_sc.append(c)
            per_fold[name].append(fold_sc)
        desc.append(dict(kind=kind, val_sessions=np.unique(sess[val_idx]).tolist(),
                         val_subjects=len(np.unique(subj[val_idx])),
                         n_fit=len(fit_idx), n_val=len(val_idx)))
        L.log(f"  wcv={wcv}{'' if wref == 'all' else ' wref=strict'} ({kind}) "
              f"fold {i + 1}/{len(folds)}: val sessions "
              f"{desc[-1]['val_sessions']} of {desc[-1]['val_subjects']} subjects "
              f"(n_fit={len(fit_idx)} n_val={len(val_idx)}) cell for w={W_GRID}: "
              + ("" if "persubject" not in per_fold else
                 f"blend {np.round(per_fold['persubject'][-1], 3).tolist()} ")
              + f"blend_calib {np.round(per_fold['calib'][-1], 3).tolist()}")
    out = {"folds": desc}
    for name, sc in scores.items():
        sc = sc / len(folds)
        k = max(range(len(W_GRID)), key=lambda i: (round(sc[i], 6), W_GRID[i]))  # ties -> pooled
        out[name] = (W_GRID[k], sc.tolist(), per_fold[name])
    return out


# ---------------------------------------------------------------------------
# EEGNet: pooled net, classifier-only fine-tune per subject
# ---------------------------------------------------------------------------
def eegnet_finetune(model, X, y, seed, epochs=15, lr=1e-3, bs=32):
    net = copy.deepcopy(model.net)
    for p in net.parameters():
        p.requires_grad = False
    for p in net.classifier.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(net.classifier.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    net.eval()          # BatchNorm statistics and dropout frozen
    for _ in range(epochs):
        for b in L._batches(len(y), bs, rng):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(Xt[b]), yt[b]).backward()
            opt.step()
            net.apply_max_norm()
    return net


def eegnet_all(d, meta, spec, Xa, seed, tr_idx, te_idx):
    y, subj = d["y"], d["subj"]
    subjects = np.unique(subj[tr_idx])
    model = L.make_model(spec, meta, seed)
    fi, vi = L.es_split(d, tr_idx, "pooled", seed)
    model.fit(Xa[fi], y[fi], Xval=Xa[vi], yval=y[vi])
    P_pool = model.predict_proba(Xa[te_idx])
    calib = np.repeat(P_pool[None], len(subjects), 0)
    Xte = torch.from_numpy(Xa[te_idx])
    for k, s in enumerate(subjects):
        mf = tr_idx[subj[tr_idx] == s]
        net = eegnet_finetune(model, Xa[mf], y[mf], seed)
        with torch.no_grad():
            calib[k] = torch.cat([torch.softmax(net(Xte[i:i + 512]), 1)
                                  for i in range(0, len(Xte), 512)]).numpy()
    return subjects, P_pool, {"calib": calib}, model.best_epoch


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--study", required=True)
    ap.add_argument("--classes", default=None)
    ap.add_argument("--family", required=True, help="riemann:... | eegnet_st")
    ap.add_argument("--align", default="router-psd:riemann")
    ap.add_argument("--seeds", nargs="+", type=int, default=[33])
    ap.add_argument("--wcv", default="last", choices=["last", "loso"],
                    help="blend-weight CV: last training session | leave-one-session-out")
    ap.add_argument("--wref", default="all", choices=["all", "strict"],
                    help="blend-weight CV whitening references: whole training set "
                         "(all) | each fold's fit rows (strict, router alignments)")
    ap.add_argument("--wvariant", default="both", choices=["both", "calib"],
                    help="both: blend + blend_calib (and persubject); calib: only "
                         "blend_calib, without per-subject full models (faster)")
    add_data_args(ap)
    args = ap.parse_args()
    classes = None if args.classes is None else [int(c) for c in args.classes.split(",")]
    out = Path.home() / f"codabench/logs/sealed_{args.tag}"
    (out / "probs").mkdir(parents=True, exist_ok=True)
    res_path = out / f"results_{args.study}.jsonl"
    done = {r["key"] for r in L.read_results(out / "results.jsonl")}

    d = L.load_study(args.study, classes, mmap=args.mmap)
    sp, drop, opts, si = prepare_data(d, args)
    if args.wref == "strict" and args.align.split(":")[0] not in ("router-psd",
                                                                  "router-psdctx"):
        # as the solver (wcv_ref="strict" equals "all" with adapt="online"): the
        # online / oracle conditions centre each session on itself by design
        L.log(f"--wref strict applies to router alignments; {args.align}: using all")
        args.wref = "all"
    opts.update(wcv=args.wcv, wvariant=args.wvariant, wref=args.wref)
    both = args.wvariant == "both"
    meta = dict(d["meta"], n_classes=int(d["y"].max() + 1))
    tr, te = sp["train"], sp["test"]
    tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
    y, subj, sess, ctx = d["y"], d["subj"], d["sess"], d["ctx"]
    L.log(f"[data] study={args.study} classes={classes or 'all'} X={d['X'].shape} "
          f"subjects={len(np.unique(subj))} train={len(tr_idx)} test={len(te_idx)} "
          f"K={meta['n_classes']} family={args.family} align={args.align} threads={L.N_THREADS} "
          f"sessions/subject={len(np.unique(sess))} dropped_ch={drop} wcv={args.wcv} "
          + ("" if both else f"wvariant={args.wvariant} ")
          + data_line(args, si))

    # personal-model router: log-PSD over subjects, or over (subject, context)
    # pairs under router-psdctx (then a window's id is its pair's subject)
    t0 = time.time()
    grp, n_ctx = ctx_groups(d, args.align)
    rname = "psdctx" if args.align.split(":")[0].endswith("-psdctx") else "psd"
    router = L.SubjectRouter("psd", meta["sfreq"], cap=args.router_cap).fit(
        d["X"][tr], grp[tr], sess[tr])
    Pr = router.predict_proba(d["X"][te])
    cache = {f"router_{rname}": (router, Pr)}
    rid = router.subjects[Pr.argmax(1)]
    if rname == "psdctx":
        rid = rid // n_ctx
    r_fb = Pr.max(1) < router.thr
    racc = float(np.mean(rid == subj[te_idx]))
    L.log(f"[router] {rname}: acc={racc:.3f} fallback={r_fb.mean():.3f} ({time.time() - t0:.0f}s)"
          + ("" if args.router_cap is None else
             f" thr={router.thr:.3f} (uncapped {router.thr_raw:.3f}, "
             f"fallback uncapped {np.mean(Pr.max(1) < router.thr_raw):.3f})"))
    Xa, info = aligned_data(d, tr, te, args.align, cache, args.router_cap)
    info.pop("router_assign", None)
    if rname == "psdctx" and n_ctx > 1:
        L.log(f"[router] psdctx: (subject, context) acc={info['router_acc']:.3f} "
              f"subject acc={info['router_subj_acc']:.3f} context acc={info['router_ctx_acc']:.3f}")

    n_tr, n_te = len(tr_idx), len(te_idx)
    neural = L.is_neural(args.family)
    for seed in (args.seeds if neural else [33]):
        key0 = config_key(args.study, classes, f"{args.family}/pooled", "none", args.align,
                          seed, **opts)
        if key0 in done:
            L.log(f"skip {key0} (done)")
            continue
        t0 = time.time()
        L.log(f"[fit] {args.family} seed={seed} X_train={(n_tr,) + d['X'].shape[1:]} "
              f"X_test={(n_te,) + d['X'].shape[1:]}")
        extra = {}
        if neural:
            subjects, P_pool, stacks, ep = eegnet_all(d, meta, args.family, Xa, seed, tr_idx, te_idx)
            extra["epochs"] = [int(ep)]
        else:
            subjects, P_pool, stacks = riemann_all(meta, args.family, Xa, y, subj, tr_idx,
                                                   te_idx, persubject=both)
            ws = choose_w(d, meta, args.family, Xa, tr_idx, args.wcv, args.wvariant,
                          args.wref, args.align, cache.get("covs"))
            if both:
                w, cv, fcv = ws["persubject"]
                stacks["blend"] = w * P_pool[None] + (1 - w) * stacks["persubject"]
                extra.update(blend_w=w, blend_cv=[round(v, 4) for v in cv])
            wc, cvc, fcvc = ws["calib"]
            stacks["blend_calib"] = wc * P_pool[None] + (1 - wc) * stacks["calib"]
            extra.update(blend_calib_w=wc, blend_calib_cv=[round(v, 4) for v in cvc])
            if both:
                extra["blend_cv_folds"] = [[round(v, 4) for v in f] for f in fcv]
            extra.update(blend_calib_cv_folds=[[round(v, 4) for v in f] for f in fcvc],
                         wcv_folds=ws["folds"])
            if both:
                L.log(f"  blend: w_pooled={w} chosen on training CV (cell scores "
                      f"{np.round(cv, 3).tolist()} for w={W_GRID})")
            L.log(f"  blend_calib: w_pooled={wc} (training CV {np.round(cvc, 3).tolist()})")
        dt = time.time() - t0
        rows = [("pooled", "none", P_pool)]
        for v, st in stacks.items():
            rows.append((v, "oracle-id", select(st, subjects, subj[te_idx], P_pool)))
            rows.append((v, "router-id", select(st, subjects, rid, P_pool, r_fb)))
        for v, mode, Pv in rows:
            key = config_key(args.study, classes, f"{args.family}/{v}", mode, args.align,
                             seed, **opts)
            sc = L.score(y[te_idx], Pv.argmax(1), subj[te_idx], sess[te_idx], ctx[te_idx])
            h = hashlib.md5(key.encode()).hexdigest()[:12]
            np.savez_compressed(out / "probs" / f"{h}.npz", P=Pv, te_idx=te_idx, key=key)
            L.append_result(res_path, dict(
                key=key, study=args.study, classes=classes, spec=f"{args.family}/{v}",
                mode=mode, align=args.align, seed=seed, probs=f"{h}.npz",
                cell=sc["cell"], pooled=sc["pooled"], per_subject=sc["per_subject"],
                n_cells=sc["n_cells"], n_train=n_tr, n_test=n_te,
                shape=list(d["X"].shape), seconds=round(dt, 1),
                time=time.strftime("%H:%M:%S"), router_psd_acc=racc,
                router_psd_fallback=float(r_fb.mean()), **opts, n_ctx=si["n_ctx"],
                **extra, **info))
            L.log(f"[done] {key} cell={sc['cell']:.4f} pooled={sc['pooled']:.4f}")
        L.log(f"  seed {seed}: {dt:.0f}s")
    L.log("ALL CONFIGS DONE")


if __name__ == "__main__":
    main()
