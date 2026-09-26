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
              chosen on TRAINING data only: the val session where subjects have
              >= 2 training sessions (Zhou), else two chronological halves of
              each subject's training session (fit one, score the other, both
              ways). Riemann only
  blend_calib as blend, with the calib model as the personal part (own weight)
  persubject  one model per subject on its own training sessions (Riemann only
              here; EEGNet per-subject comes from Phases 1-2)
Personalised variants need a subject id at test time. Every subject's
personal model predicts EVERY test window; a row then takes the model of its
TRUE subject ("oracle-id") or of the subject the log-PSD router picked
("router-id": a wrong route applies the wrong subject's model; a window the
router rejects as an outlier gets the pooled prediction).
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
from sealed_run import aligned_data, config_key  # noqa: E402

W_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def _lda(K):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
                                      priors=np.full(K, 1.0 / K))


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
def riemann_all(meta, spec, X, y, subj, fit_idx, pred_idx, own_rows_only=False):
    """pooled (n, K) and per-subject stacks (S, n, K) for calib / persubject.
    own_rows_only: each subject's models predict only its own rows (CV use)."""
    K = int(meta["n_classes"])
    subjects = np.unique(subj[fit_idx])
    pooled = L.make_model(spec, meta).fit(X[fit_idx], y[fit_idx])
    P_pool = pooled.predict_proba(X[pred_idx])
    F_fit, F_pred = pooled.features(X[fit_idx]), pooled.features(X[pred_idx])
    calib = np.repeat(P_pool[None], len(subjects), 0)
    ps = np.repeat(P_pool[None], len(subjects), 0)
    for k, s in enumerate(subjects):
        mf = subj[fit_idx] == s
        rows = (subj[pred_idx] == s) if own_rows_only else np.ones(len(pred_idx), bool)
        if not rows.any() or len(np.unique(y[fit_idx][mf])) < K:
            continue
        calib[k, rows] = _lda(K).fit(F_fit[mf], y[fit_idx][mf]).predict_proba(F_pred[rows])
        m = L.make_model(spec, meta).fit(X[fit_idx][mf], y[fit_idx][mf])
        ps[k, rows] = m.predict_proba(X[pred_idx][rows])
    return subjects, P_pool, {"calib": calib, "persubject": ps}


def blend_cv_folds(d, tr_idx):
    """(fit_idx, val_idx) pairs inside the training windows only."""
    subj, sess = d["subj"], d["sess"]
    subjects = np.unique(subj[tr_idx])
    if min(len(np.unique(sess[tr_idx][subj[tr_idx] == s])) for s in subjects) >= 2:
        last = np.zeros(subj.max() + 1, dtype=np.int64)
        for s in subjects:
            last[s] = sess[tr_idx][subj[tr_idx] == s].max()
        v = sess[tr_idx] == last[subj[tr_idx]]
        return [(tr_idx[~v], tr_idx[v])]
    first = np.zeros(len(tr_idx), bool)
    for s in subjects:                     # rows are in recording order
        i = np.where(subj[tr_idx] == s)[0]
        first[i[: len(i) // 2]] = True
    return [(tr_idx[first], tr_idx[~first]), (tr_idx[~first], tr_idx[first])]


def choose_w(d, meta, spec, Xa, tr_idx):
    """Pooled weight for blending pooled with each personal stack
    ("persubject" -> blend, "calib" -> blend_calib), chosen on training CV."""
    y, subj, sess = d["y"], d["subj"], d["sess"]
    folds = blend_cv_folds(d, tr_idx)
    scores = {"persubject": np.zeros(len(W_GRID)), "calib": np.zeros(len(W_GRID))}
    for fit_idx, val_idx in folds:
        subjects, P_pool, st = riemann_all(meta, spec, Xa, y, subj, fit_idx, val_idx,
                                           own_rows_only=True)
        for name in scores:
            P_p = select(st[name], subjects, subj[val_idx], P_pool)
            for k, w in enumerate(W_GRID):
                Pb = w * P_pool + (1 - w) * P_p
                scores[name][k] += L.score(y[val_idx], Pb.argmax(1), subj[val_idx],
                                           sess[val_idx])["cell"]
    out = {}
    for name, sc in scores.items():
        sc = sc / len(folds)
        k = max(range(len(W_GRID)), key=lambda i: (round(sc[i], 6), W_GRID[i]))  # ties -> pooled
        out[name] = (W_GRID[k], sc.tolist())
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
    ap.add_argument("--drop_ch", default="A2-A1")
    args = ap.parse_args()
    classes = None if args.classes is None else [int(c) for c in args.classes.split(",")]
    out = Path.home() / f"codabench/logs/sealed_{args.tag}"
    (out / "probs").mkdir(parents=True, exist_ok=True)
    res_path = out / f"results_{args.study}.jsonl"
    done = {r["key"] for r in L.read_results(out / "results.jsonl")}

    d = L.load_study(args.study, classes)
    drop = [c for c in args.drop_ch.split(",") if c in d["meta"]["ch_names"]]
    if drop:
        keep = [i for i, c in enumerate(d["meta"]["ch_names"]) if c not in drop]
        d["X"] = np.ascontiguousarray(d["X"][:, keep])
        d["meta"] = dict(d["meta"], ch_names=[d["meta"]["ch_names"][i] for i in keep])
    meta = dict(d["meta"], n_classes=int(d["y"].max() + 1))
    sp = L.xsess_split(d)
    tr, te = sp["train"], sp["test"]
    tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
    y, subj, sess = d["y"], d["subj"], d["sess"]
    L.log(f"[data] study={args.study} classes={classes or 'all'} X={d['X'].shape} "
          f"subjects={len(np.unique(subj))} train={len(tr_idx)} test={len(te_idx)} "
          f"K={meta['n_classes']} family={args.family} align={args.align} threads={L.N_THREADS}")

    t0 = time.time()
    router = L.SubjectRouter("psd", meta["sfreq"]).fit(d["X"][tr], subj[tr], sess[tr])
    Pr = router.predict_proba(d["X"][te])
    cache = {"router_psd": (router, Pr)}
    rid = router.subjects[Pr.argmax(1)]
    r_fb = Pr.max(1) < router.thr
    racc = float(np.mean(rid == subj[te_idx]))
    L.log(f"[router] psd: acc={racc:.3f} fallback={r_fb.mean():.3f} ({time.time() - t0:.0f}s)")
    Xa, info = aligned_data(d, tr, te, args.align, cache)
    info.pop("router_assign", None)

    neural = L.is_neural(args.family)
    for seed in (args.seeds if neural else [33]):
        key0 = config_key(args.study, classes, f"{args.family}/pooled", "none", args.align, seed)
        if key0 in done:
            L.log(f"skip {key0} (done)")
            continue
        t0 = time.time()
        L.log(f"[fit] {args.family} seed={seed} X_train={tuple(d['X'][tr].shape)} "
              f"X_test={tuple(d['X'][te].shape)}")
        extra = {}
        if neural:
            subjects, P_pool, stacks, ep = eegnet_all(d, meta, args.family, Xa, seed, tr_idx, te_idx)
            extra["epochs"] = [int(ep)]
        else:
            subjects, P_pool, stacks = riemann_all(meta, args.family, Xa, y, subj, tr_idx, te_idx)
            ws = choose_w(d, meta, args.family, Xa, tr_idx)
            (w, cv), (wc, cvc) = ws["persubject"], ws["calib"]
            stacks["blend"] = w * P_pool[None] + (1 - w) * stacks["persubject"]
            stacks["blend_calib"] = wc * P_pool[None] + (1 - wc) * stacks["calib"]
            extra.update(blend_w=w, blend_cv=[round(v, 4) for v in cv],
                         blend_calib_w=wc, blend_calib_cv=[round(v, 4) for v in cvc])
            L.log(f"  blend: w_pooled={w} chosen on training CV (cell scores "
                  f"{np.round(cv, 3).tolist()} for w={W_GRID})")
            L.log(f"  blend_calib: w_pooled={wc} (training CV {np.round(cvc, 3).tolist()})")
        dt = time.time() - t0
        rows = [("pooled", "none", P_pool)]
        for v, st in stacks.items():
            rows.append((v, "oracle-id", select(st, subjects, subj[te_idx], P_pool)))
            rows.append((v, "router-id", select(st, subjects, rid, P_pool, r_fb)))
        for v, mode, Pv in rows:
            key = config_key(args.study, classes, f"{args.family}/{v}", mode, args.align, seed)
            sc = L.score(y[te_idx], Pv.argmax(1), subj[te_idx], sess[te_idx])
            h = hashlib.md5(key.encode()).hexdigest()[:12]
            np.savez_compressed(out / "probs" / f"{h}.npz", P=Pv, te_idx=te_idx, key=key)
            L.append_result(res_path, dict(
                key=key, study=args.study, classes=classes, spec=f"{args.family}/{v}",
                mode=mode, align=args.align, seed=seed, probs=f"{h}.npz",
                cell=sc["cell"], pooled=sc["pooled"], per_subject=sc["per_subject"],
                n_cells=sc["n_cells"], n_train=len(tr_idx), n_test=len(te_idx),
                shape=list(d["X"].shape), seconds=round(dt, 1),
                time=time.strftime("%H:%M:%S"), router_psd_acc=racc,
                router_psd_fallback=float(r_fb.mean()), **extra, **info))
            L.log(f"[done] {key} cell={sc['cell']:.4f} pooled={sc['pooled']:.4f}")
        L.log(f"  seed {seed}: {dt:.0f}s")
    L.log("ALL CONFIGS DONE")


if __name__ == "__main__":
    main()
