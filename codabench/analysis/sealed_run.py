#!/usr/bin/env python
"""Run cross-session configs on a proxy cache; append one JSON row each.

    python ~/codabench/analysis/sealed_run.py --tag p1 --study zhou2016 \
        --models meanlr riemann:xd=1,fb=1 eegnet_st --modes pooled persubject \
        --aligns none --seeds 33 34 35

A config = (study, classes, model spec, mode, align, seed). Rows go to
~/codabench/logs/sealed_<tag>/results.jsonl, test-window probabilities to
.../probs/<key>.npz. Resumable: a config whose key is already in
results.jsonl is skipped. Deterministic models (meanlr, riemann) run once
(seed 33) whatever --seeds says.

Modes: pooled (one model on every subject's training sessions) or
persubject (one model per subject on its own training sessions, applied to
that subject's test session with the TRUE id: an oracle-id setting).

Aligns (signal-level whitening X <- R^-1/2 X; <kind> = euclid | riemann):
  none
  oracle:<kind>     train and test per (subject, session); the test session's
                    reference comes from its own unlabelled windows
  trainonly:<kind>  train per (subject, session); test with the global
                    training reference
  router:<kind>     train per subject (its training sessions pooled); each
                    test window goes to the nearest training subject's
                    reference (Riemannian distance), global reference beyond
                    the training-set distance threshold
  routerb:<kind>    same, one decision per contiguous 64-window test batch
                    (sum of distances), no id needed
  batch:<kind>      train per (subject, session); each contiguous 64-window
                    test batch whitened with its own statistics
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import xsess_lib as L  # noqa: E402

BATCH = 64


def config_key(study, classes, spec, mode, align, seed):
    cls = "all" if classes is None else "-".join(map(str, classes))
    return f"{study}|{cls}|{spec}|{mode}|{align}|{seed}"


# ---------------------------------------------------------------------------
# routing (no ids at test time)
# ---------------------------------------------------------------------------
def fingerprint_dist(covs_tr, subj_tr, covs_te, kind):
    """(n_test, S) Riemannian distance of each test window's covariance to
    each training subject's reference; plus the per-subject refs and the
    training-set threshold (99th pct of own-subject distances)."""
    from pyriemann.utils.distance import distance_riemann
    subjects = np.unique(subj_tr)
    refs = {s: L.mean_cov(covs_tr[subj_tr == s], kind) for s in subjects}
    D = np.stack([distance_riemann(covs_te, refs[s]) for s in subjects], 1)
    own = np.concatenate([distance_riemann(covs_tr[subj_tr == s], refs[s])
                          for s in subjects])
    return subjects, refs, D, float(np.percentile(own, 99))


def contiguous_batches(n, size=BATCH):
    return [np.arange(i, min(i + size, n)) for i in range(0, n, size)]


def aligned_data(d, tr, te, align, cache):
    """Full-size aligned X (train rows and test rows transformed as the
    condition says) + info dict."""
    X, subj, sess = d["X"], d["subj"], d["sess"]
    if align == "none":
        return X, {}
    how, kind = align.split(":")
    if "covs" not in cache:
        cache["covs"] = L.window_covs(X)
    covs = cache["covs"]
    Xa = np.empty_like(X)
    ss = subj * 1000 + sess
    info = {}
    if how in ("oracle", "trainonly", "batch") or how.startswith("online"):
        Xa[tr], _ = L.align_groups(X[tr], ss[tr], kind, covs[tr])
    elif how.split("-")[0] in ("router", "routerb"):
        Xa[tr], Ws = L.align_groups(X[tr], subj[tr], kind, covs[tr])
    else:
        raise ValueError(align)

    te_idx = np.where(te)[0]            # recording order (cache is sorted)
    if how == "oracle":
        Xa[te], _ = L.align_groups(X[te], ss[te], kind, covs[te])
    elif how == "trainonly":
        Xa[te] = L.apply_W(X[te], L.inv_sqrtm(L.mean_cov(covs[tr], kind)))
    elif how == "batch":
        for b in contiguous_batches(len(te_idx)):
            i = te_idx[b]
            Xa[i] = L.apply_W(X[i], L.inv_sqrtm(L.mean_cov(covs[i], kind)))
        info["batch_single_subject"] = float(np.mean(
            [len(np.unique(subj[te_idx[b]])) == 1 for b in contiguous_batches(len(te_idx))]))
    elif how.startswith("online"):
        # online-<N>: causal per-subject re-centring. Test windows arrive in
        # recording order, 64 per predict() call. Each window is routed (log-PSD
        # router); every routed subject keeps a buffer of its last N test-window
        # covariances (current batch included). A subject with < N/4 buffered
        # windows uses its training reference; otherwise the buffer's mean.
        # Transductive (uses unlabelled test windows seen so far): rule-dependent.
        N = int(how.split("-")[1]) if "-" in how else 128
        if "router_psd" not in cache:
            r = L.SubjectRouter("psd", d["meta"]["sfreq"]).fit(X[tr], subj[tr], sess[tr])
            cache["router_psd"] = (r, r.predict_proba(X[te]))
        r, Pr = cache["router_psd"]
        a = r.subjects[Pr.argmax(1)]
        ref_tr = {s: L.mean_cov(covs[tr & (subj == s)], kind) for s in r.subjects}
        buf = {s: [] for s in r.subjects}
        for b in contiguous_batches(len(te_idx)):
            for k in b:
                buf[a[k]].append(te_idx[k])
            for s in np.unique(a[b]):
                buf[s] = buf[s][-N:]
                ref = (ref_tr[s] if len(buf[s]) < N // 4
                       else L.mean_cov(covs[np.array(buf[s])], kind))
                rows = te_idx[b][a[b] == s]
                Xa[rows] = L.apply_W(X[rows], L.inv_sqrtm(ref))
        info.update(router_acc=float(np.mean(a == subj[te_idx])), online_buffer=N)
    else:   # router[-fp] / routerb[-fp]
        base, _, fp = how.partition("-")
        W_glob = L.inv_sqrtm(L.mean_cov(covs[tr], kind))
        if fp:      # subject classifier on a fingerprint; cost = -log posterior
            rkey = f"router_{fp}"
            if rkey not in cache:
                r = L.SubjectRouter(fp, d["meta"]["sfreq"]).fit(X[tr], subj[tr], sess[tr])
                cache[rkey] = (r, r.predict_proba(X[te]))
            r, Pr = cache[rkey]
            subjects, D, thr = r.subjects, -np.log(Pr + 1e-12), -np.log(max(r.thr, 1e-12))
            info["router_oof_acc"] = r.oof_acc
        else:       # Riemannian distance to each subject's mean covariance
            subjects, refs, D, thr = fingerprint_dist(covs[tr], subj[tr], covs[te], kind)
        if base == "router":
            a = subjects[D.argmin(1)]
            fallback = D.min(1) > thr
        else:
            a = np.empty(len(te_idx), dtype=subj.dtype)
            fallback = np.zeros(len(te_idx), bool)
            for b in contiguous_batches(len(te_idx)):
                tot = D[b].sum(0)
                a[b] = subjects[tot.argmin()]
                fallback[b] = np.median(D[b].min(1)) > thr
        for k, i in enumerate(te_idx):
            W = W_glob if fallback[k] else Ws[a[k]]
            Xa[i] = L.apply_W(X[i:i + 1], W)[0]
        info.update(router_acc=float(np.mean(a == subj[te_idx])),
                    router_fallback=float(fallback.mean()),
                    router_thr=float(thr),
                    router_acc_window=float(np.mean(subjects[D.argmin(1)] == subj[te_idx])))
        info["router_assign"] = a.tolist()
    return Xa, info


# ---------------------------------------------------------------------------
def run_config(d, spec, mode, align, seed, cache):
    meta = dict(d["meta"], n_classes=int(d["y"].max() + 1))
    sp = L.xsess_split(d)
    tr, te = sp["train"], sp["test"]
    y, subj, sess = d["y"], d["subj"], d["sess"]
    Xa, info = aligned_data(d, tr, te, align, cache)
    te_idx = np.where(te)[0]
    K = meta["n_classes"]
    P = np.zeros((len(te_idx), K))
    epochs = []

    def fit_predict(train_idx, test_rows, mode_):
        model = L.make_model(spec, meta, seed)
        kw = {}
        if L.is_neural(spec) and spec.startswith("eegnet_st"):
            fi, vi = L.es_split(d, train_idx, mode_, seed)
            kw = dict(Xval=Xa[vi], yval=y[vi])
            train_idx = fi
        model.fit(Xa[train_idx], y[train_idx], **kw)
        if hasattr(model, "best_epoch"):
            epochs.append(int(model.best_epoch))
        return model.predict_proba(Xa[te_idx[test_rows]])

    if mode == "pooled":
        P[:] = fit_predict(np.where(tr)[0], np.arange(len(te_idx)), "pooled")
    elif mode == "persubject":
        for s in np.unique(subj):
            rows = np.where(subj[te_idx] == s)[0]
            P[rows] = fit_predict(np.where(tr & (subj == s))[0], rows, "persubject")
    else:
        raise ValueError(mode)
    sc = L.score(y[te_idx], P.argmax(1), subj[te_idx], sess[te_idx])
    return P, te_idx, sc, info, epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--study", required=True)
    ap.add_argument("--classes", default=None, help="e.g. 0,1,3 (cache label ids)")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--modes", nargs="+", default=["pooled"])
    ap.add_argument("--aligns", nargs="+", default=["none"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[33])
    ap.add_argument("--drop_ch", default="A2-A1", help="comma list of channels to drop")
    args = ap.parse_args()

    classes = None if args.classes is None else [int(c) for c in args.classes.split(",")]
    out = Path.home() / f"codabench/logs/sealed_{args.tag}"
    (out / "probs").mkdir(parents=True, exist_ok=True)
    res_path = out / "results.jsonl"
    done = {r["key"] for r in L.read_results(res_path)}

    d = L.load_study(args.study, classes)
    drop = [c for c in args.drop_ch.split(",") if c in d["meta"]["ch_names"]]
    if drop:
        keep = [i for i, c in enumerate(d["meta"]["ch_names"]) if c not in drop]
        d["X"] = np.ascontiguousarray(d["X"][:, keep])
        d["meta"] = dict(d["meta"], ch_names=[d["meta"]["ch_names"][i] for i in keep])
    sp = L.xsess_split(d)
    n_s = len(np.unique(d["subj"]))
    L.log(f"[data] study={args.study} classes={classes or 'all'} X={d['X'].shape} "
          f"subjects={n_s} sessions/subject={len(np.unique(d['sess']))} "
          f"train={int(sp['train'].sum())} test={int(sp['test'].sum())} "
          f"val_session_windows={int(sp['val'].sum())} K={int(d['y'].max() + 1)} "
          f"class_counts={np.bincount(d['y']).tolist()} dropped_ch={drop} threads={L.N_THREADS}")
    cache = {}
    for spec in args.models:
        seeds = args.seeds if L.is_neural(spec) else [33]
        for mode in args.modes:
            for align in args.aligns:
                for seed in seeds:
                    key = config_key(args.study, classes, spec, mode, align, seed)
                    if key in done:
                        L.log(f"skip {key} (done)")
                        continue
                    t0 = time.time()
                    L.log(f"[fit] {key} X_train={tuple(d['X'][sp['train']].shape)} "
                          f"X_test={tuple(d['X'][sp['test']].shape)}")
                    P, te_idx, sc, info, epochs = run_config(d, spec, mode, align, seed, cache)
                    dt = time.time() - t0
                    h = hashlib.md5(key.encode()).hexdigest()[:12]
                    np.savez_compressed(out / "probs" / f"{h}.npz", P=P, te_idx=te_idx,
                                        key=key, router_assign=np.asarray(info.pop("router_assign", [])))
                    row = dict(key=key, study=args.study, classes=classes, spec=spec,
                               mode=mode, align=align, seed=seed, probs=f"{h}.npz",
                               cell=sc["cell"], pooled=sc["pooled"],
                               per_subject=sc["per_subject"], n_cells=sc["n_cells"],
                               n_train=int(sp["train"].sum()), n_test=int(sp["test"].sum()),
                               shape=list(d["X"].shape), epochs=epochs,
                               seconds=round(dt, 1), time=time.strftime("%H:%M:%S"), **info)
                    # one file per study: two lanes never append to the same file
                    L.append_result(out / f"results_{args.study}.jsonl", row)
                    extra = "".join(f" {k}={v:.3f}" for k, v in info.items()
                                    if isinstance(v, float))
                    L.log(f"[done] {key} cell={sc['cell']:.4f} pooled={sc['pooled']:.4f} "
                          f"epochs={epochs[:9]} {dt:.0f}s{extra}")
    L.log("ALL CONFIGS DONE")


if __name__ == "__main__":
    main()
