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
  router-psdctx:<kind>  (studies with a context column only) as router-psd,
                    but the unit is the (subject, context) pair: training
                    windows are whitened per (subject, context) and the
                    log-PSD router predicts the pair of each test window

Split / data options (shared with sealed_personal.py; defaults = the
proxies' protocol, and a non-default value is appended to the config key):
  --split last | calib:K | replica:K   (xsess_lib.xsess_split)
  --test_subjects 10,11,...            subjects tested under the split
  --chans eeg | eeg+eog | eeg+emg | all   channel types kept (meta ch_types;
                    none -> every channel is EEG), applied before routing
  --pool all | test                    train on everyone, or only on the test
                    subjects' own training windows
  --router_cap 0.5                     cap the log-PSD router's fallback threshold
                    (max posterior) as the solver does, min(thr, cap); default
                    uncapped (committed). The harness threshold saturates at 1.0
                    on the mock and with 1 calibration session, so up to half
                    the test windows fall back: pass 0.5 to describe the solver
  --mmap                               memory-map X (500 Hz); not in the key
                    (values are unchanged)
  --check                              preflight: the [data] line only, rc 1 if a
                    hidden-test row is in the training set (train_sealed.sh /
                    release_ablations.sh run it before any step)
Scores average over subject x session x context cells (one context on the
proxies, so their cells are subject x session as before). On a release cache
the organisers' hidden-test rows (split 2) never train (xsess_lib.xsess_split);
the [data] line shows hidden_in_train (0) and hidden_excluded, and ch_types
(channels per type; source meta, or the solver's name rule).
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
# key options in key order; a value equal to its default is left out of the key
KEY_DEFAULTS = {"split": "last", "test_subjects": None, "chans": "eeg",
                "pool": "all", "wcv": "last", "router_cap": None, "wvariant": "both"}


def config_key(study, classes, spec, mode, align, seed, **opts):
    """study|classes|spec|mode|align|seed, then |name=value for every
    non-default split / data option (so default keys never change)."""
    unknown = set(opts) - set(KEY_DEFAULTS)
    if unknown:
        raise ValueError(f"config_key: unknown options {sorted(unknown)}")
    cls = "all" if classes is None else "-".join(map(str, classes))
    key = f"{study}|{cls}|{spec}|{mode}|{align}|{seed}"
    for k, default in KEY_DEFAULTS.items():
        v = opts.get(k, default)
        if v != default:
            key += f"|{k}={v}"
    return key


# ---------------------------------------------------------------------------
# split / channel options (shared with sealed_personal.py)
# ---------------------------------------------------------------------------
def add_data_args(ap):
    ap.add_argument("--drop_ch", default="A2-A1", help="comma list of channels to drop")
    ap.add_argument("--split", default="last", help="last | calib:K | replica:K")
    ap.add_argument("--test_subjects", default=None,
                    help="comma list of subject indices tested under --split")
    ap.add_argument("--chans", default="eeg", choices=sorted(L.CHAN_SETS),
                    help="channel types kept (meta ch_types; none -> all EEG)")
    ap.add_argument("--pool", default="all", choices=["all", "test"],
                    help="train on every subject, or on the test subjects only")
    ap.add_argument("--router_cap", type=float, default=None,
                    help="cap the log-PSD router's fallback threshold at this max "
                         "posterior (0.5 = the solver's rule); default: uncapped")
    ap.add_argument("--mmap", action="store_true",
                    help="memory-map the cache's X (500 Hz sealed data); values unchanged")


def prepare_data(d, args):
    """Apply --drop_ch / --chans to d (in place), then --split /
    --test_subjects / --pool. Returns (split masks, dropped channel names,
    key options, split_info)."""
    ts = (None if args.test_subjects is None
          else sorted(int(s) for s in args.test_subjects.split(",") if s != ""))
    n_ch, types = d["X"].shape[1], L.ch_type_counts(d["meta"])
    dropped = L.select_channels(d, drop=args.drop_ch.split(","), chans=args.chans)
    sp = L.restrict_pool(d, L.xsess_split(d, args.split, ts), args.pool)
    opts = dict(split=args.split, chans=args.chans, pool=args.pool,
                test_subjects=None if ts is None else ",".join(map(str, ts)),
                router_cap=args.router_cap)
    info = L.split_info(d, sp)
    info.update(n_ch=d["X"].shape[1], n_ch_cache=n_ch, n_ctx=len(np.unique(d["ctx"])),
                ch_types=types)
    return sp, dropped, opts, info


def data_line(args, info):
    """Split / channel part of the [data] log line (ch_types = the cache's
    channels per type and their source, meta | names; hidden_in_train must
    be 0, hidden_excluded = hidden-test rows kept out of train and test)."""
    return (f"split={args.split} pool={args.pool} chans={args.chans}"
            f"({info['n_ch']}/{info['n_ch_cache']} ch) ch_types={info['ch_types']} "
            f"contexts={info['n_ctx']} "
            f"test_subjects={info['test_subjects']} test_sessions={info['test_sessions']} "
            f"test_cells={info['n_test_cells']} hidden_in_train={info['hidden_in_train']}"
            + (f" hidden_excluded={info['hidden_excluded']}" if info["hidden_excluded"] else "")
            + ("" if args.router_cap is None else f" router_cap={args.router_cap}")
            + (" mmap" if getattr(args, "mmap", False) else ""))


def ctx_groups(d, align):
    """Alignment / routing unit of an align option: the subject (grp = subj,
    n_ctx = 1), or for router-psdctx / routerb-psdctx the (subject, context)
    pair, grp = subj * n_ctx + ctx."""
    how = align.split(":")[0]
    if how.partition("-")[2] != "psdctx":
        return d["subj"], 1
    if not d.get("has_ctx"):
        raise ValueError(f"align {align!r} needs a study with a context column")
    n_ctx = int(d["ctx"].max()) + 1
    return d["subj"] * n_ctx + d["ctx"], n_ctx


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


def aligned_data(d, tr, te, align, cache, router_cap=None):
    """Full-size aligned X (train rows and test rows transformed as the
    condition says) + info dict. ``router_cap``: SubjectRouter cap for the
    log-PSD routers built here (None = the committed, uncapped threshold; a
    router already in ``cache`` is used as built)."""
    X, subj, sess = d["X"], d["subj"], d["sess"]
    if align == "none":
        return X, {}
    how, kind = align.split(":")
    t0 = time.time()
    big = X.nbytes * 2 > 512 * 2 ** 20      # > one float64 chunk (500 Hz): heartbeat

    def lap(stage):
        if big:
            L.log(f"  [align] {align}: {stage} ({time.time() - t0:.0f}s)")
    if "covs" not in cache:
        cache["covs"] = L.window_covs(X)
        lap("window covariances")
    covs = cache["covs"]
    Xa = np.empty(X.shape, dtype=X.dtype)
    ss = subj * 1000 + sess
    grp, n_ctx = ctx_groups(d, align)   # subject, or (subject, context) pair
    info = {}
    # align_rows = align_groups on X[tr] without its two train-size copies
    tr_idx = np.where(tr)[0]
    if how in ("oracle", "trainonly", "batch") or how.startswith("online"):
        L.align_rows(X, tr_idx, ss[tr], kind, covs, Xa)
    elif how.split("-")[0] in ("router", "routerb"):
        Ws = L.align_rows(X, tr_idx, grp[tr], kind, covs, Xa)
    else:
        raise ValueError(align)
    lap("training windows whitened")

    te_idx = np.where(te)[0]            # recording order (cache is sorted)
    if how == "oracle":
        L.align_rows(X, te_idx, ss[te], kind, covs, Xa)
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
            r = L.SubjectRouter("psd", d["meta"]["sfreq"], cap=router_cap).fit(
                X[tr], subj[tr], sess[tr])
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
            # (psdctx: the log-PSD router over (subject, context) pairs)
            rkey = f"router_{fp}"
            if rkey not in cache:
                r = L.SubjectRouter("psd" if fp == "psdctx" else fp, d["meta"]["sfreq"],
                                    cap=router_cap).fit(X[tr], grp[tr], sess[tr])
                cache[rkey] = (r, r.predict_proba(X[te]))
                lap(f"{fp} router fitted and applied")
            r, Pr = cache[rkey]
            subjects, D, thr = r.subjects, -np.log(Pr + 1e-12), -np.log(max(r.thr, 1e-12))
            info["router_oof_acc"] = r.oof_acc
            if getattr(r, "cap", None) is not None:     # uncapped, same -log units
                info["router_thr_raw"] = float(-np.log(max(r.thr_raw, 1e-12)))
        else:       # Riemannian distance to each subject's mean covariance
            subjects, refs, D, thr = fingerprint_dist(covs[tr], grp[tr], covs[te], kind)
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
        lap("test windows routed and whitened")
        info.update(router_acc=float(np.mean(a == grp[te_idx])),
                    router_fallback=float(fallback.mean()),
                    router_thr=float(thr),
                    router_acc_window=float(np.mean(subjects[D.argmin(1)] == grp[te_idx])))
        if n_ctx > 1:   # router_acc above is the (subject, context) pair accuracy
            info.update(router_subj_acc=float(np.mean(a // n_ctx == subj[te_idx])),
                        router_ctx_acc=float(np.mean(a % n_ctx == d["ctx"][te_idx])))
        info["router_assign"] = a.tolist()   # psdctx: pair ids subj * n_ctx + ctx
    return Xa, info


# ---------------------------------------------------------------------------
def run_config(d, spec, mode, align, seed, cache, sp=None, router_cap=None):
    """Fit / predict one config. ``sp``: split masks (default: the proxies'
    last-session split); ``router_cap``: see aligned_data."""
    meta = dict(d["meta"], n_classes=int(d["y"].max() + 1))
    sp = L.xsess_split(d) if sp is None else sp
    tr, te = sp["train"], sp["test"]
    y, subj, sess = d["y"], d["subj"], d["sess"]
    Xa, info = aligned_data(d, tr, te, align, cache, router_cap)
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
            if len(rows) == 0:      # a training-only subject (calib / replica split)
                continue
            P[rows] = fit_predict(np.where(tr & (subj == s))[0], rows, "persubject")
    else:
        raise ValueError(mode)
    sc = L.score(y[te_idx], P.argmax(1), subj[te_idx], sess[te_idx], d["ctx"][te_idx])
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
    ap.add_argument("--check", action="store_true",
                    help="preflight: print the [data] line and exit (rc 1 if any "
                         "hidden-test row is in the training set); nothing is fitted")
    add_data_args(ap)
    args = ap.parse_args()

    classes = None if args.classes is None else [int(c) for c in args.classes.split(",")]
    out = Path.home() / f"codabench/logs/sealed_{args.tag}"
    (out / "probs").mkdir(parents=True, exist_ok=True)
    res_path = out / "results.jsonl"
    done = {r["key"] for r in L.read_results(res_path)}

    d = L.load_study(args.study, classes, mmap=args.mmap)
    sp, drop, opts, si = prepare_data(d, args)
    n_s = len(np.unique(d["subj"]))
    L.log(f"[data] study={args.study} classes={classes or 'all'} X={d['X'].shape} "
          f"subjects={n_s} sessions/subject={len(np.unique(d['sess']))} "
          f"train={int(sp['train'].sum())} test={int(sp['test'].sum())} "
          f"val_session_windows={int(sp['val'].sum())} K={int(d['y'].max() + 1)} "
          f"class_counts={np.bincount(d['y']).tolist()} dropped_ch={drop} threads={L.N_THREADS} "
          + data_line(args, si))
    if args.check:
        L.log(f"[check] {'FAIL' if si['hidden_in_train'] else 'OK'}: hidden-test rows in "
              f"train = {si['hidden_in_train']}")
        sys.exit(1 if si["hidden_in_train"] else 0)
    n_tr, n_te = int(sp["train"].sum()), int(sp["test"].sum())
    cache = {}
    for spec in args.models:
        seeds = args.seeds if L.is_neural(spec) else [33]
        for mode in args.modes:
            for align in args.aligns:
                for seed in seeds:
                    key = config_key(args.study, classes, spec, mode, align, seed, **opts)
                    if key in done:
                        L.log(f"skip {key} (done)")
                        continue
                    t0 = time.time()
                    L.log(f"[fit] {key} X_train={(n_tr,) + d['X'].shape[1:]} "
                          f"X_test={(n_te,) + d['X'].shape[1:]}")
                    P, te_idx, sc, info, epochs = run_config(d, spec, mode, align, seed,
                                                             cache, sp, args.router_cap)
                    dt = time.time() - t0
                    h = hashlib.md5(key.encode()).hexdigest()[:12]
                    np.savez_compressed(out / "probs" / f"{h}.npz", P=P, te_idx=te_idx,
                                        key=key, router_assign=np.asarray(info.pop("router_assign", [])))
                    row = dict(key=key, study=args.study, classes=classes, spec=spec,
                               mode=mode, align=align, seed=seed, probs=f"{h}.npz",
                               cell=sc["cell"], pooled=sc["pooled"],
                               per_subject=sc["per_subject"], n_cells=sc["n_cells"],
                               n_train=n_tr, n_test=n_te,
                               shape=list(d["X"].shape), epochs=epochs,
                               seconds=round(dt, 1), time=time.strftime("%H:%M:%S"),
                               **opts, n_ctx=si["n_ctx"], **info)
                    # one file per study: two lanes never append to the same file
                    L.append_result(out / f"results_{args.study}.jsonl", row)
                    extra = "".join(f" {k}={v:.3f}" for k, v in info.items()
                                    if isinstance(v, float))
                    L.log(f"[done] {key} cell={sc['cell']:.4f} pooled={sc['pooled']:.4f} "
                          f"epochs={epochs[:9]} {dt:.0f}s{extra}")
    L.log("ALL CONFIGS DONE")


if __name__ == "__main__":
    main()
