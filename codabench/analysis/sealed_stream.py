#!/usr/bin/env python
"""Sprint 2026-09-28 Phase 3: online re-centring on a realistic test stream.

    python ~/codabench/analysis/sealed_stream.py --study zhou2016 --split calib:1
    python ~/codabench/analysis/sealed_stream.py --study scherer2015 --classes 0,1,3
    python ~/codabench/analysis/sealed_stream.py --report

The sealed test stream holds 3 consecutive test sessions per evaluation
participant, fed to predict() in batches of 64; the proxies test one session.
This script replays the recipe on a proxy stream (log-PSD subject router,
per-subject whitening, blend_calib personalisation: pooled Riemann xDAWN + FB
extractor, per-subject LDA on the shared features, P = w P_pooled +
(1 - w) P_subject, w chosen on training data only) and compares

  clean      router alignment: each test window whitened with its routed
             subject's training reference (stateless, so order-independent:
             computed once per split)
  online-N   RULE-DEPENDENT: each routed subject's reference = mean
             covariance of its last N routed test windows (current batch
             included; training reference until N/4 are seen); training data
             centred per (subject, session). Depends on the stream order.

Splits (--split): last (the harness split: each subject's last session is
the test session) | calib:K (each subject's first K sessions calibrate, all
later sessions form its test stream; Zhou calib:1 = test sessions 1 and 2).
Orders (--orders): rec (recording order), inter (subjects interleaved
round-robin window by window, each subject's own windows in order), shufS
(fully shuffled with seed S). --guard adds online-N+guard rows: a routed
subject's buffer is emptied when the median Riemannian distance of its
windows in the current batch to the buffer mean exceeds the 99th percentile
of the training windows' distances to their own (subject, session) mean.

Reuses the harness (xsess_lib, sealed_run.aligned_data / contiguous_batches,
sealed_personal.blend_cv_folds / select / _lda), so the recording-order rows
on the "last" split reproduce the committed Phase 3 numbers exactly. The
blend weight comes from sealed_personal.choose_w's blend_calib branch
(default --wcv last folds) without the per-subject Riemann models only its
"blend" variant needs (~2x faster; --verify_w recomputes and compares). The
fitted model and every weight-CV fold are checkpointed in state/; rows are
resumable (a key already in results_<study>.jsonl is skipped), so a run cut
by a timeout continues where it stopped. Rows -> ~/codabench/logs/
sprint0928_p3/results_<study>.jsonl, probabilities -> probs/<hash>.npz,
batch composition per order -> orders_<study>_<classes>_<split>.json;
--report writes RESULTS.md and BOOTSTRAP.md there.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

try:                                  # pyriemann >= 0.9; same function as before
    from pyriemann.geometry.distance import distance_riemann
except ImportError:
    from pyriemann.utils.distance import distance_riemann

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import xsess_lib as L  # noqa: E402
import sealed_personal as SP  # noqa: E402
from sealed_run import BATCH, aligned_data, contiguous_batches  # noqa: E402

SPEC = "riemann:xd=1,fb=1"
KIND = "riemann"
FIRST = 32                     # windows counted as "just after the boundary"
OUT = Path.home() / "codabench/logs/sprint0928_p3"
ORDERS = ["rec", "inter", "shuf0", "shuf1", "shuf2"]


# ---------------------------------------------------------------------------
# split and stream order
# ---------------------------------------------------------------------------
def stream_split(d, split):
    """(train, test) boolean masks. last = the harness split; calib:K = each
    subject's first K sessions train, its later sessions are the test stream."""
    if split == "last":
        sp = L.xsess_split(d)
        return sp["train"], sp["test"]
    how, _, k = split.partition(":")
    if how != "calib" or not k.isdigit():
        raise ValueError(split)
    subj, sess = d["subj"], d["sess"]
    tr = np.zeros(len(subj), bool)
    for s in np.unique(subj):
        m = subj == s
        tr[m] = np.isin(sess[m], np.unique(sess[m])[:int(k)])
    return tr, ~tr


def stream_order(subj_te, name):
    """Positions into the test rows (recording order) in the order the stream
    feeds them to predict()."""
    n = len(subj_te)
    if name == "rec":
        return np.arange(n)
    if name == "inter":       # round-robin over subjects, each one in order
        rank = np.zeros(n, np.int64)
        for s in np.unique(subj_te):
            m = subj_te == s
            rank[m] = np.arange(m.sum())
        return np.lexsort((subj_te, rank))
    if name.startswith("shuf"):
        return np.random.default_rng(int(name[4:])).permutation(n)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# online whitening of a stream
# ---------------------------------------------------------------------------
def stream_whiten(X, covs, te_idx, order, a, ref_tr, N, guard_thr=None):
    """Online-N whitening of the test windows fed in ``order`` (positions into
    te_idx), BATCH per predict() call, with routed ids ``a`` (per position).
    Same buffer logic as sealed_run.aligned_data's online-N (bit-identical in
    recording order). guard_thr: before a batch is added, a subject whose
    buffer is active (>= N/4) has it emptied when the median distance of its
    windows in the batch to the buffer mean exceeds guard_thr.
    Returns the whitened test windows (te_idx order) and the reset events
    [routed subject, stream position of the batch start]."""
    Xt = np.empty((len(te_idx),) + X.shape[1:], dtype=X.dtype)
    buf = {s: [] for s in ref_tr}
    resets = []
    for b in contiguous_batches(len(order)):
        pos = order[b]
        subs = np.unique(a[pos])
        if guard_thr is not None:
            for s in subs:
                if len(buf[s]) >= N // 4:
                    rows = pos[a[pos] == s]
                    ref = L.mean_cov(covs[np.array(buf[s])], KIND)
                    if np.median(distance_riemann(covs[te_idx[rows]], ref)) > guard_thr:
                        buf[s] = []
                        resets.append([int(s), int(b[0])])
        for k in pos:
            buf[a[k]].append(te_idx[k])
        for s in subs:
            buf[s] = buf[s][-N:]
            ref = (ref_tr[s] if len(buf[s]) < N // 4
                   else L.mean_cov(covs[np.array(buf[s])], KIND))
            rows = pos[a[pos] == s]
            Xt[rows] = L.apply_W(X[te_idx[rows]], L.inv_sqrtm(ref))
    return Xt, resets


def guard_threshold(covs, tr_idx, subj, sess, pct=99):
    """pct-th percentile of the training windows' Riemannian distances to
    their own (subject, session) mean (training data only)."""
    g = subj[tr_idx] * 1000 + sess[tr_idx]
    dist = [distance_riemann(covs[tr_idx[g == v]], L.mean_cov(covs[tr_idx[g == v]], KIND))
            for v in np.unique(g)]
    return float(np.percentile(np.concatenate(dist), pct))


# ---------------------------------------------------------------------------
# blend_calib: fit once, predict any whitened test stream
# ---------------------------------------------------------------------------
def fit_calib(meta, Xa, y, subj, fit_idx):
    """Pooled Riemann extractor + per-subject calib LDAs on the aligned
    training windows: the calib stack of sealed_personal.riemann_all."""
    K = int(meta["n_classes"])
    pooled = L.make_model(SPEC, meta).fit(Xa[fit_idx], y[fit_idx])
    F = pooled.features(Xa[fit_idx])
    subjects = np.unique(subj[fit_idx])
    ldas = []
    for s in subjects:
        m = subj[fit_idx] == s
        ldas.append(SP._lda(K).fit(F[m], y[fit_idx][m])
                    if len(np.unique(y[fit_idx][m])) == K else None)
    return pooled, subjects, ldas


def calib_fold_scores(d, meta, Xa, fit_idx, val_idx):
    """blend_calib cell score per W_GRID weight on one training-CV fold: the
    calib branch of sealed_personal.choose_w (riemann_all(own_rows_only=True)
    -> select -> blend -> ctx-aware cell score), without the per-subject
    Riemann models that only its "blend" variant needs."""
    y, subj, sess = d["y"], d["subj"], d["sess"]
    ctx = d["ctx"] if d.get("ctx") is not None else np.zeros(len(y), int)
    pooled, subjects, ldas = fit_calib(meta, Xa, y, subj, fit_idx)
    F_val = pooled.features(Xa[val_idx])
    P_pool = pooled.model.parts["lda"].predict_proba(F_val)
    calib = np.repeat(P_pool[None], len(subjects), 0)
    for k, (s, lda) in enumerate(zip(subjects, ldas)):
        rows = subj[val_idx] == s
        if lda is not None and rows.any():
            calib[k, rows] = lda.predict_proba(F_val[rows])
    P_p = SP.select(calib, subjects, subj[val_idx], P_pool)
    return [L.score(y[val_idx], (w * P_pool + (1 - w) * P_p).argmax(1), subj[val_idx],
                    sess[val_idx], ctx[val_idx])["cell"] for w in SP.W_GRID]


def choose_w_calib(scores):
    """Mean fold scores -> (w, cv); ties -> the larger (more pooled) w, as
    sealed_personal.choose_w."""
    sc = np.mean(scores, 0)
    k = max(range(len(SP.W_GRID)), key=lambda i: (round(sc[i], 6), SP.W_GRID[i]))
    return SP.W_GRID[k], sc.tolist()


def fit_state(d, meta, Xa, tr_idx, path, sig):
    """Model (fit_calib on every training window) + blend_calib weight chosen on
    the training CV folds of sealed_personal.blend_cv_folds (default --wcv
    last), checkpointed to ``path`` after the fit and after every fold."""
    st = joblib.load(path) if path.exists() else {}
    if st.get("sig") != sig:
        st = {"sig": sig, "fit_seconds": 0.0}
    elif "w" in st:
        L.log(f"  state loaded: {path.name} (w={st['w']})")
        return st
    y, subj = d["y"], d["subj"]
    if "parts" not in st:
        t0 = time.time()
        pooled, subjects, ldas = fit_calib(meta, Xa, y, subj, tr_idx)
        st.update(parts=pooled.model.parts, subjects=subjects, ldas=ldas,
                  fit_seconds=round(time.time() - t0, 1), fold_scores=[])
        joblib.dump(st, path)
        L.log(f"  model fitted ({st['fit_seconds']:.0f}s) -> {path.name}")
    folds = SP.blend_cv_folds(d, tr_idx)
    for i, (fit_idx, val_idx) in enumerate(folds):
        if i < len(st["fold_scores"]):
            continue
        t0 = time.time()
        st["fold_scores"].append(calib_fold_scores(d, meta, Xa, fit_idx, val_idx))
        st["fit_seconds"] = round(st["fit_seconds"] + time.time() - t0, 1)
        joblib.dump(st, path)
        L.log(f"  weight CV fold {i + 1}/{len(folds)}: val sessions "
              f"{np.unique(d['sess'][val_idx]).tolist()} n_fit={len(fit_idx)} "
              f"n_val={len(val_idx)} blend_calib cell for w={SP.W_GRID}: "
              f"{np.round(st['fold_scores'][-1], 3).tolist()} ({time.time() - t0:.0f}s)")
    st["w"], st["cv"] = choose_w_calib(st["fold_scores"])
    joblib.dump(st, path)
    L.log(f"  blend_calib: w_pooled={st['w']} (training CV {np.round(st['cv'], 3).tolist()}); "
          f"fit + weight CV {st['fit_seconds']:.0f}s")
    return st


def pooled_model(meta, parts):
    """RiemannModel around fitted parts (as RiemannModel.fit builds it)."""
    m = L.make_model(SPEC, meta)
    pre = m.RS.WindowPreproc(meta["ch_names"], meta["sfreq"], "none", m.reference)
    m.model = m.RS.RiemannStepTypeModel(parts, nfilter=4, estimator=m.estimator,
                                        preproc=pre, **m.kw)
    return m


def blend_calib_probs(state, model, Xte, rid, r_fb):
    """blend_calib with router ids (the rejected windows get pooled), exactly
    as sealed_personal builds its router-id blend_calib row."""
    F = model.features(Xte)
    P_pool = state["parts"]["lda"].predict_proba(F)
    calib = np.repeat(P_pool[None], len(state["subjects"]), 0)
    for k, lda in enumerate(state["ldas"]):
        if lda is not None:
            calib[k] = lda.predict_proba(F)
    w = state["w"]
    return SP.select(w * P_pool[None] + (1 - w) * calib, state["subjects"], rid, P_pool, r_fb)


def verify_w(d, meta, Xa, tr_idx, st):
    """Recompute the calib-only weight CV and compare it with the state's. A
    state without "fold_scores" was written by sealed_personal.choose_w itself
    (first version of this script), so on such a state this checks that the
    calib-only CV and choose_w agree."""
    t0 = time.time()
    scores = [calib_fold_scores(d, meta, Xa, f, v) for f, v in SP.blend_cv_folds(d, tr_idx)]
    w, cv = choose_w_calib(scores)
    diff = float(np.max(np.abs(np.subtract(cv, st["cv"]))))
    L.log(f"[check] calib-only weight CV vs state: w {w} vs {st['w']}, max |cv diff| = "
          f"{diff:.2e} -> {'SAME' if w == st['w'] and diff < 1e-9 else 'DIFFERENT'} "
          f"({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def boundary_stats(y, yhat, subj, sess, order):
    """Per subject with >= 2 test sessions: accuracy on the first FIRST
    windows of its second test session (stream order), on the rest of that
    session and on all of it. None when the sessions are mixed (shuffles)."""
    pos = np.empty(len(order), np.int64)
    pos[order] = np.arange(len(order))
    out = {}
    for s in np.unique(subj):
        vs = np.unique(sess[subj == s])
        if len(vs) < 2:
            continue
        r1 = np.where((subj == s) & (sess == vs[0]))[0]
        r2 = np.where((subj == s) & (sess == vs[1]))[0]
        if pos[r1].max() > pos[r2].min():
            return None
        r2 = r2[np.argsort(pos[r2])]
        ok = yhat[r2] == y[r2]
        out[str(int(s))] = dict(first=float(ok[:FIRST].mean()), rest=float(ok[FIRST:].mean()),
                                session=float(ok.mean()), n=int(len(r2)))
    return out or None


def score_all(d, te_idx, P, order):
    y, subj, sess = d["y"][te_idx], d["subj"][te_idx], d["sess"][te_idx]
    ctx = d["ctx"][te_idx] if d.get("ctx") is not None else np.zeros(len(te_idx), int)
    yhat = P.argmax(1)
    sc = L.score(y, yhat, subj, sess, ctx)
    per_session, per_ss = {}, {}
    for v in np.unique(sess):
        m = sess == v
        s2 = L.score(y[m], yhat[m], subj[m], sess[m], ctx[m])
        per_session[str(int(v))] = s2["cell"]
        for s, val in s2["per_subject"].items():
            per_ss.setdefault(str(s), {})[str(int(v))] = val
    return dict(cell=sc["cell"], pooled=sc["pooled"], per_subject=sc["per_subject"],
                n_cells=sc["n_cells"], per_session=per_session, per_subject_session=per_ss,
                boundary=boundary_stats(y, yhat, subj, sess, order))


# ---------------------------------------------------------------------------
def run_study(args):
    classes = None if args.classes is None else [int(c) for c in args.classes.split(",")]
    cls = "all" if classes is None else "-".join(map(str, classes))
    (OUT / "probs").mkdir(parents=True, exist_ok=True)
    (OUT / "state").mkdir(parents=True, exist_ok=True)
    res_path = OUT / f"results_{args.study}.jsonl"
    done = {r["key"] for r in L.read_results(res_path)}

    d = L.load_study(args.study, classes)
    drop = [c for c in args.drop_ch.split(",") if c in d["meta"]["ch_names"]]
    if drop:
        keep = [i for i, c in enumerate(d["meta"]["ch_names"]) if c not in drop]
        d["X"] = np.ascontiguousarray(d["X"][:, keep])
        d["meta"] = dict(d["meta"], ch_names=[d["meta"]["ch_names"][i] for i in keep])
    meta = dict(d["meta"], n_classes=int(d["y"].max() + 1))
    X, y, subj, sess = d["X"], d["y"], d["subj"], d["sess"]
    tr, te = stream_split(d, args.split)
    tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
    n_te_sess = sorted({len(np.unique(sess[te & (subj == s)])) for s in np.unique(subj)})
    L.log(f"[data] study={args.study} classes={classes or 'all'} split={args.split} "
          f"X={X.shape} subjects={len(np.unique(subj))} train={len(tr_idx)} test={len(te_idx)} "
          f"train_sessions={sorted(np.unique(sess[tr]).tolist())} "
          f"test_sessions={sorted(np.unique(sess[te]).tolist())} test_sessions/subject={n_te_sess} "
          f"K={meta['n_classes']} class_counts={np.bincount(y).tolist()} dropped_ch={drop} "
          f"threads={L.N_THREADS}")
    base = f"{args.study}|{cls}|{args.split}"
    sig = f"{base}|{SPEC}|{X.shape}|{len(tr_idx)}|{len(te_idx)}"

    t0 = time.time()
    router = L.SubjectRouter("psd", meta["sfreq"]).fit(X[tr], subj[tr], sess[tr])
    Pr = router.predict_proba(X[te])
    rid = router.subjects[Pr.argmax(1)]
    r_fb = Pr.max(1) < router.thr
    racc = float(np.mean(rid == subj[te_idx]))
    racc_sess = {str(int(v)): float(np.mean((rid == subj[te_idx])[sess[te_idx] == v]))
                 for v in np.unique(sess[te_idx])}
    L.log(f"[router] psd: acc={racc:.3f} fallback={r_fb.mean():.3f} thr={router.thr:.3f} "
          f"acc per test session={ {k: round(v, 3) for k, v in racc_sess.items()} } "
          f"({time.time() - t0:.0f}s)")
    covs = L.window_covs(X)
    cache = {"covs": covs, "router_psd": (router, Pr)}
    common = dict(study=args.study, classes=classes, split=args.split, spec=SPEC + "/blend_calib",
                  mode="router-id", n_train=len(tr_idx), n_test=len(te_idx), shape=list(X.shape),
                  router_psd_acc=racc, router_psd_fallback=float(r_fb.mean()),
                  router_psd_thr=float(router.thr), router_psd_acc_session=racc_sess)

    def write(key, P, order, extra, dt):
        sc = score_all(d, te_idx, P, order)
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        np.savez_compressed(OUT / "probs" / f"{h}.npz", P=P, te_idx=te_idx, order=order, key=key)
        L.append_result(res_path, dict(key=key, **common, **extra, **sc, probs=f"{h}.npz",
                                       seconds=round(dt, 1), time=time.strftime("%H:%M:%S")))
        L.log(f"[done] {key} cell={sc['cell']:.4f} pooled={sc['pooled']:.4f} "
              f"per_session={ {k: round(v, 3) for k, v in sc['per_session'].items()} } {dt:.0f}s")

    # clean: router alignment (sealed_run.aligned_data), stateless per window
    Xa_c, _ = aligned_data(d, tr, te, "router-psd:riemann", cache)
    L.log(f"[fit] clean {base} X_train={tuple(X[tr].shape)} X_test={tuple(X[te].shape)}")
    stem = OUT / "state" / base.replace("|", "_").replace(":", "")
    st_c = fit_state(d, meta, Xa_c, tr_idx, Path(f"{stem}_clean.joblib"), sig + "|clean")
    if args.verify_w:
        verify_w(d, meta, Xa_c, tr_idx, st_c)
    m_c = pooled_model(meta, st_c["parts"])
    key = f"{base}|clean|-|-|-"
    if key not in done:
        t1 = time.time()
        P = blend_calib_probs(st_c, m_c, Xa_c[te_idx], rid, r_fb)
        write(key, P, np.arange(len(te_idx)),
              dict(cond="clean", order="-", buffer=None, guard=False, align="router-psd:riemann",
                   blend_calib_w=st_c["w"], blend_calib_cv=st_c["cv"],
                   fit_seconds=st_c["fit_seconds"]), time.time() - t1)

    # online: training centred per (subject, session); test whitened by stream
    ss = subj * 1000 + sess
    Xa_o = np.array(Xa_c)
    Xa_o[tr_idx], _ = L.align_groups(X[tr_idx], ss[tr_idx], KIND, covs[tr_idx])
    if np.array_equal(Xa_o[tr_idx], Xa_c[tr_idx]):      # one training session each
        L.log("  online training alignment == clean (1 training session/subject): reuse the fit")
        st_o, m_o = st_c, m_c
    else:
        L.log(f"[fit] online {base} (training centred per (subject, session))")
        st_o = fit_state(d, meta, Xa_o, tr_idx, Path(f"{stem}_online.joblib"), sig + "|online")
        if args.verify_w:
            verify_w(d, meta, Xa_o, tr_idx, st_o)
        m_o = pooled_model(meta, st_o["parts"])
    ref_tr = {s: L.mean_cov(covs[tr & (subj == s)], KIND) for s in router.subjects}
    thr = guard_threshold(covs, tr_idx, subj, sess) if args.guard else None
    if thr is not None:
        L.log(f"  guard threshold (99th pct within-session training distance) = {thr:.3f}")

    # batch composition per order (what the online buffer sees per predict() call)
    ost_path = OUT / f"orders_{stem.name}.json"
    ost = json.loads(ost_path.read_text()) if ost_path.exists() else {}
    for oname in args.orders:
        order = stream_order(subj[te_idx], oname)
        bs = [order[b] for b in contiguous_batches(len(te_idx))]
        ost[oname] = dict(
            single_subject=float(np.mean([len(np.unique(subj[te_idx][p])) == 1 for p in bs])),
            routed_subjects=float(np.mean([len(np.unique(rid[p])) for p in bs])),
            windows_per_routed_subject=float(np.mean([len(p) / len(np.unique(rid[p])) for p in bs])))
    ost_path.write_text(json.dumps(ost, indent=1))

    for oname in args.orders:
        order = stream_order(subj[te_idx], oname)
        for N in args.buffers:
            for g in ([False, True] if args.guard else [False]):
                key = f"{base}|online|{oname}|{N}|{'guard' if g else '-'}"
                if key in done:
                    L.log(f"skip {key} (done)")
                    continue
                t1 = time.time()
                Xte, resets = stream_whiten(X, covs, te_idx, order, rid, ref_tr, N,
                                            thr if g else None)
                for ev in resets:     # + majority TRUE subject / session of the windows
                    p = order[ev[1]:ev[1] + BATCH]
                    p = p[rid[p] == ev[0]]
                    ev += [int(np.bincount(subj[te_idx][p]).argmax()),
                           int(np.bincount(sess[te_idx][p]).argmax())]
                if oname == "rec" and N == 64 and not g:
                    ref, _ = aligned_data(d, tr, te, "online-64:riemann", cache)
                    same = np.array_equal(ref[te_idx], Xte) and np.array_equal(ref[tr_idx],
                                                                                Xa_o[tr_idx])
                    L.log(f"[check] rec-order online-64 whitening == sealed_run.aligned_data: {same}")
                P = blend_calib_probs(st_o, m_o, Xte, rid, r_fb)
                write(key, P, order,
                      dict(cond="online", order=oname, buffer=N, guard=g,
                           align=f"online-{N}:riemann", blend_calib_w=st_o["w"],
                           blend_calib_cv=st_o["cv"], fit_seconds=st_o["fit_seconds"],
                           guard_thr=thr if g else None, resets=resets),
                      time.time() - t1)
    L.log("ALL CONFIGS DONE")


# ---------------------------------------------------------------------------
# report: RESULTS.md + BOOTSTRAP.md
# ---------------------------------------------------------------------------
PROXIES = [("tangermann2012", None, "last", "Tangermann (4 cl., last session)"),
           ("scherer2015", [0, 1, 3], "last", "Scherer 3-class (WORD/SUB/HAND, last session)"),
           ("zhou2016", None, "last", "Zhou (3 cl., last session)"),
           ("zhou2016", None, "calib:1", "Zhou stream (calib session 0; test sessions 1+2)")]
COMMITTED = {  # proxy -> (tag, study, cls) of the committed Phase 3 / 4 rows
    ("tangermann2012", None, "last"): ("p3b", "tangermann2012", "all"),
    ("scherer2015", (0, 1, 3), "last"): ("p4", "scherer2015", "0-1-3"),
    ("zhou2016", None, "last"): ("p3b", "zhou2016", "all")}


def _pkey(r):
    return (r["study"], tuple(r["classes"]) if r["classes"] else None, r["split"])


def _order_scores(rows, order):
    """(cell, {subject: score}) of one order; shuf = mean over the shuffle seeds."""
    if order != "shuf":
        r = rows.get(order)
        return (r["cell"], r["per_subject"]) if r else (None, None)
    rs = [v for k, v in rows.items() if k.startswith("shuf")]
    if not rs:
        return None, None
    subs = rs[0]["per_subject"].keys()
    return (float(np.mean([r["cell"] for r in rs])),
            {s: float(np.mean([r["per_subject"][s] for r in rs])) for s in subs})


def _boot(a, b, seed=0, n=10000):
    """Paired subject bootstrap of mean(a - b): mean, 95 % CI, #improved, #subjects."""
    subs = sorted(set(a) & set(b), key=int)
    dd = np.array([a[s] - b[s] for s in subs])
    boots = dd[np.random.default_rng(seed).integers(0, len(dd), (n, len(dd)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(dd.mean()), float(lo), float(hi), int((dd > 0).sum()), len(dd)


def report():
    rows = []
    for p in sorted(OUT.glob("results_*.jsonl")):
        for line in p.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:      # an interrupted append
                continue
    rows = list({r["key"]: r for r in rows}.values())
    ostats = {}
    for p in OUT.glob("orders_*.json"):
        ostats[p.stem[len("orders_"):]] = json.loads(p.read_text())
    by = {}
    for r in rows:
        k = _pkey(r)
        by.setdefault(k, {"clean": None, "online": {}, "guard": {}})
        if r["cond"] == "clean":
            by[k]["clean"] = r
        else:
            by[k]["guard" if r["guard"] else "online"].setdefault(r["buffer"], {})[r["order"]] = r
    committed = {}
    for pk, (tag, st, cls) in COMMITTED.items():
        for r in L.read_results(Path.home() / f"codabench/logs/sealed_{tag}/results.jsonl"):
            for al in ("router-psd:riemann", "online-64:riemann"):
                if r["key"] == f"{st}|{cls}|{SPEC}/blend_calib|router-id|{al}|33":
                    committed[(pk, al)] = r

    R, B = [], []
    R += ["# Sprint 0928 Phase 3: online re-centring on a realistic test stream", "",
          "Metric: **cell-averaged balanced accuracy** (mean over subject × test-session cells) "
          "unless a column says otherwise. Splits: *last* = each subject's last session is "
          "the test session (the harness split); *stream* = Zhou calibrated on session 0 only, "
          "test sessions 1 and 2 fed in recording order (the sealed structure has 3 "
          "consecutive test sessions). Recipe: Riemann xDAWN + FB, log-PSD router, "
          "blend_calib with router ids, w chosen on training data only. **clean** = router "
          "alignment (stateless per window; the recipe default). **online-N (RULE-DEPENDENT)** "
          "= each routed subject's whitening reference is the mean covariance of its last N "
          "routed test windows (training reference until N/4 are seen), fed 64 per predict() "
          "call; never a default. Single deterministic fits; uncertainty is across subjects "
          "(BOOTSTRAP.md). Code: `analysis/sealed_stream.py`.", ""]

    # sanity vs committed
    R += ["## Sanity: reproduction of the committed Phase 3 rows (last split, recording order)", "",
          "| proxy | clean committed | clean here | online-64 committed (rule-dep.) | "
          "online-64 here (rule-dep.) | max abs. diff | w clean / online |", "|---|---|---|---|---|---|---|"]
    for pk, (tag, st, cls) in COMMITTED.items():
        g = by.get(pk)
        c0, o0 = committed.get((pk, "router-psd:riemann")), committed.get((pk, "online-64:riemann"))
        c1 = g["clean"] if g else None
        o1 = g["online"].get(64, {}).get("rec") if g else None
        vals = [x["cell"] if x else None for x in (c0, c1, o0, o1)]
        diff = (max(abs(vals[0] - vals[1]), abs(vals[2] - vals[3]))
                if None not in vals else None)
        name = next(p[3] for p in PROXIES if p[:3] == (pk[0], list(pk[1]) if pk[1] else None, pk[2]))
        f = lambda v: "n/a" if v is None else f"{v:.3f}"
        wtxt = (f"{c1['blend_calib_w']} / {o1['blend_calib_w']}" if c1 and o1 else "")
        R.append(f"| {name} | {f(vals[0])} | {f(vals[1])} | {f(vals[2])} | {f(vals[3])} | "
                 f"{'n/a' if diff is None else f'{diff:.4f}'} | {wtxt} |")
    R.append("")

    # 3a stream
    pk3 = ("zhou2016", None, "calib:1")
    g3 = by.get(pk3)
    lag = None
    if g3 and g3["clean"]:
        c = g3["clean"]
        sess_ids = sorted(c["per_session"], key=int)
        subs = sorted(c["per_subject"], key=int)
        R += ["## 3a: multi-session stream (Zhou, calibrate on session 0, test sessions 1 → 2, "
              "recording order)", "",
              f"n_train={c['n_train']}, n_test={c['n_test']}, router accuracy "
              f"{c['router_psd_acc']:.3f} (per test session: "
              + ", ".join(f"session {k} {v:.3f}" for k, v in
                          sorted(c.get("router_psd_acc_session", {}).items()))
              + f"; chance {1 / len(subs):.2f}), fallback to pooled "
              f"{c['router_psd_fallback']:.3f} (threshold {c.get('router_psd_thr', float('nan')):.3f}: "
              "with one calibration session the out-of-fold posteriors saturate). "
              f"blend_calib w = {c['blend_calib_w']} (training chronological halves, "
              f"CV {[round(v, 3) for v in c['blend_calib_cv']]}). The router is the same for "
              "clean and online, so the comparison is fair, but on session 2 most windows "
              "go to the wrong subject's reference and personal LDA in both conditions.", "",
              "Cell score overall and per test session; per-subject = mean of its 2 cells. "
              "Online columns are rule-dependent; resets = buffer resets by the guard.", "",
              "| variant | cell (all) | " + " | ".join(f"session {v}" for v in sess_ids) + " | "
              + " | ".join(f"subj {s}" for s in subs) + " | resets |",
              "|---|---|" + "---|" * (len(sess_ids) + len(subs) + 1)]
        variants = [("clean", c)]
        for N in sorted(g3["online"]):
            if "rec" in g3["online"][N]:
                variants.append((f"online-{N} (rule-dep.)", g3["online"][N]["rec"]))
        for N in sorted(g3["guard"]):
            if "rec" in g3["guard"][N]:
                variants.append((f"online-{N} + guard (rule-dep.)", g3["guard"][N]["rec"]))
        for name, r in variants:
            R.append(f"| {name} | {r['cell']:.3f} | "
                     + " | ".join(f"{r['per_session'][v]:.3f}" for v in sess_ids) + " | "
                     + " | ".join(f"{r['per_subject'][s]:.3f}" for s in subs) + " | "
                     + ("" if not r.get("guard") else str(len(r.get("resets", [])))) + " |")
        R += ["", f"Session-boundary check (second test session = session {sess_ids[-1]}): "
              f"plain **accuracy** on its first {FIRST} windows in stream order vs the rest "
              "of that session and the whole session, mean over subjects (per subject in "
              "brackets); first − session mean < 0 would be a post-boundary drop; gain = "
              "variant − clean on the same windows; the last two columns use the session's "
              "balanced accuracy.", "",
              f"| variant | first {FIRST} | rest | session mean | first − session mean | "
              f"gain first {FIRST} | gain rest | session-{sess_ids[-1]} BA gain per subject | "
              "subjects with gain < 0 |",
              "|---|---|---|---|---|---|---|---|---|"]
        cb = c["boundary"]
        for name, r in variants:
            bd = r["boundary"]
            if not bd:
                continue
            fm = np.mean([bd[s]["first"] for s in subs])
            rm = np.mean([bd[s]["rest"] for s in subs])
            sm = np.mean([bd[s]["session"] for s in subs])
            gf = np.mean([bd[s]["first"] - cb[s]["first"] for s in subs])
            gr = np.mean([bd[s]["rest"] - cb[s]["rest"] for s in subs])
            v2 = sess_ids[-1]
            g2 = {s: r["per_subject_session"][s][v2] - c["per_subject_session"][s][v2]
                  for s in subs}
            neg = sum(v < 0 for v in g2.values())
            per = "; ".join(f"{bd[s]['first']:.2f}" for s in subs)
            R.append(f"| {name} | {fm:.3f} ({per}) | {rm:.3f} | {sm:.3f} | {fm - sm:+.3f} | "
                     f"{gf:+.3f} | {gr:+.3f} | "
                     + ", ".join(f"{g2[s]:+.3f}" for s in subs) + f" | "
                     f"{'-' if name == 'clean' else f'{neg}/{len(subs)}'} |")
            if name.startswith("online-64 ("):
                lag = dict(neg=neg, n=len(subs), drop=sm - fm, first=fm, sess=sm,
                           clean_drop=np.mean([cb[s]["session"] - cb[s]["first"] for s in subs]))
        R.append("")

    # 3b order robustness
    R += ["## 3b: order robustness (online-N minus clean, cell score)", "",
          "Orders: **rec** = recording order; **inter** = subjects interleaved round-robin "
          "window by window, each subject's windows in order; **shuf** = fully shuffled, "
          "mean ± SD over seeds 0, 1, 2 (sessions mixed on the stream). The clean score is "
          "the same for every order. Retained = inter gain / rec gain. All online columns "
          "are rule-dependent.", "",
          "| proxy | N | clean | online rec | online inter | online shuf | gain rec | "
          "gain inter | gain shuf | retained (inter) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    robust = {}
    for study, cls, split, name in PROXIES:
        g = by.get((study, tuple(cls) if cls else None, split))
        if not g or not g["clean"]:
            continue
        c = g["clean"]["cell"]
        for N in sorted(g["online"]):
            o = g["online"][N]
            rec, inter = o.get("rec"), o.get("inter")
            shuf = [v["cell"] for k, v in o.items() if k.startswith("shuf")]
            f = lambda r: "" if r is None else f"{r['cell']:.3f}"
            gr = rec["cell"] - c if rec else None
            gi = inter["cell"] - c if inter else None
            gs = np.mean(shuf) - c if shuf else None
            ret = (gi / gr if (gr is not None and gi is not None and gr > 0) else None)
            if N == 64:
                robust[name] = (gr, gi, ret)
            R.append(f"| {name} | {N} | {c:.3f} | {f(rec)} | {f(inter)} | "
                     + (f"{np.mean(shuf):.3f} ± {np.std(shuf):.3f} (n{len(shuf)})" if shuf else "")
                     + " | " + " | ".join("" if v is None else f"{v:+.3f}" for v in (gr, gi, gs))
                     + f" | {'n/a (rec gain ≤ 0)' if ret is None and gr is not None else ('' if ret is None else f'{ret:.0%}')} |")
    R += ["", "Batch composition per order (64 windows per predict() call): share of batches "
          "holding one true subject, mean number of routed subjects per batch and mean "
          "windows per routed subject per batch. In recording order a batch is mostly one "
          "subject, so online-64's buffer is essentially the current batch (its own "
          "statistics, look-ahead within the batch); interleaved or shuffled, each subject "
          "gets a few windows per batch and its buffer is mostly past windows.", "",
          "| proxy | order | single-subject batches | routed subjects / batch | "
          "windows / routed subject |", "|---|---|---|---|---|"]
    for study, cls, split, name in PROXIES:
        stem = f"{study}_{'all' if cls is None else '-'.join(map(str, cls))}_{split}".replace(":", "")
        for oname, v in ostats.get(stem, {}).items():
            R.append(f"| {name} | {oname} | {v['single_subject']:.2f} | "
                     f"{v['routed_subjects']:.1f} | {v['windows_per_routed_subject']:.1f} |")
    guard_rows = []
    for study, cls, split, name in PROXIES:
        g = by.get((study, tuple(cls) if cls else None, split))
        for N in sorted(g["guard"]) if g else []:
            for oname in [o for o in ORDERS if o in g["guard"][N]]:
                r, base_r = g["guard"][N][oname], g["online"].get(N, {}).get(oname)
                if base_r:
                    guard_rows.append(
                        f"| {name} | {N} | {oname} | {base_r['cell']:.3f} | {r['cell']:.3f} | "
                        f"{r['cell'] - base_r['cell']:+.3f} | {len(r.get('resets', []))} | "
                        f"{r.get('guard_thr', float('nan')):.3f} |")
    if guard_rows:
        R += ["", "Buffer-reset guard (rule-dependent, like online): a routed subject's buffer "
              "is emptied when the median Riemannian distance of its windows in the current "
              "batch to the buffer mean exceeds the 99th percentile of training windows' "
              "distances to their own (subject, session) mean (training data only).", "",
              "| proxy | N | order | online | online + guard | guard − online | resets | "
              "threshold |", "|---|---|---|---|---|---|---|---|"] + guard_rows
        ev = []
        for N, orows in sorted((g3 or {}).get("guard", {}).items()):
            for oname in [o for o in ORDERS if o in orows]:
                for e in orows[oname].get("resets", []):
                    if len(e) == 4:
                        ev.append(f"N={N} {oname}: routed subject {e[0]} at stream position "
                                  f"{e[1]}, its windows in that batch mostly true subject "
                                  f"{e[2]}, session {e[3]}"
                                  + (" (a routing error, not a session change)"
                                     if e[2] != e[0] else ""))
        if ev:
            R += ["", "Guard resets on the Zhou stream: " + "; ".join(ev) + "."]
    R.append("")

    # bootstrap
    B += ["# Sprint 0928 Phase 3: paired subject bootstrap", "",
          "Per-subject score = mean of the subject's cells (cell-averaged balanced "
          "accuracy); for shuf, the mean over seeds 0-2 first. A − B = mean per-subject "
          "difference with a 95 % percentile CI (10,000 resamples over subjects, "
          "`np.random.default_rng(0)` per comparison), and the number of subjects with "
          "A > B. Online rows are rule-dependent.", "",
          "| proxy | A | B | order | A cell | B cell | A − B (95 % CI) | subjects A > B |",
          "|---|---|---|---|---|---|---|---|"]
    boot = {}
    for study, cls, split, name in PROXIES:
        g = by.get((study, tuple(cls) if cls else None, split))
        if not g or not g["clean"]:
            continue
        c = g["clean"]
        for N in sorted(g["online"]):
            for od in ("rec", "inter", "shuf"):
                cell, ps = _order_scores(g["online"][N], od)
                if cell is None:
                    continue
                m, lo, hi, k, n = _boot(ps, c["per_subject"])
                boot[(name, N, od)] = (m, lo, hi, k, n)
                B.append(f"| {name} | online-{N} | clean | {od} | {cell:.3f} | {c['cell']:.3f} | "
                         f"{m:+.3f} ({lo:+.3f}, {hi:+.3f}) | {k}/{n} |")
        for N in sorted(g["guard"]):
            for od in ("rec", "inter", "shuf"):
                ca, pa = _order_scores(g["guard"][N], od)
                cb_, pb = _order_scores(g["online"].get(N, {}), od)
                if ca is None or cb_ is None:
                    continue
                m, lo, hi, k, n = _boot(pa, pb)
                boot[(name, N, "guard", od)] = (m, lo, hi, k, n)
                B.append(f"| {name} | online-{N}+guard | online-{N} | {od} | {ca:.3f} | "
                         f"{cb_:.3f} | {m:+.3f} ({lo:+.3f}, {hi:+.3f}) | {k}/{n} |")
    if g3 and g3["clean"]:
        v2 = sorted(g3["clean"]["per_session"], key=int)[-1]
        B += ["", f"Zhou stream, second test session (session {v2}) only, recording order:", "",
              "| A | B | A − B (95 % CI) | subjects A > B |", "|---|---|---|---|"]
        cps = {s: v[v2] for s, v in g3["clean"]["per_subject_session"].items()}
        for N in sorted(g3["online"]):
            r = g3["online"][N].get("rec")
            if r:
                m, lo, hi, k, n = _boot({s: v[v2] for s, v in r["per_subject_session"].items()}, cps)
                B.append(f"| online-{N} | clean | {m:+.3f} ({lo:+.3f}, {hi:+.3f}) | {k}/{n} |")
    B.append("")

    # decisions
    R += ["## Pre-registered decisions (brief § 5, Phase 3)", ""]
    if robust:
        ok = {k: (v[2] is not None and v[2] >= 0.75) for k, v in robust.items()}
        R.append("- **Order robustness** (online-64 keeps ≥ 75 % of its recording-order gain "
                 "under the interleaved order on every proxy): "
                 + "; ".join(f"{k}: rec {v[0]:+.3f}, inter {v[1]:+.3f}, "
                             + ("retained n/a (no rec gain)" if v[2] is None else f"retained {v[2]:.0%}")
                             for k, v in robust.items())
                 + f". → **{'order-robust' if all(ok.values()) else 'NOT order-robust'}**"
                 + ("" if all(ok.values()) else
                    f" (fails on: {', '.join(k for k, v in ok.items() if not v)})") + ".")
    trig = None
    if lag:
        trig = lag["neg"] >= 3 or lag["drop"] > 0.05
        R.append(f"- **Session-boundary lag** (3a, online-64, recording order): second-session "
                 f"gain < 0 for {lag['neg']}/{lag['n']} subjects (trigger ≥ 3); first-{FIRST} "
                 f"accuracy {lag['first']:.3f} vs session mean {lag['sess']:.3f}, i.e. "
                 + (f"a drop of {lag['drop'] * 100:.1f} points" if lag["drop"] > 0 else
                    f"{-lag['drop'] * 100:.1f} points above the mean (no drop)")
                 + f" (trigger: drop > 5 points; clean on the same windows: "
                 f"{-lag['clean_drop'] * 100:+.1f} points vs its session mean). → **"
                 + ("session-boundary lag risk" if trig else "no session-boundary lag risk") + "**.")
    gk = ("Zhou stream (calib session 0; test sessions 1+2)", 64, "guard", "rec")
    if gk in boot:
        m, lo, hi, k, n = boot[gk]
        adopt = m >= 0.02 and lo > 0
        R.append("- **Buffer-reset guard** (online-64 + guard vs online-64 on the stream, "
                 f"recording order; adopt if ≥ +0.020 with the CI excluding 0): {m:+.3f} "
                 f"({lo:+.3f}, {hi:+.3f}), {k}/{n} subjects. → **"
                 + ("adopt" if adopt else "no gain (not adopted)") + "**"
                 + ("" if trig else " (evaluated because the order-robustness rule failed: the "
                    "brief says the runbook then adds the guard; the lag rule itself did not "
                    "trigger)") + ".")
    elif trig:
        R.append("- Buffer-reset guard: not evaluated yet (run with --guard).")
    # descriptive only: which buffer did best per order (N must not be picked on test data)
    best = []
    for study, cls, split, name in PROXIES:
        g = by.get((study, tuple(cls) if cls else None, split))
        if not g or not g["clean"]:
            continue
        for od in ("rec", "inter", "shuf"):
            sc = {N: _order_scores(g["online"][N], od)[0] for N in g["online"]}
            sc = {N: v for N, v in sc.items() if v is not None}
            if sc:
                bN = max(sc, key=sc.get)
                best.append(f"{name} {od}: N={bN} ({sc[bN] - g['clean']['cell']:+.3f})")
    if best:
        R.append("- Descriptive, not a decision (choosing N on test sessions would be tuning "
                 "on test data): best buffer per order and its gain over clean — "
                 + "; ".join(best) + ".")
    R.append("")
    # timings
    R += ["## Timings (4 threads, other agents sharing the CPU)", "",
          "Weight CV: *choose_w* = the state was written by `sealed_personal.choose_w` itself "
          "(which also fits the per-subject Riemann models only its blend variant uses); "
          "*calib-only* = `calib_fold_scores` (the same blend_calib computation without "
          "them; `--verify_w` found identical CV scores and w on the three Zhou states).", "",
          "| proxy | fit + weight CV clean (s) | fit + weight CV online (s) | weight CV | "
          "per-variant predict (s, median) | rows |", "|---|---|---|---|---|---|"]
    for study, cls, split, name in PROXIES:
        g = by.get((study, tuple(cls) if cls else None, split))
        if not g or not g["clean"]:
            continue
        on = [r for N in g["online"] for r in g["online"][N].values()]
        on += [r for N in g["guard"] for r in g["guard"][N].values()]
        fo = on[0]["fit_seconds"] if on else None
        stem = f"{study}_{'all' if cls is None else '-'.join(map(str, cls))}_{split}".replace(":", "")
        sp_ = OUT / "state" / f"{stem}_clean.joblib"
        impl = ("" if not sp_.exists() else
                "calib-only" if "fold_scores" in joblib.load(sp_) else "choose_w")
        R.append(f"| {name} | {g['clean']['fit_seconds']} | "
                 f"{'same fit' if fo == g['clean']['fit_seconds'] else fo} | {impl} | "
                 f"{np.median([r['seconds'] for r in on]) if on else ''} | {1 + len(on)} |")
    (OUT / "RESULTS.md").write_text("\n".join(R) + "\n")
    (OUT / "BOOTSTRAP.md").write_text("\n".join(B) + "\n")
    print("\n".join(R + [""] + B))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study")
    ap.add_argument("--classes", default=None, help="e.g. 0,1,3 (cache label ids)")
    ap.add_argument("--split", default="last", help="last | calib:K")
    ap.add_argument("--orders", nargs="+", default=ORDERS)
    ap.add_argument("--buffers", nargs="+", type=int, default=[32, 64, 128])
    ap.add_argument("--guard", action="store_true", help="also online-N + buffer-reset guard")
    ap.add_argument("--drop_ch", default="A2-A1")
    ap.add_argument("--verify_w", action="store_true",
                    help="recompute the weight CV and compare it with the saved state")
    ap.add_argument("--report", action="store_true", help="write RESULTS.md + BOOTSTRAP.md")
    args = ap.parse_args()
    if args.report:
        report()
    elif args.study:
        run_study(args)
    else:
        ap.error("--study or --report")


if __name__ == "__main__":
    main()
