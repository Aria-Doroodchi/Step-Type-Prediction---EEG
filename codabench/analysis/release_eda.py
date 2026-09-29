#!/usr/bin/env python
"""Release-day EDA of a cross-session cache (RELEASE_DAY.md section 4).

    python ~/codabench/analysis/release_eda.py <study> \\
        [--split calib:3 --test_subjects 0,1,2,3,4,5,6,7,8,9] [--chans eeg] \\
        [--router_cap 0.5] [--no_psdctx] [--out logs/release_eda_<study>.md]

Reads the cache built by ``xsess_cache.py`` (``xsess_lib.load_study``; X is
memory-mapped) and applies the harness's own channel / split / hidden-row
logic through ``sealed_run.prepare_data`` (= ``xsess_lib.select_channels`` +
``xsess_split`` + ``restrict_pool``), with the same --split / --test_subjects
/ --chans / --drop_ch / --pool / --router_cap / --classes options and the
same ``[data]`` line. It then writes a markdown report, plus a ``.json``
sidecar with the numbers, to --out (default
``~/codabench/logs/release_eda_<study>.md``; a relative --out is relative to
the current directory):

1. data shape and release structure (subjects, sessions per subject,
   evaluation / fully labelled participants, contexts, hidden rows, channel
   types, the split's train / test / cell counts), next to the sealed
   structure the tracks page announces;
2. windows and class counts per subject x session x context cell (the
   metric's cells); flags cells with < 10 windows or a class imbalance
   > 1.5:1 (labelled cells only: on the release the hidden rows carry
   placeholder labels);
3. how contexts are laid out (within sessions by run, or one context per
   session) and any (subject, context) pair missing from training, which the
   router-psdctx alignment would need;
4. the log-PSD subject router, ``xsess_lib.SubjectRouter("psd", sfreq,
   cap=--router_cap)`` fitted on the split's training rows exactly as
   ``sealed_personal.py`` fits it (default cap 0.5 = the solver's rule,
   i.e. the harness's ``--router_cap 0.5``), scored PER TEST SESSION and per
   context: accuracy of the argmax subject (the harness's ``[router] psd:
   acc=``), fallback rate at the capped threshold and at the uncapped one;
   and, for a study with contexts, the (subject, context) router of
   router-psdctx (``--no_psdctx`` skips it);
5. drift: Riemannian distance of each (subject, session) mean covariance
   (``xsess_lib.window_covs``; Riemannian means, ``--drift_mean euclid`` for
   arithmetic ones) to that subject's calibration mean (sessions < K on
   calib:K / replica:K; the subject's sessions before its last on the
   "last" split), averaged per session index; with the between-subject
   distance and the within-subject context shift for scale. Hidden
   (split 2) sessions enter here as X only; no label is read;
6. the evoked response to the cue per class (xDAWN's premise), on the
   split's training windows: GFP (spatial SD) of each subject's class-mean
   ERP against a +/- reference (every other window sign-flipped: the
   residual noise of an average of the same size), and the mean over
   windows of the single-window GFP time course; peak time and amplitude
   per class, and a coarse time course.

Every table is descriptive. The release-day decisions are the
pre-registered rules on the replica split (RELEASE_DAY.md section 6); the
flags at the top of the report only say where to look. At 120 Hz full size
(14,400 windows x 43 ch) it takes ~3 min at 3 threads.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HOME = Path.home()
sys.path.insert(0, str(HOME / "codabench/analysis"))
import sealed_run as SR  # noqa: E402  (add_data_args / prepare_data / data_line / ctx_groups)
import xsess_lib as L  # noqa: E402

MIN_WINDOWS = 10          # cell flag: fewer windows than this
MAX_IMBALANCE = 1.5       # cell flag: largest / smallest class count above this
ROUTER_WARN_ACC = 0.80    # router flag: a test session routed worse than this
ROUTER_WARN_FALLBACK = 0.20
EVOKED_PEAK_RATIO = 1.5   # ERP GFP peak / +/- reference peak (~1 for noise): "evoked"
ROLES = ("train", "test", "hidden", "unused")
# the sealed data as the tracks page describes it (2026-09-28)
SEALED = {"subjects": "20", "sessions per subject": "6",
          "evaluation participants (3 labelled sessions)": "10",
          "classes": "3 (MI / CALC / WORD)", "contexts": "2 (Graz / BrainHero)",
          "channels": "43 EEG + 2 EMG + 2 EOG (NeuralBench default pick: 43 EEG)",
          "sfreq": "500 Hz raw (NeuralBench default: 120 Hz)"}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def f3(v):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def f2(v):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"


def md_table(head, rows):
    out = ["| " + " | ".join(map(str, head)) + " |",
           "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def sess_label(v):
    """0-based cache session index -> 'idx (Nth)'; the sealed docs count 1..6."""
    return f"{v} ({v + 1}.)"


def mean_or_nan(a):
    a = np.asarray(a, dtype=float)
    return float(a.mean()) if a.size else float("nan")


def to_json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and np.isnan(o):
        return None
    raise TypeError(type(o))


class Timer:
    def __init__(self):
        self.t0, self.laps = time.time(), {}

    def lap(self, name, t_start):
        self.laps[name] = round(time.time() - t_start, 1)
        L.log(f"  {name}: {self.laps[name]:.1f}s")


# ---------------------------------------------------------------------------
# 1. structure
# ---------------------------------------------------------------------------
def row_roles(d, sp):
    """Per window: 0 train, 1 test, 2 hidden (split 2, kept out of train and
    test), 3 unused (e.g. --pool test leaves the other subjects out)."""
    hid = L.hidden_rows(d)
    role = np.full(len(d["y"]), 3, np.int64)
    role[hid] = 2
    role[sp["train"]] = 0
    role[sp["test"]] = 1
    return role, hid


def structure(d, sp, si, hid, K, dropped, cache_shape, args):
    meta, subj, sess, y = d["meta"], d["subj"], d["sess"], d["y"]
    subjects = np.unique(subj)
    n_sess = {int(s): int(len(np.unique(sess[subj == s]))) for s in subjects}
    n_lab = {int(s): int(len(np.unique(sess[(subj == s) & ~hid]))) for s in subjects}
    hid_lab = bool(meta.get("hidden_labelled", "mock" in meta))
    split_codes = np.bincount(np.asarray(d["split"]), minlength=3).tolist()
    ctx_counts = {d["ctx_names"][c]: int(n) for c, n in
                  zip(*np.unique(d["ctx"], return_counts=True))}
    lab = ~hid | hid_lab
    cls_names = [str(meta["classes"].get(str(k), k)) for k in range(K)]
    win = meta.get("window") or {}
    return dict(
        study=meta.get("study"), built=meta.get("built"), sfreq=float(meta["sfreq"]),
        window_start=win.get("start"), window_duration=win.get("duration"),
        shape_cache=list(cache_shape), shape_used=list(d["X"].shape),
        dropped_channels=list(dropped), ch_types_cache=si["ch_types"],
        n_ch_used=int(d["X"].shape[1]),
        n_subjects=int(len(subjects)),
        sessions_per_subject=dict(sorted(Counter(n_sess.values()).items())),
        labelled_sessions_per_subject=dict(sorted(Counter(n_lab.values()).items())),
        eval_subjects=meta.get("eval_subjects"), full_subjects=meta.get("full_subjects"),
        calib_sessions=meta.get("calib_sessions"),
        release_cache=bool(meta.get("hidden_split", "eval_subjects" in meta
                                    or "context_column" in meta)),
        context_column=meta.get("context_column"), contexts=ctx_counts,
        has_ctx=bool(d["has_ctx"]),
        hidden_rows=int(hid.sum()), hidden_labelled=hid_lab,
        hidden_subjects=np.unique(subj[hid]).tolist(),
        hidden_sessions=np.unique(sess[hid]).tolist(),
        cache_split_codes=split_codes,
        classes=cls_names,
        class_counts_labelled=np.bincount(y[lab & (y >= 0) & (y < K)], minlength=K).tolist(),
        split=args.split, pool=args.pool, chans=args.chans,
        n_train=si["n_train"], n_test=si["n_test"], test_subjects=si["test_subjects"],
        test_sessions=si["test_sessions"], n_test_cells=si["n_test_cells"],
        hidden_in_train=si["hidden_in_train"], hidden_excluded=si["hidden_excluded"],
        n_ctx=si["n_ctx"])


def structure_md(st, meta):
    subj_names = meta.get("subjects") or []
    rows = [
        ("cache", f"`{st['study']}` built {st['built']}"),
        ("X (cache → used)", f"{tuple(st['shape_cache'])} → {tuple(st['shape_used'])} "
                             f"(dropped: {', '.join(st['dropped_channels']) or 'none'})"),
        ("sfreq / window", f"{st['sfreq']:g} Hz; window start {st['window_start']} s, "
                           f"duration {st['window_duration']} s"),
        ("channel types (cache)", st["ch_types_cache"]),
        ("classes (labelled windows)", ", ".join(f"{k} {n}={c}" for k, (n, c) in enumerate(
            zip(st["classes"], st["class_counts_labelled"])))),
        ("subjects", f"{st['n_subjects']}"
                     + (f" (ids {subj_names[0]} … {subj_names[-1]})" if subj_names else "")),
        ("sessions per subject", ", ".join(f"{k} sessions: {v} subjects"
                                           for k, v in st["sessions_per_subject"].items())),
        ("sessions outside the hidden split, per subject", ", ".join(
            f"{k}: {v} subjects" for k, v in st["labelled_sessions_per_subject"].items())),
        ("release-structure cache", "yes" if st["release_cache"] else "no (a proxy)"),
        ("evaluation participants (meta)", st["eval_subjects"] if st["eval_subjects"]
         is not None else "-"),
        ("fully labelled participants (meta)", st["full_subjects"] if st["full_subjects"]
         is not None else "-"),
        ("calib_sessions (meta)", st["calib_sessions"] if st["calib_sessions"] is not None
         else "-"),
        ("contexts", (f"column `{st['context_column']}`: " + ", ".join(
            f"{k}={v}" for k, v in st["contexts"].items())) if st["has_ctx"]
         else "none (no context column: cells are subject × session)"),
        ("hidden (split 2) rows", f"{st['hidden_rows']}"
         + (f": subjects {st['hidden_subjects']}, sessions {st['hidden_sessions']}, "
            f"labels {'known (hidden_labelled)' if st['hidden_labelled'] else 'placeholders'}"
            if st["hidden_rows"] else "")),
        ("cache split codes 0/1/2", "/".join(map(str, st["cache_split_codes"]))),
        ("split", f"`{st['split']}` pool={st['pool']} chans={st['chans']}"),
        ("train / test windows", f"{st['n_train']} / {st['n_test']}"),
        ("test subjects / sessions", f"{st['test_subjects']} / {st['test_sessions']}"),
        ("test cells (subject × session × context)", st["n_test_cells"]),
        ("hidden_in_train (must be 0) / hidden_excluded",
         f"{st['hidden_in_train']} / {st['hidden_excluded']}"),
    ]
    out = [md_table(("item", "value"), rows), "",
           "Against the sealed structure on the tracks page:", ""]
    sp_cnt = st["sessions_per_subject"]
    ev = st["eval_subjects"]
    found = {"subjects": str(st["n_subjects"]),
             "sessions per subject": ", ".join(f"{k}×{v}" for k, v in sp_cnt.items()),
             "evaluation participants (3 labelled sessions)":
                 (f"{len(ev)} (meta)" if ev is not None else "-")
                 + "; non-hidden sessions per subject "
                 + ", ".join(f"{k}×{v}" for k, v in st["labelled_sessions_per_subject"].items()),
             "classes": f"{len(st['classes'])} ({' / '.join(st['classes'])})",
             "contexts": (f"{len(st['contexts'])} ({' / '.join(st['contexts'])})"
                          if st["has_ctx"] else "none"),
             "channels": f"{st['ch_types_cache']}; used {st['n_ch_used']}",
             "sfreq": f"{st['sfreq']:g} Hz"}
    out.append(md_table(("item", "tracks page", "this cache"),
                        [(k, v, found[k]) for k, v in SEALED.items()]))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 2. cells
# ---------------------------------------------------------------------------
def cells(d, role, K, hid_lab):
    """One dict per subject x session x context cell: windows, class counts
    (an extra last count for labels outside 0..K-1), roles, flags."""
    y = np.where((d["y"] >= 0) & (d["y"] < K), d["y"], K)
    keys = np.stack([d["subj"], d["sess"], d["ctx"]], 1)
    u, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    nc = len(u)
    cnt = np.bincount(inv * (K + 1) + y, minlength=nc * (K + 1)).reshape(nc, K + 1)
    rc = np.bincount(inv * 4 + role, minlength=nc * 4).reshape(nc, 4)
    out = []
    for i, (s, v, c) in enumerate(u):
        n = int(cnt[i].sum())
        labelled = not (rc[i, 2] and not hid_lab)
        flags = []
        if n < MIN_WINDOWS:
            flags.append(f"{n} windows < {MIN_WINDOWS}")
        if labelled:
            k_cnt = cnt[i, :K]
            if k_cnt.min() == 0:
                flags.append("a class is missing")
            elif k_cnt.max() / k_cnt.min() > MAX_IMBALANCE:
                flags.append(f"imbalance {k_cnt.max() / k_cnt.min():.2f}:1")
            if cnt[i, K]:
                flags.append(f"{int(cnt[i, K])} labels outside 0..{K - 1}")
        out.append(dict(subj=int(s), sess=int(v), ctx=int(c), n=n,
                        counts=cnt[i, :K].tolist(), other=int(cnt[i, K]),
                        roles=[ROLES[j] for j in range(4) if rc[i, j]],
                        labelled=bool(labelled), flags=flags))
    return out


def cells_md(cl, d, K, cls_names):
    ctx_names = d["ctx_names"]
    subjects = sorted({c["subj"] for c in cl})
    sessions = sorted({c["sess"] for c in cl})
    by = {(c["subj"], c["sess"], c["ctx"]): c for c in cl}
    ns = np.array([c["n"] for c in cl])
    ratios = [max(c["counts"]) / max(min(c["counts"]), 1) for c in cl if c["labelled"]]
    flagged = [c for c in cl if c["flags"]]
    lines = [f"{len(cl)} cells; windows per cell min {ns.min()}, median {np.median(ns):g}, "
             f"max {ns.max()}; largest class imbalance in a labelled cell "
             f"{max(ratios) if ratios else float('nan'):.2f}:1; "
             f"**{len(flagged)} flagged** (< {MIN_WINDOWS} windows, or imbalance "
             f"> {MAX_IMBALANCE}:1 / a missing class in a labelled cell).", ""]
    marks = {"test": "T", "hidden": "h", "unused": "u"}
    head = ["subject"] + [f"s{v}" for v in sessions]
    rows = []
    subj_names = d["meta"].get("subjects") or []
    for s in subjects:
        r = [f"{s}" + (f" ({subj_names[s]})" if s < len(subj_names) else "")]
        for v in sessions:
            cs = [by.get((s, v, c)) for c in range(len(ctx_names))]
            if not any(cs):
                r.append("")
                continue
            roles = set().union(*[set(c["roles"]) for c in cs if c])
            mk = "".join(marks[x] for x in ("test", "hidden", "unused") if x in roles)
            flag = "!" if any(c and c["flags"] for c in cs) else ""
            r.append("/".join(str(c["n"]) if c else "0" for c in cs) + (f" {mk}" if mk else "")
                     + flag)
        rows.append(r)
    lines.append("Windows per subject × session" + (
        f" (per context: {' / '.join(ctx_names)})" if len(ctx_names) > 1 else "")
        + ". Marks: T = test cell of the split, h = hidden (split 2) rows kept out of "
          "train and test, u = unused (not train, not test), ! = flagged; unmarked = "
          "training. Sessions are 0-based cache indices (s3 = the 4th session).")
    lines += ["", md_table(head, rows), ""]
    if flagged:
        lines += ["Flagged cells:", "",
                  md_table(("subject", "session", "context", "windows",
                            "class counts (" + "/".join(cls_names) + ")", "roles", "flags"),
                           [(c["subj"], c["sess"], ctx_names[c["ctx"]], c["n"],
                             "/".join(map(str, c["counts"])), ",".join(c["roles"]),
                             "; ".join(c["flags"])) for c in flagged[:60]])]
        if len(flagged) > 60:
            lines.append(f"\n… {len(flagged) - 60} more flagged cells in the JSON sidecar.")
    else:
        lines.append("Flagged cells: none.")
    lines += ["", "<details><summary>Every cell: class counts</summary>", "",
              md_table(("subject", "session", "context", "windows",
                        "class counts (" + "/".join(cls_names) + ")", "roles", "flags"),
                       [(c["subj"], c["sess"], ctx_names[c["ctx"]], c["n"],
                         "/".join(map(str, c["counts"]))
                         + ("" if c["labelled"] else " (placeholder labels)"),
                         ",".join(c["roles"]), "; ".join(c["flags"])) for c in cl]),
              "", "</details>"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. contexts
# ---------------------------------------------------------------------------
def context_layout(d, sp):
    if not d["has_ctx"]:
        return dict(mode="none")
    subj, sess, run, ctx = d["subj"], d["sess"], d["run"], d["ctx"]
    n_ctx = len(d["ctx_names"])
    u = np.unique(np.stack([subj, sess, ctx], 1), axis=0)
    _, per_sess = np.unique(u[:, :2], axis=0, return_counts=True)
    u3 = np.unique(np.stack([subj, sess, run, ctx], 1), axis=0)
    _, per_run = np.unique(u3[:, :3], axis=0, return_counts=True)
    if per_sess.max() == 1:
        mode = "one context per session"
    elif per_run.max() == 1:
        mode = "within sessions, by run (every run holds one context)"
    else:
        mode = "mixed within runs (a run holds several contexts)"
    sessions, runs = np.unique(sess), np.unique(run)
    sess_tab = [[int(((sess == v) & (ctx == c)).sum()) for c in range(n_ctx)] for v in sessions]
    sess_subj = [[int(len(np.unique(subj[(sess == v) & (ctx == c)]))) for c in range(n_ctx)]
                 for v in sessions]
    run_tab = [[int(((run == r) & (ctx == c)).sum()) for c in range(n_ctx)] for r in runs]
    tr, te = sp["train"], sp["test"]
    missing_train = {}
    for s in np.unique(subj[tr]):
        have = set(np.unique(ctx[tr & (subj == s)]).tolist())
        miss = sorted(set(range(n_ctx)) - have)
        if miss:
            missing_train[int(s)] = [d["ctx_names"][c] for c in miss]
    pairs_tr = set(zip(subj[tr].tolist(), ctx[tr].tolist()))
    test_unseen = sorted({(s, c) for s, c in zip(subj[te].tolist(), ctx[te].tolist())}
                         - pairs_tr)
    return dict(mode=mode, sessions_by_n_contexts=dict(Counter(per_sess.tolist())),
                runs_by_n_contexts=dict(Counter(per_run.tolist())),
                sessions=sessions.tolist(), sess_tab=sess_tab, sess_subj=sess_subj,
                runs=runs.tolist(), run_tab=run_tab, missing_train=missing_train,
                test_pairs_unseen=[(int(s), d["ctx_names"][c]) for s, c in test_unseen])


def context_md(cx, d):
    if cx["mode"] == "none":
        return ("No context column in this cache (a proxy): one context (`none`), so the "
                "metric's cells are subject × session and every per-context column below "
                "has one entry. On the release cache a missing context column is an "
                "error: see RELEASE_DAY.md § 3.2.")
    names = d["ctx_names"]
    lines = [f"**Layout: {cx['mode']}.** Subject-sessions by number of contexts: "
             + ", ".join(f"{k} context(s): {v}" for k, v in sorted(cx["sessions_by_n_contexts"].items()))
             + "; subject-session-runs by number of contexts: "
             + ", ".join(f"{k}: {v}" for k, v in sorted(cx["runs_by_n_contexts"].items())) + ".",
             "",
             md_table(["session"] + [f"{c} windows (subjects)" for c in names],
                      [[sess_label(v)] + [f"{n} ({m})" for n, m in zip(t, ts)]
                       for v, t, ts in zip(cx["sessions"], cx["sess_tab"], cx["sess_subj"])]),
             "",
             md_table(["run (within session)"] + [f"{c} windows" for c in names],
                      [[r] + t for r, t in zip(cx["runs"], cx["run_tab"])]),
             ""]
    if cx["missing_train"]:
        lines.append("**(subject, context) pairs missing from training** (router-psdctx "
                     "cannot whiten them per pair): " + "; ".join(
                         f"subject {s}: {', '.join(c)}" for s, c in cx["missing_train"].items()))
    else:
        lines.append("Every training subject has training windows in every context "
                     "(router-psdctx can whiten every pair).")
    if cx["test_pairs_unseen"]:
        lines.append("Test (subject, context) pairs with no training windows: "
                     + ", ".join(f"({s}, {c})" for s, c in cx["test_pairs_unseen"]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. router per test session
# ---------------------------------------------------------------------------
def router_rows(d, te_idx, ok, fb, fbr, extra=None):
    """Per test session index: n, acc, fallback, uncapped fallback, acc of
    the windows that do not fall back, per-context acc, worst subject."""
    subj, sess, ctx = d["subj"][te_idx], d["sess"][te_idx], d["ctx"][te_idx]
    out = []
    for v in [int(x) for x in np.unique(sess)] + ["all"]:
        m = np.ones(len(te_idx), bool) if isinstance(v, str) else sess == v
        per_s = {int(s): float(ok[m & (subj == s)].mean()) for s in np.unique(subj[m])}
        worst = min(per_s, key=per_s.get)
        r = dict(session=v, n=int(m.sum()), acc=float(ok[m].mean()),
                 fallback=float(fb[m].mean()), fallback_uncapped=float(fbr[m].mean()),
                 acc_routed=mean_or_nan(ok[m & ~fb]),
                 acc_ctx={d["ctx_names"][c]: float(ok[m & (ctx == c)].mean())
                          for c in np.unique(ctx[m])},
                 worst_subject=worst, worst_acc=per_s[worst], per_subject=per_s)
        for k, arr in (extra or {}).items():
            r[k] = float(arr[m].mean())
        out.append(r)
    return out


def router_section(d, sp, args, psdctx, timer):
    tr, te = sp["train"], sp["test"]
    te_idx = np.where(te)[0]
    if not len(te_idx) or not tr.any():
        return None
    sfreq = d["meta"]["sfreq"]
    subj = d["subj"]
    t0 = time.time()
    Xtr = d["X"][tr]
    Xte = d["X"][te]
    r = L.SubjectRouter("psd", sfreq, cap=args.router_cap).fit(Xtr, subj[tr], d["sess"][tr])
    Pr = r.predict_proba(Xte)
    a = r.subjects[Pr.argmax(1)]
    ok, mp = a == subj[te_idx], Pr.max(1)
    fb, fbr = mp < r.thr, mp < r.thr_raw
    res = dict(n_train_subjects=int(len(r.subjects)), thr=float(r.thr),
               thr_raw=float(r.thr_raw), oof_acc=float(r.oof_acc), cap=args.router_cap,
               rows=router_rows(d, te_idx, ok, fb, fbr))
    timer.lap("router psd (fit + OOF threshold + predict)", t0)
    L.log(f"[router] psd: acc={ok.mean():.3f} fallback={fb.mean():.3f} thr={r.thr:.3f} "
          f"(uncapped {r.thr_raw:.3f}, fallback uncapped {fbr.mean():.3f})")
    for row in res["rows"][:-1]:
        L.log(f"  test session {row['session']}: n={row['n']} acc={row['acc']:.3f} "
              f"fallback={row['fallback']:.3f} (uncapped {row['fallback_uncapped']:.3f})")
    if psdctx and d["has_ctx"] and len(d["ctx_names"]) > 1:
        t0 = time.time()
        grp, n_ctx = SR.ctx_groups(d, "router-psdctx:riemann")
        r2 = L.SubjectRouter("psd", sfreq, cap=args.router_cap).fit(Xtr, grp[tr], d["sess"][tr])
        P2 = r2.predict_proba(Xte)
        a2 = r2.subjects[P2.argmax(1)]
        mp2 = P2.max(1)
        okp = a2 == grp[te_idx]
        res["psdctx"] = dict(
            thr=float(r2.thr), thr_raw=float(r2.thr_raw), oof_acc=float(r2.oof_acc),
            n_pairs=int(len(r2.subjects)),
            rows=router_rows(d, te_idx, okp, mp2 < r2.thr, mp2 < r2.thr_raw,
                             extra=dict(subject_acc=a2 // n_ctx == subj[te_idx],
                                        context_acc=a2 % n_ctx == d["ctx"][te_idx])))
        timer.lap("router psdctx (fit + OOF threshold + predict)", t0)
        al = res["psdctx"]["rows"][-1]
        L.log(f"[router] psdctx: pair acc={al['acc']:.3f} subject acc={al['subject_acc']:.3f} "
              f"context acc={al['context_acc']:.3f} fallback={al['fallback']:.3f}")
    del Xtr, Xte
    return res


def router_md(rr, d, st):
    if rr is None:
        return "No test rows (or no training rows) under this split: router not evaluated."
    ctx_names = d["ctx_names"]
    multi = len(ctx_names) > 1
    cap = "uncapped" if rr["cap"] is None else f"min(OOF 1st percentile, cap {rr['cap']:g})"
    lines = [f"`SubjectRouter(\"psd\")` fitted on the split's {st['n_train']} training windows "
             f"({rr['n_train_subjects']} subjects, chance {1 / rr['n_train_subjects']:.3f}); "
             f"out-of-fold accuracy on training {rr['oof_acc']:.3f}; fallback threshold on the "
             f"max posterior {rr['thr']:.3f} ({cap}; uncapped {rr['thr_raw']:.3f}). "
             "acc = the argmax subject is the true one (the harness's `[router] psd: acc=`); "
             "fallback = max posterior below the threshold (the window is whitened with the "
             "global reference and gets the pooled prediction); acc routed = accuracy of the "
             "windows that do not fall back.", ""]
    head = (["test session", "windows", "acc", "fallback", "fallback uncapped", "acc routed"]
            + ([f"acc {c}" for c in ctx_names] if multi else []) + ["worst subject (acc)"])
    rows = []
    for r in rr["rows"]:
        lab = "**all**" if r["session"] == "all" else sess_label(r["session"])
        rows.append([lab, r["n"], f3(r["acc"]), f3(r["fallback"]), f3(r["fallback_uncapped"]),
                     f3(r["acc_routed"])]
                    + ([f3(r["acc_ctx"].get(c)) for c in ctx_names] if multi else [])
                    + [f"{r['worst_subject']} ({r['worst_acc']:.3f})"])
    lines += [md_table(head, rows), ""]
    per = rr["rows"][:-1]
    if len(per) > 1:
        lines.append(f"Change from the first to the last test session: "
                     f"{per[-1]['acc'] - per[0]['acc']:+.3f} accuracy, "
                     f"{per[-1]['fallback'] - per[0]['fallback']:+.3f} fallback.")
    if "psdctx" in rr:
        pc = rr["psdctx"]
        lines += ["", f"**router-psdctx** (the (subject, context) router of the context "
                      f"alignment ablation; {pc['n_pairs']} pairs, OOF accuracy "
                      f"{pc['oof_acc']:.3f}, threshold {pc['thr']:.3f}, uncapped "
                      f"{pc['thr_raw']:.3f}). pair acc = subject and context both right; "
                      "subject acc = what sealed_personal logs for psdctx:", "",
                  md_table(["test session", "windows", "pair acc", "subject acc", "context acc",
                            "fallback", "fallback uncapped"],
                           [["**all**" if r["session"] == "all" else sess_label(r["session"]),
                             r["n"], f3(r["acc"]), f3(r["subject_acc"]), f3(r["context_acc"]),
                             f3(r["fallback"]), f3(r["fallback_uncapped"])]
                            for r in pc["rows"]])]
    subjects = sorted({s for r in per for s in r["per_subject"]})
    lines += ["", "<details><summary>Router accuracy per test subject × session</summary>", "",
              md_table(["subject"] + [sess_label(r["session"]) for r in per],
                       [[s] + [f3(r["per_subject"].get(s)) for r in per] for s in subjects]),
              "", "</details>"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. drift
# ---------------------------------------------------------------------------
def drift_section(d, sp, args, kind, hid, timer):
    try:
        from pyriemann.geometry.distance import distance_riemann
    except ImportError:     # pyriemann < 0.12
        from pyriemann.utils.distance import distance_riemann
    t0 = time.time()
    covs = L.window_covs(d["X"])
    timer.lap("window covariances", t0)
    t0 = time.time()
    subj, sess, ctx = d["subj"], d["sess"], d["ctx"]
    name, _, k = args.split.partition(":")
    test_subjects = set(np.asarray(sp["test_subjects"]).tolist())
    refs, ctx_refs, cellrows = {}, {}, []
    for s in np.unique(subj):
        ms = subj == s
        kref = int(sess[ms].max()) if name == "last" else int(k)
        ref_rows = ms & (sess < kref)
        if not ref_rows.any():
            continue
        refs[int(s)] = L.mean_cov(covs[ref_rows], kind)
        if d["has_ctx"]:
            cr = {int(c): L.mean_cov(covs[ref_rows & (ctx == c)], kind)
                  for c in np.unique(ctx[ref_rows])}
            if len(cr) > 1:
                ctx_refs[int(s)] = cr
        for v in np.unique(sess[ms]):
            m = ms & (sess == v)
            dist = float(distance_riemann(L.mean_cov(covs[m], kind), refs[int(s)]))
            cellrows.append(dict(subj=int(s), sess=int(v), calib=bool(v < kref),
                                 hidden=bool(hid[m].all()), test_subject=int(s) in test_subjects,
                                 dist=dist))
    subs = sorted(refs)
    between = [float(distance_riemann(refs[a], refs[b]))
               for i, a in enumerate(subs) for b in subs[i + 1:]]
    ctx_shift = []
    for s, cr in ctx_refs.items():
        cs = sorted(cr)
        ctx_shift += [float(distance_riemann(cr[a], cr[b]))
                      for i, a in enumerate(cs) for b in cs[i + 1:]]
    per_v = []
    for v in sorted({c["sess"] for c in cellrows}):
        cs = [c for c in cellrows if c["sess"] == v]
        dv = np.array([c["dist"] for c in cs])
        per_v.append(dict(
            session=v, n=len(cs), calib=sum(c["calib"] for c in cs),
            mean=float(dv.mean()), sd=float(dv.std()),
            mean_test_subjects=mean_or_nan([c["dist"] for c in cs if c["test_subject"]]),
            mean_other_subjects=mean_or_nan([c["dist"] for c in cs if not c["test_subject"]]),
            n_hidden=sum(c["hidden"] for c in cs)))
    calib_d = [c["dist"] for c in cellrows if c["calib"]]
    later = [r for r in per_v if r["calib"] == 0]
    later_means = [r["mean"] for r in later]
    res = dict(kind=kind, per_session=per_v, cells=cellrows,
               calib_mean=mean_or_nan(calib_d), between_subject_mean=mean_or_nan(between),
               context_shift_mean=mean_or_nan(ctx_shift) if ctx_shift else None,
               later_means=later_means,
               grows=bool(len(later_means) > 1 and all(np.diff(later_means) > 0)),
               slope=(float(np.polyfit([r["session"] for r in later], later_means, 1)[0])
                      if len(later_means) > 1 else None))
    timer.lap(f"drift ({kind} means, Riemannian distances)", t0)
    return res


def drift_md(dr, split):
    lines = [f"Riemannian distance δ(cell mean, calibration mean) per subject, where both "
             f"means are {dr['kind']} means of the `window_covs` covariances (the harness's "
             f"alignment statistics) and the calibration mean pools the subject's "
             + ("sessions before its last" if split == "last" else "sessions < K")
             + ". A calibration session's own distance is the within-calibration spread "
               "(the baseline). Hidden sessions use X only.", "",
             md_table(["session", "kind", "subjects", "mean δ", "sd", "mean δ test subjects",
                       "mean δ other subjects", "hidden cells"],
                      [[sess_label(r["session"]),
                        "calibration" if r["calib"] == r["n"] else
                        ("later" if r["calib"] == 0 else "mixed"),
                        r["n"], f3(r["mean"]), f3(r["sd"]), f3(r["mean_test_subjects"]),
                        f3(r["mean_other_subjects"]), r["n_hidden"]]
                       for r in dr["per_session"]]), ""]
    scale = (f"Scale: calibration sessions' mean δ {dr['calib_mean']:.3f}; between-subject "
             f"δ of the calibration means {dr['between_subject_mean']:.3f}")
    if dr["context_shift_mean"] is not None:
        scale += (f"; within-subject context shift (δ between a subject's per-context "
                  f"calibration means) {dr['context_shift_mean']:.3f}")
    lines.append(scale + ".")
    if dr["later_means"]:
        lm, base = dr["later_means"], dr["calib_mean"]
        if not base > 1e-6:     # one calibration session: its distance to itself is 0
            rel = (" (one calibration session, so there is no within-calibration baseline; "
                   f"{lm[0] / dr['between_subject_mean']:.2f}× to "
                   f"{lm[-1] / dr['between_subject_mean']:.2f}× the between-subject δ)")
        elif len(lm) == 1:
            rel = f" ({lm[0] / base:.2f}× the calibration baseline)"
        else:
            rel = f" ({lm[0] / base:.2f}× to {lm[-1] / base:.2f}× the calibration baseline)"
        verdict = ("drift **grows** with the session index" if dr["grows"] else
                   ("drift does not grow monotonically" if len(lm) > 1
                    else "one later session, so no trend"))
        lines.append("Later sessions: mean δ " + ", ".join(f"{v:.3f}" for v in lm) + rel
                     + f": {verdict}"
                     + (f", slope {dr['slope']:+.3f} per session." if dr["slope"] is not None
                        else "."))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. evoked response
# ---------------------------------------------------------------------------
def evoked_section(d, sp, K, timer, chunk=512):
    """Per-subject class-mean ERPs (and +/- reference averages) of the
    split's training windows, and the per-class mean single-window GFP."""
    t0 = time.time()
    rows = np.where(sp["train"])[0]
    y_all = d["y"][rows]
    keep = (y_all >= 0) & (y_all < K)
    rows, y = rows[keep], y_all[keep]
    if not len(rows):
        return None
    subjects, si = np.unique(d["subj"][rows], return_inverse=True)
    g = si.reshape(-1) * K + y
    G = len(subjects) * K
    order = np.argsort(g, kind="stable")
    gs = g[order]
    starts = np.r_[0, np.flatnonzero(np.diff(gs)) + 1]
    rank = np.arange(len(gs)) - np.repeat(starts, np.diff(np.r_[starts, len(gs)]))
    sign = np.empty(len(g))
    sign[order] = np.where(rank % 2 == 0, 1.0, -1.0)
    n_g = np.bincount(g, minlength=G).astype(float)
    C, T = d["X"].shape[1:]
    S = np.zeros((G, C * T))
    P = np.zeros((G, C * T))
    gfp_sum = np.zeros((K, T))
    for i in range(0, len(rows), chunk):
        Xc = np.asarray(d["X"][rows[i:i + chunk]], dtype=np.float64)
        nc = len(Xc)
        M = np.zeros((G, nc))
        M[g[i:i + chunk], np.arange(nc)] = 1.0
        Xf = Xc.reshape(nc, -1)
        S += M @ Xf
        P += (M * sign[i:i + chunk]) @ Xf
        Mk = np.zeros((K, nc))
        Mk[y[i:i + chunk], np.arange(nc)] = 1.0
        gfp_sum += Mk @ Xc.std(axis=1)
    ok = n_g > 0
    S[ok] /= n_g[ok, None]
    P[ok] /= n_g[ok, None]
    erp = S.reshape(len(subjects), K, C, T)
    pm = P.reshape(len(subjects), K, C, T)
    have = ok.reshape(len(subjects), K)
    gfp_erp = erp.std(axis=2)           # (S, K, T)
    gfp_pm = pm.std(axis=2)
    # between-class part: each class ERP minus the subject's mean over classes
    full = have.all(1)
    dev = erp[full] - erp[full].mean(1, keepdims=True)
    dev_pm = pm[full] - pm[full].mean(1, keepdims=True)
    sfreq = float(d["meta"]["sfreq"])
    start = float((d["meta"].get("window") or {}).get("start") or 0.0)
    t = start + np.arange(T) / sfreq
    n_k = np.bincount(y, minlength=K)
    cls_names = [str(d["meta"]["classes"].get(str(k), k)) for k in range(K)]
    per_class, curves = [], {}
    for k in range(K):
        hk = have[:, k]
        if not hk.any():
            continue
        cur, flo = gfp_erp[hk, k].mean(0), gfp_pm[hk, k].mean(0)
        sw = gfp_sum[k] / max(n_k[k], 1)
        i_pk, i_sw = int(cur.argmax()), int(sw.argmax())
        per_class.append(dict(
            cls=cls_names[k], windows=int(n_k[k]), subjects=int(hk.sum()),
            windows_per_subject=float(n_g.reshape(len(subjects), K)[hk, k].mean()),
            erp_peak_t=float(t[i_pk]), erp_peak=float(cur[i_pk]),
            floor_peak=float(flo.max()), floor_mean=float(flo.mean()),
            peak_ratio=float(cur[i_pk] / flo.max()), mean_ratio=float(cur.mean() / flo.mean()),
            sw_peak_t=float(t[i_sw]), sw_peak=float(sw[i_sw]), sw_median=float(np.median(sw)),
            sw_ratio=float(sw[i_sw] / np.median(sw))))
        curves[cls_names[k]] = (cur, flo)
    between = None
    if full.any():
        cur = dev.std(axis=2).mean((0, 1))
        flo = dev_pm.std(axis=2).mean((0, 1))
        i_pk = int(cur.argmax())
        between = dict(peak_t=float(t[i_pk]), peak=float(cur[i_pk]), floor_peak=float(flo.max()),
                       floor_mean=float(flo.mean()), peak_ratio=float(cur[i_pk] / flo.max()),
                       mean_ratio=float(cur.mean() / flo.mean()))
        curves["between-class"] = (cur, flo)
    edges = np.arange(t[0], t[-1] + 1e-9, 0.5)
    edges = np.r_[edges, t[-1] + 1.0 / sfreq] if edges[-1] < t[-1] + 1.0 / sfreq else edges
    bins = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (t >= a) & (t < b)
        if m.any():
            bins.append(dict(label=f"{a:.1f}–{b:.1f}", **{
                name: [float(c[m].mean()), float(f[m].mean())] for name, (c, f) in curves.items()}))
    timer.lap("evoked (ERP / GFP per class)", t0)
    return dict(n_windows=int(len(rows)), n_subjects=int(len(subjects)), per_class=per_class,
                between=between, bins=bins, curve_names=list(curves))


def evoked_md(ev):
    if ev is None:
        return "No labelled training windows: evoked response not computed."
    lines = [f"On the split's {ev['n_windows']} labelled training windows "
             f"({ev['n_subjects']} subjects), as delivered (robust-scaled units, no extra "
             "filter; t = 0 is the window start, i.e. the cue when the window start is 0). "
             "**ERP GFP** = spatial SD of each subject's class-mean window, averaged over "
             "subjects: a phase-locked (evoked) response. **floor** = the same for the "
             "+/- reference (every other window sign-flipped), i.e. what an average of that "
             "many windows gives with no evoked response. **peak ratio** = ERP GFP peak / "
             "floor peak and **mean ratio** = ERP GFP mean / floor mean: both ≈ 1 (0.9–1.2) "
             f"for noise only; a peak ratio ≥ {EVOKED_PEAK_RATIO} is a clear evoked response. "
             "**single-window GFP** = the mean over windows of each window's GFP time course "
             "(evoked + induced + ongoing activity; peak / median ≈ 1 = flat).", "",
             md_table(["class", "windows (per subject)", "ERP GFP peak t (s)", "ERP GFP peak",
                       "floor peak (mean)", "peak ratio", "mean ratio",
                       "single-window GFP peak t (s)", "peak", "peak / median"],
                      [[c["cls"], f"{c['windows']} ({c['windows_per_subject']:.0f})",
                        f2(c["erp_peak_t"]), f3(c["erp_peak"]),
                        f"{c['floor_peak']:.3f} ({c['floor_mean']:.3f})",
                        f2(c["peak_ratio"]), f2(c["mean_ratio"]), f2(c["sw_peak_t"]),
                        f3(c["sw_peak"]), f2(c["sw_ratio"])] for c in ev["per_class"]])]
    if ev["between"]:
        b = ev["between"]
        lines += ["", f"Between-class evoked part (class ERP minus the subject's mean ERP "
                      f"over classes; the class-specific evoked signal xDAWN can separate): "
                      f"peak {b['peak']:.3f} at {b['peak_t']:.2f} s, floor peak "
                      f"{b['floor_peak']:.3f} (mean {b['floor_mean']:.3f}), peak ratio "
                      f"{b['peak_ratio']:.2f}, mean ratio {b['mean_ratio']:.2f}."]
    names = ev["curve_names"]
    lines += ["", "Time course, ERP GFP (floor) per 0.5 s bin:", "",
              md_table(["t (s)"] + names,
                       [[b["label"]] + [f"{b[n][0]:.3f} ({b[n][1]:.3f})" for n in names]
                        for b in ev["bins"]])]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# flags
# ---------------------------------------------------------------------------
def flags(st, cl, cx, rr, dr, ev):
    out = []
    if st["hidden_in_train"]:
        out.append(f"ERROR: {st['hidden_in_train']} hidden (split 2) rows in the training "
                   "set; the harness refuses this split")
    if st["release_cache"] and not st["has_ctx"]:
        out.append("WARN: a release-structure cache without a context column: the cell "
                   "metric would ignore contexts (RELEASE_DAY.md § 3.2)")
    few = [c for c in cl if any("windows <" in f for f in c["flags"])]
    imb = [c for c in cl if any(("imbalance" in f or "missing" in f) for f in c["flags"])]
    if few:
        out.append(f"WARN: {len(few)} cells with < {MIN_WINDOWS} windows (section 2)")
    if imb:
        out.append(f"WARN: {len(imb)} labelled cells with a class imbalance > "
                   f"{MAX_IMBALANCE}:1 or a missing class (section 2; the LDA priors are "
                   "uniform)")
    if cx.get("missing_train"):
        out.append(f"WARN: {len(cx['missing_train'])} training subjects lack a context in "
                   "their training windows (section 3; router-psdctx cannot whiten those pairs)")
    if rr is not None:
        for r in rr["rows"][:-1]:
            if r["acc"] < ROUTER_WARN_ACC or r["fallback"] > ROUTER_WARN_FALLBACK:
                out.append(f"WARN: router on test session {r['session']}: acc {r['acc']:.3f}, "
                           f"fallback {r['fallback']:.3f} (flag below {ROUTER_WARN_ACC} / "
                           f"above {ROUTER_WARN_FALLBACK}; section 4)")
        per = rr["rows"][:-1]
        if len(per) > 1 and per[0]["acc"] - per[-1]["acc"] > 0.10:
            out.append(f"WARN: router accuracy falls {per[0]['acc'] - per[-1]['acc']:.3f} "
                       "from the first to the last test session (section 4)")
    if dr is not None and dr["grows"]:
        base = (f"calibration {dr['calib_mean']:.3f}" if dr["calib_mean"] > 1e-6 else
                f"between-subject {dr['between_subject_mean']:.3f}")
        out.append(f"INFO: covariance drift grows with the session index (later sessions "
                   f"{', '.join(f'{v:.3f}' for v in dr['later_means'])} vs {base}; "
                   "section 5)")
    if ev is not None and ev["per_class"]:
        best = max(c["peak_ratio"] for c in ev["per_class"])
        if best < EVOKED_PEAK_RATIO:
            out.append(f"INFO: no clear evoked response (ERP GFP peak ratio at most "
                       f"{best:.2f} < {EVOKED_PEAK_RATIO}; section 6): xDAWN has little "
                       "evoked signal to work with (the xDAWN rule on the replica decides)")
        else:
            out.append(f"INFO: an evoked response is present (ERP GFP peak ratio up to "
                       f"{best:.2f}; section 6)")
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study")
    ap.add_argument("--classes", default=None, help="keep these label indices (e.g. 0,1,3)")
    ap.add_argument("--out", default=None,
                    help="report path (.md; a .json sidecar is written next to it); default "
                         "~/codabench/logs/release_eda_<study>.md")
    ap.add_argument("--no_psdctx", action="store_true",
                    help="skip the (subject, context) router table")
    ap.add_argument("--drift_mean", default="riemann", choices=["riemann", "euclid"],
                    help="mean of the window covariances per cell / calibration")
    SR.add_data_args(ap)     # --drop_ch --split --test_subjects --chans --pool --router_cap --mmap
    ap.set_defaults(router_cap=0.5)   # the solver's rule (harness: --router_cap 0.5)
    args = ap.parse_args()
    t_all = time.time()
    timer = Timer()
    out = (Path(args.out) if args.out else HOME / f"codabench/logs/release_eda_{args.study}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    classes = None if args.classes is None else [int(c) for c in args.classes.split(",")]
    L.log(f"release_eda: study={args.study} split={args.split} chans={args.chans} "
          f"router_cap={args.router_cap} threads={L.N_THREADS} "
          f"OMP={os.environ.get('OMP_NUM_THREADS')} -> {out}")

    # ---- load + the harness's channel / split logic
    t0 = time.time()
    d = L.load_study(args.study, classes, mmap=True)   # values unchanged; no up-front read
    cache_shape = d["X"].shape
    sp, dropped, _opts, si = SR.prepare_data(d, args)
    K = len(d["meta"]["classes"])
    role, hid = row_roles(d, sp)
    timer.lap("load + channel pick + split", t0)
    L.log(f"[data] study={args.study} classes={classes or 'all'} X={d['X'].shape} "
          f"subjects={len(np.unique(d['subj']))} sessions/subject="
          f"{sorted(set(len(np.unique(d['sess'][d['subj'] == s])) for s in np.unique(d['subj'])))} "
          f"train={si['n_train']} test={si['n_test']} K={K} dropped_ch={dropped} "
          + SR.data_line(args, si))
    if si["hidden_in_train"]:
        raise SystemExit("hidden-test rows in the training set: refusing (see [data])")

    # ---- 1-3: structure, cells, contexts (cheap)
    t0 = time.time()
    st = structure(d, sp, si, hid, K, dropped, cache_shape, args)
    cl = cells(d, role, K, st["hidden_labelled"])
    cx = context_layout(d, sp)
    timer.lap("structure, cells, contexts", t0)
    L.log(f"  cells={len(cl)} flagged={sum(bool(c['flags']) for c in cl)} "
          f"context layout: {cx['mode']}")

    # ---- 4-6
    rr = router_section(d, sp, args, not args.no_psdctx, timer)
    dr = drift_section(d, sp, args, args.drift_mean, hid, timer)
    L.log(f"  drift per session: " + ", ".join(
        f"s{r['session']}={r['mean']:.3f}" for r in dr["per_session"]))
    ev = evoked_section(d, sp, K, timer)
    if ev:
        L.log("  evoked peak/floor: " + ", ".join(
            f"{c['cls']}={c['peak_ratio']:.2f}" for c in ev["per_class"]))
    fl = flags(st, cl, cx, rr, dr, ev)
    total = round(time.time() - t_all, 1)

    cls_names = st["classes"]
    cmd = "python ~/codabench/analysis/release_eda.py " + " ".join(sys.argv[1:])
    md = [f"# Release EDA: {args.study}", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `{cmd}` in {total:.0f} s "
          f"(threads {L.N_THREADS}). Split `{args.split}`"
          + (f", test subjects {args.test_subjects}" if args.test_subjects else "")
          + f", chans {args.chans}, pool {args.pool}, router cap {args.router_cap}. "
            "Descriptive only: the release-day decisions are the pre-registered replica "
            "rules (RELEASE_DAY.md § 6).", "",
          "## Flags", "", "\n".join(f"- {f}" for f in fl) if fl else "- none", "",
          "## 1. Data and release structure", "", structure_md(st, d["meta"]), "",
          "## 2. Cells: windows and class counts (subject × session × context)", "",
          cells_md(cl, d, K, cls_names), "",
          "## 3. Contexts", "", context_md(cx, d), "",
          "## 4. Subject router per test session", "", router_md(rr, d, st), "",
          "## 5. Drift across sessions", "", drift_md(dr, args.split.partition(":")[0]), "",
          "## 6. Evoked response to the cue per class", "", evoked_md(ev), "",
          "## Timings", "",
          md_table(("step", "seconds"), list(timer.laps.items()) + [("**total**", total)]), ""]
    out.write_text("\n".join(md), encoding="utf-8")
    side = dict(study=args.study, argv=sys.argv[1:], generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                seconds=total, timings=timer.laps, flags=fl, structure=st, cells=cl,
                contexts=cx, router=rr, drift=dr, evoked=ev)
    out.with_suffix(".json").write_text(json.dumps(side, indent=1, default=to_json),
                                        encoding="utf-8")
    L.log(f"wrote {out} (+ .json) in {total:.1f}s")
    for f in fl:
        L.log(f"  {f}")


if __name__ == "__main__":
    main()
