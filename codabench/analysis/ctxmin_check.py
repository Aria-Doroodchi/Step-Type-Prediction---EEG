#!/usr/bin/env python
"""Sprint 2026-10-01 Phase 8 check: the harness's router-psdctx alignment
with the solver's ctx_min rule (sealed_run.aligned_data).

Synthetic study: 4 subjects x 2 contexts x 2 sessions (session 0 trains,
session 1 tests); the pair (subject 1, context 1) has only 6 training
windows (< CTX_MIN_WINDOWS = 16). Checks, against a reference copy of
sealed_run.py from before the change (``--ref``):
  1. every training row of the other pairs and every test row not routed to
     the small pair is bit-identical;
  2. the small pair's training rows are whitened with its subject's
     reference (all the subject's training windows), exactly as the solver's
     align="subject_context" builds W_pair (``_inv_sqrtm(_mean_cov(...))``);
  3. test windows routed to the small pair use that reference too;
  4. info["ctx_small_pairs"] == 1, and 0 on a study without small pairs
     (then the whole aligned X is bit-identical to the reference).

    python ~/codabench/analysis/ctxmin_check.py --ref <pre-change sealed_run.py copy>
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import xsess_lib as L  # noqa: E402
import sealed_run as SR  # noqa: E402
import riemann_sealed as R  # noqa: E402


def synth(small=6, n=40, C=8, T=480, seed=0):
    rng = np.random.default_rng(seed)
    X, subj, sess, ctx = [], [], [], []
    for s in range(4):
        A = np.eye(C) + 0.3 * rng.standard_normal((C, C))
        for c in range(2):
            B = A @ (np.eye(C) + 0.2 * rng.standard_normal((C, C)))
            for v in range(2):
                m = small if (s, c, v) == (1, 1, 0) else n
                Z = rng.standard_normal((m, C, T)) * (1 + 0.5 * s)
                X.append(np.einsum("ij,njt->nit", B, Z))
                subj += [s] * m
                sess += [v] * m
                ctx += [c] * m
    X = np.concatenate(X).astype(np.float32)
    subj, sess, ctx = map(np.asarray, (subj, sess, ctx))
    d = {"X": X, "subj": subj, "sess": sess, "ctx": ctx, "has_ctx": True,
         "y": rng.integers(0, 3, len(subj)), "meta": {"sfreq": 120.0}}
    return d, sess == 0, sess == 1


def run(mod, d, tr, te):
    return mod.aligned_data(d, tr, te, "router-psdctx:riemann", {}, 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("sealed_run_ref", a.ref)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    ok = True
    # no small pair: bit-identical
    d, tr, te = synth(small=40)
    Xn, info_n = run(SR, d, tr, te)
    Xr, _ = run(ref, d, tr, te)
    same = np.array_equal(Xn, Xr)
    ok &= same and info_n.get("ctx_small_pairs") == 0
    print(f"[1] no small pair: aligned X bit-identical {same}; "
          f"ctx_small_pairs={info_n.get('ctx_small_pairs')}")
    # one small pair
    d, tr, te = synth(small=6)
    Xn, info_n = run(SR, d, tr, te)
    Xr, info_r = run(ref, d, tr, te)
    grp = d["subj"] * 2 + d["ctx"]
    small = tr & (grp == 3)                       # subject 1, context 1
    other_tr = tr & ~small
    covs = L.window_covs(d["X"])
    W_subj = R._inv_sqrtm(R._mean_cov(R._window_covs(d["X"][tr & (d["subj"] == 1)]), "riemann"))
    W_h = L.inv_sqrtm(L.mean_cov(covs[tr & (d["subj"] == 1)], "riemann"))
    exp = L.apply_W(d["X"][small], W_h)
    te_idx = np.where(te)[0]
    a_n = np.array(info_n["router_assign"])
    to_small = te_idx[a_n == 3]
    not_small = te_idx[a_n != 3]
    c1 = np.array_equal(Xn[other_tr], Xr[other_tr])
    c2 = np.array_equal(Xn[small], exp)
    c2s = float(np.abs(W_subj - W_h).max())
    c3 = (np.array_equal(Xn[to_small], L.apply_W(d["X"][to_small], W_h)) if len(to_small)
          else True)
    c3b = np.array_equal(Xn[not_small], Xr[not_small])
    c4 = info_n.get("ctx_small_pairs") == 1
    print(f"[2] other pairs' training rows bit-identical {c1}; small pair = subject "
          f"reference {c2} (solver W_subj vs harness W: max |d| {c2s:.1e}); "
          f"test rows routed to it ({len(to_small)}) use it {c3}; other test rows "
          f"identical {c3b}; ctx_small_pairs={info_n.get('ctx_small_pairs')}; the "
          f"reference kept the pair's own W: {not np.array_equal(Xr[small], exp)}")
    ok &= c1 and c2 and c3 and c3b and c4 and c2s < 1e-12
    print(f"CTXMIN CHECK: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
