#!/usr/bin/env python
"""Sprint 2026-10-01: does a Riemann-Sealed joblib (pickled with sklearn 1.9.1,
pyriemann 0.12, numpy 2.5) load and predict identically under other library
versions? The scoring image installs scikit-learn unpinned at build time, so
its version is unknown. Run this once per environment; each run writes the
outputs of every pickled estimator on fixed random inputs, and --compare
reports the largest difference against a reference run.

    <python> joblib_compat_check.py <riemann_sealed.joblib> <out.npz>
    <python> joblib_compat_check.py --compare ref.npz other.npz

Needs only numpy, scipy, scikit-learn, pyriemann, joblib (no torch).
"""

import sys
import warnings

import numpy as np


def run(path, out):
    import joblib
    import scipy
    import sklearn
    import pyriemann
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        parts = joblib.load(path)
    msgs = sorted({f"{x.category.__name__}: {str(x.message)[:120]}" for x in w})
    rng = np.random.default_rng(0)
    res = {}
    lda = parts["lda"]
    F = rng.standard_normal((50, lda.n_features_in_))
    res["lda"] = lda.predict_proba(F)
    for k, m in enumerate(parts.get("lda_subject") or []):
        if m is not None:
            res[f"lda_subject_{k}"] = m.predict_proba(F)
    r = parts["router"]["lda"]
    res["router"] = r.predict_proba(rng.standard_normal((50, r.n_features_in_)))
    C = len(parts["chan_idx"]) if parts.get("chan_idx") is not None else parts["W_global"].shape[0]
    A = rng.standard_normal((50, C, C))
    covs = A @ A.transpose(0, 2, 1) + C * np.eye(C)[None]
    res["broad_ts"] = parts["broad_ts"].transform(covs)
    for i, ts in enumerate(parts.get("fb_ts") or []):
        res[f"fb_ts_{i}"] = ts.transform(covs)
    if parts.get("xdawn") is not None:
        T = int(round(4 * parts["sfreq"]))
        Xd = parts["xdawn"].transform(rng.standard_normal((20, C, T)))
        res["xdawn"] = Xd
        res["xdawn_ts"] = parts["xdawn_ts"].transform(Xd)
    np.savez(out, **res)
    print(f"python {sys.version.split()[0]} numpy {np.__version__} scipy {scipy.__version__} "
          f"sklearn {sklearn.__version__} pyriemann {pyriemann.__version__}: loaded, "
          f"{len(res)} outputs; load warnings: {msgs or 'none'}")


def compare(a, b):
    A, B = np.load(a), np.load(b)
    worst = 0.0
    for k in A.files:
        if k not in B.files:
            print(f"MISSING {k}")
            worst = np.inf
            continue
        d = float(np.abs(A[k] - B[k]).max())
        worst = max(worst, d)
    print(f"{b} vs {a}: {len(A.files)} outputs, max |diff| {worst:.2e} -> "
          f"{'SAME' if worst < 1e-9 else 'DIFFERENT'}")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1], sys.argv[2])
