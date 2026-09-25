#!/usr/bin/env python
"""Subject routing without ids: how often does each fingerprint send a test
(last-session) window to its true training subject? No decoder is fitted.

    python ~/codabench/analysis/router_eval.py [study ...] > router.md

Fingerprints: dist-euclid / dist-riemann (Riemannian distance to each
subject's mean covariance), ts / fbts / lv (shrinkage-LDA subject classifier on
tangent-space, filter-bank tangent-space, per-band log-variance features).
Reports per-window accuracy, 64-window contiguous batch-vote accuracy
(recording order), fallback fraction and the training OOF accuracy.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import xsess_lib as L  # noqa: E402
from sealed_run import contiguous_batches, fingerprint_dist  # noqa: E402


def batch_vote(D, subjects):
    a = np.empty(len(D), dtype=subjects.dtype)
    for b in contiguous_batches(len(D)):
        a[b] = subjects[D[b].sum(0).argmin()]
    return a


FPS = ["dist-euclid", "dist-riemann", "ts", "lv", "fbts", "psd", "lv+fbts", "psd+lv"]


def main():
    global FPS
    args = [a for a in sys.argv[1:] if not a.startswith("--fp=")]
    for a in sys.argv[1:]:
        if a.startswith("--fp="):
            FPS = a[5:].split(",")
    studies = args or ["zhou2016", "tangermann2012", "scherer2015"]
    print("| study | fingerprint | window acc | batch-64 vote acc | fallback | acc on fallback windows | train OOF acc | s |")
    print("|---|---|---|---|---|---|---|---|")
    for study in studies:
        d = L.load_study(study)
        sp = L.xsess_split(d)
        tr, te = sp["train"], sp["test"]
        X, subj, sess = d["X"], d["subj"], d["sess"]
        covs = L.window_covs(X)
        for fp in FPS:
            t0 = time.time()
            if fp.startswith("dist"):
                subjects, _, D, thr = fingerprint_dist(covs[tr], subj[tr], covs[te], fp.split("-")[1])
                oof = float("nan")
            else:
                r = L.SubjectRouter(fp, d["meta"]["sfreq"]).fit(X[tr], subj[tr], sess[tr])
                P = r.predict_proba(X[te])
                subjects, D, thr = r.subjects, -np.log(P + 1e-12), -np.log(max(r.thr, 1e-12))
                oof = r.oof_acc
            win = np.mean(subjects[D.argmin(1)] == subj[te])
            bat = np.mean(batch_vote(D, subjects) == subj[te])
            fbm = D.min(1) > thr
            fb = np.mean(fbm)
            ok = subjects[D.argmin(1)] == subj[te]
            acc_fb = np.mean(ok[fbm]) if fbm.any() else float("nan")
            print(f"| {study} | {fp} | {win:.3f} | {bat:.3f} | {fb:.3f} | {acc_fb:.3f} | {oof:.3f} | "
                  f"{time.time() - t0:.0f} |", flush=True)


if __name__ == "__main__":
    main()
