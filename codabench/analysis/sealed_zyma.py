#!/usr/bin/env python
"""Phase 4 sanity check: is "mental calculation" detectable at all?
Zyma 2019: mental arithmetic vs rest, 35 people, one session each, 5-s
windows (19 EEG channels after dropping A2-A1). Cross-subject only: 5-fold
GroupKFold by subject. Riemann-StepType variants (+ block ablations),
MeanLogReg, and the same with per-subject Euclidean/Riemannian alignment
(each test subject whitened with its own unlabelled windows = oracle, since a
cross-subject split has no calibration session).

    python ~/codabench/analysis/sealed_zyma.py
Rows -> ~/codabench/logs/sealed_p4/results_zyma2019.jsonl.
"""

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import xsess_lib as L  # noqa: E402

SPECS = ["meanlr", "riemann:xd=1,fb=1", "riemann:xd=0,fb=1", "riemann:xd=1,fb=0",
         "riemann:xd=1,fb=1,blocks=fb", "riemann:xd=1,fb=1,blocks=xdawn",
         "riemann:xd=1,fb=1,blocks=broad+logvar"]


def main():
    out = Path.home() / "codabench/logs/sealed_p4"
    out.mkdir(parents=True, exist_ok=True)
    res = out / "results_zyma2019.jsonl"
    done = {r["key"] for r in L.read_results(out / "results.jsonl")}
    d = L.load_study("zyma2019")
    keep = [i for i, c in enumerate(d["meta"]["ch_names"]) if c != "A2-A1"]
    X = np.ascontiguousarray(d["X"][:, keep])
    meta = dict(d["meta"], ch_names=[d["meta"]["ch_names"][i] for i in keep], n_classes=2)
    y, subj = d["y"], d["subj"]
    L.log(f"[data] zyma2019 X={X.shape} subjects={len(np.unique(subj))} "
          f"classes={np.bincount(y).tolist()} (0 arithmetic, 1 rest); 5-fold GroupKFold")
    covs = None
    for align in ("none", "subject:euclid", "subject:riemann"):
        Xa = X
        if align != "none":
            covs = L.window_covs(X) if covs is None else covs
            Xa, _ = L.align_groups(X, subj, align.split(":")[1], covs)
        for spec in SPECS:
            key = f"zyma2019|all|{spec}|xsubject5|{align}|33"
            if key in done:
                continue
            t0 = time.time()
            yhat = np.zeros(len(y), int)
            for tr, te in GroupKFold(5).split(Xa, y, subj):
                m = L.make_model(spec, meta).fit(Xa[tr], y[tr])
                yhat[te] = m.predict_proba(Xa[te]).argmax(1)
            per = [balanced_accuracy_score(y[subj == s], yhat[subj == s])
                   for s in np.unique(subj) if len(np.unique(y[subj == s])) > 1]
            row = dict(key=key, study="zyma2019", classes=None, spec=spec, mode="xsubject5",
                       align=align, seed=33, cell=float(np.mean(per)),
                       pooled=float(balanced_accuracy_score(y, yhat)), n_cells=len(per),
                       per_subject={}, n_train=int(len(y) * 0.8), n_test=len(y),
                       shape=list(X.shape), seconds=round(time.time() - t0, 1),
                       time=time.strftime("%H:%M:%S"))
            L.append_result(res, row)
            L.log(f"[done] {key} per-subject mean={row['cell']:.4f} pooled={row['pooled']:.4f} "
                  f"{row['seconds']:.0f}s")
    L.log("ALL CONFIGS DONE")


if __name__ == "__main__":
    main()
