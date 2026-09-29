#!/usr/bin/env python
"""Self-test of the xfeat hook (sprint 2026-09-29): a spec without ``x=`` builds
the plain RiemannModel; a spec with ``x=`` builds RiemannXModel; a trivial
registered block runs end to end on a small slice of zhou2016.

    python ~/codabench/analysis/xfeat_selftest.py [block_id ...]

With block ids, also fits/predicts each named block on the slice and prints
its feature count, time and a train-slice accuracy (a pipeline check only).
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import xfeat  # noqa: E402
import xsess_lib as L  # noqa: E402


@xfeat.register(r"_selftest")
def _selftest(meta):
    class LogVar(xfeat.Block):
        def transform(self, X):
            return np.log(np.var(X, axis=2) + 1e-12)
    return LogVar(meta)


def main():
    d = L.load_study("zhou2016")
    meta = dict(d["meta"], n_classes=int(d["y"].max() + 1))
    m0 = L.make_model("riemann:xd=1,fb=1", meta)
    assert type(m0) is L.RiemannModel, type(m0)
    m1 = L.make_model("riemann:xd=1,fb=1,sl=1", meta)
    assert type(m1) is L.RiemannModel and m1.kw.get("slow_block") is True
    assert "slow_block" not in m0.kw          # default kwargs unchanged
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(d["y"]), 240, replace=False))
    X, y = d["X"][idx], d["y"][idx]
    print(f"slice X={X.shape} classes={np.unique(y).tolist()} ch={len(meta['ch_names'])}")
    ids = ["_selftest"] + sys.argv[1:]
    for bid in ids:
        t0 = time.time()
        m = L.make_model(f"riemann:xd=1,fb=1,x={bid}", meta)
        assert type(m) is xfeat.RiemannXModel
        m.fit(X[:160], y[:160])
        P = m.predict_proba(X[160:])
        F = m.features(X[:5])
        assert P.shape == (80, meta["n_classes"]) and np.all(np.isfinite(P))
        acc = (m.predict_proba(X[:160]).argmax(1) == y[:160]).mean()
        print(f"OK {bid}: features={F.shape[1]} train-acc={acc:.3f} "
              f"test-acc={(P.argmax(1) == y[160:]).mean():.3f} {time.time() - t0:.1f}s")
    print("SELFTEST PASSED")


if __name__ == "__main__":
    main()
