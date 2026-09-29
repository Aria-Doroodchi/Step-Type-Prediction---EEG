#!/usr/bin/env python
"""Sprint 2026-09-29 Phase 4 gates for the solver's opt-in ``xblocks``.

1. Defaults bit-identical: a reference copy of riemann_sealed.py (``--ref``,
   e.g. ``git show HEAD:codabench/solvers/bci_decoding/riemann_sealed.py``)
   vs the working copy, recipe settings (align=subject, personal=blend,
   blend_w=0.5, filter bank) on zhou2016, last-session split: max |dP| must be
   0, with xblocks left at its default and with xblocks="" passed explicitly.
2. Harness parity: the solver (align=none, personal=pooled, xblocks=<X>) vs
   the harness spec riemann:xd=1,fb=1,x=<X> on Scherer 3-class (train =
   training sessions, predict = test session): features and probabilities.

    python ~/codabench/analysis/xblocks_gate.py --ref <head copy> --xblocks tseg3_bpt4
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import xsess_lib as L  # noqa: E402  (also puts the solvers + benchmark_utils on the path)
import riemann_sealed as R  # noqa: E402


def loader(X, y, subj, bs=256):
    return [(torch.from_numpy(X[i:i + bs]), torch.from_numpy(y[i:i + bs]),
             {"subject_id": torch.from_numpy(subj[i:i + bs])}) for i in range(0, len(y), bs)]


def model(mod, meta, **kw):
    pre = mod.WindowPreproc(meta["ch_names"], meta["sfreq"], "none", "none")
    return mod.RiemannSealedModel(None, pre, ch_names=list(meta["ch_names"]), **kw)


def split(study, classes=None):
    d = L.load_study(study, classes)
    sp = L.xsess_split(d)
    tr, te = np.where(sp["train"])[0], np.where(sp["test"])[0]
    return d, tr, te


def gate_defaults(ref_path):
    spec = importlib.util.spec_from_file_location("riemann_sealed_ref", ref_path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    d, tr, te = split("zhou2016")
    X, y, s, meta = d["X"], d["y"], d["subj"], d["meta"]
    print(f"[gate 1] zhou2016 X={X.shape} train={len(tr)} test={len(te)}", flush=True)
    kw = dict(personal="blend", blend_w=0.5)
    P = {}
    for name, mod, extra in (("ref", ref, {}), ("new", R, {}), ("new-empty", R, {"xblocks": ""})):
        m = model(mod, meta, **kw, **extra).fit(loader(X[tr], y[tr], s[tr]))
        P[name] = m.predict_proba(X[te])
    ok = True
    for k in ("new", "new-empty"):
        dp = float(np.abs(P[k] - P["ref"]).max())
        ok &= dp == 0.0
        print(f"[gate 1] max |dP| ref vs {k} = {dp!r}", flush=True)
    print(f"GATE 1 (defaults bit-identical): {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def gate_parity(xblocks):
    d, tr, te = split("scherer2015", [0, 1, 3])
    X, y, s, meta = d["X"], d["y"], d["subj"], dict(d["meta"], n_classes=3)
    print(f"[gate 2] scherer2015 [0,1,3] X={X.shape} train={len(tr)} test={len(te)} "
          f"xblocks={xblocks}", flush=True)
    t0 = time.time()
    h = L.make_model("riemann:xd=1,fb=1,x=" + "+".join(R._parse_xblocks(xblocks)), meta)
    h.fit(X[tr], y[tr])
    Fh, Ph = h.features(X[te]), h.predict_proba(X[te])
    t1 = time.time()
    m = model(R, meta, align="none", personal="pooled", xblocks=xblocks)
    m.fit(loader(X[tr], y[tr], s[tr]))
    Xp = m._prep(X[te], m.parts.get("chan_idx"))
    Fs, Ps = R._features(m.parts, Xp), m.predict_proba(X[te])
    t2 = time.time()
    df = float(np.abs(Fh - Fs).max() / max(np.abs(Fh).max(), 1e-300))
    dp = float(np.abs(Ph - Ps).max())
    acc_h = float((Ph.argmax(1) == y[te]).mean())
    acc_s = float((Ps.argmax(1) == y[te]).mean())
    print(f"[gate 2] features {Fh.shape} vs {Fs.shape}; max rel |dF| = {df:.3e}; "
          f"max |dP| = {dp:.3e}; acc harness {acc_h:.4f} solver {acc_s:.4f}; "
          f"harness {t1 - t0:.0f} s, solver {t2 - t1:.0f} s", flush=True)
    ok = Fh.shape == Fs.shape and df <= 1e-10 and dp <= 1e-8
    print(f"GATE 2 (harness parity, {xblocks}): {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference riemann_sealed.py (HEAD copy)")
    ap.add_argument("--xblocks", nargs="+", default=["tseg3_bpt4"])
    ap.add_argument("--skip_defaults", action="store_true")
    args = ap.parse_args()
    ok = True if args.skip_defaults else gate_defaults(args.ref)
    for xb in args.xblocks:
        ok &= gate_parity(xb)
    print("ALL GATES PASS" if ok else "SOME GATE FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
