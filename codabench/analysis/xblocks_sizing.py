#!/usr/bin/env python
"""Sprint 2026-09-29 Phase 4 sizing gate for the solver's opt-in ``xblocks``
at the sealed size: Riemann-Sealed fitted on the full-size 500 Hz mock
(``mock_sealed_500``: 20 participants x 6 sessions, 47 ch -> 43 EEG, 2000
samples) through the mock's own benchopt dataset class, so the solver sees
what benchopt gives it (shuffled train loader, trigger table with session and
context ids): personal="blend", blend_w="auto" (6 LOSO folds), chans="eeg".
Prints the solver's own stage timings and one SIZING line: fit seconds, peak
RSS, feature count, predict ms per window (first 1,200 test windows).

Gate (brief § 5 Phase 4): fit <= 1.5 x the recipe's fit and peak RSS
<= 16 GiB. Run each variant in its own process (clean peak RSS):

    python ~/codabench/analysis/xblocks_sizing.py --xblocks ""       # the recipe
    python ~/codabench/analysis/xblocks_sizing.py --xblocks bpt4
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import xsess_lib as L  # noqa: E402,F401  (solver + benchmark_utils on the path)
import riemann_sealed as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xblocks", default="")
    ap.add_argument("--study", default="mock_sealed_500")
    ap.add_argument("--blend_w", default="auto")
    ap.add_argument("--n_pred", type=int, default=1200)
    ap.add_argument("--wcv_ref", default="all", help="all | strict (sprint 2026-10-01)")
    args = ap.parse_args()
    p = Path.home() / "codabench/datasets/mock_sealed.py"
    spec = importlib.util.spec_from_file_location("mock_sealed_ds", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ds = mod.Dataset.__new__(mod.Dataset)
    ds.study, ds.batch_size, ds.split = args.study, 64, "organisers"
    data = ds.get_data()
    names = data["ch_names"]
    print(f"[sizing] {args.study}: train {len(data['train_loader'].dataset)} windows, "
          f"test {len(data['test_loader'].dataset)}, {data['n_chans']} ch x "
          f"{data['n_times']} samples @ {data['sfreq']} Hz, xblocks={args.xblocks!r}, "
          f"blend_w={args.blend_w}, wcv_ref={args.wcv_ref}, threads={L.N_THREADS}", flush=True)
    pre = R.WindowPreproc(names, data["sfreq"], "none", "none")
    m = R.RiemannSealedModel(None, pre, personal="blend", blend_w=args.blend_w,
                             chans="eeg", ch_names=names, chs_info=data["chs_info"],
                             xblocks=args.xblocks, wcv_ref=args.wcv_ref)
    t0 = time.time()
    m.fit(data["train_loader"])
    fit_s = time.time() - t0
    Xs, n = [], 0
    for Xb, _, _ in data["test_loader"]:
        Xs.append(Xb.numpy())
        n += len(Xb)
        if n >= args.n_pred:
            break
    X = np.concatenate(Xs)[:args.n_pred]
    t1 = time.time()
    P = m.predict_proba(X)
    pred_ms = 1000 * (time.time() - t1) / len(X)
    n_feat = m.parts["lda"].coef_.shape[1]
    print(f"SIZING xblocks={args.xblocks or 'none'}{'' if args.wcv_ref == 'all' else ' wcv_ref=' + args.wcv_ref} fit_s={fit_s:.1f} "
          f"maxrss_gb={R._maxrss_gb():.2f} features={n_feat} "
          f"blend_w={m.parts.get('blend_w')} predict_ms_per_window={pred_ms:.1f} "
          f"(n={len(X)}, finite={bool(np.isfinite(P).all())})", flush=True)


if __name__ == "__main__":
    main()
