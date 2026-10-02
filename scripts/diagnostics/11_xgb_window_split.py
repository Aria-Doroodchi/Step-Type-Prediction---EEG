"""Does the XGB headline feature set also get its information from the first 0.5 s?

Uses the cached rich_mean_0125 full-window features (columns end in _bin_<k>, 125 ms bins; k < 4 = 0-0.5 s).
Per participant, fixed shallow XGB, repeated stratified 5-fold (x3), AUC, for column sets:
all / bins>=4 (0.5-2 s) / bins<4 (0-0.5 s) / bins>=8 (1-2 s, = the 'late' window).
Resumable (appends per participant). Usage: python 11_xgb_window_split.py [feature_dir] [out_csv]
"""
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
fdir = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Ali D\Documents\ML\data\features")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "outputs/diagnostics/window_effect/xgb_split.csv")
out.parent.mkdir(parents=True, exist_ok=True)
SUFFIX = "_t0p0-2p0_bin_rich_mean_0125.parquet"
pids = sorted({f.name.split("_")[0] for f in fdir.glob("P*_One_features" + SUFFIX)})
done = set(pd.read_csv(out)["participant"]) if out.exists() else set()
META = {"epoch", "condition", "participant_id", "block_id"}
print(time.strftime("%H:%M:%S"), len(pids), "participants", flush=True)

for pid in pids:
    if pid in done:
        continue
    t0 = time.time()
    d = pd.concat([pd.read_parquet(fdir / f"{pid}_{c}_features{SUFFIX}").assign(y=i) for i, c in enumerate(["One", "Two"])],
                  ignore_index=True)
    y = d["y"].to_numpy()
    cols = [c for c in d.columns if c not in META and c != "y"]
    kbin = {c: (int(m.group(1)) if (m := re.search(r"_bin_(\d+)$", c)) else None) for c in cols}
    sets = {
        "all": [c for c in cols],
        "ge4": [c for c in cols if kbin[c] is not None and kbin[c] >= 4],
        "lt4": [c for c in cols if kbin[c] is not None and kbin[c] < 4],
        "ge8": [c for c in cols if kbin[c] is not None and kbin[c] >= 8],
    }
    X = d[cols].to_numpy(dtype=np.float32)
    ci = {c: i for i, c in enumerate(cols)}
    rows = []
    for name, cs in sets.items():
        Xs = X[:, [ci[c] for c in cs]]
        aucs = []
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1).split(Xs, y):
            m = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.3,
                              tree_method="hist", n_jobs=8, verbosity=0).fit(Xs[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(Xs[te])[:, 1]))
        rows.append(dict(participant=pid, set=name, n_feat=len(cs), auc=float(np.mean(aucs))))
    pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
    print(time.strftime("%H:%M:%S"), pid, {r["set"]: round(r["auc"], 3) for r in rows}, f"{time.time() - t0:.0f}s", flush=True)

r = pd.read_csv(out)
print(r.groupby("set").auc.agg(["mean", "std", "count"]))
