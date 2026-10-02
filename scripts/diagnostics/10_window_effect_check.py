"""Window-effect integrity check (sprint 2026-10-02).

Questions: is the full-CNV (0-2 s) vs late (1-2 s) AUC gap real, or an artefact of
(a) shuffled CV on temporally autocorrelated epochs, (b) a trigger-locked sensory response in 0-0.5 s?
See docs/sprints/2026-10-02_window_effect_check.md for the pre-registered rules.

Usage:
    python scripts/diagnostics/10_window_effect_check.py run  [--epochs DIR] [--out DIR] [--limit N]
    python scripts/diagnostics/10_window_effect_check.py summarize --out DIR
Resumable: finished participants are skipped.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BIN = 0.125
EDGES = np.round(np.arange(0.0, 2.0 + 1e-9, BIN), 4)  # 16 bins over 0-2 s
WINDOWS = {  # name -> (start, end) in seconds; "base" is the single pre-trigger bin
    "full": (0.0, 2.0), "late": (1.0, 2.0), "early": (0.0, 0.5), "mid": (0.5, 1.0),
    "no_early": (0.5, 2.0), "base": (-0.1, 0.0),
}
for i in range(8):  # sliding 250 ms
    WINDOWS[f"s{i * 0.25:.2f}"] = (i * 0.25, i * 0.25 + 0.25)
SLIDING = [k for k in WINDOWS if k.startswith("s")]


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_participant(epochs_dir: Path, pid: str):
    import mne
    mne.set_log_level("ERROR")
    xs, ys, ts = [], [], []
    for lab, cond in enumerate(["One", "Two"]):
        f = epochs_dir / f"{pid}_CNV_{cond}-epo.fif"
        if not f.exists():
            return None
        ep = mne.read_epochs(f, preload=True).pick("csd")
        xs.append(ep.get_data() * 1e6)
        ys.append(np.full(len(ep), lab))
        ts.append(ep.events[:, 0])
        times = ep.times
    x, y, t = np.concatenate(xs), np.concatenate(ys), np.concatenate(ts)
    o = np.argsort(t, kind="stable")  # recording order
    return x[o], y[o], t[o], times


def bin_epochs(x: np.ndarray, times: np.ndarray) -> dict[str, np.ndarray]:
    """Mean amplitude per channel per 125 ms bin -> (n, ch, nbins) and the baseline bin."""
    out = np.stack(
        [x[:, :, (times >= a) & (times < b)].mean(-1) for a, b in zip(EDGES[:-1], EDGES[1:])], axis=-1
    )
    base = x[:, :, (times >= -0.1) & (times < 0.0)].mean(-1)[:, :, None]
    return {"bins": out, "base": base}


def window_features(b: dict, win: str) -> np.ndarray:
    a, c = WINDOWS[win]
    if win == "base":
        f = b["base"]
    else:
        idx = [i for i in range(len(EDGES) - 1) if EDGES[i] >= a - 1e-9 and EDGES[i + 1] <= c + 1e-9]
        f = b["bins"][:, :, idx]
    return f.reshape(len(f), -1)


def model():
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


def cv_auc(X, y, scheme: str, seed: int = 1) -> float:
    aucs = []
    if scheme == "shuffled":
        sp = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed).split(X, y)
    else:  # contiguous folds in recording order, purge 1 epoch on each side of the test block
        def gen():
            n = len(y)
            for _, te in KFold(5, shuffle=False).split(np.arange(n)):
                tr = np.setdiff1d(np.arange(n), np.arange(max(te[0] - 1, 0), min(te[-1] + 2, n)))
                yield tr, te
        sp = gen()
    for tr, te in sp:
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        m = model().fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.decision_function(X[te])))
    return float(np.mean(aucs)) if aucs else np.nan


def run(args) -> None:
    epochs_dir, out = Path(args.epochs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pids = sorted({f.name.split("_")[0] for f in epochs_dir.glob("P*_CNV_One-epo.fif")})
    if args.limit:
        pids = pids[: args.limit]
    log(f"epochs dir {epochs_dir}: {len(pids)} participants")
    within_csv, perm_csv, binned_npz = out / "within.csv", out / "perm.csv", out / "binned.npz"
    done = set(pd.read_csv(within_csv)["participant"]) if within_csv.exists() else set()
    rng = np.random.default_rng(0)
    t_all = time.time()
    for k, pid in enumerate(pids):
        if pid in done:
            continue
        t0 = time.time()
        d = load_participant(epochs_dir, pid)
        if d is None:
            log(f"{pid}: missing a condition file, skipped")
            continue
        x, y, t, times = d
        b = bin_epochs(x, times)
        rows = []
        for win in WINDOWS:
            X = window_features(b, win)
            for scheme in ["shuffled", "chrono"]:
                rows.append(dict(participant=pid, window=win, scheme=scheme, n=len(y), n_feat=X.shape[1],
                                 auc=cv_auc(X, y, scheme)))
        Xf = window_features(b, "full")
        prow = []
        for p in range(args.perms):
            yp = rng.permutation(y)
            prow.append(dict(participant=pid, perm=p, auc=_quick_auc(Xf, yp, seed=p)))
        np.savez_compressed(out / f"binned_{pid}.npz", bins=b["bins"], base=b["base"], y=y, t=t)
        pd.DataFrame(prow).to_csv(perm_csv, mode="a", header=not perm_csv.exists(), index=False)
        pd.DataFrame(rows).to_csv(within_csv, mode="a", header=not within_csv.exists(), index=False)
        full = [r["auc"] for r in rows if r["window"] == "full"]
        log(f"[{k + 1}/{len(pids)}] {pid} n={len(y)} full shuffled={full[0]:.3f} chrono={full[1]:.3f} "
            f"({time.time() - t0:.0f}s)")
    log(f"within-participant part done in {time.time() - t_all:.0f}s; starting LOSO")
    loso(out)


def _quick_auc(X, y, seed: int) -> float:
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = model().fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.decision_function(X[te])))
    return float(np.mean(aucs))


def loso(out: Path) -> None:
    files = sorted(out.glob("binned_P*.npz"))
    data = {}
    for f in files:
        z = np.load(f)
        data[f.stem.split("_")[1]] = (z["bins"], z["base"], z["y"])
    pids = sorted(data)
    rows = []
    for win in WINDOWS:
        feats = {}
        for p in pids:
            bins, base, y = data[p]
            X = window_features({"bins": bins, "base": base}, win)
            feats[p] = ((X - X.mean(0)) / (X.std(0) + 1e-9), y)  # per-participant z-score (label-free)
        for p in pids:
            tr = [q for q in pids if q != p]
            Xtr = np.concatenate([feats[q][0] for q in tr])
            ytr = np.concatenate([feats[q][1] for q in tr])
            m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, ytr)
            rows.append(dict(participant=p, window=win, auc=roc_auc_score(feats[p][1], m.decision_function(feats[p][0]))))
        log(f"LOSO {win}: mean AUC {np.mean([r['auc'] for r in rows if r['window'] == win]):.3f}")
    pd.DataFrame(rows).to_csv(out / "loso.csv", index=False)


def boot_ci(v: np.ndarray, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    m = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(args) -> None:
    out = Path(args.out)
    w = pd.read_csv(out / "within.csv")
    piv = w.pivot_table(index=["participant", "scheme"], columns="window", values="auc").reset_index()
    perm = pd.read_csv(out / "perm.csv").groupby("participant")["auc"].mean()
    L = []
    fmt = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
    L.append(f"participants: {w.participant.nunique()}\n")
    L.append("## Within-participant AUC (cohort mean, participant-bootstrap 95 % CI)\n")
    L.append("| window | shuffled CV | chronological CV | shuffled - chrono |")
    L.append("|---|---|---|---|")
    P = {s: piv[piv.scheme == s].set_index("participant") for s in ["shuffled", "chrono"]}
    for win in WINDOWS:
        a, c = P["shuffled"][win], P["chrono"][win]
        L.append(f"| {win} | {fmt(boot_ci(a))} | {fmt(boot_ci(c))} | {fmt(boot_ci(a - c))} |")
    L.append(f"\npermutation null (full window, shuffled CV): mean {perm.mean():.3f}\n")
    full_gap = P["shuffled"]["full"] - P["chrono"]["full"]
    base_ex = P["shuffled"]["base"].mean() - perm.mean()
    d_chrono = P["chrono"]["full"] - P["chrono"]["late"]
    d_shuf = P["shuffled"]["full"] - P["shuffled"]["late"]
    g = boot_ci(full_gap)
    r1 = "FLAG" if (g[0] > 0.04 and g[1] > 0) else ("NO LEAKAGE" if g[0] <= 0.02 else "inconclusive")
    r2 = "FLAG" if base_ex > 0.03 else "clean"
    dc = boot_ci(d_chrono)
    r3 = "SURVIVES" if (dc[0] > 0.03 and dc[1] > 0) else "NOT CONFIRMED"
    L.append("## Pre-registered rules\n")
    L.append(f"- R1 temporal leakage (full: shuffled - chrono): {fmt(g)} -> **{r1}**")
    L.append(f"- R2 baseline (shuffled AUC {P['shuffled']['base'].mean():.3f} vs permutation null {perm.mean():.3f}, excess {base_ex:+.3f}) -> **{r2}**")
    L.append(f"- R3 window effect under chronological CV (full - late): {fmt(dc)} -> **{r3}**  "
             f"(shuffled CV: {fmt(boot_ci(d_shuf))})")
    lo = out / "loso.csv"
    if lo.exists():
        l = pd.read_csv(lo).pivot_table(index="participant", columns="window", values="auc")
        L.append("\n## Cross-subject (LOSO) AUC by window\n")
        L.append("| window | LOSO AUC |")
        L.append("|---|---|")
        for win in WINDOWS:
            L.append(f"| {win} | {fmt(boot_ci(l[win]))} |")
        e = l["early"].mean()
        r4 = "cue/sensory response FLAG" if e >= 0.60 else "no consistent early response"
        L.append(f"\n- R4 early (0-0.5 s) LOSO AUC {e:.3f} (threshold 0.60) -> **{r4}**; "
                 f"no_early (0.5-2 s) LOSO {l['no_early'].mean():.3f}, full {l['full'].mean():.3f}")
    text = "\n".join(L)
    (out / "RESULTS.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summarize"])
    ap.add_argument("--epochs", default=r"C:\Users\Ali D\Documents\ML\data\interim\epochs")
    ap.add_argument("--out", default="outputs/diagnostics/window_effect")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--perms", type=int, default=100)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else summarize(a)
