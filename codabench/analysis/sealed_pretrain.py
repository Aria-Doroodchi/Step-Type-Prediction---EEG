#!/usr/bin/env python
"""Phase 5: cross-dataset pre-training of EEGNet-StepType on the motor-imagery
datasets, fine-tuned on Scherer 2015 (the sealed-like 3-class subset).

    python sealed_pretrain.py pretrain --align none --seed 33
    python sealed_pretrain.py pretrain --align euclid --seed 33
    python sealed_pretrain.py finetune --align none --seeds 33 34 35
    python sealed_pretrain.py finetune --align trainonly:euclid --seeds 33 34 35
    python sealed_pretrain.py riemann

Channels: the 11 shared by Dreyer 2023, Tangermann 2012 and Scherer 2015
(Fz FC3 FCz FC4 C3 Cz C4 CP3 CPz CP4 Pz). Zhou 2016 lacks Fz and Pz; they are
spline-interpolated from its 14 channels (MNE, the interpolate_bads matrix).
Everything is already 120 Hz / 4 s = 480 samples (NeuralBench resampling).

Pre-training: one EEGNet-StepType trunk shared by all MI datasets, one linear
head per dataset (Dreyer 2-class, Tangermann 4-class, Zhou 3-class), every
window of every session, CE loss; early stopping on 10 % held-out subjects
per dataset (max 60 epochs, patience 10). ``--align euclid|riemann`` whitens
each (subject, session) group before pre-training (Euclidean alignment).
Trunk weights -> ~/neuralbench/pretrain/eegnet_mi11_<align>_s<seed>.pt.

Fine-tuning on Scherer (labels WORD, SUB, HAND; 11 channels): the usual
cross-session split (session 0 -> train, session 1 -> test), per-subject and
pooled, (a) from scratch (lr 1e-3) vs (b) initialised from the pre-trained
trunk with a new head (lr 5e-4), same ES-then-refit protocol as Phase 1;
(c) = (b) with the aligned pre-training and the given Scherer alignment.
Riemann analogue: filter-bank tangent space (+ broadband) whose reference
point is the Riemannian mean of Scherer's training covariances vs of the
pooled MI + Scherer covariances.
Rows -> ~/codabench/logs/sealed_p5/results_scherer2015.jsonl.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import xsess_lib as L  # noqa: E402
from sealed_run import aligned_data, config_key  # noqa: E402

HOME = Path.home()
CH11 = ["Fz", "FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4", "Pz"]
SCHERER3 = [0, 1, 3]          # WORD, SUB, HAND
import os  # noqa: E402
PRE_DIR = Path(os.environ.get("XS_PRE_DIR", HOME / "neuralbench/pretrain"))
OUT = Path(os.environ.get("XS_P5_OUT", HOME / "codabench/logs/sealed_p5"))
SMOKE_EPOCHS = int(os.environ.get("XS_SMOKE_EPOCHS", "0"))   # >0: cap epochs (smoke tests)


# ---------------------------------------------------------------------------
# channel harmonisation
# ---------------------------------------------------------------------------
def interp_matrix(src, dst):
    """(len(dst), len(src)) linear map: known channels copied, missing ones
    spherical-spline interpolated (MNE interpolate_bads on an identity
    'recording')."""
    import mne
    low = {c.lower(): i for i, c in enumerate(src)}
    missing = [c for c in dst if c.lower() not in low]
    M = np.zeros((len(dst), len(src)))
    for r, c in enumerate(dst):
        if c.lower() in low:
            M[r, low[c.lower()]] = 1.0
    if missing:
        names = list(src) + missing
        info = mne.create_info(names, 100.0, "eeg")
        data = np.zeros((len(names), len(src)))
        data[: len(src)] = np.eye(len(src))
        raw = mne.io.RawArray(data, info, verbose=False)
        raw.set_montage(mne.channels.make_standard_montage("standard_1005"),
                        match_case=False, verbose=False)
        raw.info["bads"] = missing
        raw.interpolate_bads(reset_bads=True, verbose=False)
        out = raw.get_data()
        for r, c in enumerate(dst):
            if c in missing:
                M[r] = out[names.index(c)]
    return M


def harmonise(X, ch_names, dst=CH11):
    M = interp_matrix(ch_names, dst).astype(np.float32)
    return np.einsum("ij,njt->nit", M, X).astype(np.float32)


def load_pool():
    """MI pre-training pool on CH11: list of dict(X, y, subj, sess, name, K)."""
    pool = []
    dm = json.loads((HOME / "neuralbench/eda_cache/meta.json").read_text())
    Xd = np.load(HOME / "neuralbench/eda_cache/X.npy", mmap_mode="r")
    sd = np.load(HOME / "neuralbench/eda_cache/subj.npy")
    Xh = np.concatenate([harmonise(np.asarray(Xd[i:i + 4096]), dm["ch_names"])
                         for i in range(0, len(Xd), 4096)])
    pool.append(dict(name="dreyer2023", X=Xh, y=np.load(HOME / "neuralbench/eda_cache/y.npy"),
                     subj=np.unique(sd, return_inverse=True)[1], sess=np.zeros(len(sd), int)))
    for study in ("tangermann2012", "zhou2016"):
        d = L.load_study(study)
        pool.append(dict(name=study, X=harmonise(d["X"], d["meta"]["ch_names"]),
                         y=d["y"], subj=d["subj"], sess=d["sess"]))
    for p in pool:
        p["K"] = int(p["y"].max() + 1)
    return pool


def align_pool(pool, kind):
    for p in pool:
        groups = p["subj"] * 1000 + p["sess"]
        p["X"], _ = L.align_groups(p["X"], groups, kind)
    return pool


# ---------------------------------------------------------------------------
# multi-head pre-training
# ---------------------------------------------------------------------------
def trunk_forward(net, x):
    """EEGNetTorch.forward without the classifier -> (B, n_flat)."""
    import torch.nn.functional as F
    x = x.unsqueeze(1)
    x = net.temporal_bn(net.temporal_conv(net.temporal_pad(x)))
    x = F.elu(net.spatial_bn(net.spatial_depthwise(x)))
    x = net.dropout_1(net.pool_1(x))
    x = net.separable_pointwise(net.separable_depthwise(net.separable_pad(x)))
    x = F.elu(net.separable_bn(x))
    x = net.dropout_2(net.pool_2(x))
    return x.permute(0, 2, 3, 1).flatten(1)


def pretrain(align, seed, max_epochs=60, patience=10, lr=1e-3, bs=64):
    import copy
    import eegnet_steptype as ES
    PRE_DIR.mkdir(parents=True, exist_ok=True)
    path = PRE_DIR / f"eegnet_mi11_{align}_s{seed}.pt"
    if path.exists():
        L.log(f"pretrained trunk exists: {path}")
        return
    t0 = time.time()
    pool = load_pool()
    if align != "none":
        pool = align_pool(pool, align)
    rng = np.random.default_rng(seed)
    Xs, ys, ds, val = [], [], [], []
    for k, p in enumerate(pool):
        subs = np.unique(p["subj"])
        held = rng.choice(subs, max(1, round(0.1 * len(subs))), replace=False)
        Xs.append(p["X"]); ys.append(p["y"]); ds.append(np.full(len(p["y"]), k))
        val.append(np.isin(p["subj"], held))
        L.log(f"[data] pretrain {p['name']}: X={p['X'].shape} K={p['K']} "
              f"subjects={len(subs)} held_out={len(held)}")
    X, y, dsid, val = (np.concatenate(a) for a in (Xs, ys, ds, val))
    L.log(f"[data] pretrain pool X={X.shape} train={int((~val).sum())} val={int(val.sum())} "
          f"align={align} seed={seed} ({time.time() - t0:.0f}s to load)")
    torch.manual_seed(seed)
    net = ES.EEGNetTorch(n_channels=X.shape[1], n_times=X.shape[2], n_classes=2,
                         kernel_length=60, separable_kernel_length=15)
    n_flat = net.classifier.in_features
    heads = torch.nn.ModuleList([torch.nn.Linear(n_flat, p["K"]) for p in pool])
    params = list(net.parameters()) + list(heads.parameters())
    opt = torch.optim.Adam(params, lr=lr, eps=ES._KERAS_ADAM_EPSILON)
    tr_idx, va_idx = np.where(~val)[0], np.where(val)[0]
    Xt, yt, dt_ = torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(dsid)

    def loss_on(idx):
        """Mean CE over the batch, each window scored by its dataset's head."""
        feats = trunk_forward(net, Xt[idx])
        tot, n = 0.0, 0
        for k, h in enumerate(heads):
            m = dt_[idx] == k
            if m.any():
                tot = tot + torch.nn.functional.cross_entropy(
                    h(feats[m]), yt[idx][m], reduction="sum")
                n += int(m.sum())
        return tot / max(n, 1)

    best, best_state, wait = np.inf, None, 0
    for ep in range(max_epochs):
        net.train(); heads.train()
        for b in L._batches(len(tr_idx), bs, rng):
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss_on(tr_idx[b]).backward()
            opt.step()
            net.apply_max_norm()
        net.eval(); heads.eval()
        with torch.no_grad():
            vl = float(np.mean([loss_on(va_idx[i:i + 1024]).item()
                                for i in range(0, len(va_idx), 1024)]))
            acc = []
            for k, h in enumerate(heads):
                m = va_idx[dsid[va_idx] == k]
                pr = torch.cat([h(trunk_forward(net, Xt[m[i:i + 1024]])).argmax(1)
                                for i in range(0, len(m), 1024)]).numpy()
                acc.append(float(np.mean(pr == y[m])))
        L.log(f"    [pretrain] epoch {ep + 1} val_loss={vl:.4f} val_acc="
              f"{[round(a, 3) for a in acc]} ({time.time() - t0:.0f}s)")
        if vl < best:
            best, wait = vl, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break
    torch.save(best_state, path)
    L.log(f"[done] pretrain align={align} seed={seed} best_val={best:.4f} -> {path} "
          f"({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------
# fine-tuning on Scherer
# ---------------------------------------------------------------------------
def scherer11():
    d = L.load_study("scherer2015", SCHERER3)
    d["X"] = harmonise(d["X"], d["meta"]["ch_names"])
    d["meta"] = dict(d["meta"], ch_names=list(CH11))
    return d


def finetune(align, seeds):
    """align: none | <how>:<kind> for Scherer; the pretrained trunk used for
    'pretrained' is the one pre-trained with the same kind (none / euclid /
    riemann)."""
    d = scherer11()
    meta = dict(d["meta"], n_classes=3)
    sp = L.xsess_split(d)
    tr, te = sp["train"], sp["test"]
    tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
    y, subj, sess = d["y"], d["subj"], d["sess"]
    Xa, info = aligned_data(d, tr, te, align, {})
    info.pop("router_assign", None)
    pre_kind = "none" if align == "none" else align.split(":")[1]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probs").mkdir(exist_ok=True)
    res = OUT / "results_scherer2015.jsonl"
    done = {r["key"] for r in L.read_results(OUT / "results.jsonl")}
    L.log(f"[data] finetune scherer2015 3-class on CH11 X={d['X'].shape} train={len(tr_idx)} "
          f"test={len(te_idx)} align={align} pretrained_kind={pre_kind}")
    for seed in seeds:
        state = torch.load(PRE_DIR / f"eegnet_mi11_{pre_kind}_s{seed}.pt")
        for init in ("scratch", "pretrained"):
            for mode in ("persubject", "pooled"):
                spec = f"eegnet_st/{init}"
                key = config_key("scherer2015", SCHERER3, spec + "@ch11", mode, align, seed)
                if key in done:
                    continue
                t0 = time.time()
                kw = dict(init_state=state, lr=5e-4) if init == "pretrained" else {}
                P = np.zeros((len(te_idx), 3))
                eps = []
                groups = [None] if mode == "pooled" else list(np.unique(subj))
                for s in groups:
                    ti = tr_idx if s is None else tr_idx[subj[tr_idx] == s]
                    rows = np.arange(len(te_idx)) if s is None else np.where(subj[te_idx] == s)[0]
                    if SMOKE_EPOCHS:
                        kw["n_epochs"] = SMOKE_EPOCHS
                    m = L.EEGNetSTModel(meta, seed=seed, **kw)
                    fi, vi = L.es_split(d, ti, mode, seed)
                    m.fit(Xa[fi], y[fi], Xval=Xa[vi], yval=y[vi])
                    eps.append(int(m.best_epoch))
                    P[rows] = m.predict_proba(Xa[te_idx[rows]])
                sc = L.score(y[te_idx], P.argmax(1), subj[te_idx], sess[te_idx])
                h = hashlib.md5(key.encode()).hexdigest()[:12]
                np.savez_compressed(OUT / "probs" / f"{h}.npz", P=P, te_idx=te_idx, key=key)
                L.append_result(res, dict(
                    key=key, study="scherer2015", classes=SCHERER3, spec=spec + "@ch11",
                    mode=mode, align=align, seed=seed, probs=f"{h}.npz", cell=sc["cell"],
                    pooled=sc["pooled"], per_subject=sc["per_subject"], n_cells=sc["n_cells"],
                    n_train=len(tr_idx), n_test=len(te_idx), shape=list(d["X"].shape),
                    epochs=eps, seconds=round(time.time() - t0, 1),
                    time=time.strftime("%H:%M:%S"), **info))
                L.log(f"[done] {key} cell={sc['cell']:.4f} pooled={sc['pooled']:.4f} "
                      f"epochs={eps} {time.time() - t0:.0f}s")


def riemann_reference():
    """Filter-bank TS (+ broadband) on CH11 Scherer 3-class; tangent-space
    reference = Riemannian mean of Scherer training covariances vs of the
    pooled MI + Scherer training covariances. Pooled and per-subject LDA."""
    from pyriemann.estimation import Covariances
    from pyriemann.utils.mean import mean_riemann
    d = scherer11()
    sp = L.xsess_split(d)
    tr, te = sp["train"], sp["test"]
    y, subj, sess = d["y"], d["subj"], d["sess"]
    pool = load_pool()
    bands = [None, "4to8", "8to13", "13to30", "30to45"]
    res = OUT / "results_scherer2015.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    feats = {"scherer": [], "mi+scherer": []}
    for band in bands:
        C = Covariances("oas").transform(L._band_filter(d["X"], 120.0, band).astype(np.float64))
        Cmi = np.concatenate([Covariances("oas").transform(
            L._band_filter(p["X"][::4], 120.0, band).astype(np.float64)) for p in pool])
        for name, ref_covs in (("scherer", C[tr]), ("mi+scherer", np.concatenate([C[tr], Cmi]))):
            feats[name].append(_ts_at(C, mean_riemann(ref_covs, maxiter=50)))
    for name, fl in feats.items():
        F = np.concatenate(fl, 1)
        for mode in ("pooled", "persubject"):
            P = np.zeros((int(te.sum()), 3))
            te_idx = np.where(te)[0]
            groups = [None] if mode == "pooled" else list(np.unique(subj))
            for s in groups:
                ti = np.where(tr)[0] if s is None else np.where(tr & (subj == s))[0]
                rows = np.arange(len(te_idx)) if s is None else np.where(subj[te_idx] == s)[0]
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
                                                 priors=np.full(3, 1 / 3)).fit(F[ti], y[ti])
                P[rows] = lda.predict_proba(F[te_idx[rows]])
            sc = L.score(y[te_idx], P.argmax(1), subj[te_idx], sess[te_idx])
            key = config_key("scherer2015", SCHERER3, f"riemann_fbts@ch11/ref={name}", mode, "none", 33)
            L.append_result(res, dict(key=key, study="scherer2015", classes=SCHERER3,
                                      spec=f"riemann_fbts@ch11/ref={name}", mode=mode, align="none",
                                      seed=33, cell=sc["cell"], pooled=sc["pooled"],
                                      per_subject=sc["per_subject"], n_cells=sc["n_cells"],
                                      n_train=int(tr.sum()), n_test=int(te.sum()),
                                      shape=list(d["X"].shape), seconds=0,
                                      time=time.strftime("%H:%M:%S")))
            L.log(f"[done] {key} cell={sc['cell']:.4f} pooled={sc['pooled']:.4f}")


def _ts_at(C, ref):
    from pyriemann.utils.tangentspace import tangent_space
    return tangent_space(C, ref, metric="riemann")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["pretrain", "finetune", "riemann", "channels"])
    ap.add_argument("--align", default="none")
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--seeds", nargs="+", type=int, default=[33, 34, 35])
    ap.add_argument("--max_epochs", type=int, default=60)
    a = ap.parse_args()
    if a.what == "pretrain":
        pretrain(a.align, a.seed, max_epochs=a.max_epochs)
    elif a.what == "finetune":
        finetune(a.align, a.seeds)
    elif a.what == "riemann":
        riemann_reference()
    else:
        z = L.load_study("zhou2016")
        M = interp_matrix(z["meta"]["ch_names"], CH11)
        for c, row in zip(CH11, M):
            nz = {z["meta"]["ch_names"][j]: round(float(v), 3) for j, v in enumerate(row) if abs(v) > 0.05}
            print(c, nz)


if __name__ == "__main__":
    main()
