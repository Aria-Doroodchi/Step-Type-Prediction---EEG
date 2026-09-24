#!/usr/bin/env python
"""Exploratory analysis of the Track 2 warm-up data (Dreyer 2023, as delivered).

Everything here is computed on exactly what a Track 2 solver receives: the
NeuralBench-preprocessed 4-s windows (27 EEG ch x 480 samples at 120 Hz,
robust-scaled per recording/channel, cue at t=0). Labels: 0 = left hand,
1 = right hand (NeuralBench's LabelEncoder sorts label names).

Run in WSL after `source ~/codabench/env.sh`:

    python ~/codabench/analysis/dreyer_eda.py

Outputs: codabench/reports/dreyer_eda/*.png + summary.json. The first run
caches all windows to ~/neuralbench/eda_cache/ (~1.1 GB); later runs reuse it.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HOME = Path.home()
OUT = HOME / "codabench/reports/dreyer_eda"
CACHE = HOME / "neuralbench/eda_cache"
sys.path.insert(0, str(HOME / "codabench/2026-competition/tracks/bci_decoding"))
mne.set_log_level("ERROR")

# --- palette (dataviz reference instance) ----------------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
CLASS_NAMES = ["left hand", "right hand"]
SPLIT_NAMES = ["train", "val", "test"]
DIV = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#0d366b", "#3987e5", "#f0efec", "#e34948", "#8f2322"])
SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 9.5,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 1.8, "legend.frameon": False, "legend.fontsize": 9,
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(fig, name, title=None):
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.savefig(OUT / name)
    plt.close(fig)
    log(f"  wrote {name}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load():
    CACHE.mkdir(parents=True, exist_ok=True)
    if (CACHE / "X.npy").exists():
        meta = json.loads((CACHE / "meta.json").read_text())
        arrs = {k: np.load(CACHE / f"{k}.npy", mmap_mode=None)
                for k in ("X", "y", "subj", "split")}
        return arrs["X"], arrs["y"], arrs["subj"], arrs["split"], meta
    import os
    from benchmark_utils.nb_task import load_task
    loaders, meta = load_task(
        "eeg", "motor_imagery",
        data_dir=Path(os.environ["BENCHOPT_DATA_HOME"]) / "neural_compet",
        dataset="dreyer2023", device="cpu", batch_size=256, seed=33,
        num_workers=0, subset="full", target_transform=lambda y: y.argmax(-1))
    Xs, ys, ss, sp = [], [], [], []
    for k, name in enumerate(SPLIT_NAMES):
        for X, y, info in loaders[name]:
            Xs.append(X.numpy().astype(np.float32))
            ys.append(y.numpy())
            ss.append(np.asarray(info["subject_id"]).reshape(-1))
            sp.append(np.full(len(y), k))
        log(f"  loaded {name}")
    X, y, subj, split = (np.concatenate(a) for a in (Xs, ys, ss, sp))
    meta = {"sfreq": float(meta["sfreq"]), "ch_names": list(meta["ch_names"])}
    for k, v in dict(X=X, y=y, subj=subj, split=split).items():
        np.save(CACHE / f"{k}.npy", v)
    (CACHE / "meta.json").write_text(json.dumps(meta))
    return X, y, subj, split, meta


def butter(X, lo, hi, sf):
    if lo is None:
        sos = signal.butter(4, hi, "low", fs=sf, output="sos")
    elif hi is None:
        sos = signal.butter(4, lo, "high", fs=sf, output="sos")
    else:
        sos = signal.butter(4, [lo, hi], "band", fs=sf, output="sos")
    return signal.sosfiltfilt(sos, X, axis=-1).astype(np.float32)


def cohens_d(a, b):
    """d = (mean b - mean a) / pooled sd, along axis 0."""
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(0, ddof=1) + (nb - 1) * b.var(0, ddof=1))
                 / (na + nb - 2))
    return (b.mean(0) - a.mean(0)) / (sp + 1e-12)


def per_subject_d(F, y, subj, subjects):
    """Mean over subjects of within-subject d(right - left); + sign consistency."""
    ds = np.stack([cohens_d(F[(subj == s) & (y == 0)], F[(subj == s) & (y == 1)])
                   for s in subjects])
    return ds.mean(0), (np.sign(ds) == np.sign(ds.mean(0))).mean(0), ds


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    log("loading windows")
    X, y, subj, split, meta = load()
    sf, chs = meta["sfreq"], meta["ch_names"]
    n, C, T = X.shape
    t = np.arange(T) / sf
    subjects = np.unique(subj)
    subj_split = {s: int(split[subj == s][0]) for s in subjects}
    dev_subjects = [s for s in subjects if subj_split[s] < 2]       # train+val
    log(f"X={X.shape} subjects={len(subjects)} windows/split="
        f"{[int((split == k).sum()) for k in range(3)]} classes={np.bincount(y).tolist()}")
    summary = {"n_windows": int(n), "n_channels": C, "n_times": T, "sfreq": sf,
               "windows_per_split": {SPLIT_NAMES[k]: int((split == k).sum()) for k in range(3)},
               "subjects_per_split": {SPLIT_NAMES[k]: int(sum(v == k for v in subj_split.values())) for k in range(3)},
               "class_counts": np.bincount(y).tolist()}

    montage = mne.channels.make_standard_montage("standard_1005")
    pos = montage.get_positions()["ch_pos"]
    order = sorted(range(C), key=lambda i: (-round(pos[chs[i]][1], 3), pos[chs[i]][0]))
    info = mne.create_info(chs, sf, "eeg")
    info.set_montage(montage)
    ci = {c: i for i, c in enumerate(chs)}
    groups = {
        "frontal (F3 Fz F4)": ["F3", "Fz", "F4"],
        "fronto-central (FC)": ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6"],
        "central (C)": ["C5", "C3", "C1", "Cz", "C2", "C4", "C6"],
        "centro-parietal / parietal": ["CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "P3", "Pz", "P4"],
        "all 27": chs,
    }

    # ------------------------------------------------------------ inventory
    log("fig 0: inventory")
    per_sub = np.array([[((subj == s) & (y == k)).sum() for k in (0, 1)] for s in subjects])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [3, 1]})
    idx = np.argsort([subj_split[s] for s in subjects], kind="stable")
    xs = np.arange(len(subjects))
    for k in (0, 1):
        ax[0].bar(xs, per_sub[idx, k], bottom=per_sub[idx, 0] if k else 0,
                  color=CAT[k], width=0.8, label=CLASS_NAMES[k], edgecolor=SURFACE, linewidth=0.6)
    for k, name in enumerate(SPLIT_NAMES):
        m = np.where(np.array([subj_split[subjects[i]] for i in idx]) == k)[0]
        ax[0].annotate(name, (m.mean(), per_sub.sum(1).max() * 1.04), ha="center", color=INK2)
        if k:
            ax[0].axvline(m.min() - 0.5, color=INK2, lw=0.8, ls=":")
    ax[0].set(xlabel="subject (grouped by split)", ylabel="windows", xticks=[],
              title="Windows per subject and class")
    ax[0].legend(loc="upper right", bbox_to_anchor=(1.0, -0.06), ncol=2)
    tot = [int((split == k).sum()) for k in range(3)]
    ax[1].barh(SPLIT_NAMES[::-1], tot[::-1], color=CAT[0], height=0.55)
    for i, v in enumerate(tot[::-1]):
        ax[1].text(v, i, f" {v:,}", va="center", color=INK)
    ax[1].set(title="Windows per split", xticks=[])
    ax[1].spines["bottom"].set_visible(False)
    save(fig, "00_inventory.png")

    # ------------------------------------------------------------ recording
    log("fig 1: example recording")
    s_ex = dev_subjects[len(dev_subjects) // 2]
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.2), gridspec_kw={"width_ratios": [4, 4, 2.2]})
    for k in (0, 1):
        i = np.where((subj == s_ex) & (y == k))[0][0]
        ax = axes[k]
        for r, c in enumerate(order):
            ax.plot(t, X[i, c] / 4 - r, color=CAT[k], lw=0.8)
        ax.set_yticks(-np.arange(C), [chs[c] for c in order], fontsize=7.5)
        ax.set(xlabel="time after cue (s)", title=f"One {CLASS_NAMES[k]} window, subject {s_ex}",
               xlim=(0, 4), ylim=(-C, 1.5))
        ax.spines["left"].set_visible(False)
    xy = np.array([pos[c][:2] for c in chs])
    ax = axes[2]
    ax.scatter(xy[:, 0], xy[:, 1], s=430, color=CAT[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for c, (px, py) in zip(chs, xy):
        ax.text(px, py, c, ha="center", va="center", fontsize=6, color="#ffffff",
                fontweight="bold", zorder=4)
    ax.add_patch(plt.Circle((0, 0), 0.095, fill=False, color=INK2, lw=1))
    ax.set(aspect="equal", xticks=[], yticks=[], title="27 electrodes (nose up)")
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    save(fig, "01_recording_examples.png", "What a model receives: robust-scaled 4-s windows, 120 Hz")

    # ------------------------------------------------------------ spectra
    log("PSD (Welch, 2-s segments)")
    f, P = signal.welch(X, fs=sf, nperseg=240, noverlap=120, axis=-1)
    logP = np.log10(P.astype(np.float64) + 1e-20).astype(np.float32)   # (n, C, F)
    del P
    sub_psd = np.stack([logP[subj == s].mean((0, 1)) for s in subjects])  # (S, F)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for row in sub_psd:
        ax.plot(f, row, color="#b9b8b2", lw=0.6, alpha=0.6)
    med = np.median(sub_psd, 0)
    q1, q3 = np.percentile(sub_psd, [25, 75], 0)
    ax.fill_between(f, q1, q3, color=CAT[0], alpha=0.18, lw=0)
    ax.plot(f, med, color=CAT[0], lw=2.2, label="median subject (IQR band)")
    for fr, lab in [(10, "alpha / mu"), (50, "50 Hz notch")]:
        ax.axvline(fr, color=INK2, ls=":", lw=0.8)
        ax.annotate(lab, (fr, ax.get_ylim()[1]), xytext=(3, -12), textcoords="offset points", color=INK2)
    ax.set(xlabel="frequency (Hz)", ylabel="log10 power (robust units² / Hz)",
           title="Power spectrum: 87 subjects (grey), median + IQR (blue)", xlim=(0.5, 60))
    ax.legend(loc="lower left")
    save(fig, "02_power_spectrum.png")
    # alpha peak per subject (7-14 Hz on central/parietal, after 1/f removal)
    band_fit = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
    post = [ci[c] for c in ["C3", "Cz", "C4", "CP1", "CP2", "P3", "Pz", "P4"]]
    apf = []
    for s in subjects:
        spec = logP[subj == s][:, post].mean((0, 1))
        coef = np.polyfit(np.log10(f[band_fit]), spec[band_fit], 1)
        resid = spec - np.polyval(coef, np.log10(np.maximum(f, 0.5)))
        m = (f >= 7) & (f <= 14)
        apf.append((f[m][np.argmax(resid[m])], resid[m].max()))
    apf = np.array(apf)

    # ------------------------------------------------------------ noise
    log("fig 3: noise / quality")
    var = X.var(-1)                                          # (n, C)
    art_win = np.abs(X).max(-1) > 6                          # (n, C): > 6 robust units
    art = np.stack([100 * art_win[subj == s].mean(0) for s in subjects])
    hf = (f >= 30) & (f <= 45)
    hf_pow = np.stack([logP[subj == s][:, :, hf].mean((0, 2)) for s in subjects])  # (S, C)
    hf_rel = 10 * (hf_pow - np.median(hf_pow))
    lv = np.log(var.mean(1))
    out_pct, clip_pct = [], []
    for s in subjects:
        m = subj == s
        med_, mad = np.median(lv[m]), np.median(np.abs(lv[m] - np.median(lv[m])))
        out_pct.append(100 * np.mean(lv[m] > med_ + 3 * 1.4826 * mad))
        clip_pct.append(100 * np.mean(np.abs(X[m]) >= 19.99))
    out_pct, clip_pct = np.array(out_pct), np.array(clip_pct)
    sidx = np.argsort(out_pct)[::-1]
    fig = plt.figure(figsize=(13, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.4], hspace=0.45, wspace=0.18)
    for j, (M, ttl, lab) in enumerate([
            (art, "Artifact windows: % with |x| > 6 robust units", "% windows"),
            (hf_rel, "30–45 Hz power vs grand median (muscle / noise)", "dB")]):
        ax = fig.add_subplot(gs[0, j])
        if j == 0:
            im = ax.imshow(M[sidx][:, order].T, aspect="auto", cmap=SEQ, vmin=0,
                           vmax=np.percentile(M, 99), interpolation="nearest")
        else:
            vm = np.percentile(np.abs(M), 98)
            im = ax.imshow(M[sidx][:, order].T, aspect="auto", cmap=DIV, vmin=-vm, vmax=vm,
                           interpolation="nearest")
        ax.set_yticks(range(C), [chs[c] for c in order], fontsize=7)
        ax.set(xlabel="subject (sorted by % outlier windows, worst left)", title=ttl, xticks=[])
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01, label=lab)
    ax = fig.add_subplot(gs[1, :])
    cols = [CAT[subj_split[subjects[i]]] for i in sidx]
    ax.bar(np.arange(len(subjects)), out_pct[sidx], color=cols, width=0.8)
    for k in range(3):
        ax.bar(0, 0, color=CAT[k], label=SPLIT_NAMES[k])
    ax.set(ylabel="% windows", xticks=[], xlabel="subject",
           title="Outlier windows per subject (log-variance > median + 3 robust SD)")
    ax.legend(ncol=3, loc="upper right")
    save(fig, "03_noise_quality.png", "Noise and data quality")

    # ------------------------------------------------------------ features: where is the info
    log("fig 4: effect sizes (spectral + temporal)")
    fsel = (f >= 1) & (f <= 45)
    d_spec, cons_spec, _ = per_subject_d(logP[:, :, fsel], y, subj, dev_subjects)
    Xslow = butter(X, None, 4.0, sf)
    Xs10 = Xslow[:, :, ::12]                                 # 10 Hz, 40 points
    t10 = t[::12]
    d_time, cons_time, ds_time = per_subject_d(Xs10, y, subj, dev_subjects)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, D, xv, xl, ttl in [
            (axes[0], d_spec, f[fsel], "frequency (Hz)", "Spectral: log band power, d(right − left)"),
            (axes[1], d_time, t10, "time after cue (s)", "Temporal: slow waveform < 4 Hz, d(right − left)")]:
        vm = np.abs(D).max()
        im = ax.imshow(D[order], aspect="auto", cmap=DIV, vmin=-vm, vmax=vm,
                       extent=[xv[0], xv[-1], C - 0.5, -0.5], interpolation="nearest")
        ax.set_yticks(range(C), [chs[c] for c in order], fontsize=7.5)
        ax.set(xlabel=xl, title=ttl)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01, label="mean within-subject Cohen's d")
    save(fig, "04_where_is_the_class_information.png",
         "Where left and right hand differ (mean over 66 train/val subjects; red = larger for right hand)")
    top = np.unravel_index(np.argmax(np.abs(d_time)), d_time.shape)
    summary["max_temporal_d"] = {"channel": chs[top[0]], "time_s": float(t10[top[1]]),
                                 "d": float(d_time[top]), "sign_consistency": float(cons_time[top])}
    topsp = np.unravel_index(np.argmax(np.abs(d_spec)), d_spec.shape)
    summary["max_spectral_d"] = {"channel": chs[topsp[0]], "freq_hz": float(f[fsel][topsp[1]]),
                                 "d": float(d_spec[topsp]), "sign_consistency": float(cons_spec[topsp])}
    mu = (f[fsel] >= 8) & (f[fsel] <= 13)
    summary["mu_d_C3_C4"] = {"C3": float(d_spec[ci["C3"], mu].mean()), "C4": float(d_spec[ci["C4"], mu].mean())}

    # ------------------------------------------------------------ ERPs + topomaps
    log("fig 5: ERPs + topographies")
    erp = np.stack([[Xslow[(subj == s) & (y == k)].mean(0) for k in (0, 1)] for s in dev_subjects])  # (S,2,C,T)
    fig = plt.figure(figsize=(13, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.2], hspace=0.35)
    top_gs = gs[0].subgridspec(2, 3, hspace=0.45, wspace=0.12)
    show = ["F3", "Fz", "F4", "C3", "Cz", "C4"]
    for j, c in enumerate(show):
        ax = fig.add_subplot(top_gs[j // 3, j % 3])
        for k in (0, 1):
            w = erp[:, k, ci[c]]
            m_, se = w.mean(0), w.std(0) / np.sqrt(len(w))
            ax.fill_between(t, m_ - se, m_ + se, color=CAT[k], alpha=0.18, lw=0)
            ax.plot(t, m_, color=CAT[k], label=CLASS_NAMES[k])
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        ax.set(title=c, xlim=(0, 4))
        if j % 3 == 0:
            ax.set_ylabel("robust units")
        if j >= 3:
            ax.set_xlabel("time after cue (s)")
        if j == 0:
            ax.legend(loc="lower left")
    bot = gs[1].subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.08], wspace=0.15)
    wins = [(0.2, 0.8), (0.8, 1.6), (1.6, 2.6), (2.6, 4.0)]
    diff = (erp[:, 1] - erp[:, 0]).mean(0)                          # (C, T)
    vm = np.abs(diff).max()
    for j, (a, b) in enumerate(wins):
        ax = fig.add_subplot(bot[j])
        m = (t >= a) & (t < b)
        im, _ = mne.viz.plot_topomap(diff[:, m].mean(1), info, axes=ax, cmap=DIV,
                                     vlim=(-vm, vm), contours=0, show=False, sensors=True)
        ax.set_title(f"{a}–{b} s", fontsize=9.5, fontweight="normal", loc="center")
    fig.colorbar(im, cax=fig.add_subplot(bot[4]), label="right − left")
    save(fig, "05_erp_and_topography.png",
         "Slow waveforms (< 4 Hz): grand average ± SE across 66 subjects, and the right − left difference")

    # ------------------------------------------------------------ ERD/ERS time-frequency
    log("fig 6: time-frequency")
    tf_ch = ["C3", "Cz", "C4"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    maps = []
    for c in tf_ch:
        ft, tt, S = signal.spectrogram(X[:, ci[c]], fs=sf, nperseg=60, noverlap=54, axis=-1)
        L = np.log10(S + 1e-20)                                   # (n, F, T)
        dmap = np.stack([L[(subj == s) & (y == 1)].mean(0) - L[(subj == s) & (y == 0)].mean(0)
                         for s in dev_subjects]).mean(0)
        maps.append((ft, tt, dmap))
    vm = max(np.abs(m_[2][(m_[0] >= 4) & (m_[0] <= 40)]).max() for m_ in maps)
    for ax, c, (ft, tt, dmap) in zip(axes, tf_ch, maps):
        fm = (ft >= 4) & (ft <= 40)
        im = ax.imshow(10 * dmap[fm], aspect="auto", origin="lower", cmap=DIV,
                       vmin=-10 * vm, vmax=10 * vm, extent=[tt[0], tt[-1], ft[fm][0], ft[fm][-1]])
        ax.set(title=c, xlabel="time after cue (s)")
    axes[0].set_ylabel("frequency (Hz)")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="dB, right − left")
    save(fig, "06_time_frequency_C3_Cz_C4.png",
         "Power difference right − left hand over time (blue = less power for right-hand imagery)")

    # ------------------------------------------------------------ frontal vs central check
    log("frontal-vs-central check (cross-subject: train+val -> test)")
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    tr, te = split < 2, split == 2
    feats = {"slow waveform < 4 Hz": None, "power 0.5–4 Hz": (0.5, 4), "power 4–8 Hz": (4, 8),
             "power 8–13 Hz (mu)": (8, 13), "power 13–30 Hz (beta)": (13, 30)}
    acc = np.zeros((len(groups), len(feats)))
    acc_cache = CACHE / "frontal_acc.npy"
    if acc_cache.exists():
        acc = np.load(acc_cache)
        log("    (cached)")
    for j, (fname, band) in enumerate(feats.items() if not acc_cache.exists() else []):
        Xb = None if band is None else butter(X, band[0], band[1], sf)
        for i, (gname, gch) in enumerate(groups.items()):
            gi = [ci[c] for c in gch]
            if band is None:
                F_ = Xs10[:, gi].reshape(n, -1)[:, ::1]
                clf = make_pipeline(StandardScaler(),
                                    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
                clf.fit(F_[tr], y[tr])
                pred = clf.predict(F_[te])
            else:
                cov = Covariances("oas").transform(Xb[:, gi].astype(np.float64))
                ts = TangentSpace(metric="riemann").fit(cov[tr])
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
                clf.fit(ts.transform(cov[tr]), y[tr])
                pred = clf.predict(ts.transform(cov[te]))
            acc[i, j] = balanced_accuracy_score(y[te], pred)
            log(f"    {gname:28s} {fname:22s} bal_acc={acc[i, j]:.3f}")
        del Xb
    np.save(acc_cache, acc)
    summary["frontal_vs_central"] = {g: dict(zip(feats, map(float, acc[i]))) for i, g in enumerate(groups)}
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    im = ax.imshow(acc, cmap=SEQ, vmin=0.5, vmax=max(0.75, acc.max()), aspect="auto")
    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            v = acc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="#ffffff" if v > 0.65 else INK, fontsize=10)
    ax.set_xticks(range(len(feats)), list(feats), rotation=15, ha="right")
    ax.set_yticks(range(len(groups)), [f"{g} ({len(v)} ch)" for g, v in groups.items()])
    ax.set_title("Cross-subject balanced accuracy on the test subjects (chance 0.50)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    save(fig, "07_frontal_vs_central_check.png",
         "Which channels and frequency bands carry the left/right information?")

    # ------------------------------------------------------------ within-subject decoding
    log("within-subject 5-fold CV per subject (slow LDA; mu/beta Riemann)")
    Xmb = butter(X, 8, 30, sf)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    ws = []
    ws_cache = CACHE / "within.npy"
    for s in (subjects if not ws_cache.exists() else []):
        m = subj == s
        Fs = Xs10[m].reshape(m.sum(), -1)
        a1 = balanced_accuracy_score(y[m], cross_val_predict(
            make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            Fs, y[m], cv=cv))
        cov = Covariances("oas").transform(Xmb[m].astype(np.float64))
        a2 = balanced_accuracy_score(y[m], cross_val_predict(
            make_pipeline(TangentSpace(metric="riemann"), StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=2000)), cov, y[m], cv=cv))
        ws.append((a1, a2))
    ws = np.load(ws_cache) if ws_cache.exists() else np.array(ws)
    np.save(ws_cache, ws)
    nwin = np.array([(subj == s).sum() for s in subjects])
    chance95 = 0.5 + 1.96 * np.sqrt(0.25 / np.median(nwin))
    summary["within_subject"] = {
        "slow_median": float(np.median(ws[:, 0])), "mubeta_median": float(np.median(ws[:, 1])),
        "chance_95pct_upper": float(chance95),
        "pct_subjects_above_chance_slow": float(100 * np.mean(ws[:, 0] > chance95)),
        "pct_subjects_above_chance_mubeta": float(100 * np.mean(ws[:, 1] > chance95)),
        "corr_slow_vs_mubeta": float(np.corrcoef(ws[:, 0], ws[:, 1])[0, 1])}

    # ------------------------------------------------------------ fig 8: trial-to-trial variability
    log("fig 8: trial-to-trial variability")
    c_top, t_top = top
    win = (t10 >= t10[t_top] - 0.3) & (t10 <= t10[t_top] + 0.3)
    feat = Xs10[:, c_top][:, win].mean(1)                            # single-trial slow amplitude
    best = np.argsort(ws.max(1))
    pick = [subjects[i] for i in best[np.linspace(0, len(best) - 1, 8).astype(int)]]
    s_med = subjects[best[len(best) // 2]]
    fig = plt.figure(figsize=(13, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.25)
    ax = fig.add_subplot(gs[0])
    m = subj == s_med
    ord_ = np.argsort(y[m], kind="stable")
    img = Xslow[m][ord_, c_top]
    vm = np.percentile(np.abs(img), 98)
    im = ax.imshow(img, aspect="auto", cmap=DIV, vmin=-vm, vmax=vm, extent=[0, 4, len(img), 0],
                   interpolation="nearest")
    nl = int((y[m] == 0).sum())
    ax.axhline(nl, color=INK, lw=1)
    ax.text(-0.45, nl / 2, "left\nhand", va="center", ha="right", color=CAT[0],
            fontweight="bold")
    ax.text(-0.45, nl + (len(img) - nl) / 2, "right\nhand", va="center", ha="right",
            color=CAT[1], fontweight="bold")
    ax.set(xlabel="time after cue (s)", ylabel="trial", title=f"Every trial, {chs[c_top]} < 4 Hz, median subject {s_med}")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.08, label="robust units")
    ax = fig.add_subplot(gs[1])
    rng = np.random.default_rng(0)
    for j, s in enumerate(pick):
        for k in (0, 1):
            v = feat[(subj == s) & (y == k)]
            ax.scatter(j + (k - 0.5) * 0.36 + rng.uniform(-0.1, 0.1, len(v)), v, s=9,
                       color=CAT[k], alpha=0.55, lw=0, label=CLASS_NAMES[k] if j == 0 else None)
            ax.plot([j + (k - 0.5) * 0.36 - 0.14, j + (k - 0.5) * 0.36 + 0.14], [np.median(v)] * 2,
                    color=INK, lw=1.6)
    shown = feat[np.isin(subj, pick)]
    lo_, hi_ = np.percentile(shown, [0.5, 99.5])
    pad = 0.15 * (hi_ - lo_)
    n_out = int(((shown < lo_ - pad) | (shown > hi_ + pad)).sum())
    ax.set_ylim(lo_ - pad, hi_ + pad)
    ax.text(0.99, 0.02, f"{n_out} extreme trials outside the axis", transform=ax.transAxes,
            ha="right", color=INK2, fontsize=8.5)
    ax.set_xticks(range(len(pick)), [f"s{s}" for s in pick])
    ax.set(xlabel="8 subjects, worst → best within-subject accuracy",
           ylabel=f"mean {chs[c_top]} < 4 Hz, {t10[t_top] - 0.3:.1f}–{t10[t_top] + 0.3:.1f} s",
           title="Single-trial values of the strongest feature (black = median)")
    ax.legend(loc="upper left", markerscale=2)
    save(fig, "08_trial_to_trial_variability.png", "Trial-to-trial variability")

    # ------------------------------------------------------------ fig 9: between-person variability
    log("fig 9: between-person variability")
    mean_cov = np.stack([Covariances("oas").transform(X[subj == s].astype(np.float64)).mean(0)
                         for s in subjects])
    from pyriemann.utils.distance import distance_riemann
    Dm = np.array([[distance_riemann(a, b) for b in mean_cov] for a in mean_cov])
    emb = MDS(n_components=2, metric="precomputed", init="random", random_state=0, n_init=4).fit_transform(Dm)
    fig = plt.figure(figsize=(13, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.42, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    o = np.argsort(ws.max(1))
    xs = np.arange(len(subjects))
    ax.axhspan(0.5 - (chance95 - 0.5), chance95, color=GRID, alpha=0.7, lw=0)
    ax.text(len(xs) - 0.5, chance95 + 0.005, "95 % chance band", ha="right", va="bottom", color=INK2, fontsize=8.5)
    ax.vlines(xs, ws[o].min(1), ws[o].max(1), color="#c9c8c2", lw=1)
    ax.scatter(xs, ws[o, 0], s=26, color=CAT[0], zorder=3, label="slow waveform < 4 Hz (LDA)", edgecolor=SURFACE, lw=0.8)
    ax.scatter(xs, ws[o, 1], s=26, color=CAT[1], zorder=3, label="mu/beta 8–30 Hz (Riemann)", edgecolor=SURFACE, lw=0.8)
    ax.set(xticks=[], xlabel="subject (sorted by best of the two)", ylabel="balanced accuracy",
           title="Within-subject 5-fold accuracy: how decodable is each person?", ylim=(0.3, 1.0))
    ax.legend(loc="upper left", ncol=2)
    ax = fig.add_subplot(gs[1, 0])
    for k in range(3):
        m = np.array([subj_split[s] == k for s in subjects])
        ax.scatter(emb[m, 0], emb[m, 1], s=34, color=CAT[k], label=SPLIT_NAMES[k], edgecolor=SURFACE, lw=0.8)
    ax.set(xticks=[], yticks=[], title="Subjects by EEG covariance (MDS)", xlabel="Riemannian distance map")
    ax.legend(loc="best", fontsize=8.5)
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(apf[:, 0], bins=np.arange(6.75, 14.5, 0.5), color=CAT[0], edgecolor=SURFACE, lw=1.5)
    ax.set(xlabel="alpha/mu peak frequency (Hz)", ylabel="subjects", title="Individual alpha peak")
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(ws[:, 0], ws[:, 1], s=26, color=CAT[0], edgecolor=SURFACE, lw=0.8)
    lim = (0.3, 1.0)
    ax.plot(lim, lim, color=GRID, lw=1, zorder=0)
    ax.set(xlim=lim, ylim=lim, xlabel="slow waveform accuracy", ylabel="mu/beta accuracy",
           title=f"Same people good at both? r = {summary['within_subject']['corr_slow_vs_mubeta']:.2f}")
    save(fig, "09_between_person_variability.png", "Between-person variability")
    summary["alpha_peak_hz"] = {"median": float(np.median(apf[:, 0])),
                                "p10": float(np.percentile(apf[:, 0], 10)),
                                "p90": float(np.percentile(apf[:, 0], 90))}
    summary["noise"] = {"median_outlier_pct": float(np.median(out_pct)),
                        "max_outlier_pct": float(out_pct.max()),
                        "median_clip_pct": float(np.median(clip_pct)),
                        "max_clip_pct": float(clip_pct.max()),
                        "median_artifact_window_pct_per_channel": float(np.median(art)),
                        "p95_artifact_window_pct_per_channel": float(np.percentile(art, 95)),
                        "hf_power_range_db_p5_p95": [float(np.percentile(hf_rel, 5)), float(np.percentile(hf_rel, 95))]}
    summary["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"done in {summary['runtime_s']} s")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
