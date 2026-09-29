#!/usr/bin/env python
"""Activation EDA on the sealed-like proxy (sprint 2026-09-29, brief
``prompts/2026-09-29_temporal_spatial_features.md`` deliverable 2).

WHERE (scalp region), in WHICH band and WHEN (time in the 4 s window) do the
three sealed-like mental tasks differ? Scherer 2015 restricted to cache labels
0, 1, 3 = WORD (word association), SUB (mental subtraction), HAND (right-hand
kinesthetic MI); ``load_study(classes=[0, 1, 3])`` re-indexes them 0 = WORD,
1 = SUB, 2 = HAND (asserted against the cache's own class codes 1, 2, 4).

Data discipline: ONLY each subject's training session is analysed
(``xsess_lib.xsess_split(d)["train"]``; the last session per subject is the
test session). ``load_study`` reads the whole cache from disk, so the test rows
are dropped right after loading, before any computation, and the script
asserts 9 subjects x 120 training windows = 1080 rows, 40 per class, all from
a session that is not the subject's test session, and no test row. Windows are
exactly what a Track 2 solver receives (NeuralBench: 0.1-75 Hz, notch, 120 Hz,
robust-scaled per recording, clamp 20): 30 ch x 480 samples, t = 0 is 3.0 s
after trial start, i.e. the cue / start of the task period.

Band-pass = the solver's own FFT mask (``xfeat.band_pass`` ->
``riemann_steptype._band_chunks``), so bands match the model. Bands: 1-4,
4-8, 8-13, 13-30, 30-45 Hz; the evoked waveform uses the slow block's
0.1-4 Hz. Group statistics: per-subject values first, then the mean and a
one-sample t across the 9 subjects (df = 8).

  (a) band-power topographies: per band x class, the per-channel difference in
      mean log band power (dB) between the class and the mean of the three
      class means; group t-map, |t| > 3 marked; family-wise 95 % threshold of
      max |t| per band from the exact sign-flip distribution (2^9 flips).
  (b) ERD/ERS-style time courses: log band power in 0.25 s bins, class minus
      the all-class mean of the same bin (dB), 4-8 / 8-13 / 13-30 Hz at
      F3 Fz F4 T3 C3 Cz C4 T4 P3 Pz P4; plus the class-agnostic time course
      (each bin minus the window mean) per band.
  (c) separability, per subject on the training session: 5-fold stratified CV
      (shuffled, random_state=0) of shrinkage LDA (lsqr, auto; the solvers'
      ``fit_shrinkage_lda`` = sklearn's fit, Cholesky solve above 500
      features) on (i) per-channel log band power, (ii) tangent space of the
      OAS band covariance (TangentSpace fitted inside each CV training fold),
      per band x segment (4 x 1 s, whole window, 4-segment concatenation) +
      broadband and multi-band rows; balanced accuracy averaged over
      subjects. Chance floor from 50 within-subject label permutations per
      band (power, whole window). Per-channel ANOVA F (group mean of
      per-subject F) per band x segment as topomaps, with a permutation
      family-wise threshold.
  (d) class-average 0.1-4 Hz waveforms at Fz Cz Pz O1 O2 (cue response in the
      first second) + LDA on 0.25 s waveform bins (0-1 s, the solver's slow
      block 0.5-4 s, 0-4 s).
  (e) artefact check: class-difference spectra at lateral-edge (EMG-prone)
      vs central channels; delta power of the F7-F8 bipolar (horizontal-EOG
      proxy) and AFz; lateral vs central 30-45 Hz power.

Run in WSL (2 threads, ~6-10 min):

    source ~/codabench/env.sh
    export XS_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUTF8=1
    python ~/codabench/analysis/activation_eda.py            # full run
    python ~/codabench/analysis/activation_eda.py --replot   # figures from cached stats
    python ~/codabench/analysis/activation_eda.py --quick    # 2-subject smoke test (scratch dir)

Outputs (``codabench/reports/features_0929/``): ``activation_*.png`` (150 dpi),
``activation_eda_numbers.md`` (every number, auto-generated) and
``activation_eda_summary.json``. The interpreted report ``activation_eda.md``
is written by hand from these and is never overwritten by this script.
Cached statistics for ``--replot``: ``~/codabench/logs/sealed_f0929/
activation_eda_stats.pkl`` (gitignored).
"""

import argparse
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy import signal  # noqa: E402
from sklearn.feature_selection import f_classif  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

HOME = Path.home()
sys.path.insert(0, str(HOME / "codabench/analysis"))
import xsess_lib as L  # noqa: E402  (also puts the solver on sys.path)
import xfeat  # noqa: E402
import riemann_steptype as RS  # noqa: E402

mne.set_log_level("ERROR")

OUT = HOME / "codabench/reports/features_0929"
STATS = HOME / "codabench/logs/sealed_f0929/activation_eda_stats.pkl"
QUICK_OUT = HOME / "codabench/logs/sealed_f0929/activation_eda_quick"

STUDY, CLASSES = "scherer2015", [0, 1, 3]
CLASS_NAMES = ["WORD", "SUB", "HAND"]
CLASS_CODES = {"0": "1", "1": "2", "2": "4"}       # cache codes: 1 WORD, 2 SUB, 4 HAND
N_SUBJ, N_TRAIN_PER_SUBJ, N_CH, N_T, SFREQ = 9, 120, 30, 480, 120.0
BANDS = ["1to4", "4to8", "8to13", "13to30", "30to45"]
BAND_LABEL = {"1to4": "1-4 Hz (delta)", "4to8": "4-8 Hz (theta)",
              "8to13": "8-13 Hz (alpha/mu)", "13to30": "13-30 Hz (beta)",
              "30to45": "30-45 Hz (low gamma)"}
SLOW_BAND = "0.1to4"                                # riemann_steptype slow block
BIN = 30                                            # 0.25 s at 120 Hz
N_BINS = N_T // BIN                                 # 16
SEGS = [(0, 4), (4, 8), (8, 12), (12, 16), (0, 16)]  # in bins: 4 x 1 s + whole
SEG_LABEL = ["0-1 s", "1-2 s", "2-3 s", "3-4 s", "0-4 s"]
ERD_CH = ["F3", "Fz", "F4", "T3", "C3", "Cz", "C4", "T4", "P3", "Pz", "P4"]
ERD_GRID = [[None, "F3", "Fz", "F4", None], ["T3", "C3", "Cz", "C4", "T4"],
            [None, "P3", "Pz", "P4", None]]
ERD_BANDS = ["4to8", "8to13", "13to30"]
EVOKED_CH = ["Fz", "Cz", "Pz", "O1", "O2"]
LATERAL_CH = ["F7", "F8", "T3", "T4"]              # EMG-prone edge channels
CENTRAL_CH = ["C3", "Cz", "C4", "CPz"]
ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
N_PERM_ACC, N_PERM_F = 50, 200

# --- palette (dataviz reference instance; same as dreyer_eda.py) ------------
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3de"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]           # slots 1-3: all-pairs validated
DIV = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#0d366b", "#3987e5", "#f0efec", "#e34948", "#8f2322"])
SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
SEQ.set_bad("#efeeea")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 9.5,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 1.8, "legend.frameon": False, "legend.fontsize": 9,
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
})

log = L.log


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load(quick=False):
    """Training-session windows only; asserts the split discipline."""
    d = L.load_study(STUDY, classes=CLASSES)
    meta = d["meta"]
    assert {k: str(v) for k, v in meta["classes"].items()} == CLASS_CODES, meta["classes"]
    assert float(meta["sfreq"]) == SFREQ and float(meta["window"]["start"]) == 3.0, meta
    sp = L.xsess_split(d)
    tr, te = sp["train"], sp["test"]
    assert not (tr & te).any()
    subj_all, sess_all = d["subj"], d["sess"]
    test_sess = {int(s): set(np.unique(sess_all[te & (subj_all == s)]).tolist())
                 for s in np.unique(subj_all)}
    X = np.ascontiguousarray(d["X"][tr], dtype=np.float64)
    y, subj, sess = d["y"][tr], subj_all[tr], sess_all[tr]
    n_test_rows = int(te.sum())
    del d, sp, tr, te                                   # test rows gone from here on
    assert X.shape == (N_SUBJ * N_TRAIN_PER_SUBJ, N_CH, N_T), X.shape
    assert np.unique(subj).tolist() == list(range(N_SUBJ))
    for s in range(N_SUBJ):
        m = subj == s
        assert m.sum() == N_TRAIN_PER_SUBJ and np.bincount(y[m]).tolist() == [40, 40, 40]
        assert not set(np.unique(sess[m]).tolist()) & test_sess[s], (s, test_sess[s])
    ch_names = list(meta["ch_names"])
    shape_line = (f"[data] {STUDY} classes {CLASSES} -> {'/'.join(CLASS_NAMES)}; "
                  f"TRAIN rows {len(y)} = {N_SUBJ} subjects x {N_TRAIN_PER_SUBJ} "
                  f"(40/class, sessions {sorted(set(sess.tolist()))}); test rows used 0 "
                  f"(of {n_test_rows} dropped); X {X.shape} @ {SFREQ:g} Hz; "
                  f"{len(ch_names)} ch; window start 3.0 s")
    log(shape_line)
    if quick:
        keep = subj < 2
        X, y, subj = X[keep], y[keep], subj[keep]
        log(f"[quick] subjects 0-1 only: X {X.shape}")
    return X, y, subj, ch_names, shape_line


def make_info(ch_names):
    info = mne.create_info([ALIASES.get(c, c) for c in ch_names], SFREQ, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"),
                     match_case=False)                  # raises if a channel is missing
    return info


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------
def tstat(a, axis=0):
    n = a.shape[axis]
    return a.mean(axis) / (a.std(axis, ddof=1) / np.sqrt(n) + 1e-300)


def sem(a, axis=0):
    return a.std(axis, ddof=1) / np.sqrt(a.shape[axis])


def class_means(v, y, subj):
    """(n, ...) -> (S, 3, ...) per-subject class means."""
    return np.stack([np.stack([v[(subj == s) & (y == k)].mean(0) for k in range(3)])
                     for s in np.unique(subj)])


def class_rel(v, y, subj):
    """(S, 3, ...) class mean minus the mean of the three class means."""
    m = class_means(v, y, subj)
    return m - m.mean(1, keepdims=True)


def signflip_max_t(D, q=95):
    """q-th percentile of max |t| over all non-subject dims, exact sign flips."""
    S = D.shape[0]
    shape = (S,) + (1,) * (D.ndim - 1)
    mx = [np.abs(tstat(D * np.asarray(sg, float).reshape(shape))).max()
          for sg in itertools.product([1, -1], repeat=S)]
    return float(np.percentile(mx, q))


def db(p):
    return 10.0 * np.log10(p + 1e-20)


def lda_fit(F, y):
    return RS.fit_shrinkage_lda(F, y, fast=F.shape[1] > 500)


def cv_bacc(F, y, folds):
    pred = np.empty_like(y)
    for tri, tei in folds:
        pred[tei] = lda_fit(F[tri], y[tri]).predict(F[tei])
    return balanced_accuracy_score(y, pred)


def folds_for(y, seed=0):
    return list(StratifiedKFold(5, shuffle=True, random_state=seed)
                .split(np.zeros(len(y)), y))


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
def compute(X, y, subj, ch_names):
    t0 = time.time()
    ci = {c: i for i, c in enumerate(ch_names)}
    n = len(y)
    subjects = np.unique(subj)
    R = dict(ch_names=ch_names, subjects=subjects.tolist())

    # --- per band: 0.25 s bin powers + OAS covariances per segment ---------
    binpow, covs = {}, {}
    for band in BANDS:
        Xb = xfeat.band_pass(X, SFREQ, band)
        binpow[band] = (Xb ** 2).reshape(n, N_CH, N_BINS, BIN).mean(-1)
        covs[band] = [xfeat.covs(Xb[:, :, a * BIN:b * BIN]) for a, b in SEGS]
        if band == "1to4":
            heog = Xb[:, ci["F7"]] - Xb[:, ci["F8"]]
            R["heog_delta_db"] = db((heog ** 2).mean(-1))
        del Xb
        log(f"  band {band}: bin powers + {len(SEGS)} segment covariances "
            f"({time.time() - t0:.0f}s)")
    # broadband = the preprocessed window itself (recipe's logvar / broad blocks)
    broad_var = np.stack([X[:, :, a * BIN:b * BIN].var(-1) for a, b in SEGS], 1)
    covs["broad"] = [xfeat.covs(X[:, :, a * BIN:b * BIN]) for a, b in SEGS]
    # slow waveform (0.1-4 Hz, the solver's slow block band)
    Xs = xfeat.band_pass(X, SFREQ, SLOW_BAND)
    slow_bins = Xs.reshape(n, N_CH, N_BINS, BIN).mean(-1)
    R["slow_wave"] = class_means(Xs, y, subj)                    # (S, 3, C, T)
    # +/- reference: alternate windows sign-flipped, i.e. the same average with
    # the phase-locked (evoked) part cancelled = the noise floor of an average
    sgn = np.zeros(n)
    for s in subjects:
        for k in range(3):
            idx = np.flatnonzero((subj == s) & (y == k))
            sgn[idx] = np.where(np.arange(len(idx)) % 2 == 0, 1.0, -1.0)
    R["slow_floor"] = class_means(Xs * sgn[:, None, None], y, subj)  # (S, 3, C, T)
    del Xs
    # Welch spectra (1 s segments, 50 % overlap -> 1 Hz resolution)
    f, P = signal.welch(X, fs=SFREQ, nperseg=120, noverlap=60, axis=-1)
    R["psd_f"] = f
    R["psd_rel"] = class_rel(db(P), y, subj)                      # (S, 3, C, F)
    R["psd_all"] = class_means(db(P), y, subj).mean(1)            # (S, C, F)
    del P
    log(f"  filtering / covariances / spectra done ({time.time() - t0:.0f}s)")

    # --- log powers per band x segment ------------------------------------
    lp = {}                                                        # (band, seg) -> (n, C)
    for band in BANDS:
        for j, (a, b) in enumerate(SEGS):
            lp[(band, j)] = db(binpow[band][:, :, a:b].mean(-1))
    for j in range(len(SEGS)):
        lp[("broad", j)] = db(broad_var[:, j])

    # (a) topographies: class - mean of classes, whole window
    D = np.stack([class_rel(lp[(b, 4)], y, subj) for b in BANDS], 1)   # (S, 5, 3, C)
    R["topo_diff"] = D
    R["topo_t"] = tstat(D)
    R["topo_fwe95"] = [signflip_max_t(D[:, i]) for i in range(len(BANDS))]
    log(f"  (a) topographies; FWE95 max|t| per band {np.round(R['topo_fwe95'], 2).tolist()}")

    # (b) time courses
    lbin = {b: db(binpow[b]) for b in BANDS}
    R["erd_rel"] = {b: class_rel(lbin[b], y, subj) for b in BANDS}   # (S, 3, C, 16)
    allc = {b: class_means(lbin[b], y, subj).mean(1) for b in BANDS}  # (S, C, 16)
    R["tc_all"] = {b: allc[b] - allc[b].mean(-1, keepdims=True) for b in BANDS}
    log("  (b) time courses")

    # (c) separability -------------------------------------------------
    pow_rows = BANDS + ["broad", "fb4", "fb5"]
    ts_rows = BANDS + ["broad", "fb4", "fb5"]
    cols = list(range(len(SEGS))) + ["cat"]

    def pow_feat(r, c):
        bands = {"fb4": BANDS[1:], "fb5": BANDS}.get(r, [r])
        segs = range(4) if c == "cat" else [c]
        return np.concatenate([lp[(b, s)] for b in bands for s in segs], 1)

    def ts_cell_parts(r, c):
        """(band, seg) parts of a TS cell, or None when not computed."""
        if r in ("fb4", "fb5"):
            return None if c != 4 else [(b, 4) for b in ({"fb4": BANDS[1:], "fb5": BANDS}[r])]
        return [(r, s) for s in range(4)] if c == "cat" else [(r, c)]

    acc_pow = np.full((len(subjects), len(pow_rows), len(cols)), np.nan)
    acc_ts = np.full_like(acc_pow, np.nan)
    acc_slow = np.full((len(subjects), 3), np.nan)
    slow_sets = {"0-1 s": slice(0, 4), "0.5-4 s (solver slow block)": slice(2, 16),
                 "0-4 s": slice(0, 16)}
    anova = np.zeros((len(subjects), len(BANDS), len(SEGS), N_CH))
    null_acc = np.zeros((N_PERM_ACC, len(BANDS)))
    null_F = np.zeros((N_PERM_F, len(BANDS)))
    rng = np.random.default_rng(0)
    perm_idx = [[rng.permutation(N_TRAIN_PER_SUBJ) for _ in subjects] for _ in range(N_PERM_F)]
    null_anova = np.zeros((N_PERM_F, len(subjects), len(BANDS), len(SEGS), N_CH))
    null_acc_s = np.zeros((N_PERM_ACC, len(subjects), len(BANDS)))

    for si, s in enumerate(subjects):
        m = np.flatnonzero(subj == s)
        ys = y[m]
        folds = folds_for(ys)
        for ri, r in enumerate(pow_rows):
            for cj, c in enumerate(cols):
                acc_pow[si, ri, cj] = cv_bacc(pow_feat(r, c)[m], ys, folds)
        for k, (lab, sl) in enumerate(slow_sets.items()):
            acc_slow[si, k] = cv_bacc(slow_bins[m][:, :, sl].reshape(len(m), -1), ys, folds)
        for bi, b in enumerate(BANDS):
            for j in range(len(SEGS)):
                anova[si, bi, j] = f_classif(lp[(b, j)][m], ys)[0]
        # tangent space, fitted inside each training fold
        oof = {}
        for tri, tei in folds:
            ts = {}
            for r in BANDS + ["broad"]:
                for j in range(len(SEGS)):
                    C = covs[r][j][m]
                    tsm = xfeat.tangent_space().fit(C[tri])
                    ts[(r, j)] = (tsm.transform(C[tri]), tsm.transform(C[tei]))
            for ri, r in enumerate(ts_rows):
                for cj, c in enumerate(cols):
                    parts = ts_cell_parts(r, c)
                    if parts is None:
                        continue
                    Ftr = np.concatenate([ts[p][0] for p in parts], 1)
                    Fte = np.concatenate([ts[p][1] for p in parts], 1)
                    pred = oof.setdefault((ri, cj), np.empty_like(ys))
                    pred[tei] = lda_fit(Ftr, ys[tri]).predict(Fte)
        for (ri, cj), pred in oof.items():
            acc_ts[si, ri, cj] = balanced_accuracy_score(ys, pred)
        # permutation nulls (labels shuffled within subject)
        for p in range(N_PERM_F):
            yp = ys[perm_idx[p][si]]
            for bi, b in enumerate(BANDS):
                for j in range(len(SEGS)):
                    null_anova[p, si, bi, j] = f_classif(lp[(b, j)][m], yp)[0]
            if p < N_PERM_ACC:
                fp = folds_for(yp)
                for bi, b in enumerate(BANDS):
                    null_acc_s[p, si, bi] = cv_bacc(lp[(b, 4)][m], yp, fp)
        log(f"  (c) subject {s}: power whole-window "
            f"{np.round(acc_pow[si, :5, 4], 3).tolist()} | TS whole "
            f"{np.round(acc_ts[si, :5, 4], 3).tolist()} ({time.time() - t0:.0f}s)")

    null_acc = null_acc_s.mean(1)                                  # (P, 5) group means
    null_F = null_anova.mean(1).max(axis=(-1, -2))                 # (P, 5) max over seg x ch
    R.update(acc_pow=acc_pow, acc_ts=acc_ts, pow_rows=pow_rows, ts_rows=ts_rows,
             cols=[SEG_LABEL[c] if c != "cat" else "4 x 1 s concat" for c in cols],
             acc_slow=acc_slow, slow_sets=list(slow_sets), anova=anova,
             null_acc=null_acc, null_acc95=float(np.percentile(null_acc, 95)),
             null_acc_mean=float(null_acc.mean()),
             anova_fwe95=np.percentile(null_F, 95, axis=0).tolist(),
             anova_null_mean=float(null_anova.mean()))

    # (d) slow waveform bins: group t of class vs mean, first second vs rest
    R["slow_bin_t"] = tstat(class_rel(slow_bins, y, subj))          # (3, C, 16)

    # (e) artefact measures, class - mean (dB), per subject
    art = {
        "hEOG proxy\nF7-F8, 1-4 Hz": R.pop("heog_delta_db"),
        "AFz\n1-4 Hz": lp[("1to4", 4)][:, ci["AFz"]],
        "lateral F7 F8\nT3 T4, 30-45 Hz": lp[("30to45", 4)][:, [ci[c] for c in LATERAL_CH]].mean(1),
        "central C3 Cz\nC4 CPz, 30-45 Hz": lp[("30to45", 4)][:, [ci[c] for c in CENTRAL_CH]].mean(1),
    }
    R["art"] = {k: class_rel(v, y, subj) for k, v in art.items()}   # (S, 3)
    R["runtime_compute_s"] = time.time() - t0
    return R


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def save(fig, out, name, title=None, sub=None):
    """Title + subtitle are stacked ABOVE the figure box (bbox 'tight' keeps
    them), so they can never collide with axes titles or legends."""
    H = fig.get_size_inches()[1]
    y = 1.0 + 0.05 / H
    if sub:
        fig.text(0.01, y, sub, ha="left", va="bottom", fontsize=9, color=INK2,
                 linespacing=1.35)
        y += ((sub.count("\n") + 1) * 0.19 + 0.08) / H
    if title:
        fig.text(0.01, y, title, ha="left", va="bottom", fontsize=12.5, fontweight="bold")
    fig.savefig(out / name)
    plt.close(fig)
    log(f"  wrote {name}")


def topo(ax, data, info, vlim, cmap, mask=None, names=None, msize=5):
    im, _ = mne.viz.plot_topomap(
        data, info, axes=ax, cmap=cmap, vlim=vlim, mask=mask, names=names,
        mask_params=dict(marker="o", markerfacecolor="white", markeredgecolor=INK,
                         linewidth=0, markersize=msize, markeredgewidth=1.2),
        contours=0, sensors=True, show=False)
    return im


def fig_montage(R, info, out):
    ch = R["ch_names"]
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    mne.viz.plot_sensors(info, kind="topomap", show_names=False, axes=ax, show=False,
                         pointsize=18, linewidth=0)
    pos = ax.collections[0].get_offsets()                  # 2-D topomap coords, ch order
    ax.collections[0].set_color(MUTED)
    erd = np.array([c in ERD_CH for c in ch])
    ax.scatter(pos[erd, 0], pos[erd, 1], s=42, color=CAT[0], zorder=3, lw=0)
    below = {"P5", "P6", "O1", "O2"}                        # de-crowd the posterior rows
    for (x, yy), c, e in zip(pos, ch, erd):
        dy, va = (-0.006, "top") if c in below else (0.006, "bottom")
        ax.text(x, yy + dy, c, ha="center", va=va, fontsize=8,
                fontweight="bold" if e else "normal", color=INK if e else INK2)
    ax.set_title("blue = channels of the time-course figures (b)", fontsize=9,
                 fontweight="normal", color=INK2, loc="center")
    save(fig, out, "activation_montage.png", "Scherer 2015 montage (30 EEG channels)",
         "Positions: standard_1005 (T3 = T7, T4 = T8).")


def fig_topo_t(R, info, out):
    T, D = R["topo_t"], R["topo_diff"]
    fig, axes = plt.subplots(len(BANDS), 3, figsize=(8.2, 13.2),
                             gridspec_kw=dict(hspace=0.22, wspace=0.05, top=0.97,
                                              bottom=0.03))
    for bi, b in enumerate(BANDS):
        v = max(3.5, float(np.abs(T[bi]).max()))
        for k in range(3):
            ax = axes[bi, k]
            im = topo(ax, T[bi, k], info, (-v, v), DIV, mask=np.abs(T[bi, k]) > 3)
            md = D[:, bi, k].mean(0)
            j = int(np.argmax(np.abs(T[bi, k])))
            if bi == 0:
                ax.set_title(CLASS_NAMES[k], loc="center", fontsize=11.5, pad=10)
            ax.text(0.5, -0.03, f"peak {R['ch_names'][j]}  t {T[bi, k, j]:+.1f}  "
                    f"({md[j]:+.2f} dB)", transform=ax.transAxes, ha="center", va="top",
                    fontsize=8, color=INK2)
            if k == 0:
                ax.text(-0.18, 0.5, BAND_LABEL[b], rotation=90, va="center", ha="center",
                        transform=ax.transAxes, fontsize=10, fontweight="bold")
        cb = fig.colorbar(im, ax=axes[bi, :].tolist(), shrink=0.8, pad=0.02, aspect=12)
        cb.set_label("group t (df 8)", fontsize=8.5)
        cb.ax.axhline(3, color=INK, lw=0.8)
        cb.ax.axhline(-3, color=INK, lw=0.8)
    save(fig, out, "activation_bandpower_tmaps.png",
         "(a) Where each task differs in band power: class minus the mean of the 3 classes",
         "Log band power (dB), whole 4 s window, training sessions only; group t across 9 "
         "subjects; white dots |t| > 3 (uncorrected).\nRed = more power than the other tasks "
         "(ERS), blue = less (ERD). Family-wise 95 % max|t| per band: "
         + ", ".join(f"{b.replace('to', '-')} {v:.1f}" for b, v in zip(BANDS, R["topo_fwe95"])))


def fig_erd(R, out, band, name):
    E = R["erd_rel"][band]                                          # (S, 3, C, 16)
    ci = {c: i for i, c in enumerate(R["ch_names"])}
    t = (np.arange(N_BINS) + 0.5) * BIN / SFREQ
    m, se = E.mean(0), sem(E)
    lim = 1.08 * np.nanmax(np.abs(np.stack([m + se, m - se]))[..., [ci[c] for c in ERD_CH if c in ci], :])
    fig, axes = plt.subplots(3, 5, figsize=(12.5, 7.4), sharex=True, sharey=True,
                             gridspec_kw=dict(hspace=0.32, wspace=0.14, top=0.95))
    for r, row in enumerate(ERD_GRID):
        for c, ch in enumerate(row):
            ax = axes[r, c]
            if ch is None or ch not in ci:
                ax.axis("off")
                continue
            if c == 1 and r != 1:
                ax.tick_params(labelleft=True)
            j = ci[ch]
            for sec in (1, 2, 3):
                ax.axvline(sec, color=GRID, lw=0.8, zorder=0)
            ax.axhline(0, color=MUTED, lw=0.8, zorder=0)
            for k in range(3):
                ax.fill_between(t, m[k, j] - se[k, j], m[k, j] + se[k, j], color=CAT[k],
                                alpha=0.16, lw=0)
                ax.plot(t, m[k, j], color=CAT[k], lw=2, label=CLASS_NAMES[k])
            tmax = np.abs(tstat(E[:, :, j])).max()
            ax.set_title(ch, loc="left")
            ax.text(0.98, 0.04, f"max|t| {tmax:.1f}", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=8, color=INK2)
            ax.set_ylim(-lim, lim)
            ax.set_xlim(0, 4)
            if c == 0 or (r != 1 and c == 1):
                ax.set_ylabel("class - all (dB)")
            if r == 2 or (r == 1 and c in (0, 4)):
                ax.set_xlabel("time from window start (s)")
                ax.tick_params(labelbottom=True)
    handles = [plt.Line2D([], [], color=CAT[k], lw=2.5) for k in range(3)]
    axes[0, 0].legend(handles, CLASS_NAMES, loc="center left", bbox_to_anchor=(0.0, 0.5),
                      fontsize=10.5, title="class", title_fontsize=9.5)   # empty top-left slot
    save(fig, out, name,
         f"(b) {BAND_LABEL[band]} power over time, class minus the all-class mean of each bin",
         "0.25 s bins; group mean ± SEM over 9 subjects (training sessions). t = 0 is the "
         "cue (3.0 s after trial start). Below 0 = ERD relative to the other tasks.")


def fig_tc_all(R, out):
    t = (np.arange(N_BINS) + 0.5) * BIN / SFREQ
    fig, axes = plt.subplots(1, len(BANDS), figsize=(13.5, 3.3), sharey=True,
                             gridspec_kw=dict(wspace=0.08, top=0.9))
    for bi, b in enumerate(BANDS):
        A = R["tc_all"][b].mean(1)                                   # (S, 16) mean over ch
        ax = axes[bi]
        ax.axhline(0, color=MUTED, lw=0.8)
        for sa in A:
            ax.plot(t, sa, color=CAT[0], lw=0.7, alpha=0.3)
        ax.fill_between(t, A.mean(0) - sem(A), A.mean(0) + sem(A), color=CAT[0], alpha=0.2, lw=0)
        ax.plot(t, A.mean(0), color=CAT[0], lw=2.2)
        ax.set_title(BAND_LABEL[b], fontsize=9.5)
        ax.set_xlim(0, 4)
        ax.set_xlabel("time (s)")
        if bi == 0:
            ax.set_ylabel("bin - window mean (dB)")
    save(fig, out, "activation_timecourse_allclass.png",
         "(b) Task-agnostic power time course: each 0.25 s bin minus the 4 s window mean",
         "Mean over all 30 channels and the 3 classes; thick = group mean ± SEM, thin = "
         "subjects. Shows how non-stationary the window is (cue response, edge effects).")


def _heat(ax, A, rows, cols, vmin, vmax, thr, title):
    im = ax.imshow(A, cmap=SEQ, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = A[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", color=MUTED, fontsize=9)
                continue
            dark = (v - vmin) / (vmax - vmin) > 0.55
            col = "white" if dark else (INK if v >= thr else MUTED)
            ax.text(j, i, f"{v:.3f}".lstrip("0"), ha="center", va="center", color=col,
                    fontsize=8.8, fontweight="bold" if v >= thr else "normal")
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for yy in (4.5, 5.5):                    # single bands | broadband | multi-band
        ax.axhline(yy, color=SURFACE, lw=3.5)
    for xx in (3.5, 4.5):                    # 1 s segments | whole window | concat
        ax.axvline(xx, color=SURFACE, lw=3.5)
    ax.set_title(title, fontsize=10.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


ROW_LABEL = {**{b: b.replace("to", "-") + " Hz" for b in BANDS},
             "broad": "broadband", "fb4": "4 bands 4-45 Hz", "fb5": "5 bands 1-45 Hz"}


def fig_separability(R, out):
    Ap, At = R["acc_pow"].mean(0), R["acc_ts"].mean(0)
    vmin, vmax = 1 / 3, max(0.55, float(np.nanmax([Ap.max(), np.nanmax(At)])))
    thr = R["null_acc95"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4),
                             gridspec_kw=dict(wspace=0.42, top=0.92))
    im = _heat(axes[0], Ap, [ROW_LABEL[r] for r in R["pow_rows"]], R["cols"], vmin, vmax, thr,
               "Log band power per channel (30 per cell)")
    _heat(axes[1], At, [ROW_LABEL[r] for r in R["ts_rows"]], R["cols"], vmin, vmax, thr,
          "Tangent space of the band covariance (465 per cell)")
    cb = fig.colorbar(im, ax=axes.tolist(), shrink=0.85, pad=0.02, aspect=25)
    cb.set_label("balanced accuracy (chance 0.333)")
    cb.ax.axhline(thr, color=INK, lw=1.2)     # permutation-null 95th percentile
    save(fig, out, "activation_separability.png",
         "(c) How separable the 3 tasks are, by band x time segment (within-subject CV)",
         "5-fold stratified CV of shrinkage LDA on each subject's training session "
         f"(120 windows), mean over 9 subjects. Bold = above the label-permutation null 95 % "
         f"({thr:.3f}, black line on the colour bar); '–' = not computed.\n"
         "TS reference fitted inside each training fold. "
         "Last column: the 4 one-second segments concatenated (time-resolved features).")


def fig_anova(R, info, out):
    F = R["anova"].mean(0)                                          # (5 bands, 5 segs, C)
    vmax = float(F.max())
    fig, axes = plt.subplots(len(BANDS), len(SEGS), figsize=(11, 11.2),
                             gridspec_kw=dict(hspace=0.12, wspace=0.04, top=0.96, bottom=0.03))
    for bi, b in enumerate(BANDS):
        thr = R["anova_fwe95"][bi]
        for j in range(len(SEGS)):
            ax = axes[bi, j]
            im = topo(ax, F[bi, j], info, (1.0, vmax), SEQ, mask=F[bi, j] > thr, msize=4)
            if bi == 0:
                ax.set_title(SEG_LABEL[j], loc="center", fontsize=10)
            if j == 0:
                ax.text(-0.14, 0.5, BAND_LABEL[b], rotation=90, va="center", ha="center",
                        transform=ax.transAxes, fontsize=9.5, fontweight="bold")
            jj = int(np.argmax(F[bi, j]))
            ax.text(0.5, -0.04, f"max {R['ch_names'][jj]} {F[bi, j, jj]:.1f}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=INK2)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, pad=0.02, aspect=30)
    cb.set_label("group mean of per-subject ANOVA F (df 2, 117); null mean ≈ 1.0")
    save(fig, out, "activation_anova_topomaps.png",
         "(c) Where the discriminative band power sits, per band x time segment",
         "Per-channel one-way ANOVA F of log band power across WORD / SUB / HAND, per subject, "
         "averaged over 9 subjects.\nWhite dots: above the per-band family-wise 95 % "
         "permutation threshold (max over channels x segments; "
         + ", ".join(f"{b.replace('to', '-')} {v:.2f}" for b, v in zip(BANDS, R["anova_fwe95"]))
         + ").")


def fig_evoked(R, out):
    W = R["slow_wave"]                                              # (S, 3, C, T)
    ci = {c: i for i, c in enumerate(R["ch_names"])}
    t = np.arange(N_T) / SFREQ
    chans = [c for c in EVOKED_CH if c in ci]
    fig = plt.figure(figsize=(14, 6.8))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 0.95], hspace=0.5, top=0.93)
    gtop = outer[0].subgridspec(1, len(chans), wspace=0.08)
    gbot = outer[1].subgridspec(1, 3, width_ratios=[1, 1, 0.62], wspace=0.32)
    axes = [fig.add_subplot(gtop[0, 0])]
    axes += [fig.add_subplot(gtop[0, i], sharey=axes[0]) for i in range(1, len(chans))]
    for a in axes[1:]:
        a.tick_params(labelleft=False)
    for a, ch in zip(axes, chans):
        j = ci[ch]
        a.axvspan(0, 1, color=GRID, alpha=0.45, lw=0, zorder=0)
        a.axhline(0, color=MUTED, lw=0.8, zorder=0)
        for k in range(3):
            mu, se = W[:, k, j].mean(0), sem(W[:, k, j])
            a.fill_between(t, mu - se, mu + se, color=CAT[k], alpha=0.16, lw=0)
            a.plot(t, mu, color=CAT[k], lw=1.8, label=CLASS_NAMES[k])
        tt = np.abs(R["slow_bin_t"][:, j])
        a.set_title(ch, loc="left")
        a.text(0.98, 0.04, f"max|t| 0-1 s {tt[:, :4].max():.1f}\n1-4 s {tt[:, 4:].max():.1f}",
               transform=a.transAxes, ha="right", va="bottom", fontsize=7.8, color=INK2)
        a.set_xlim(0, 4)
        a.set_xlabel("time from cue (s)")
    axes[0].set_ylabel("0.1-4 Hz amplitude\n(robust units)")
    axes[-1].legend(loc="upper right", ncol=1, fontsize=8.5)
    # bottom: global field power (spatial SD over the 30 channels) vs the +/- floor
    Fl = R["slow_floor"]

    def gfp(A):                                                  # (..., C, T) -> (..., T)
        return A.std(axis=-2)
    curves = [
        ("evoked response, all classes", gfp(W.mean(1)), gfp(Fl.mean(1)), INK),
        ("class-specific part (class - all-class ERP)",
         gfp(W - W.mean(1, keepdims=True)).mean(1),
         gfp(Fl - Fl.mean(1, keepdims=True)).mean(1), CAT[0]),
    ]
    for i, (lab, g, gf, col) in enumerate(curves):
        a = fig.add_subplot(gbot[0, i])
        a.axvspan(0, 1, color=GRID, alpha=0.45, lw=0, zorder=0)
        a.fill_between(t, g.mean(0) - sem(g), g.mean(0) + sem(g), color=col, alpha=0.18, lw=0)
        a.plot(t, g.mean(0), color=col, lw=1.8, label="class averages")
        a.plot(t, gf.mean(0), color=MUTED, lw=1.4, ls="--", label="± floor (no evoked part)")
        r1 = g[:, :120].mean() / gf[:, :120].mean()
        r2 = g[:, 120:].mean() / gf[:, 120:].mean()
        a.set_title(f"{lab}\nmean GFP / floor: 0-1 s {r1:.2f}, 1-4 s {r2:.2f}", fontsize=9.5)
        a.set_xlim(0, 4)
        a.set_xlabel("time from cue (s)")
        a.set_ylabel("GFP (robust units)")
        a.legend(loc="upper right", fontsize=8)
    ax_t = fig.add_subplot(gbot[0, 2])
    acc = R["acc_slow"].mean(0)
    ax_t.barh(range(3), acc - 1 / 3, left=1 / 3, color=CAT[0], height=0.55,
              xerr=sem(R["acc_slow"]), error_kw=dict(ecolor=INK2, lw=1))
    ax_t.axvline(1 / 3, color=MUTED, lw=0.8)
    ax_t.axvline(R["null_acc95"], color=INK, lw=0.8, ls=":")
    ax_t.set_yticks(range(3), ["0-1 s", "0.5-4 s\n(solver slow)", "0-4 s"], fontsize=8.5)
    for i, v in enumerate(acc):
        ax_t.text(v + sem(R["acc_slow"][:, i]) + 0.01, i, f"{v:.3f}", va="center",
                  fontsize=8.5, color=INK)
    ax_t.set_xlim(0.3, 0.6)
    ax_t.invert_yaxis()
    ax_t.set_xlabel("balanced acc. (dotted: null 95 %)")
    ax_t.set_title("LDA on waveform bins", fontsize=9.5)
    save(fig, out, "activation_evoked.png",
         "(d) Class-average slow waveforms (0.1-4 Hz): is there a class-specific cue response?",
         "Top: group mean ± SEM over 9 subjects (training sessions), first second shaded; "
         "max|t| = class vs mean on 0.25 s bin means.\nBottom: global field power of the "
         "subject averages vs the ± reference (alternate windows sign-flipped: same noise, "
         "evoked part cancelled); right: within-subject CV of LDA on 0.25 s waveform bins.")


def fig_artefact(R, out):
    f, P = R["psd_f"], R["psd_rel"]                                 # (S, 3, C, F)
    ci = {c: i for i, c in enumerate(R["ch_names"])}
    fm = (f >= 1) & (f <= 55)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.5], wspace=0.22, top=0.9))
    groups = [(LATERAL_CH, "lateral edge (EMG-prone): " + " ".join(LATERAL_CH)),
              (CENTRAL_CH, "central: " + " ".join(CENTRAL_CH))]
    lim = 0
    for ax, (chs, lab) in zip(axes[:2], groups):
        G = P[:, :, [ci[c] for c in chs]].mean(2)[..., fm]          # (S, 3, F)
        for lo, hi in ((4, 8), (13, 30)):
            ax.axvspan(lo, hi, color=GRID, alpha=0.35, lw=0, zorder=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        for k in range(3):
            mu, se = G[:, k].mean(0), sem(G[:, k])
            ax.fill_between(f[fm], mu - se, mu + se, color=CAT[k], alpha=0.16, lw=0)
            ax.plot(f[fm], mu, color=CAT[k], lw=1.8, label=CLASS_NAMES[k])
            lim = max(lim, np.abs(mu).max() + se.max())
        ax.set_title(lab, fontsize=9.5)
        ax.set_xlabel("frequency (Hz)")
        ax.set_xlim(1, 55)
    for ax in axes[:2]:
        ax.set_ylim(-1.1 * lim, 1.1 * lim)
    axes[0].set_ylabel("class - all-class mean (dB)")
    axes[0].legend(loc="upper left", fontsize=8.5)
    ax = axes[2]
    keys = list(R["art"])
    ax.axhline(0, color=MUTED, lw=0.8)
    rng = np.random.default_rng(1)
    ticks = []
    for gi, key in enumerate(keys):
        A = R["art"][key]                                           # (S, 3)
        T = tstat(A)
        for k in range(3):
            x = gi + (k - 1) * 0.26
            ax.scatter(x + rng.uniform(-0.05, 0.05, len(A)), A[:, k], s=16, color=CAT[k],
                       alpha=0.55, lw=0)
            ax.plot([x - 0.1, x + 0.1], [A[:, k].mean()] * 2, color=CAT[k], lw=2.6)
        ticks.append(key + "\nt " + " / ".join(f"{v:+.1f}" for v in T))
    ax.set_xticks(range(len(keys)), ticks, fontsize=7.8)
    ax.set_xlim(-0.55, len(keys) - 0.45)
    ax.set_ylabel("class - all-class mean (dB), per subject")
    ax.set_title("Artefact-sensitive measures (t: WORD / SUB / HAND)", fontsize=9.5)
    save(fig, out, "activation_artefact_check.png",
         "(e) Artefact check: do the class differences look like muscle (EMG) or eye (EOG)?",
         "Welch spectra (1 s, 1 Hz resolution) of each class minus the all-class mean, "
         "group mean ± SEM (shaded bands: 4-8 and 13-30 Hz). EMG rises with frequency above "
         "~20 Hz and peaks at the lateral edge;\nhorizontal eye movements load the F7-F8 "
         "difference at < 4 Hz. Dots = subjects, bars = group means.")


def make_figures(R, out):
    info = make_info(R["ch_names"])
    fig_montage(R, info, out)
    fig_topo_t(R, info, out)
    for b, name in zip(ERD_BANDS, ["activation_erd_theta.png", "activation_erd_alpha.png",
                                   "activation_erd_beta.png"]):
        fig_erd(R, out, b, name)
    fig_tc_all(R, out)
    fig_separability(R, out)
    fig_anova(R, info, out)
    fig_evoked(R, out)
    fig_artefact(R, out)


# ---------------------------------------------------------------------------
# numbers (auto-generated; the interpreted report is written by hand)
# ---------------------------------------------------------------------------
def fmt_ms(a):
    return f"{np.nanmean(a):.3f} ± {sem(a):.3f}"


def write_numbers(R, out, shape_line, runtime, replot=False):
    ch = R["ch_names"]
    L_ = [f"# Activation EDA: numbers (auto-generated by `analysis/activation_eda.py`)",
          "", f"Generated {time.strftime('%Y-%m-%d %H:%M')}; "
          + (f"figures re-plotted from cached stats in {runtime:.0f} s; " if replot
             else f"runtime {runtime:.0f} s; ")
          + f"statistics computed in {R['runtime_compute_s']:.0f} s (2 threads).",
          "", "```", shape_line, "```", "",
          "Training sessions only. Mean ± SEM over subjects; t = one-sample t across "
          "subjects (df 8).", "",
          "## (a) Band-power topographies (class minus mean of the 3 classes, whole window)", "",
          "| band | FWE95 max|t| | class | channels with |t| > 3 (t, mean dB) | peak |",
          "|---|---|---|---|---|"]
    for bi, b in enumerate(BANDS):
        for k in range(3):
            t = R["topo_t"][bi, k]
            md = R["topo_diff"][:, bi, k].mean(0)
            sig = [f"{ch[j]} {t[j]:+.1f} ({md[j]:+.2f})" for j in np.argsort(-np.abs(t))
                   if abs(t[j]) > 3]
            j = int(np.argmax(np.abs(t)))
            L_.append(f"| {b} | {R['topo_fwe95'][bi]:.2f} | {CLASS_NAMES[k]} | "
                      f"{', '.join(sig) or '-'} | {ch[j]} {t[j]:+.1f} ({md[j]:+.2f} dB) |")
    L_ += ["", "## (b) Time courses: peak |t| over 0.25 s bins (class vs all-class mean)", "",
           "| band | channel | WORD peak t @ s | SUB peak t @ s | HAND peak t @ s |",
           "|---|---|---|---|---|"]
    tc = (np.arange(N_BINS) + 0.5) * BIN / SFREQ
    for b in BANDS:
        T = tstat(R["erd_rel"][b])                                   # (3, C, 16)
        for c in ERD_CH:
            j = ch.index(c)
            cells = []
            for k in range(3):
                i = int(np.argmax(np.abs(T[k, j])))
                cells.append(f"{T[k, j, i]:+.1f} @ {tc[i]:.2f}")
            L_.append(f"| {b} | {c} | " + " | ".join(cells) + " |")
    L_ += ["", "Task-agnostic time course (mean over channels, bin minus window mean, dB; "
           "group mean per 1 s segment):", "",
           "| band | 0-0.25 s | 0-1 s | 1-2 s | 2-3 s | 3-4 s | 3.75-4 s |", "|---|---|---|---|---|---|---|"]
    for b in BANDS:
        A = R["tc_all"][b].mean(1).mean(0)
        L_.append(f"| {b} | {A[0]:+.2f} | " + " | ".join(f"{A[a:a + 4].mean():+.2f}"
                                                         for a in (0, 4, 8, 12))
                  + f" | {A[-1]:+.2f} |")
    L_ += ["", "## (c) Separability (balanced accuracy, mean ± SEM over subjects)", "",
           f"Label-permutation null (power, whole window, {N_PERM_ACC} perms x 5 bands, "
           f"group mean): mean {R['null_acc_mean']:.3f}, 95th percentile "
           f"**{R['null_acc95']:.3f}**.", ""]
    for key, rows, lab in (("acc_pow", R["pow_rows"], "log band power"),
                           ("acc_ts", R["ts_rows"], "tangent space")):
        A = R[key]
        L_ += [f"### {lab}", "", "| rows \\ cols | " + " | ".join(R["cols"]) + " |",
               "|---" * (len(R["cols"]) + 1) + "|"]
        for ri, r in enumerate(rows):
            cells = ["-" if np.all(np.isnan(A[:, ri, cj])) else fmt_ms(A[:, ri, cj])
                     for cj in range(len(R["cols"]))]
            L_.append(f"| {ROW_LABEL[r]} | " + " | ".join(cells) + " |")
        L_.append("")
    L_ += ["### TS minus power (same cell), mean over subjects [n subjects TS > power]", "",
           "| rows \\ cols | " + " | ".join(R["cols"]) + " |", "|---" * (len(R["cols"]) + 1) + "|"]
    for ri, r in enumerate(R["ts_rows"]):
        cells = []
        for cj in range(len(R["cols"])):
            dd = R["acc_ts"][:, ri, cj] - R["acc_pow"][:, ri, cj]
            cells.append("-" if np.all(np.isnan(dd)) else
                         f"{np.nanmean(dd):+.3f} [{int((dd > 0).sum())}/{len(dd)}]")
        L_.append(f"| {ROW_LABEL[r]} | " + " | ".join(cells) + " |")
    L_ += ["", "### Per-subject balanced accuracy, selected cells", "",
           "| cell | " + " | ".join(f"S{s}" for s in R["subjects"]) + " |",
           "|---" * (len(R["subjects"]) + 1) + "|"]
    sel = [("power 5 bands, 0-4 s", R["acc_pow"][:, R["pow_rows"].index("fb5"), 4]),
           ("power 5 bands, 4 x 1 s concat", R["acc_pow"][:, R["pow_rows"].index("fb5"), 5]),
           ("power broadband, 0-4 s", R["acc_pow"][:, R["pow_rows"].index("broad"), 4]),
           ("TS broadband, 0-4 s", R["acc_ts"][:, R["ts_rows"].index("broad"), 4]),
           ("TS 4 bands 4-45, 0-4 s", R["acc_ts"][:, R["ts_rows"].index("fb4"), 4]),
           ("TS 5 bands 1-45, 0-4 s", R["acc_ts"][:, R["ts_rows"].index("fb5"), 4])]
    for lab, v in sel:
        L_.append(f"| {lab} | " + " | ".join(f"{x:.3f}" for x in v) + " |")
    L_ += ["", "### Per-channel ANOVA F (group mean), top 3 channels per band x segment", "",
           f"Null: mean F under permutation {R['anova_null_mean']:.3f}; FWE95 of max over "
           "channels x segments per band: "
           + ", ".join(f"{b} {v:.2f}" for b, v in zip(BANDS, R["anova_fwe95"])), "",
           "| band | " + " | ".join(SEG_LABEL) + " |", "|---" * (len(SEG_LABEL) + 1) + "|"]
    F = R["anova"].mean(0)
    for bi, b in enumerate(BANDS):
        cells = [", ".join(f"{ch[j]} {F[bi, s, j]:.2f}" for j in np.argsort(-F[bi, s])[:3])
                 for s in range(len(SEGS))]
        L_.append(f"| {b} | " + " | ".join(cells) + " |")
    L_ += ["", "## (d) Slow waveform (0.1-4 Hz)", "",
           "LDA on 0.25 s waveform-bin means (30 ch), balanced accuracy:", "",
           "| bins | mean ± SEM | per subject |", "|---|---|---|"]
    for k, lab in enumerate(R["slow_sets"]):
        v = R["acc_slow"][:, k]
        L_.append(f"| {lab} | {fmt_ms(v)} | " + " ".join(f"{x:.2f}" for x in v) + " |")
    W, Fl = R["slow_wave"], R["slow_floor"]
    L_ += ["", "Global field power of the averages over the ± reference (1.0 = noise only):", ""]
    for lab, g, gf in (("ERP, all classes", W.mean(1).std(-2), Fl.mean(1).std(-2)),
                       ("class-specific part", (W - W.mean(1, keepdims=True)).std(-2).mean(1),
                        (Fl - Fl.mean(1, keepdims=True)).std(-2).mean(1))):
        r = [g[:, a:b].mean() / gf[:, a:b].mean() for a, b in ((0, 60), (60, 120), (0, 120),
                                                               (120, 480))]
        L_.append(f"- GFP / ± floor, {lab}: 0-0.5 s {r[0]:.2f}, 0.5-1 s {r[1]:.2f}, "
                  f"0-1 s {r[2]:.2f}, 1-4 s {r[3]:.2f}")
    L_ += ["", "Class vs mean, group t on 0.25 s bin means (max |t|, 0-1 s and 1-4 s):", "",
           "| channel | WORD 0-1 / 1-4 | SUB 0-1 / 1-4 | HAND 0-1 / 1-4 |", "|---|---|---|---|"]
    for c in EVOKED_CH + ["AFz", "F7", "F8"]:
        j = ch.index(c)
        T = R["slow_bin_t"][:, j]
        L_.append(f"| {c} | " + " | ".join(
            f"{np.abs(T[k, :4]).max():.1f} / {np.abs(T[k, 4:]).max():.1f}" for k in range(3)) + " |")
    L_ += ["", "## (e) Artefact-sensitive measures (class minus mean, dB)", "",
           "| measure | WORD mean (t) | SUB mean (t) | HAND mean (t) |", "|---|---|---|---|"]
    for key, A in R["art"].items():
        T = tstat(A)
        L_.append(f"| {key.replace(chr(10), ' ')} | " + " | ".join(
            f"{A[:, k].mean():+.2f} ({T[k]:+.1f})" for k in range(3)) + " |")
    f, P = R["psd_f"], R["psd_rel"]
    ci = {c: i for i, c in enumerate(ch)}
    L_ += ["", "Class-difference spectra, mean dB (group t) per frequency range:", "",
           "| channels | class | 1-4 | 4-8 | 8-13 | 13-20 | 20-30 | 30-45 | 45-55 |",
           "|---|---|---|---|---|---|---|---|---|"]
    rngs = [(1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45), (45, 55)]
    for chs, lab in ((LATERAL_CH, "lateral"), (CENTRAL_CH, "central"), (["AFz", "Fz"], "AFz Fz"),
                     (["O1", "O2", "PO3", "PO4"], "occipital")):
        G = P[:, :, [ci[c] for c in chs]].mean(2)
        for k in range(3):
            cells = []
            for lo, hi in rngs:
                v = G[:, k, (f >= lo) & (f <= hi)].mean(-1)
                cells.append(f"{v.mean():+.2f} ({tstat(v):+.1f})")
            L_.append(f"| {lab} | {CLASS_NAMES[k]} | " + " | ".join(cells) + " |")
    (out / "activation_eda_numbers.md").write_text("\n".join(L_) + "\n", encoding="utf-8")
    log("  wrote activation_eda_numbers.md")

    def js(v):
        if isinstance(v, np.ndarray):
            return np.round(v, 5).tolist()
        if isinstance(v, dict):
            return {str(k).replace("\n", " "): js(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [js(x) for x in v]
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v
    keep = ["subjects", "ch_names", "topo_t", "topo_fwe95", "pow_rows", "ts_rows", "cols",
            "acc_pow", "acc_ts", "acc_slow", "slow_sets", "null_acc95", "null_acc_mean",
            "anova_fwe95", "anova_null_mean", "art"]
    summ = {k: js(R[k]) for k in keep}
    summ["anova_group_mean"] = js(R["anova"].mean(0))
    summ["shape_line"] = shape_line
    summ["runtime_s"] = runtime
    summ["runtime_compute_s"] = R["runtime_compute_s"]
    summ["replot"] = replot
    (out / "activation_eda_summary.json").write_text(json.dumps(summ, indent=1))
    log("  wrote activation_eda_summary.json")


# ---------------------------------------------------------------------------
def main():
    global N_PERM_ACC, N_PERM_F
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="2 subjects, few perms, scratch dir")
    ap.add_argument("--replot", action="store_true", help="figures + numbers from cached stats")
    a = ap.parse_args()
    t0 = time.time()
    out, stats = (QUICK_OUT, QUICK_OUT / "stats.pkl") if a.quick else (OUT, STATS)
    out.mkdir(parents=True, exist_ok=True)
    stats.parent.mkdir(parents=True, exist_ok=True)
    log(f"activation EDA -> {out} (threads XS_THREADS={L.N_THREADS})")
    if a.replot:
        with open(stats, "rb") as fh:
            R, shape_line = pickle.load(fh)
        log(f"[replot] loaded {stats}")
        log(shape_line)
    else:
        if a.quick:
            N_PERM_ACC, N_PERM_F = 3, 5
        X, y, subj, ch_names, shape_line = load(quick=a.quick)
        R = compute(X, y, subj, ch_names)
        with open(stats, "wb") as fh:
            pickle.dump((R, shape_line), fh)
        log(f"  cached stats -> {stats}")
    make_figures(R, out)
    runtime = time.time() - t0
    write_numbers(R, out, shape_line, runtime, replot=a.replot)
    log(f"done in {runtime:.0f} s")


if __name__ == "__main__":
    main()
