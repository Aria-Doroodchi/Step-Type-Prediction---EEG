#!/usr/bin/env python
"""Mock sealed study: synthetic windows with the Graz + BrainHero structure.

The sealed Track 2 data (20 participants x 6 sessions, 43 EEG + 2 EMG + 2 EOG,
3 cued commands, Graz vs BrainHero contexts) is not released yet. This writes
a cache with the same *structure*, in the format of ``xsess_cache.py``
build(), so the release pipeline (harness splits, context cells, solver,
benchopt dataset, ablations, runbook) can run end to end today. **Its
accuracies mean nothing for the recipe**: they only check plumbing, timing
and that the ablation machinery recovers effects injected by construction.

    python ~/codabench/analysis/mock_sealed.py mock_sealed_s [--seed 0] \\
        [--ctx_mode runs|sessions] [--force] [--check [--check_variants ...]] \\
        [--probe_subjects N]

Studies (4-s windows, 3 classes, 20 subjects x 6 sessions x 4 runs):
    mock_sealed_s     120 Hz,  8 trials/class/session ->  2,880 windows (smoke)
    mock_sealed_120   120 Hz, 40 trials/class/session -> 14,400 windows
    mock_sealed_500   500 Hz, 40 trials/class/session -> 14,400 windows
                      (2000 samples, ~5.4 GB; X is written into an
                      ``open_memmap`` mapped one subject at a time, never
                      held in RAM: peak RSS ~0.5 GB)
Build times (2026-09-28, 20-thread box under load ~12): mock_sealed_s 5-8 s,
mock_sealed_120 40-47 s, mock_sealed_500 ~2.5 min estimated from a 2-subject
probe (4.4-7.3 s per subject).

Output: ~/neuralbench/xsess_cache/<study>/ with the xsess_cache arrays
(X, y, subj, session, run, split, onset, code) plus ``context.npy``
("graz" | "brainhero"), and meta.json with the xsess_cache keys plus
``ch_types``, ``context_column``, ``full_subjects``, ``eval_subjects``,
``calib_sessions`` and ``mock`` (generator parameters, seed, ground truth).
Rows are sorted by (subject, session, run, onset) = recording order. The
build is deterministic for a seed (bit-identical across runs) and skipped
when meta.json exists (unless ``--force``). Every file is written to a temp
name and renamed, meta.json last: a killed build leaves no meta.json and
simply rebuilds, and a reader during a ``--force`` rebuild never sees a
half-written file. After a rebuild under the same study name, run benchopt
with ``--no-cache``: its result cache is keyed on the dataset parameters,
not on the data, and silently returns the old score.

Structure (the sealed-phase description on the tracks page):
- subjects 0..9 = fully labelled training participants (all 6 sessions);
  10..19 = evaluation participants: sessions 0..2 labelled (calibration),
  3..5 hidden (``split == 2``; every other window has split 0);
- contexts, ``--ctx_mode runs`` (default): runs 0 and 2 = "graz", runs 1 and
  3 = "brainhero", so each subject x session has both contexts and the
  metric has 20 x 6 x 2 cells. ``--ctx_mode sessions``: one context per
  session (even sessions graz, odd brainhero), written to
  ``<study>_ctxsess`` so the default cache is never replaced.

Generative model (every parameter is in meta["mock"]):
- EEG: 12 latent sources per subject (frontal midline, left/right motor,
  parietal midline, left/right temporal, occipital, ...), each a 1/f (pink)
  background plus theta / alpha(mu) / beta / low-gamma spectral peaks,
  mixed into the 43 EEG channels by a subject-specific smooth spatial
  mixing (Gaussian bump around a jittered source direction on the fitted
  head sphere, a tangential-dipole asymmetry and a broad volume-conduction
  term), plus white sensor noise;
- class effects on source band power (log-power shifts scaled by a subject
  strength; three subjects near chance): MI = mu/beta ERD on both motor
  sources; CALC = frontal-midline theta up, parietal alpha down; WORD =
  left-temporal beta/low-gamma up. Window-to-window state noise on every
  band power and a per-trial effect jitter keep single windows ambiguous;
- session drift growing with the session index: a cap rotation along a
  subject-specific axis (linear in the session index), per-channel gain and
  impedance-noise changes, source band-power and alpha-frequency drift;
- context: per subject a different BrainHero cap (rotation + electrode
  jitter, per-channel gains and noise levels) and a BrainHero-only
  common-mode artefact; large enough that per-(subject, context)
  re-centring helps a Riemann pipeline (``--check`` measures it). Pure
  per-channel gains (context and session) cancel in the per-run robust
  scaling below, as they would in the NeuralBench pipeline; what survives is
  the geometry, the noise levels and the common-mode artefact;
- EMG1/2: broadband 20 Hz..(Nyquist - 5) noise with power raised for WORD;
  the coupling is strong in sessions 0..2 and decays to ~0 by 3..5 (the
  drifting muscle confound trap). No leak into the EEG channels;
- EOG1/2: slow drift, blinks (vertical, EOG1; leaking into Fp1/Fp2/AF3/AF4)
  and saccades (horizontal, EOG2). The blink rate is doubled for CALC in
  every session (a stable, genuinely non-neural cue); saccades are more
  frequent in BrainHero (not class-coupled);
- as the NeuralBench EegExtractor delivers windows: per recording (= run)
  and per channel robust scaling (median / IQR), then clamp at +-20.

``--check`` (after the build): pooled Riemann tangent space + LDA
(``xsess_lib`` spec riemann:xd=1,fb=1) trained on every window with
split != 2 and scored on split == 2 with the subject x session x context
cell metric. Default variants: EEG only (the calibration target, 0.45-0.65;
chance 0.333) and EEG with per-(subject, session) vs per-(subject, session,
context) oracle re-centring; ``--check_variants`` adds the training-data
references (eeg|s, eeg|sc) and channel sets (all, eeg+eog, eeg+emg). One
fit takes 1.5-3.5 min at 43 ch on 4 loaded threads, so run a few variants
per call.

Calibration with the committed KNOBS (seed 0, mock_sealed_s, 2026-09-28,
cell BA on split == 2, 60 cells, EEG only): no alignment 0.472; re-centring
with training references per subject 0.481 vs per (subject, context) 0.537;
oracle per (subject, session) 0.507 vs per (subject, session, context)
0.569 (9/10 evaluation subjects up); EEG + EMG 0.472, EEG + EOG 0.492 (on
this split the full subjects' uncoupled sessions 3..5 are in training, so
EMG costs nothing). Replica split (every subject's sessions 0..2 -> full
subjects' 3..5), log band-power LDA (scratch proxy): EEG 0.676, EEG + EMG
0.628 (the drifting confound costs 4.8 points), EEG + EOG 0.690. The mock
numbers only check machinery; they say nothing about the recipe.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HOME = Path.home()
CACHE_ROOT = HOME / "neuralbench/xsess_cache"

# study -> (sfreq, trials per class per session)
STUDIES = {
    "mock_sealed_s": (120.0, 8),
    "mock_sealed_120": (120.0, 40),
    "mock_sealed_500": (500.0, 40),
}
DURATION = 4.0                          # s, as the proxies' 4-s windows
CLASSES = ["CALC", "MI", "WORD"]        # NeuralBench LabelEncoder = sorted names
N_SUBJ, N_SESS, N_RUNS, CALIB = 20, 6, 4, 3
FULL, EVAL = list(range(10)), list(range(10, 20))
CONTEXTS = ["graz", "brainhero"]

# a plausible 43-channel cap (standard_1005 names), then the non-EEG channels
EEG_NAMES = [
    "Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "P7", "P3", "P1", "Pz", "P2", "P4", "P8", "PO3", "POz", "PO4", "Oz",
]
CH_NAMES = EEG_NAMES + ["EMG1", "EMG2", "EOG1", "EOG2"]
CH_TYPES = ["eeg"] * len(EEG_NAMES) + ["emg"] * 2 + ["eog"] * 2
C_EEG = len(EEG_NAMES)

# latent sources: name, anchor electrode (standard_1005), spatial spread (rad),
# band-peak amplitudes relative to the pink floor at the peak (theta, alpha/mu,
# beta, low gamma)
SOURCES = [
    ("frontal_midline", "Fz", 0.40, (2.0, 0.5, 0.3, 0.1)),
    ("left_motor", "C3", 0.30, (0.3, 3.0, 1.0, 0.1)),
    ("right_motor", "C4", 0.30, (0.3, 3.0, 1.0, 0.1)),
    ("parietal_midline", "Pz", 0.40, (0.3, 4.0, 0.4, 0.1)),
    ("left_temporal", "FT7", 0.35, (0.3, 0.8, 0.8, 0.5)),
    ("occipital", "Oz", 0.40, (0.3, 6.0, 0.3, 0.1)),
    ("right_temporal", "FT8", 0.35, (0.3, 0.8, 0.8, 0.5)),
    ("left_frontal", "F3", 0.40, (0.5, 1.0, 0.5, 0.1)),
    ("right_frontal", "F4", 0.40, (0.5, 1.0, 0.5, 0.1)),
    ("left_parietal", "P3", 0.40, (0.3, 2.0, 0.5, 0.1)),
    ("right_parietal", "P4", 0.40, (0.3, 2.0, 0.5, 0.1)),
    ("vertex", "Cz", 0.45, (0.5, 1.5, 0.8, 0.1)),
]
BANDS = ["theta", "alpha", "beta", "gamma"]
BAND_WIDTH = [1.0, 1.0, 2.5, 5.0]       # Hz (Gaussian peak sd); centres below
# class effects: (class, source, band, relative band-power change at strength 1)
EFFECTS = [
    ("MI", "left_motor", "alpha", -0.5), ("MI", "left_motor", "beta", -0.4),
    ("MI", "right_motor", "alpha", -0.5), ("MI", "right_motor", "beta", -0.4),
    ("CALC", "frontal_midline", "theta", +0.8),
    ("CALC", "parietal_midline", "alpha", -0.4),
    ("WORD", "left_temporal", "beta", +0.6), ("WORD", "left_temporal", "gamma", +0.8),
]
# WORD <-> EMG power coupling per session index (strong in calibration
# sessions 0..2, ~0 in the test sessions 3..5)
EMG_COUPLING = [1.0, 0.9, 0.8, 0.1, 0.03, 0.0]

# generator knobs (all recorded in meta["mock"]), calibrated on mock_sealed_s
# (seed 0, 2026-09-28) against the EEG-only pooled Riemann (xd=1,fb=1) cell BA
# target 0.45-0.65: effect_scale 1.0 -> 0.368, 2.5 -> 0.508 with the first
# context knobs (cm 0.6, noise 0.4), whose context gain was small (oracle
# +2.8 points); ctx_cm_amp 1.5 / ctx_noise_sd 0.6 -> 0.472, context gain
# +6.2 (oracle) / +5.6 (training references). See the module docstring.
KNOBS = dict(
    effect_scale=2.5,        # global class-effect multiplier
    state_sd=0.3,            # window-to-window log band-power noise
    trial_sd=0.4,            # per-trial log jitter of the class effect
    subj_strength=(1.0, 0.3, 0.4, 1.6),   # N(mean, sd) clipped to [lo, hi]
    near_chance_strength=0.05,
    class_gain_sd=0.3,       # per-subject per-class log effect gain
    src_jitter=0.10,         # rad, subject jitter of source directions
    sensor_noise=0.4,        # white noise sd relative to the mean channel signal sd
    sess_rot=0.015,          # rad per session index along a subject axis
    sess_gain_sd=0.04,       # per session index (log per-channel gain)
    sess_noise_sd=0.08,      # per session index (log per-channel noise level)
    sess_band_sd=0.06,       # per session index (log source band power)
    sess_alpha_sd=0.15,      # Hz x sqrt(session index), alpha peak shift
    ctx_rot=0.06,            # rad sd per axis, BrainHero cap rotation
    ctx_jitter=0.03,         # rad, BrainHero per-electrode position jitter
    ctx_gain_sd=0.25,        # log per-channel gain, BrainHero vs Graz
    ctx_noise_sd=0.6,        # log per-channel noise level, BrainHero vs Graz
    ctx_cm_amp=1.5,          # BrainHero common-mode artefact, rel. signal sd
    emg_word_gain=2.0,       # WORD EMG power = 1 + gain * coupling[session] * jitter
    emg_tonic_sd=0.35,       # per-window log EMG power
    blink_rate=0.25,         # Hz (subject log-sd 0.3); x2 for CALC
    blink_calc_mult=2.0,
    blink_amp=6.0,           # rel. signal sd
    sacc_rate=0.3,           # Hz (subject log-sd 0.3); x1.5 in BrainHero
    sacc_brainhero_mult=1.5,
    eog_leak={"Fp1": 0.15, "Fp2": 0.15, "AF3": 0.07, "AF4": 0.07},
    robust_scale=True,       # per run, per channel median / IQR, like EegExtractor
    clamp=20.0,
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _geometry():
    """Unit directions of the cap electrodes and source anchors, from the
    standard_1005 montage re-centred on its least-squares sphere."""
    import mne
    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    P = np.array(list(pos.values()))
    # sphere fit: |p|^2 = 2 p.c + (r^2 - |c|^2)
    A = np.c_[2 * P, np.ones(len(P))]
    c = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)[0][:3]
    d = {n: _unit(np.asarray(p) - c) for n, p in pos.items()}
    return dict(elec=np.array([d[n] for n in EEG_NAMES]),
                anchor={a: d[a] for _, a, _, _ in SOURCES})


def _tangent(u, sd, rng):
    """Random tangent vector(s) at unit vector(s) u, per-axis sd."""
    g = rng.normal(0, sd, np.shape(u))
    return g - (g * u).sum(-1, keepdims=True) * u


def _rotate(elec, rotvec):
    from scipy.spatial.transform import Rotation
    return elec @ Rotation.from_rotvec(rotvec).as_matrix().T


def _topography(elec, src_dir, spread, tang, tang_w):
    """(C,) smooth scalp pattern of one source: Gaussian bump in angular
    distance, a tangential-dipole asymmetry and a broad conduction term."""
    d = np.arccos(np.clip(elec @ src_dir, -1, 1))
    bump = np.exp(-0.5 * (d / spread) ** 2)
    return bump * (1 + tang_w * (elec @ tang) / spread) + 0.15 * np.exp(-0.5 * (d / 1.2) ** 2)


# ---------------------------------------------------------------------------
# per-subject parameters (independent of the study's rate and size)
# ---------------------------------------------------------------------------
def _near_chance(seed):
    rng = np.random.default_rng([seed, 999])
    return sorted(int(v) for v in np.r_[rng.choice(FULL, 1, replace=False),
                                        rng.choice(EVAL, 2, replace=False)])


def _subject_params(seed, s, geo):
    k = KNOBS
    rng = np.random.default_rng([seed, s, 101])
    K = len(SOURCES)
    mu, sd, lo, hi = k["subj_strength"]
    p = dict(
        strength=(k["near_chance_strength"] if s in _near_chance(seed)
                  else float(np.clip(rng.normal(mu, sd), lo, hi))),
        class_gain=np.exp(rng.normal(0, k["class_gain_sd"], len(CLASSES))),
        alpha_hz=rng.uniform(9.0, 11.5),
        beta_exp=rng.uniform(1.0, 1.6),
    )
    anchors = np.array([geo["anchor"][a] for _, a, _, _ in SOURCES])
    p["src_dir"] = _unit(anchors + _tangent(anchors, k["src_jitter"], rng))
    p["spread"] = np.array([sp for _, _, sp, _ in SOURCES]) * np.exp(rng.normal(0, 0.15, K))
    p["tang"] = _unit(_tangent(p["src_dir"], 1.0, rng))
    p["tang_w"] = rng.uniform(0, 0.8, K)
    p["band_amp"] = (np.array([a for _, _, _, a in SOURCES])
                     * np.exp(rng.normal(0, 0.25, (K, len(BANDS)))))
    p["noise_ch"] = np.exp(rng.normal(0, 0.2, C_EEG))
    # context: the BrainHero cap
    p["ctx_rot"] = rng.normal(0, k["ctx_rot"], 3)
    p["ctx_jit"] = _tangent(geo["elec"], k["ctx_jitter"], rng)
    p["ctx_gain"] = np.exp(rng.normal(0, k["ctx_gain_sd"], C_EEG))
    p["ctx_noise"] = np.exp(rng.normal(0, k["ctx_noise_sd"], C_EEG))
    p["cm_dir"] = geo["elec"][rng.integers(C_EEG)]
    # session drift, growing with the session index
    axis = _unit(rng.normal(size=3))
    p["sess_rot"] = [v * k["sess_rot"] * axis + (rng.normal(0, 0.3 * k["sess_rot"], 3)
                                                  if v else 0.0) for v in range(N_SESS)]
    p["sess_gain"] = [np.exp(rng.normal(0, k["sess_gain_sd"] * v, C_EEG)) for v in range(N_SESS)]
    p["sess_noise"] = [np.exp(rng.normal(0, k["sess_noise_sd"] * v, C_EEG)) for v in range(N_SESS)]
    p["sess_band"] = [rng.normal(0, k["sess_band_sd"] * v, (K, len(BANDS))) for v in range(N_SESS)]
    p["sess_alpha"] = [rng.normal(0, k["sess_alpha_sd"] * np.sqrt(v)) for v in range(N_SESS)]
    # non-EEG
    p["emg_gain"] = float(np.exp(rng.normal(0, 0.3)))
    p["emg_amp"] = np.exp(rng.normal(0, 0.2, (N_SESS, len(CONTEXTS))))
    p["blink_rate"] = k["blink_rate"] * float(np.exp(rng.normal(0, 0.3)))
    p["sacc_rate"] = k["sacc_rate"] * float(np.exp(rng.normal(0, 0.3)))
    # signal scale: mean channel sd of the Graz session-0 mixing (fixed per
    # subject so sensor noise does not follow the mixing drift)
    A0 = _mixing(p, geo, 0, "graz")[0]
    p["sig"] = float(np.sqrt((A0 ** 2).sum(1).mean()))
    return p


def _mixing(p, geo, v, ctx):
    """(A (C_EEG, K), per-channel gain, per-channel noise multiplier) for one
    subject x session x context."""
    elec = geo["elec"]
    bh = ctx == "brainhero"
    if bh:
        elec = _unit(_rotate(elec, p["ctx_rot"]) + p["ctx_jit"])
    elec = _rotate(elec, p["sess_rot"][v])
    A = np.stack([_topography(elec, p["src_dir"][k], p["spread"][k], p["tang"][k],
                              p["tang_w"][k]) for k in range(len(SOURCES))], 1)
    gain = p["sess_gain"][v] * (p["ctx_gain"] if bh else 1.0)
    noise = p["noise_ch"] * p["sess_noise"][v] * (p["ctx_noise"] if bh else 1.0)
    cm = _topography(elec, p["cm_dir"], 0.9, np.zeros(3), 0.0) if bh else None
    return A, gain, noise, cm


# ---------------------------------------------------------------------------
# one run = one recording
# ---------------------------------------------------------------------------
def _cnormal(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def _shaped(rng, shape, amp, T):
    """Gaussian series with spectral amplitude ``amp`` (..., F), broadcast to
    ``shape + (F,)``; ~unit variance for a flat unit amplitude."""
    Z = _cnormal(rng, shape + (amp.shape[-1],)) * amp
    return np.fft.irfft(Z, n=T, axis=-1, norm="ortho")


def _gen_run(p, geo, v, ctx, y, sfreq, T, rng):
    """(n, 47, T) float32 windows of one run (labels y), robust-scaled."""
    k = KNOBS
    n, K, B = len(y), len(SOURCES), len(BANDS)
    f = np.fft.rfftfreq(T, 1 / sfreq)
    nyq = sfreq / 2
    hp = (f >= 0.5).astype(float)

    # ---- source spectra (n, K, F)
    pink = 1.0 / np.maximum(f, 1.0) ** p["beta_exp"]
    fa = p["alpha_hz"] + p["sess_alpha"][v]
    centres = [6.0, fa, 2 * fa, 35.0]
    peaks = np.stack([np.exp(-0.5 * ((f - c) / w) ** 2) / max(c, 1.0) ** p["beta_exp"]
                      for c, w in zip(centres, BAND_WIDTH)])                # (B, F)
    amp = p["band_amp"] * np.exp(p["sess_band"][v])                          # (K, B)
    norm = (pink[None] + amp @ peaks)[:, hp > 0].mean(1)                     # (K,)
    logm = rng.normal(0, k["state_sd"], (n, K, B))
    jit = np.exp(rng.normal(0, k["trial_sd"], n))
    src = [s[0] for s in SOURCES]
    for cls, sname, band, eff in EFFECTS:
        c = CLASSES.index(cls)
        m = y == c
        logm[m, src.index(sname), BANDS.index(band)] += (
            k["effect_scale"] * p["strength"] * p["class_gain"][c] * jit[m] * np.log1p(eff))
    P = pink[None, None] + np.einsum("nkb,kb,bf->nkf", np.exp(logm), amp, peaks)
    P /= norm[None, :, None]                                                  # (n, K, F)
    S = _shaped(rng, (n, K), np.sqrt(P) * hp, T)                              # (n, K, T)

    # ---- EEG
    A, gain, noise, cm = _mixing(p, geo, v, ctx)
    sig = p["sig"]
    eeg = np.matmul(A[None], S)
    eeg += (k["sensor_noise"] * sig * noise)[None, :, None] * rng.standard_normal((n, C_EEG, T))
    if cm is not None:
        cms = _shaped(rng, (n,), pink ** 0.5 * hp, T)
        cms /= cms.std() + 1e-12
        eeg += k["ctx_cm_amp"] * sig * cm[None, :, None] * cms[:, None, :]

    # ---- EMG: broadband 20 Hz..(Nyquist - 5), WORD power x (1 + g * coupling)
    band = ((f >= 20) & (f <= nyq - 5)).astype(float)
    E = _shaped(rng, (n, 2), band, T) / np.sqrt(2 * band.sum() / T)
    tonic = np.exp(rng.normal(0, k["emg_tonic_sd"], n))
    wj = np.exp(rng.normal(0, 0.4, n))
    word = (y == CLASSES.index("WORD")) * k["emg_word_gain"] * EMG_COUPLING[v] * p["emg_gain"] * wj
    emg_amp = p["emg_amp"][v, CONTEXTS.index(ctx)] * np.sqrt(tonic * (1 + word))
    emg = sig * emg_amp[:, None, None] * E

    # ---- EOG: drift + blinks (EOG1) / saccades (EOG2)
    t = np.arange(T) / sfreq
    drift = _shaped(rng, (n, 2), (f >= 0.25) / np.maximum(f, 0.25), T)
    drift /= drift.std() + 1e-12
    eog = drift.copy()
    calc = y == CLASSES.index("CALC")
    brate = p["blink_rate"] * np.where(calc, k["blink_calc_mult"], 1.0)
    srate = p["sacc_rate"] * (k["sacc_brainhero_mult"] if ctx == "brainhero" else 1.0)
    for i in range(n):
        nb = rng.poisson(brate[i] * DURATION)
        for tb, ab in zip(rng.uniform(-0.2, DURATION, nb),
                          k["blink_amp"] * np.exp(rng.normal(0, 0.2, nb))):
            pulse = ab * np.exp(-0.5 * ((t - tb) / 0.07) ** 2)
            eog[i, 0] += pulse
            eog[i, 1] += 0.2 * pulse
        ns = rng.poisson(srate * DURATION)
        for ts, a_s in zip(rng.uniform(0, DURATION, ns), rng.normal(0, 2.5, ns)):
            eog[i, 1] += a_s * 0.5 * (1 + np.tanh((t - ts) / 0.02))
    eog *= sig
    for name, c in k["eog_leak"].items():
        eeg[:, EEG_NAMES.index(name)] += c * eog[:, 0]
    eeg *= gain[None, :, None]

    X = np.concatenate([eeg, emg, eog], 1)
    if k["robust_scale"]:
        flat = X.transpose(1, 0, 2).reshape(X.shape[1], -1)
        q1, med, q3 = np.percentile(flat, [25, 50, 75], axis=1)
        X = (X - med[None, :, None]) / (q3 - q1 + 1e-12)[None, :, None]
    return np.clip(X, -k["clamp"], k["clamp"]).astype(np.float32)


def _run_layout(seed, s, v, r, n_per_run_class):
    """Labels (balanced, shuffled) and onsets (s) of one run."""
    rng = np.random.default_rng([seed, s, v, r, 1])
    y = rng.permutation(np.repeat(np.arange(len(CLASSES)), n_per_run_class))
    onset = 3.0 + 8.0 * np.arange(len(y)) + rng.uniform(0, 1.0, len(y))
    return y, np.round(onset, 3)


def _context(ctx_mode, v, r):
    if ctx_mode == "runs":
        return CONTEXTS[r % 2]
    return CONTEXTS[v % 2]


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def study_dir(study, ctx_mode="runs"):
    return CACHE_ROOT / (study if ctx_mode == "runs" else f"{study}_ctxsess")


def build(study, seed=0, ctx_mode="runs", force=False, max_subjects=None):
    """Write the cache (``max_subjects``: timing probe only, writes to a
    ``<name>_probe`` folder without meta.json)."""
    sfreq, n_per_class = STUDIES[study]
    assert n_per_class % N_RUNS == 0, n_per_class
    npr = n_per_class // N_RUNS
    T = int(round(DURATION * sfreq))
    out = study_dir(study, ctx_mode)
    name = out.name
    if max_subjects is not None:
        out = out.with_name(name + "_probe")
    if (out / "meta.json").exists() and not force:
        log(f"{name}: cache exists at {out}, skipping (--force to rebuild)")
        return out
    out.mkdir(parents=True, exist_ok=True)
    n_subj = N_SUBJ if max_subjects is None else max_subjects
    n = n_subj * N_SESS * N_RUNS * npr * len(CLASSES)
    C = len(CH_NAMES)
    log(f"{name}: building seed={seed} ctx_mode={ctx_mode} sfreq={sfreq} "
        f"X=({n}, {C}, {T}) = {n * C * T * 4 / 1e9:.2f} GB -> {out}")
    t0 = time.time()
    geo = _geometry()
    tmp = out / "X.tmp.npy"
    X = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, C, T))
    del X                       # header + sparse file; filled subject by subject
    cols = {c: [] for c in ("y", "subj", "session", "run", "onset", "context")}
    i = 0
    for s in range(n_subj):
        ts = time.time()
        p = _subject_params(seed, s, geo)
        # map the file per subject: only one subject's pages are ever mapped,
        # so RSS stays ~ one subject (~0.3 GB at 500 Hz) whatever the size
        X = np.load(tmp, mmap_mode="r+")
        for v in range(N_SESS):
            for r in range(N_RUNS):
                y, onset = _run_layout(seed, s, v, r, npr)
                ctx = _context(ctx_mode, v, r)
                rng = np.random.default_rng([seed, s, v, r, 2])
                X[i:i + len(y)] = _gen_run(p, geo, v, ctx, y, sfreq, T, rng)
                i += len(y)
                cols["y"].append(y)
                cols["subj"].append(np.full(len(y), s))
                cols["session"].append(np.full(len(y), v))
                cols["run"].append(np.full(len(y), r))
                cols["onset"].append(onset)
                cols["context"].append(np.full(len(y), ctx))
        X.flush()
        del X
        log(f"  subject {s + 1}/{n_subj} ({time.time() - ts:.1f} s, "
            f"total {time.time() - t0:.0f} s)")
    assert i == n, (i, n)
    a ={k: np.concatenate(v) for k, v in cols.items()}
    y = a["y"].astype(np.int64)
    subj, session, run = (a[k].astype(np.int64) for k in ("subj", "session", "run"))
    split = np.where(np.isin(subj, EVAL) & (session >= CALIB), 2, 0).astype(np.int64)
    arrays = dict(y=y, subj=subj, session=session, run=run, split=split,
                  onset=a["onset"].astype(np.float64),
                  code=np.array(CLASSES)[y], context=a["context"].astype(str))
    # every file is written to a temp name and renamed (atomic), so a reader
    # running during a --force rebuild never sees a half-written file
    for key, val in arrays.items():
        np.save(out / f"{key}.tmp.npy", val)
        os.replace(out / f"{key}.tmp.npy", out / f"{key}.npy")
    os.replace(tmp, out / "X.npy")
    if max_subjects is not None:
        log(f"{name}: probe of {n_subj} subjects written in {time.time() - t0:.0f} s "
            f"(no meta.json) -> {out}")
        return out

    subjects = [f"S{s + 1:02d}" for s in range(N_SUBJ)]
    near = _near_chance(seed)
    strengths = [round(_subject_params(seed, s, geo)["strength"], 3) for s in range(N_SUBJ)]
    meta = {
        "study": name, "modality": "eeg", "task": "mock_sealed", "overlay": None,
        "sfreq": sfreq, "ch_names": CH_NAMES, "shape": [n, C, T],
        "classes": {str(k): c for k, c in enumerate(CLASSES)},
        "class_counts": np.bincount(y, minlength=len(CLASSES)).tolist(),
        "subjects": subjects,
        "sessions": {sj: [str(v) for v in range(N_SESS)] for sj in subjects},
        "windows_per_subject_session": {
            f"{subjects[s]}|{v}": int(((subj == s) & (session == v)).sum())
            for s in range(N_SUBJ) for v in range(N_SESS)},
        "window": {"start": 0.0, "duration": DURATION, "stride": None},
        "default_split_counts": np.bincount(split, minlength=3).tolist(),
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ch_types": CH_TYPES, "context_column": "context",
        "full_subjects": FULL, "eval_subjects": EVAL, "calib_sessions": CALIB,
        "mock": {
            "seed": seed, "ctx_mode": ctx_mode, "generator": "analysis/mock_sealed.py",
            "trials_per_class_per_session": n_per_class, "runs_per_session": N_RUNS,
            "knobs": KNOBS,
            "sources": [{"name": nm, "anchor": an, "spread": sp,
                         "band_amp": dict(zip(BANDS, am))} for nm, an, sp, am in SOURCES],
            "class_effects": [{"class": c, "source": sn, "band": b, "rel_power_change": e}
                              for c, sn, b, e in EFFECTS],
            "subject_strength": strengths, "near_chance_subjects": near,
            "emg_word_coupling_per_session": EMG_COUPLING,
            "notes": [
                "Synthetic: accuracies say nothing about the recipe.",
                "EMG1/2: WORD power raised with coupling strong in sessions 0..2, "
                "~0 in 3..5 (drifting muscle confound); a channel ablation on the "
                "replica split should reject EMG.",
                "EOG1/2: blink rate x2 for CALC in every session (stable non-neural "
                "cue); blinks leak into Fp1/Fp2/AF3/AF4.",
                "Context: BrainHero = different cap per subject (rotation, jitter, "
                "gains, noise) + common-mode artefact; per-(subject, context) "
                "re-centring should beat per-subject re-centring.",
                "Session drift grows with the session index (cap rotation, gains, "
                "impedance noise, band power, alpha frequency).",
                "Per run (= recording) per channel robust scaling + clamp 20, as "
                "the NeuralBench EegExtractor.",
            ],
        },
    }
    (out / "meta.json.tmp").write_text(json.dumps(meta, indent=1))
    os.replace(out / "meta.json.tmp", out / "meta.json")
    log(f"{name}: built in {time.time() - t0:.0f} s -> {out}")
    return out


def data_line(out):
    """Log the [data] shape line of a built cache."""
    meta = json.loads((out / "meta.json").read_text())
    ld = {k: np.load(out / f"{k}.npy") for k in ("y", "subj", "session", "split", "context")}
    ctx, cnt = np.unique(ld["context"], return_counts=True)
    types = {t: meta["ch_types"].count(t) for t in ("eeg", "emg", "eog")}
    n_ss = [len(np.unique(ld["session"][ld["subj"] == s])) for s in np.unique(ld["subj"])]
    cells = len({(a, b, c) for a, b, c in zip(ld["subj"][ld["split"] == 2],
                                               ld["session"][ld["split"] == 2],
                                               ld["context"][ld["split"] == 2])})
    log(f"[data] study={meta['study']} subjects={len(np.unique(ld['subj']))} "
        f"sessions/subject={sorted(set(n_ss))} contexts={dict(zip(ctx.tolist(), cnt.tolist()))} "
        f"X={tuple(meta['shape'])} sfreq={meta['sfreq']} classes={meta['classes']} "
        f"class_counts={meta['class_counts']} ch_types={types} "
        f"split_counts={meta['default_split_counts']} test_cells={cells}")


# ---------------------------------------------------------------------------
# calibration check (EEG-only pooled Riemann; see module docstring)
# ---------------------------------------------------------------------------
CHECK_VARIANTS = ("eeg", "eeg|ss", "eeg|ssc")


def check(out, variants=CHECK_VARIANTS):
    """Variants "<chans>[|<align>]": chans = eeg | all | eeg+eog | eeg+emg;
    align = ss / ssc: oracle re-centring per subject x session (x context) on
    each group's own unlabelled windows; s / sc: reference per subject (x
    context) from labelled (split != 2) windows only, applied to all of the
    group's windows. One xd=1,fb=1 fit takes ~2.5 min at 43 ch on 4 loaded
    threads."""
    sys.path.insert(0, str(HOME / "codabench/analysis"))
    import xsess_lib as L
    meta = json.loads((out / "meta.json").read_text())
    X = np.load(out / "X.npy", mmap_mode="r")
    ld = {k: np.load(out / f"{k}.npy") for k in
          ("y", "subj", "session", "split", "context")}
    y, subj, sess, ctx = ld["y"], ld["subj"], ld["session"], ld["context"]
    tr, te = ld["split"] != 2, ld["split"] == 2
    types = np.array(meta["ch_types"])
    res = {}
    for var in variants:
        chans, _, align = var.partition("|")
        keep = np.isin(types, {"eeg": ["eeg"], "all": ["eeg", "emg", "eog"],
                               "eeg+eog": ["eeg", "eog"], "eeg+emg": ["eeg", "emg"]}[chans])
        t0 = time.time()
        Xc = np.ascontiguousarray(X[:, keep], dtype=np.float32)
        bh = (ctx == "brainhero") if align in ("ssc", "sc") else 0
        if align in ("ss", "ssc"):
            Xc, _ = L.align_groups(Xc, subj * 100 + sess * 10 + bh, "euclid")
        elif align in ("s", "sc"):
            g, covs = subj * 10 + bh, L.window_covs(Xc)
            for gv in np.unique(g):
                mg = g == gv
                Xc[mg] = L.apply_W(Xc[mg], L.inv_sqrtm(covs[mg & tr].mean(0)))
        elif align:
            raise ValueError(f"unknown align {align!r}")
        m = dict(meta, ch_names=[c for c, k in zip(meta["ch_names"], keep) if k])
        model = L.make_model("riemann:xd=1,fb=1", m).fit(Xc[tr], y[tr])
        yhat = model.predict_proba(Xc[te]).argmax(1)
        sc = L.score(y[te], yhat, subj[te], sess[te], ctx[te])
        res[var] = sc
        log(f"[check] {meta['study']} {var:8s} cell={sc['cell']:.3f} pooled={sc['pooled']:.3f} "
            f"n_cells={sc['n_cells']} per_subject="
            f"{ {k: round(v, 2) for k, v in sc['per_subject'].items()} } "
            f"({time.time() - t0:.0f} s)")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("study", choices=sorted(STUDIES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctx_mode", choices=["runs", "sessions"], default="runs")
    ap.add_argument("--force", action="store_true", help="rebuild an existing cache")
    ap.add_argument("--check", action="store_true",
                    help="after the build: pooled Riemann calibration check")
    ap.add_argument("--check_variants", nargs="+", default=list(CHECK_VARIANTS),
                    help="e.g. eeg 'eeg|ss' 'eeg|ssc' 'eeg|s' 'eeg|sc' all eeg+eog eeg+emg")
    ap.add_argument("--probe_subjects", type=int, default=None,
                    help="timing probe: build only the first N subjects into "
                         "<study>_probe (no meta.json; not a usable cache)")
    a = ap.parse_args()
    out = build(a.study, a.seed, a.ctx_mode, a.force, a.probe_subjects)
    if a.probe_subjects is None:
        data_line(out)
        if a.check:
            check(out, a.check_variants)
