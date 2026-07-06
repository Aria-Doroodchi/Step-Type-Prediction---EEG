# -*- coding: utf-8 -*-
"""Build the supervisor HTML report for the 3-state (standing/straight/diagonal)
SCREEN run. Mirrors the style of scripts/_report_build_html.py (self-contained,
figures embedded as base64), with 3-class content + the SEP/src ablation.

Figures: reuses the generic schematics (xgb tree/boosting, binning, nested CV,
funnel) from the step-type report, the state data figures from
outputs/state_module/figs/, and one new 3-state paradigm schematic generated here.
"""
import base64
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(r"C:/Users/Ali D/Documents/ML")
SRC_FIGS = ROOT / "outputs" / "reports" / "supervisor_2026-06-25" / "figures"
STATE_FIGS = ROOT / "outputs" / "state_module" / "figs"
DIR = ROOT / "outputs" / "reports" / "state_3class_2026-06-26"
FIGS = DIR / "figures"
OUT = DIR / "EEG_State_Report.html"

INK, BLUE, ACCENT, GREEN, WARN = "#1f2933", "#1b3a6b", "#1f77b4", "#2f7d4f", "#a85a00"
C_STAND, C_STR, C_DIAG = "#5b6b7b", "#1f77b4", "#d6582b"
ESTIM = "#e6a100"   # foot-sole e-stim ticks (yellow-gold, high-contrast on white)


# --------------------------------------------------------------------------
# New schematic: the 3-state paradigm + how the 2 s windows are extracted.
# Both recordings carry a *continuous* ~2 Hz foot-sole train (gold ticks);
# stepping locks a response-triggered window, standing tiles the record.
# --------------------------------------------------------------------------
def make_paradigm(path):
    fig, ax = plt.subplots(figsize=(10.2, 5.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    x0, x1 = 0.55, 9.65          # strip extent
    sh = 1.35                    # strip height

    def strip(y, color, title):
        ax.add_patch(FancyBboxPatch((x0, y), x1 - x0, sh,
                                    boxstyle="round,pad=0.02,rounding_size=0.10",
                                    fc="#f7f9fc", ec=color, lw=1.6, zorder=2))
        ax.text(x0 + 0.05, y + sh + 0.44, title, fontsize=10.5, fontweight="bold",
                color=color, zorder=5)

    def tick(x, y, c, h, lw=2.0, z=4):
        ax.plot([x, x], [y, y + h], color=c, lw=lw, solid_capstyle="round", zorder=z)

    def estim_train(ybase, y_h):
        # continuous foot-sole e-stim train (~2 Hz, 0.52 s ISI) across the strip
        for x in np.arange(x0 + 0.30, x1 - 0.05, 0.52):
            tick(x, ybase, ESTIM, y_h, lw=1.7, z=3)

    # ================= STEPPING (Stim.bdf) =================
    ys = 6.55
    strip(ys, C_STR, "STEPPING   ·   Pxx_Stim.bdf   (straight = 256 cue · diagonal = 512 cue)")
    # response-locked 2 s analysis window (shaded)
    win_l = 2.55
    ax.add_patch(plt.Rectangle((win_l, ys + 0.04), 4.15, sh - 0.08,
                               fc=C_STR, alpha=0.10, ec=C_STR, lw=1.0, zorder=2.5))
    estim_train(ys + 0.16, sh - 0.55)
    # S1 direction cue + response/gait-onset markers (labels ABOVE the strip)
    tick(1.35, ys + 0.08, BLUE, sh - 0.16, lw=3, z=6)
    tick(win_l, ys + 0.08, GREEN, sh - 0.16, lw=3, z=6)
    ax.text(1.35, ys + sh + 0.10, "S1 cue", fontsize=8, color=BLUE, ha="center", zorder=6)
    ax.text(win_l, ys + sh + 0.10, "response (t = 0)", fontsize=8, color=GREEN, ha="center", zorder=6)
    # analysis-window label BELOW the shaded region only
    ax.text(win_l + 2.07, ys - 0.30, "2 s analysis window  [−0.1, +2.0 s]",
            fontsize=8.5, color=C_STR, ha="center", zorder=6)

    # ================= STANDING (Standing.bdf) =================
    yst = 3.35
    strip(yst, C_STAND, "STANDING   ·   Pxx_Standing.bdf   (no step cues — continuous stance)")
    ax.add_patch(plt.Rectangle((3.15, yst + 0.04), 4.15, sh - 0.08,
                               fc=C_STAND, alpha=0.16, ec=C_STAND, lw=1.0, zorder=2.5))
    estim_train(yst + 0.16, sh - 0.55)
    ax.text(5.22, yst - 0.30, "random / tiled 2 s window (balanced to the stepping count)",
            fontsize=8.5, color=C_STAND, ha="center", zorder=6)

    # e-stim legend note (between the strips, clear of both)
    ax.text(x0 + 0.02, 5.62, "gold ticks = continuous foot-sole electrical stimulation  "
            "(~2 Hz, 0.52 s ISI; 273 ms trigger→pulse offset) throughout BOTH recordings",
            fontsize=7.8, color="#8a6100", ha="left", style="italic", zorder=6)

    # ================= SEP inset (bottom-right, clear of all labels) =================
    box_l, box_b, box_w, box_h = 5.55, 0.35, 4.10, 1.95
    # short vertical connector from a standing e-stim tick straight down to the box
    cx = 8.05
    ax.annotate("", xy=(cx, box_b + box_h), xytext=(cx, yst + 0.02),
                arrowprops=dict(arrowstyle="-|>", color=ESTIM, lw=1.4, ls=(0, (3, 2))), zorder=3)
    ax.add_patch(FancyBboxPatch((box_l, box_b), box_w, box_h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="#fdf7ea", ec=ESTIM, lw=1.4, zorder=4))
    t = np.linspace(0, 1, 200)
    # schematic vertex foot-SEP: small P50 positivity then a larger N90 trough
    sep = 0.35 * np.exp(-((t - 0.28) ** 2) / 0.004) - 0.85 * np.exp(-((t - 0.52) ** 2) / 0.006)
    ax.plot(box_l + 0.55 + t * 2.15, box_b + 1.02 + sep * 0.42, color=WARN, lw=1.6, zorder=5)
    ax.plot([box_l + 0.55, box_l + 0.55 + 2.15], [box_b + 1.02, box_b + 1.02],
            color="#c9b48f", lw=0.6, ls=":", zorder=4.5)
    ax.text(box_l + box_w / 2, box_b + box_h - 0.24, "evoked response per pulse (vertex foot-SEP)",
            fontsize=8, color=WARN, ha="center", fontweight="bold", zorder=6)
    ax.text(box_l + box_w / 2, box_b + 0.18, "vertex P50 / N90 complex, sub-µV",
            fontsize=7.5, color=INK, ha="center", zorder=6)

    ax.text(5.0, 9.55, "Three motor states — one model per participant", fontsize=13,
            fontweight="bold", color=BLUE, ha="center")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def prep_figures():
    FIGS.mkdir(parents=True, exist_ok=True)
    # reusable generic schematics from the step-type report
    for src, dst in [("fig_xgb_tree.png", "fig_xgb_tree.png"),
                     ("fig_xgb_boosting.png", "fig_xgb_boosting.png"),
                     ("fig_binning.png", "fig_binning.png"),
                     ("fig_nestedcv.png", "fig_nestedcv.png"),
                     ("fig_pipeline.png", "fig_pipeline.png")]:
        if (SRC_FIGS / src).exists():
            shutil.copy(SRC_FIGS / src, FIGS / dst)
    # state data figures
    for src, dst in [("condition_window_erp.png", "fig_condition_erp.png"),
                     ("sep_grandaverage.png", "fig_sep.png"),
                     ("ablation_xgb.png", "fig_ablation.png"),
                     ("confusion_screen_xgb.png", "fig_confusion.png"),
                     ("per_participant_screen_xgb.png", "fig_perparticipant.png")]:
        if (STATE_FIGS / src).exists():
            shutil.copy(STATE_FIGS / src, FIGS / dst)
    make_paradigm(FIGS / "fig_paradigm.png")


def b64(name):
    return "data:image/png;base64," + base64.b64encode((FIGS / name).read_bytes()).decode()


def fig(name, caption, num):
    return f"""
    <figure>
      <img src="{b64(name)}" alt="{caption}"/>
      <figcaption><span class="fignum">Figure {num}.</span> {caption}</figcaption>
    </figure>"""


CSS = """
:root{--ink:#1f2933;--muted:#5b6b7b;--line:#e1e6ec;--blue:#1b3a6b;--accent:#1f77b4;
  --soft:#f4f7fb;--warn:#a85a00;--green:#2f7d4f;}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);
  line-height:1.62;margin:0;background:#fff;font-size:16.5px;}
.wrap{max-width:860px;margin:0 auto;padding:0 28px 80px;}
h1,h2,h3{line-height:1.25;color:var(--blue);font-weight:700;}
h2{font-size:1.5rem;margin-top:2.6em;padding-bottom:.25em;border-bottom:2px solid var(--line);}
h3{font-size:1.18rem;margin-top:1.8em;color:#243b56;}
p{margin:.7em 0;}a{color:var(--accent);text-decoration:none;}strong{color:#16324f;}
figure{margin:1.7em 0;text-align:center;}
figure img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:8px;
  box-shadow:0 1px 4px rgba(20,40,70,.06);}
figcaption{font-size:.9rem;color:var(--muted);margin-top:.6em;text-align:left;padding:0 6px;}
.fignum{color:var(--blue);font-weight:600;}
.title{padding:90px 0 40px;text-align:center;border-bottom:1px solid var(--line);margin-bottom:10px;}
.title .eyebrow{letter-spacing:.16em;text-transform:uppercase;font-size:.8rem;color:var(--accent);font-weight:700;}
.title h1{font-size:2.35rem;margin:.35em 0 .2em;}
.title .sub{font-size:1.15rem;color:var(--muted);max-width:660px;margin:0 auto 1.6em;}
.title .meta{font-size:.96rem;color:#3d4d5c;margin-top:1.4em;}.title .meta div{margin:.15em 0;}
.placeholder{background:#fff6da;border-bottom:1px dashed #d9a400;padding:0 3px;color:#7a5b00;}
.callout{background:var(--soft);border-left:4px solid var(--accent);border-radius:0 8px 8px 0;
  padding:.9em 1.1em;margin:1.4em 0;}
.callout.key{border-color:var(--green);background:#f1f8f3;}
.callout.warn{border-color:var(--warn);background:#fdf6ee;}
.callout p{margin:.35em 0;}
.callout .lab{font-weight:700;color:var(--blue);font-size:.82rem;letter-spacing:.05em;
  text-transform:uppercase;display:block;margin-bottom:.2em;}
.callout.key .lab{color:var(--green);}.callout.warn .lab{color:var(--warn);}
.cards{display:flex;gap:14px;flex-wrap:wrap;margin:1.5em 0;}
.card{flex:1;min-width:150px;background:#fff;border:1px solid var(--line);border-radius:10px;
  padding:16px 14px;text-align:center;box-shadow:0 1px 3px rgba(20,40,70,.05);}
.card .big{font-size:2rem;font-weight:800;color:var(--blue);line-height:1;}
.card .lab{font-size:.82rem;color:var(--muted);margin-top:.45em;}
.toc{background:var(--soft);border:1px solid var(--line);border-radius:10px;padding:18px 26px;margin:24px 0;}
.toc h2{border:none;margin:.1em 0 .5em;font-size:1.1rem;padding:0;}
.toc ol{margin:.3em 0;padding-left:1.3em;}.toc li{margin:.28em 0;}
table{border-collapse:collapse;width:100%;margin:1.3em 0;font-size:.95rem;}
th,td{border:1px solid var(--line);padding:8px 11px;text-align:left;}
th{background:var(--soft);color:var(--blue);font-weight:700;}
tr:nth-child(even) td{background:#fafcfe;}
.gloss dt{font-weight:700;color:#16324f;margin-top:.7em;}.gloss dd{margin:0 0 .2em 0;color:#33424f;}
.feat-list li{margin:.5em 0;}
hr.soft{border:none;border-top:1px solid var(--line);margin:2.4em 0;}
.foot{color:var(--muted);font-size:.86rem;margin-top:3em;border-top:1px solid var(--line);padding-top:1em;}
@media print{.title{padding-top:40px;}h2{page-break-after:avoid;}figure{page-break-inside:avoid;}.card{box-shadow:none;}}
"""

def build_html():
  return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Reading Motor State from Brain Signals — 3-Class Report</title>
<style>{CSS}</style></head><body><div class="wrap">

  <header class="title">
    <div class="eyebrow">EEG &middot; Motor State &middot; Machine Learning</div>
    <h1>Reading Motor State from Brain Activity</h1>
    <p class="sub">Single-trial classification of three motor states — <strong>quiet stance</strong>,
      <strong>straight stepping</strong>, and <strong>diagonal stepping</strong> — from 64-channel
      scalp EEG, with a per-participant model and nested cross-validation. A companion to the
      straight-vs-diagonal step-type project.</p>
    <div class="meta">
      <div><strong>Prepared by:</strong> <span class="placeholder">[your name]</span></div>
      <div><strong>For:</strong> <span class="placeholder">[supervisor name]</span></div>
      <div><strong>Affiliation:</strong> <span class="placeholder">Sunnybrook Research Institute</span></div>
      <div><strong>Date:</strong> June 2026 &middot; full 32-participant cohort</div>
    </div>
  </header>

  <nav class="toc"><h2>What's in this report</h2><ol>
    <li><a href="#summary">The short version</a></li>
    <li><a href="#goal">What the project is trying to do</a></li>
    <li><a href="#data">The three states and where the data come from</a></li>
    <li><a href="#engine">The prediction engine: a 3-way XGBoost model</a></li>
    <li><a href="#features">What the model looks at: the features</a></li>
    <li><a href="#arch">How the model is built and tested</a></li>
    <li><a href="#results">Results: which states are separable, and what helps</a></li>
    <li><a href="#caveat">An honest caveat: the standing&ndash;stepping movement asymmetry</a></li>
    <li><a href="#next">Summary and next steps</a></li>
    <li><a href="#glossary">Glossary (modelling &amp; statistics)</a></li>
  </ol></nav>

  <h2 id="summary">1 &nbsp;The short version</h2>
  <p>This is the three-way sibling of the step-type project. Instead of asking <em>straight or
  diagonal?</em>, it poses a broader decoding question: <strong>from a single two-second epoch of
  64-channel EEG, can we recover the participant's motor state — quiet stance, straight stepping, or
  diagonal stepping?</strong> Each participant is modelled separately and scored only on their own
  held-out epochs.</p>

  <div class="cards">
    <div class="card"><div class="big">0.88</div><div class="lab">average score (macro-AUC), where 0.50&nbsp;=&nbsp;chance</div></div>
    <div class="card"><div class="big">73%</div><div class="lab">three-way accuracy (chance&nbsp;=&nbsp;33%)</div></div>
    <div class="card"><div class="big">32</div><div class="lab">participants (the full cohort), each with their own model</div></div>
  </div>

  <div class="callout key"><span class="lab">Headline</span>
    <p>Across all thirty-two participants, the model separates the three motor states with an average
    <strong>macro-AUC of 0.878</strong> (chance = 0.50) and <strong>73% accuracy</strong> (chance =
    33%), and it does so honestly — the gap between its optimistic internal score and its real
    held-out score is tiny (0.05). <strong>Standing is almost always identified correctly; telling a
    straight step from a diagonal one is the hard part.</strong> A controlled comparison (Section 7)
    shows that a planned new <em>foot-stimulation</em> feature adds essentially nothing on top of the
    basic scalp features, while the <em>brain-source</em> feature adds a small but real boost to the
    hard straight-vs-diagonal distinction.</p>
  </div>

  <h2 id="goal">2 &nbsp;What the project is trying to do</h2>
  <p>The step-type project showed that the pre-movement <strong>contingent negative variation
  (CNV)</strong> — the slow anticipatory potential of the warning&ndash;imperative foreperiod — carries
  a faint trace of <em>which way</em> a participant is about to step. This project widens the scope from
  two step-types to <strong>three motor states</strong>: quiet stance, straight stepping, and diagonal
  stepping, decoded from the ongoing sensorimotor EEG rather than only the preparatory window. It is a
  step toward decoding a richer state repertoire from a single montage.</p>

  <p>With <strong>three</strong> classes, a naive model would reach only <strong>one third
  (33%)</strong> accuracy; anything reliably above that indicates the EEG genuinely separates the
  states. We retain the project's core design: <strong>one model per participant</strong>, each
  evaluated only on epochs it never saw during training or tuning.</p>

  <h2 id="data">3 &nbsp;The three states and where the data come from</h2>
  <p>Each participant contributes two continuous 64-channel recordings (BioSemi, 1024 or 2048&nbsp;Hz)
  that between them supply the three states. Throughout, a single <strong>2&nbsp;s epoch</strong> of the
  64-channel montage is one exemplar for the classifier.</p>

  {fig("fig_paradigm.png", "The three-state paradigm and epoching. <strong>Stepping</strong> (top, Stim recording): a warning&ndash;imperative structure in which an S1 direction cue (event code 256&nbsp;=&nbsp;straight, 512&nbsp;=&nbsp;diagonal) precedes the response / gait onset (code 96); the analysis epoch is response-locked over [&minus;0.1, +2.0]&nbsp;s. <strong>Standing</strong> (bottom, Standing recording): no cue structure, so non-overlapping 2&nbsp;s windows are tiled from the continuous record and count-balanced to the stepping trials. A <em>continuous</em> foot-sole electrical-stimulation train (gold ticks; ~2&nbsp;Hz, 0.52&nbsp;s ISI) runs through <em>both</em> recordings; each pulse evokes a low-amplitude vertex somatosensory evoked potential (inset; P50/N90), with the stimulator trigger offset from the true cutaneous pulse by a calibrated 273&nbsp;ms.", 1)}

  <ul class="feat-list">
    <li><strong>Straight / diagonal stepping</strong> — from the <em>Stim</em> recording. The S1 cue
      signals the direction; the participant steps ~2&nbsp;s later. Epochs are locked to the paired
      response (code 96) over [&minus;0.1, +2.0]&nbsp;s, exactly as in the step-type project
      (~40 trials per direction).</li>
    <li><strong>Quiet stance</strong> — from the <em>Standing</em> recording. With no cue structure,
      non-overlapping 2&nbsp;s windows are tiled from the continuous record and down-sampled to the
      stepping count, giving a balanced ~40/40/40 three-class problem per participant.</li>
  </ul>

  <p>A methodological feature of this dataset is that a foot-sole electrical stimulator runs
  <em>continuously</em> — at ~2&nbsp;Hz (0.52&nbsp;s ISI) — across <em>both</em> the stepping and
  standing recordings. Because the stimulation train is present and matched in every state, it is not a
  between-state cue in itself; rather, each pulse evokes a small cortical <strong>somatosensory evoked
  potential (SEP)</strong> at the vertex, which we exploited as an additional per-epoch feature family
  (Section 5). The stimulator trigger is offset from the true cutaneous pulse by a fixed
  <strong>273&nbsp;ms</strong> (calibrated cohort-wide; 95% CI [271.6, 276.1]&nbsp;ms), applied per
  recording as <code>round(0.273&nbsp;&times;&nbsp;sfreq)</code> samples to accommodate the mixed
  1024/2048&nbsp;Hz sampling.</p>

  <h2 id="engine">4 &nbsp;The prediction engine: a 3-way XGBoost model</h2>
  <p>As in the step-type project, the workhorse is <strong>XGBoost</strong> — a model built from many
  small <strong>decision trees</strong> (yes/no flowcharts over the measured numbers) added together,
  each new tree correcting the previous ones' mistakes. The only change for this project is the output:
  instead of one probability (“diagonal?”), the model now produces <strong>three probabilities that add
  to 100%</strong> — one for each state — and predicts whichever is largest.</p>

  {fig("fig_xgb_tree.png", "A single decision tree — the building block. Each box asks a yes/no question about one measurement (e.g. “was the signal at the top of the head below this level at 1.2 s?”); following the answers leads to a verdict. The model invents its own thresholds from the continuous data.", 2)}

  {fig("fig_xgb_boosting.png", "Gradient boosting: hundreds of simple trees are added together, each correcting the running total left by the trees before it. For this project the combined scores are turned into three competing probabilities (standing / straight / diagonal) instead of one.", 3)}

  <p>Performance is summarised with <strong>macro-AUC</strong> — the natural three-class version of the
  AUC score used in the step-type report. It asks, for each state in turn, "how well does the model
  separate this state from the other two?", then averages. As before, <strong>0.50 is chance and 1.00 is
  perfect.</strong></p>

  <h2 id="features">5 &nbsp;What the model looks at: the features</h2>
  <p>Each 2&nbsp;s epoch is reduced to a large feature vector of interpretable single-trial measures.
  We reuse the four families from the step-type pipeline, each computed per channel and per
  <strong>62.5&nbsp;ms (1/16&nbsp;s) time bin</strong>:</p>

  <ul class="feat-list">
    <li><strong>Amplitude</strong> — the mean potential within each bin: the binned single-trial ERP.</li>
    <li><strong>Slopes</strong> — the within-bin temporal gradient (first derivative), indexing the
      <em>rate</em> of change rather than the absolute level.</li>
    <li><strong>Spectral power</strong> — single-trial band power from a Morlet-wavelet
      time&ndash;frequency decomposition, summarised in the canonical bands (&delta;, &theta;, &alpha;,
      &beta;, &gamma;).</li>
    <li><strong>Source estimates</strong> — a distributed inverse solution (<strong>eLORETA</strong>)
      projected onto the <em>fsaverage</em> template cortex and parcellated with the Destrieux atlas
      (<em>aparc.a2009s</em>, ~150 ROIs).</li>
  </ul>

  {fig("fig_binning.png", "Turning a continuous waveform into features. Each 2&nbsp;s epoch is divided into non-overlapping 62.5&nbsp;ms bins and summarised per bin; across channels, bins, bands and source ROIs this yields ~19,500 candidate features per epoch.", 4)}

  <div class="callout">
    <span class="lab">A note on referencing — CSD (surface Laplacian)</span>
    <p>The scalp-electrode features (amplitude, slopes, power) are computed not on the raw
    average-referenced potentials but on their <strong>current source density (CSD)</strong> transform —
    the surface Laplacian of the scalp field. CSD is a reference-free spatial high-pass filter that
    sharpens the topography of superficial radial generators and suppresses volume-conducted and
    far-field activity; it is expressed here in arbitrary units. The trade-off is that, as a spatial
    second derivative, CSD also amplifies high-spatial-frequency and broadband noise — which is why the
    grand-average display below (Figure 5) is drawn from the average-reference (µV) epochs rather than
    the CSD ones, even though the classifier's window features use CSD.</p>
  </div>

  <p><strong>The novel block — a per-epoch foot-SEP.</strong> Because the sole is stimulated throughout,
  we added a fifth feature family unique to this project, read from the <em>average-reference</em>
  (non-CSD) epochs — the vertex foot-SEP is a deep, largely tangential paracentral generator that the
  surface Laplacian would attenuate. For each analysis epoch we average the SEPs of its own in-epoch
  pulses (~4 per 2&nbsp;s window), baseline-correct (&minus;50&ndash;0&nbsp;ms), and read fixed a-priori
  vertex components — an early positivity (<strong>P50</strong>, 40&ndash;50&nbsp;ms window) and the
  dominant negativity (<strong>N90</strong>, 75&ndash;90&nbsp;ms window; grand-average trough
  ~65&ndash;75&nbsp;ms), plus peak-to-peak and RMS over 15&ndash;130&nbsp;ms, across the vertex montage
  [Cz, C1, C2, FCz, CPz], with the read window starting &ge;15&nbsp;ms to exclude the stimulus artifact.
  (<em>P50/N90</em> are latency-defined labels for the vertex foot-SEP peaks, not the canonical
  tibial-nerve P37/N45 nomenclature — plantar cutaneous stimulation evokes a later, sub-µV complex.) The
  block is <strong>leakage-safe by construction</strong>: every epoch's SEP is built only from that
  epoch's own pulses, never a per-condition average broadcast across trials.</p>

  {fig("fig_condition_erp.png", "Grand-average vertex window ERP per state (average reference, µV; zero-phase 6&nbsp;Hz low-pass for display), across all thirty-two participants. Standing (blue) stays near baseline, whereas both stepping states show a large early positivity (~0.2&nbsp;s, around response / gait onset) followed by a sustained negativity that is deepest for diagonal. The two stepping waveforms share a similar early positivity but diverge through the sustained negativity; at the single-trial level, though, quiet stance is trivially separable whereas straight vs diagonal remains subtle. The classifier's window features are the CSD transform of these signals; the display is in µV for legibility.", 5)}

  {fig("fig_sep.png", "Grand-average vertex foot-SEP per state (average reference, µV), time-locked to the true (offset-corrected) pulse. The evoked complex is low-amplitude (sub-µV): a weak P50 positivity and a dominant N90 trough. The three states are near-identical — consistent with the ablation (Section 7), where the SEP block adds nothing beyond the window features.", 6)}

  <h2 id="arch">6 &nbsp;How the model is built and tested</h2>
  <p>The testing machinery is identical to the step-type project, because honesty matters more than any
  single number. Two principles run through it: <strong>never let the model see the data it will be
  scored on</strong>, and <strong>trim the ~19,500 raw features down to the few dozen that matter</strong>,
  doing both <em>inside</em> each training fold so nothing leaks.</p>

  {fig("fig_nestedcv.png", "Nested cross-validation. An outer split sets aside a test block used only for the final score; all choices (which features to keep, which settings to use) are made on a separate inner split. Because the test block is never used to make a choice, the score is not inflated.", 7)}

  {fig("fig_pipeline.png", "The feature funnel, applied fresh inside every training fold: from ~19,500 candidates down to a few dozen, via a correlation filter, a quick statistical screen, stability selection (keep only features that repeatedly prove useful), and gain pruning. The three-class run uses a multiclass-safe version of each step.", 8)}

  <p>To keep the three states evenly matched, the standing windows are down-sampled to the number of
  stepping trials, and the cross-validation is <strong>stratified</strong> so every split contains all
  three states. Each per-person model is then scored on held-out epochs it never saw.</p>

  <h2 id="results">7 &nbsp;Results: which states are separable, and what helps</h2>
  <p>The headline numbers come from all thirty-two participants with the full honest test. To find out <em>which
  features actually matter</em>, we ran the same analysis four times, each time with a different feature
  set — an <strong>ablation</strong>. This is the most informative result in the report.</p>

  {fig("fig_ablation.png", "The ablation. Each bar is the cohort macro-AUC for a different feature set (chance = 0.50, dashed). The full set (“combined”) and the set without the foot-SEP (“window”) are identical, and dropping the eLORETA source features too (“electrode”) costs a small but real ~0.015 macro-AUC. The foot-SEP on its own (“sep”) lands only modestly above chance.", 9)}

  <table>
    <thead><tr><th>Feature set</th><th>What's included</th><th>Macro-AUC</th><th>3-way accuracy</th></tr></thead>
    <tbody>
      <tr><td><strong>Combined</strong></td><td>everything (scalp + brain-source + foot-SEP)</td><td><strong>0.878</strong></td><td>73%</td></tr>
      <tr><td><strong>Window</strong></td><td>scalp + brain-source (no foot-SEP)</td><td>0.877</td><td>73%</td></tr>
      <tr><td><strong>Electrode</strong></td><td>scalp features only (amplitude, slopes, power)</td><td>0.862</td><td>71%</td></tr>
      <tr><td><strong>SEP only</strong></td><td>just the foot-stimulation feature</td><td>0.584</td><td>40%</td></tr>
    </tbody>
  </table>

  <div class="callout key"><span class="lab">Two clean answers</span>
    <p><strong>Does the new foot-SEP feature add anything?</strong> No. "Combined" and "Window" score
    identically (0.878 vs 0.877), so once the ordinary scalp features are present, the foot-SEP is
    redundant — although on its own it does carry a weak signal (0.58). <strong>Do the expensive
    brain-source features earn their keep?</strong> Yes, modestly — removing them drops the score by
    0.015 (0.877 → 0.862), and more tellingly it costs ~4 points of accuracy on the hard
    straight-vs-diagonal distinction (63/61% &rarr; 59/57%). (In an early 8-person preview this gap
    looked negligible; across the full cohort the brain-source contribution is small but consistent.)</p>
  </div>

  <p>The <strong>confusion matrix</strong> shows <em>where</em> the accuracy comes from: rows are the
  true state, columns are the model's guess, and the diagonal counts correct calls.</p>

  {fig("fig_confusion.png", "Three-by-three confusion matrix across all thirty-two participants. Standing is identified almost perfectly (~96% of standing epochs correct). The errors concentrate in the bottom-right block: straight and diagonal steps are most often confused <em>with each other</em> (~61–63% correct each), not with standing.", 10)}

  <p>So the three states are <strong>not</strong> equally easy. Per-class accuracy is about
  <strong>96% for standing, 63% for straight, and 61% for diagonal</strong>. Standing is trivially
  separable from movement; the genuinely hard sub-problem — telling a straight step from a diagonal one —
  is exactly the original step-type question, and it sits modestly above its own two-way chance line.</p>

  {fig("fig_perparticipant.png", "Each participant's own model. Top: three-class macro-AUC (chance 0.50, dashed). Bottom: overall accuracy (chance 0.333, dashed). Every participant is clearly above chance on both measures, with the usual person-to-person variation.", 11)}

  <div class="callout key"><span class="lab">The honesty check</span>
    <p>An overfit model scores far higher on its own tuning data than on unseen data. Here the
    inner-vs-outer gap is only <strong>0.05</strong>, and every participant clears chance on the truly
    held-out epochs — so the 0.878 figure is a fair, not inflated, estimate for the full
    32-participant cohort.</p>
  </div>

  <h2 id="caveat">8 &nbsp;An honest caveat: the standing&ndash;stepping movement asymmetry</h2>
  <div class="callout warn"><span class="lab">Read this before over-interpreting "standing"</span>
    <p>Near-perfect detection of standing should not be read as near-perfect decoding of a
    <em>cortical</em> state. Because the foot-stimulation train is continuous and matched across all
    three states (Section 3), it is <em>not</em> a between-state confound here — and we additionally
    blank the &plusmn;12&nbsp;ms stimulus-artifact intervals before extracting the window features. What
    still separates standing from stepping is the obvious factor: <strong>gross movement</strong>.
    Stepping entails whole-body motion, and with it postural EMG, motion and cable artifact, and genuine
    sensorimotor cortical engagement; quiet stance has none of these. Some of the standing&ndash;stepping
    separability is therefore <strong>non-neural</strong> (movement-related artifact) rather than a pure
    motor-state readout — an asymmetry no reference scheme fully removes.</p>
    <p>This is precisely why the <strong>straight-vs-diagonal</strong> contrast is the scientifically
    clean one: both are stepping, with matched movement <em>and</em> matched stimulation, so any
    separation there reflects genuine direction-related sensorimotor activity — the original step-type
    signal, now embedded in the three-class setting. Partitioning the standing result into
    movement-related artifact versus true motor-state signal (e.g. via EMG / accelerometer regressors or
    artifact-matched controls) is the natural next check.</p>
  </div>

  <h2 id="next">9 &nbsp;Summary and next steps</h2>
  <p>In summary: <strong>a single 2&nbsp;s EEG epoch separates quiet stance, straight stepping and
  diagonal stepping well above chance</strong> (macro-AUC 0.878, accuracy 73% vs 33% chance), under an
  honest nested cross-validation estimate. Standing is near-ceiling — subject to the movement-asymmetry
  caveat above — while the meaningful, artifact-matched problem is straight vs diagonal. The ablation was
  decisive: the per-epoch foot-SEP adds nothing beyond the window features, whereas the eLORETA source
  estimates give a small but consistent lift (+0.015 macro-AUC) to the hard straight-vs-diagonal
  distinction.</p>

  <p><strong>Where this goes next:</strong></p>
  <ul>
    <li><strong>The full 32-participant cohort is complete.</strong> Ablation-guided production feature
      set: <strong>keep the eLORETA source block</strong> (a small, consistent lift to
      straight-vs-diagonal) and <strong>drop the foot-SEP block</strong> (redundant against the window
      features).</li>
    <li><strong>Partition the standing result</strong> into movement-related artifact versus genuine
      motor-state signal — e.g. with EMG / accelerometer regressors or artifact-matched controls — since
      the continuous, matched stimulation already rules out a stimulation-rhythm explanation.</li>
    <li><strong>Focus on straight-vs-diagonal</strong>, the artifact-matched core problem, and relate it
      back to the CNV step-type findings.</li>
  </ul>

  <hr class="soft"/>

  <h2 id="glossary">Glossary (modelling &amp; statistics)</h2>
  <dl class="gloss">
    <dt>Feature</dt><dd>A single number summarising one aspect of an epoch (e.g. band power at one channel in one time bin).</dd>
    <dt>Macro-AUC</dt><dd>Macro-averaged one-vs-rest area under the ROC curve: for each state, how well it separates from the other two, then averaged. 0.50 = chance, 1.00 = perfect.</dd>
    <dt>Accuracy</dt><dd>Fraction of epochs assigned the correct state. Chance is 1/3 (33%) for three balanced classes.</dd>
    <dt>Ablation</dt><dd>Re-running the analysis with feature blocks removed, to isolate each block's contribution.</dd>
    <dt>Confusion matrix</dt><dd>A table of true versus predicted state; the diagonal counts correct calls, the off-diagonal the errors.</dd>
    <dt>Overfitting</dt><dd>When a model fits its training data rather than a generalisable rule — high internal scores, poor held-out scores.</dd>
    <dt>Nested cross-validation</dt><dd>An outer split scores the finished model; a separate inner split makes all tuning and feature-selection choices, so the reported score is not inflated.</dd>
    <dt>XGBoost</dt><dd>The classifier used here: an additive ensemble of gradient-boosted decision trees.</dd>
  </dl>

  <p class="foot">The full 32-participant cohort run of the 3-class motor-state classifier, the
  three-way sibling of the EEG step-type project. Per-participant nested cross-validation; cohort
  macro-AUC 0.878, 3-way accuracy 73% (chance 33%), inner-vs-outer gap 0.05. Figures 2&ndash;4,
  7&ndash;8 are explanatory schematics shared with the step-type report; Figures 1, 5&ndash;6, 9&ndash;11
  are generated from this project's own data and results. Bracketed yellow fields on the title page are
  placeholders to be filled in. Cohort complete (32/32).</p>

</div></body></html>"""


def main():
    prep_figures()
    DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_html(), encoding="utf-8")
    print(f"HTML written: {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
    print(f"figures: {sorted(p.name for p in FIGS.glob('*.png'))}")


if __name__ == "__main__":
    main()
