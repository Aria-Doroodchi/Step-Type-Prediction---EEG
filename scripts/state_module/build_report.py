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


# --------------------------------------------------------------------------
# New schematic: the 3-state paradigm + how the 2 s windows are extracted
# --------------------------------------------------------------------------
def make_paradigm(path):
    fig, ax = plt.subplots(figsize=(9.2, 4.7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def strip(y, color, title):
        ax.add_patch(FancyBboxPatch((0.5, y), 9.0, 1.5, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="#f4f7fb", ec=color, lw=1.6))
        ax.text(0.7, y + 1.75, title, fontsize=11, fontweight="bold", color=color)

    def tick(x, y, c, h=0.55, lw=2.2):
        ax.plot([x, x], [y, y + h], color=c, lw=lw, solid_capstyle="round")

    # --- Stepping (Stim.bdf) ---
    strip(7.0, C_STR, "STEPPING  —  Pxx_Stim.bdf  (straight = 256 cue · diagonal = 512 cue)")
    ax.text(0.95, 7.75, "cue", fontsize=8.5, color=INK, ha="center")
    tick(0.95, 7.1, BLUE, 0.9, 3)
    ax.text(2.0, 7.75, "response\n(t = 0)", fontsize=8, color=INK, ha="center")
    tick(2.0, 7.1, GREEN, 0.9, 3)
    ax.add_patch(plt.Rectangle((1.85, 7.05), 4.0, 1.4, fc=C_STR, alpha=0.12, ec=C_STR, lw=1))
    ax.text(3.85, 7.0 - 0.28, "2 s analysis window  [-0.1, +2.0 s]", fontsize=8.5, color=C_STR, ha="center")
    for x in (3.0, 3.5, 4.0, 4.5):      # 4 foot-sole e-stims during the step
        tick(x, 7.15, WARN, 0.5, 1.8)
    ax.text(3.75, 8.05, "4 e-stims", fontsize=8, color=WARN, ha="center")

    # --- Standing (Standing.bdf) ---
    strip(3.4, C_STAND, "STANDING  —  Pxx_Standing.bdf  (no cues — continuous foot-sole e-stims)")
    for x in np.arange(1.0, 9.3, 0.52):     # continuous e-stims ~0.52 s ISI
        tick(x, 3.55, WARN, 0.5, 1.6)
    ax.add_patch(plt.Rectangle((3.0, 3.45), 4.0, 1.4, fc=C_STAND, alpha=0.16, ec=C_STAND, lw=1))
    ax.text(5.0, 3.4 - 0.28, "random 2 s window (balanced to the stepping count)", fontsize=8.5,
            color=C_STAND, ha="center")

    # --- SEP inset ---
    ax.annotate("", xy=(7.7, 1.6), xytext=(4.5, 3.5),
                arrowprops=dict(arrowstyle="-|>", color=WARN, lw=1.3, ls=(0, (3, 2))))
    ax.add_patch(FancyBboxPatch((6.7, 0.2), 2.9, 1.7, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#fdf6ee", ec=WARN, lw=1.4))
    t = np.linspace(0, 1, 100)
    sep = -0.5 * np.exp(-((t - 0.45) ** 2) / 0.004) + 0.7 * np.exp(-((t - 0.62) ** 2) / 0.006)
    ax.plot(6.85 + t * 2.6, 1.0 + sep * 0.45, color=WARN, lw=1.5)
    ax.text(8.15, 1.72, "foot-SEP per e-stim", fontsize=8, color=WARN, ha="center", fontweight="bold")
    ax.text(8.15, 0.33, "P50 / N90 at the vertex", fontsize=7.5, color=INK, ha="center")

    ax.text(5.0, 9.4, "Three motor states, one model per participant", fontsize=12.5,
            fontweight="bold", color=BLUE, ha="center")
    fig.savefig(path, dpi=130, bbox_inches="tight")
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
    <p class="sub">Telling apart three states — <strong>standing</strong>, <strong>stepping
      straight</strong>, and <strong>stepping diagonally</strong> — from EEG, with one model per
      person. A companion to the straight-vs-diagonal step-type project.</p>
    <div class="meta">
      <div><strong>Prepared by:</strong> <span class="placeholder">[your name]</span></div>
      <div><strong>For:</strong> <span class="placeholder">[supervisor name]</span></div>
      <div><strong>Affiliation:</strong> <span class="placeholder">Sunnybrook Research Institute</span></div>
      <div><strong>Date:</strong> June 2026 &middot; 20-participant run</div>
    </div>
  </header>

  <nav class="toc"><h2>What's in this report</h2><ol>
    <li><a href="#summary">The short version</a></li>
    <li><a href="#goal">What the project is trying to do</a></li>
    <li><a href="#data">The three states and where the data come from</a></li>
    <li><a href="#engine">The prediction engine: a 3-way XGBoost model</a></li>
    <li><a href="#features">What the model looks at: the features (and the new foot-SEP)</a></li>
    <li><a href="#arch">How the model is built and tested</a></li>
    <li><a href="#results">Results: which states are separable, and what helps</a></li>
    <li><a href="#caveat">An honest caveat: the stimulation-rhythm confound</a></li>
    <li><a href="#next">Summary and next steps</a></li>
    <li><a href="#glossary">Plain-language glossary</a></li>
  </ol></nav>

  <h2 id="summary">1 &nbsp;The short version</h2>
  <p>This is the three-way sibling of the step-type project. Instead of asking <em>straight or
  diagonal?</em>, it asks a broader question: <strong>from a two-second window of EEG, can we tell
  whether a person is standing still, stepping straight, or stepping diagonally?</strong> Each person
  gets their own model, tested only on their own held-out data.</p>

  <div class="cards">
    <div class="card"><div class="big">0.89</div><div class="lab">average score (macro-AUC), where 0.50&nbsp;=&nbsp;chance</div></div>
    <div class="card"><div class="big">75%</div><div class="lab">three-way accuracy (chance&nbsp;=&nbsp;33%)</div></div>
    <div class="card"><div class="big">20</div><div class="lab">participants, each with their own model</div></div>
  </div>

  <div class="callout key"><span class="lab">Headline</span>
    <p>Across twenty participants, the model separates the three motor states with an average
    <strong>macro-AUC of 0.886</strong> (chance = 0.50) and <strong>75% accuracy</strong> (chance =
    33%), and it does so honestly — the gap between its optimistic internal score and its real
    held-out score is tiny (0.04). <strong>Standing is almost always identified correctly; telling a
    straight step from a diagonal one is the hard part.</strong> A controlled comparison (Section 7)
    shows that a planned new <em>foot-stimulation</em> feature adds essentially nothing on top of the
    basic scalp features, while the <em>brain-source</em> feature adds a small but real boost to the
    hard straight-vs-diagonal distinction.</p>
  </div>

  <h2 id="goal">2 &nbsp;What the project is trying to do</h2>
  <p>The step-type project showed that the brain's <strong>preparation signal</strong> carries a faint
  trace of <em>which way</em> a person is about to step. This project widens the lens from two
  step-types to <strong>three motor states</strong>: standing still, stepping straight, and stepping
  diagonally. The motivation is the same — to read movement-related information directly from brain
  activity — but the three-way version is a stepping stone toward decoding a richer set of states.</p>

  <p>Because there are now <strong>three</strong> possible answers, a model that guessed randomly would
  be right only <strong>one third (33%)</strong> of the time. Anything reliably above that means the
  EEG genuinely distinguishes the states. We keep the project's core principle: <strong>one model per
  person</strong>, each judged only on data it has never seen.</p>

  <h2 id="data">3 &nbsp;The three states and where the data come from</h2>
  <p>The data come from two separate EEG recordings per participant, which between them supply the three
  states. Throughout, a single <strong>two-second slice</strong> of the 64-sensor recording is called an
  <strong>epoch</strong> — one example the model learns from.</p>

  {fig("fig_paradigm.png", "How the three states are defined. <strong>Stepping</strong> (top, from the Stim recording): after a cue tells the person which way to step, we lock a 2-second window to their response; “straight” and “diagonal” come from two different cues. <strong>Standing</strong> (bottom, from the Standing recording): there are no cues, so we cut random 2-second windows from the continuous recording. In <em>both</em> recordings the sole of the foot is gently electrically stimulated every half-second — the orange ticks — which evokes a tiny brain response (inset) that we turn into an optional extra feature.", 1)}

  <ul class="feat-list">
    <li><strong>Stepping straight / diagonal</strong> — from the <em>Stim</em> recording. A cue tells
      the person the direction; about two seconds later they step. We take the 2-second window locked to
      their response, exactly as in the step-type project (~40 of each per person).</li>
    <li><strong>Standing</strong> — from the <em>Standing</em> recording. Here the person simply stands;
      there are no cues, so we cut non-overlapping 2-second windows from the continuous recording and
      balance their number to the stepping trials, so the three states are evenly represented.</li>
  </ul>

  <p>One quirk of this study is that throughout <em>both</em> recordings, the sole of the foot receives a
  gentle electrical pulse roughly twice a second. Each pulse evokes a small, brief response in the
  brain's sensory areas — a <strong>somatosensory evoked potential (SEP)</strong> — which we explored as
  an extra source of information (Section 5).</p>

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
  <p>Each 2-second epoch is summarised into a long list of <strong>features</strong> — interpretable
  numbers describing the brain signal. We reuse the four families from the step-type project, computed
  by chopping the window into short <strong>time bins</strong> (each ~1/16<sup>th</sup> of a second):</p>

  <ul class="feat-list">
    <li><strong>Amplitude</strong> — how high or low the signal sits at each sensor and time-slice.</li>
    <li><strong>Slopes</strong> — whether the signal is rising or falling, and how steeply.</li>
    <li><strong>Power (frequency bands)</strong> — the strength of the brain's rhythms (Delta…Gamma).</li>
    <li><strong>Source localisation</strong> — an estimate (via "eLORETA") of <em>where</em> in the
      brain the activity arises, across ~150 regions.</li>
  </ul>

  {fig("fig_binning.png", "Turning a continuous wave into numbers. The window is chopped into short equal time-slices and the signal in each is summarised — every bar becomes one feature. Repeated over every sensor, band and brain region, this yields roughly 19,500 candidate features per epoch.", 4)}

  <p><strong>The new ingredient — a foot-stimulation feature (“SEP”).</strong> Because the foot is being
  stimulated throughout, we added a fifth, novel feature family unique to this project: for each epoch we
  average the tiny brain responses to the ~4 foot-pulses that fall inside it, and measure the size and
  timing of the two characteristic peaks (around 50 and 90 milliseconds after the pulse) over the
  vertex sensors. Critically, each epoch's SEP is built <em>only from its own pulses</em> — never shared
  across trials — so it cannot secretly leak the answer.</p>

  {fig("fig_condition_erp.png", "The average brain signal over the 2-second window at the top of the head, for each state, across the twenty participants. Standing (grey) sits apart from the two stepping states; straight (blue) and diagonal (orange) overlap heavily — a first visual hint that standing is easy to spot but the two step types are hard to tell apart.", 5)}

  {fig("fig_sep.png", "The foot-stimulation response (SEP) at the vertex, averaged across participants, for each state. The small deflections after the pulse are the brain's sensory response. The traces differ only modestly between states — consistent with the result (Section 7) that this feature adds little once the ordinary scalp features are present.", 6)}

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
  <p>The headline numbers come from twenty participants with the full honest test. To find out <em>which
  features actually matter</em>, we ran the same analysis four times, each time with a different feature
  set — an <strong>ablation</strong>. This is the most informative result in the report.</p>

  {fig("fig_ablation.png", "The ablation. Each bar is the cohort macro-AUC for a different feature set (chance = 0.50, dashed). The full set (“combined”) and the set without the foot-SEP (“window”) are identical, and dropping the expensive brain-source features too (“electrode”) costs almost nothing. The foot-SEP on its own (“sep”) lands only modestly above chance.", 9)}

  <table>
    <thead><tr><th>Feature set</th><th>What's included</th><th>Macro-AUC</th><th>3-way accuracy</th></tr></thead>
    <tbody>
      <tr><td><strong>Combined</strong></td><td>everything (scalp + brain-source + foot-SEP)</td><td><strong>0.886</strong></td><td>75%</td></tr>
      <tr><td><strong>Window</strong></td><td>scalp + brain-source (no foot-SEP)</td><td>0.885</td><td>74%</td></tr>
      <tr><td><strong>Electrode</strong></td><td>scalp features only (amplitude, slopes, power)</td><td>0.862</td><td>71%</td></tr>
      <tr><td><strong>SEP only</strong></td><td>just the foot-stimulation feature</td><td>0.573</td><td>40%</td></tr>
    </tbody>
  </table>

  <div class="callout key"><span class="lab">Two clean answers</span>
    <p><strong>Does the new foot-SEP feature add anything?</strong> No. "Combined" and "Window" score
    identically (0.886 vs 0.885), so once the ordinary scalp features are present, the foot-SEP is
    redundant — although on its own it does carry a weak signal (0.57). <strong>Do the expensive
    brain-source features earn their keep?</strong> Yes, modestly — removing them drops the score by
    0.023 (0.885 → 0.862), and more tellingly it costs ~6&ndash;7 points of accuracy on the hard
    straight-vs-diagonal distinction (66/63% &rarr; 59/58%). (In an earlier 8-person preview this gap
    looked negligible; with more people the brain-source contribution became clear.)</p>
  </div>

  <p>The <strong>confusion matrix</strong> shows <em>where</em> the accuracy comes from: rows are the
  true state, columns are the model's guess, and the diagonal counts correct calls.</p>

  {fig("fig_confusion.png", "Three-by-three confusion matrix across the twenty participants. Standing is identified almost perfectly (~95% of standing epochs correct). The errors concentrate in the bottom-right block: straight and diagonal steps are most often confused <em>with each other</em> (~64–66% correct each), not with standing.", 10)}

  <p>So the three states are <strong>not</strong> equally easy. Per-class accuracy is about
  <strong>95% for standing, 66% for straight, and 64% for diagonal</strong>. Standing is trivially
  separable from movement; the genuinely hard sub-problem — telling a straight step from a diagonal one —
  is exactly the original step-type question, and it sits modestly above its own two-way chance line.</p>

  {fig("fig_perparticipant.png", "Each participant's own model. Top: three-class macro-AUC (chance 0.50, dashed). Bottom: overall accuracy (chance 0.333, dashed). Every participant is clearly above chance on both measures, with the usual person-to-person variation.", 11)}

  <div class="callout key"><span class="lab">The honesty check</span>
    <p>A model that is overfitting scores far higher on its own tuning data than on unseen data. Here the
    gap between the two is only <strong>0.04</strong>, and every participant clears chance on the truly
    held-out epochs. The 0.87 figure is a real, not inflated, estimate for this eight-person preview.</p>
  </div>

  <h2 id="caveat">8 &nbsp;An honest caveat: the stimulation-rhythm confound</h2>
  <div class="callout warn"><span class="lab">Read this before over-interpreting "standing"</span>
    <p>Standing being identified almost perfectly is partly <em>too</em> easy. During standing the foot
    is stimulated in a steady half-second rhythm; during stepping the pulses come in a short cluster and
    then stop. That difference in <strong>stimulation rhythm</strong> — not necessarily brain state — can
    by itself separate standing from stepping. We actively blank the electrical-pulse artefacts before
    computing features, but the blanking pattern itself still differs between the states, so some of the
    "standing" performance may reflect the experiment's structure rather than the brain.</p>
    <p>This is why the <strong>straight-vs-diagonal</strong> comparison is the scientifically clean one:
    both are stepping, with the identical pulse structure, so any separation there is genuine
    movement-related brain signal. A planned "stimulation-artefact-only" control will quantify exactly
    how much of the standing result is confound versus real motor state.</p>
  </div>

  <h2 id="next">9 &nbsp;Summary and next steps</h2>
  <p>In plain terms: <strong>a 2-second EEG window distinguishes standing, straight stepping, and
  diagonal stepping well above chance</strong> (macro-AUC 0.89, accuracy 75% versus 33%), with an honest
  test. Standing is easy (with the confound caveat above); separating the two step types is the hard,
  meaningful part. The controlled comparison was decisive: the new foot-SEP feature adds nothing beyond
  the basic scalp measurements, whereas the brain-source features give a small but real boost to the
  hard straight-vs-diagonal distinction.</p>

  <p><strong>Where this goes next:</strong></p>
  <ul>
    <li><strong>Finish the full group.</strong> Twenty of the thirty-two participants are done; the
      remaining twelve are the next step. The ablation says to <strong>keep the brain-source features</strong>
      (they help straight-vs-diagonal) but <strong>drop the foot-SEP block</strong> (redundant), trimming
      the feature set without losing accuracy.</li>
    <li><strong>Run the stimulation-artefact control</strong> to pin down how much of the standing result
      is the rhythm confound versus genuine motor state.</li>
    <li><strong>Focus on straight-vs-diagonal</strong>, the confound-free core problem, and connect it back
      to the step-type project's findings.</li>
  </ul>

  <hr class="soft"/>

  <h2 id="glossary">Plain-language glossary</h2>
  <dl class="gloss">
    <dt>EEG</dt><dd>Recording the brain's electrical activity with sensors on the scalp.</dd>
    <dt>Epoch / trial</dt><dd>One ~2-second recording across all sensors — one example for the model.</dd>
    <dt>Motor state</dt><dd>Here, one of three: standing, stepping straight, or stepping diagonally.</dd>
    <dt>Feature</dt><dd>A single number summarising one aspect of an epoch.</dd>
    <dt>SEP (somatosensory evoked potential)</dt><dd>The brain's small response to a touch/stimulation of the body — here, the electrical pulse to the sole of the foot.</dd>
    <dt>Macro-AUC</dt><dd>The three-class performance score: for each state, how well it is separated from the other two, averaged. 0.50 = chance, 1.00 = perfect.</dd>
    <dt>Accuracy</dt><dd>The fraction of epochs whose state the model gets exactly right. Chance here is 1/3 (33%).</dd>
    <dt>Ablation</dt><dd>Re-running the analysis with parts of the feature set removed, to see what each part contributes.</dd>
    <dt>Confusion matrix</dt><dd>A table of true state versus predicted state; the diagonal counts correct calls.</dd>
    <dt>Overfitting</dt><dd>When a model memorises its training data instead of learning a general rule.</dd>
    <dt>Cross-validation</dt><dd>Repeatedly holding out part of the data as an unseen test to estimate real-world performance fairly.</dd>
    <dt>XGBoost</dt><dd>The model used here: hundreds of small decision trees added together, each correcting the previous ones' mistakes.</dd>
  </dl>

  <p class="foot">A 20-participant run of the 3-class motor-state classifier, the three-way sibling of
  the EEG step-type project. Per-participant nested cross-validation; cohort macro-AUC 0.886, 3-way
  accuracy 75% (chance 33%), inner-vs-outer gap 0.04. Figures 2&ndash;4, 7&ndash;8 are explanatory
  schematics shared with the step-type report; Figures 1, 5&ndash;6, 9&ndash;11 are generated from this
  project's own data and results. Bracketed yellow fields on the title page are placeholders to be
  filled in. The remaining 12 of 32 participants are pending.</p>

</div></body></html>"""


def main():
    prep_figures()
    DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_html(), encoding="utf-8")
    print(f"HTML written: {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
    print(f"figures: {sorted(p.name for p in FIGS.glob('*.png'))}")


if __name__ == "__main__":
    main()
