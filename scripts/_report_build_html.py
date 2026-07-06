# -*- coding: utf-8 -*-
"""Build the self-contained supervisor HTML report (figures embedded as base64)."""
import base64
from pathlib import Path

ROOT = Path(r"C:/Users/Ali D/Documents/ML")
DIR = ROOT / "outputs" / "reports" / "supervisor_2026-06-25"
FIGS = DIR / "figures"
OUT = DIR / "EEG_StepType_Report.html"

def b64(name):
    data = (FIGS / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode()

def fig(name, caption, num):
    return f"""
    <figure>
      <img src="{b64(name)}" alt="{caption}"/>
      <figcaption><span class="fignum">Figure {num}.</span> {caption}</figcaption>
    </figure>"""

CSS = """
:root{
  --ink:#1f2933; --muted:#5b6b7b; --line:#e1e6ec; --blue:#1b3a6b; --accent:#1f77b4;
  --soft:#f4f7fb; --warn:#a85a00; --green:#2f7d4f;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);
  line-height:1.62; margin:0; background:#fff; font-size:16.5px;}
.wrap{max-width:860px; margin:0 auto; padding:0 28px 80px;}
h1,h2,h3{line-height:1.25; color:var(--blue); font-weight:700;}
h2{font-size:1.5rem; margin-top:2.6em; padding-bottom:.25em; border-bottom:2px solid var(--line);}
h3{font-size:1.18rem; margin-top:1.8em; color:#243b56;}
p{margin:.7em 0;}
a{color:var(--accent); text-decoration:none;}
strong{color:#16324f;}
figure{margin:1.7em 0; text-align:center;}
figure img{max-width:100%; height:auto; border:1px solid var(--line); border-radius:8px;
  box-shadow:0 1px 4px rgba(20,40,70,.06);}
figcaption{font-size:.9rem; color:var(--muted); margin-top:.6em; text-align:left;
  padding:0 6px;}
.fignum{color:var(--blue); font-weight:600;}
/* title page */
.title{padding:90px 0 40px; text-align:center; border-bottom:1px solid var(--line); margin-bottom:10px;}
.title .eyebrow{letter-spacing:.16em; text-transform:uppercase; font-size:.8rem; color:var(--accent); font-weight:700;}
.title h1{font-size:2.35rem; margin:.35em 0 .2em;}
.title .sub{font-size:1.15rem; color:var(--muted); max-width:640px; margin:0 auto 1.6em;}
.title .meta{font-size:.96rem; color:#3d4d5c; margin-top:1.4em;}
.title .meta div{margin:.15em 0;}
.placeholder{background:#fff6da; border-bottom:1px dashed #d9a400; padding:0 3px; color:#7a5b00;}
/* callouts */
.callout{background:var(--soft); border-left:4px solid var(--accent); border-radius:0 8px 8px 0;
  padding:.9em 1.1em; margin:1.4em 0;}
.callout.key{border-color:var(--green); background:#f1f8f3;}
.callout.warn{border-color:var(--warn); background:#fdf6ee;}
.callout p{margin:.35em 0;}
.callout .lab{font-weight:700; color:var(--blue); font-size:.82rem; letter-spacing:.05em;
  text-transform:uppercase; display:block; margin-bottom:.2em;}
.callout.key .lab{color:var(--green);}
.callout.warn .lab{color:var(--warn);}
/* metric cards */
.cards{display:flex; gap:14px; flex-wrap:wrap; margin:1.5em 0;}
.card{flex:1; min-width:150px; background:#fff; border:1px solid var(--line); border-radius:10px;
  padding:16px 14px; text-align:center; box-shadow:0 1px 3px rgba(20,40,70,.05);}
.card .big{font-size:2rem; font-weight:800; color:var(--blue); line-height:1;}
.card .lab{font-size:.82rem; color:var(--muted); margin-top:.45em;}
/* toc */
.toc{background:var(--soft); border:1px solid var(--line); border-radius:10px; padding:18px 26px; margin:24px 0;}
.toc h2{border:none; margin:.1em 0 .5em; font-size:1.1rem; padding:0;}
.toc ol{margin:.3em 0; padding-left:1.3em;}
.toc li{margin:.28em 0;}
/* table */
table{border-collapse:collapse; width:100%; margin:1.3em 0; font-size:.95rem;}
th,td{border:1px solid var(--line); padding:8px 11px; text-align:left;}
th{background:var(--soft); color:var(--blue); font-weight:700;}
tr:nth-child(even) td{background:#fafcfe;}
/* glossary */
.gloss dt{font-weight:700; color:#16324f; margin-top:.7em;}
.gloss dd{margin:0 0 .2em 0; color:#33424f;}
.feat-list li{margin:.5em 0;}
hr.soft{border:none; border-top:1px solid var(--line); margin:2.4em 0;}
.foot{color:var(--muted); font-size:.86rem; margin-top:3em; border-top:1px solid var(--line); padding-top:1em;}
@media print{
  .title{padding-top:40px;}
  h2{page-break-after:avoid;} figure{page-break-inside:avoid;}
  .card{box-shadow:none;}
}
"""

HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Predicting Step Type from Brain Signals — Project Report</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

  <header class="title">
    <div class="eyebrow">EEG &middot; Movement Intention &middot; Machine Learning</div>
    <h1>Reading the Next Step from Brain Activity</h1>
    <p class="sub">Predicting whether a person is about to step <strong>straight</strong> or
      <strong>diagonally</strong> — from the brain's "getting ready" signal, before they move.</p>
    <div class="meta">
      <div><strong>Prepared by:</strong> <span class="placeholder">[your name]</span></div>
      <div><strong>For:</strong> <span class="placeholder">[supervisor name]</span></div>
      <div><strong>Affiliation:</strong> <span class="placeholder">Sunnybrook Research Institute</span></div>
      <div><strong>Date:</strong> June 2026</div>
    </div>
  </header>

  <nav class="toc">
    <h2>What's in this report</h2>
    <ol>
      <li><a href="#summary">The short version</a></li>
      <li><a href="#goal">What the project is trying to do</a></li>
      <li><a href="#xgb">The prediction engine: what an "XGBoost" model is</a></li>
      <li><a href="#features">What the model looks at: the features</a></li>
      <li><a href="#pooling">Borrowing strength across people: partial pooling</a></li>
      <li><a href="#arch">How the model is built and tested, step by step</a></li>
      <li><a href="#results">Current results</a></li>
      <li><a href="#features-informative">Which features carry the signal?</a></li>
      <li><a href="#next">Summary and next steps</a></li>
      <li><a href="#glossary">Plain-language glossary</a></li>
    </ol>
  </nav>

  <!-- 1 -->
  <h2 id="summary">1 &nbsp;The short version</h2>
  <p>This project asks a simple question: <strong>can we tell, from brain activity alone, which way
  a person is about to step</strong> — straight ahead or off to the side — <em>before the movement
  actually happens?</em> We record the electrical activity of the brain (an EEG) during the brief
  "get ready" window after a person is told which way to step but before they are told to go, and we
  train a computer model to read that signal.</p>

  <div class="cards">
    <div class="card"><div class="big">0.71</div><div class="lab">average accuracy score (AUC), where 0.50&nbsp;=&nbsp;a coin flip</div></div>
    <div class="card"><div class="big">30</div><div class="lab">participants, each with their own model</div></div>
    <div class="card"><div class="big">19/30</div><div class="lab">participants clearly above chance (AUC&nbsp;&ge;&nbsp;0.70)</div></div>
  </div>

  <div class="callout key">
    <span class="lab">Headline</span>
    <p>On the most recent full-group analysis, the model separates "straight" from "diagonal"
    steps with an average score of <strong>0.71</strong> (a coin-flip would score 0.50, a perfect
    model 1.00). Results vary a lot from person to person — some are read almost perfectly, a few
    barely above chance. Crucially, the score holds up under a strict honesty check, so it is
    <strong>not an artefact of the model "memorising" the data.</strong></p>
  </div>

  <!-- 2 -->
  <h2 id="goal">2 &nbsp;What the project is trying to do</h2>
  <p>When a person prepares a planned movement, a slow negative-going cortical potential develops over
  the scalp — the <strong>Contingent Negative Variation (CNV)</strong>, first described by Walter and
  colleagues in 1964. It arises during the <em>foreperiod</em> of a warning&ndash;imperative
  (S1&ndash;S2) paradigm: a <strong>direction cue</strong> (the warning stimulus, S1) signals which way
  the participant will step, and ~2&nbsp;s later a <strong>"go" cue</strong> (the imperative stimulus,
  S2) triggers the movement. The CNV unfolds in the S1&ndash;S2 interval and is generally taken to index
  anticipatory attention and motor preparation, its later phase reflecting sensorimotor readiness over
  fronto-central cortex.</p>

  {fig("fig_timeline.png", "The structure of a single trial. The CNV develops across the foreperiod between the direction cue (S1) and the “go” cue (S2). The model only ever uses this pre-movement interval — it predicts <em>before</em> the imperative stimulus and any overt movement.", 1)}

  <p>Our goal is to look only at that preparation window and predict the <strong>step type</strong>:
  a <strong>straight</strong> step (labelled "One" in the data) versus a <strong>diagonal</strong>
  step (labelled "Two"). Because there are exactly two possible answers, this is a
  <strong>binary classification</strong> problem, and a model that guessed randomly would be right
  about half the time. Anything reliably above 50% means the preparation signal genuinely carries
  information about the planned direction.</p>

  <p>The data come from <strong>30 participants</strong>, each contributing roughly <strong>80 trials</strong>
  (about 40 straight, 40 diagonal), recorded on a <strong>64-channel EEG montage</strong>. The continuous
  data were preprocessed with a standard artifact-handling pipeline (line-noise removal, ICA-based
  artifact rejection, automated bad-trial repair) and re-expressed as <strong>current-source density</strong>
  (CSD / surface Laplacian), which sharpens spatial topography and removes the reference. The continuous
  record is segmented into <strong>epochs</strong>: ~2&nbsp;s single-trial windows time-locked to the
  direction cue (S1).</p>

  <!-- 3 -->
  <h2 id="xgb">3 &nbsp;The prediction engine: what an "XGBoost" model is</h2>
  <p>The workhorse of this project is a model called <strong>XGBoost</strong>. To understand it, start
  with its building block: the <strong>decision tree</strong>. A decision tree is just a flowchart of
  yes/no questions. Each question looks at one number and compares it to a threshold, and the answers
  guide you down the tree to a final verdict.</p>

  {fig("fig_xgb_tree.png", "A single decision tree. Each box asks a yes/no question about one measurement from the brain signal (for example, “was the signal at the top of the head below this level at 1.2 seconds?”). Following the answers leads to a leaf at the bottom, which gives a probability that the step was diagonal.", 2)}

  <h3>How it handles "continuous" measurements (not just categories)</h3>
  <p>A common question is how a method built on yes/no questions can work with our data, which are not
  neat categories like "red / blue / green" but <strong>continuous numbers</strong> — voltages,
  slopes, and powers that can take any value. The answer is that the tree <strong>invents its own
  thresholds</strong>. During training it searches through the data and finds the most informative
  cut-points automatically — for instance, "signal below &minus;1.8" versus "above &minus;1.8." The
  modeller never has to define categories; the model discovers where the meaningful boundaries lie in
  the raw numbers. This is exactly why it suits brain-signal data, which is entirely numeric.</p>

  <h3>From one tree to many: "gradient boosting"</h3>
  <p>A single small tree is a weak guesser. XGBoost's trick — the "boosting" in the name — is to build
  <strong>hundreds of small trees in sequence, where each new tree focuses on fixing the mistakes the
  previous trees made.</strong> The first tree makes a rough guess; the second tree learns where the
  first went wrong and nudges those cases; the third corrects what remains; and so on. The final
  prediction adds up the contributions of all the trees and converts the total into a probability
  between 0% and 100%.</p>

  {fig("fig_xgb_boosting.png", "Gradient boosting in a nutshell. Many simple trees are added together, each one correcting the running total left by the trees before it. The combined score is squashed into a probability of “diagonal.” In our models there are up to a thousand such trees.", 3)}

  <p>This combination — many simple trees, each improving on the last — is what makes XGBoost both
  accurate and good at spotting subtle <em>combinations</em> of signals (e.g. "high power in one place
  <em>and</em> a rising slope in another"). It is one of the most successful methods for the kind of
  table-shaped data we have here, where each trial is a row and each measurement is a column.</p>

  <!-- 4 -->
  <h2 id="features">4 &nbsp;What the model looks at: the features</h2>
  <p>The model never sees the raw squiggly EEG trace. Instead, we summarise each trial into a long
  list of numbers called <strong>features</strong> — each one capturing a specific, interpretable
  aspect of the brain signal. We compute four families of features.</p>

  <ul class="feat-list">
    <li><strong>Amplitude</strong> — the mean CSD-transformed potential within each time bin at each
      electrode: the binned single-trial ERP, i.e. the CNV waveform itself.</li>
    <li><strong>Slopes</strong> — the local temporal gradient (linear slope / first derivative) of the
      potential within each bin, sensitive to the <em>rate</em> of the CNV's evolving negativity rather
      than its absolute level.</li>
    <li><strong>Spectral power (PSD)</strong> — single-trial power from a <strong>Morlet-wavelet</strong>
      time-frequency decomposition, summarised in the canonical bands: delta (0.5&ndash;4&nbsp;Hz), theta
      (4&ndash;8), alpha (8&ndash;13), beta (13&ndash;30) and gamma (30&ndash;40).</li>
    <li><strong>Source localisation</strong> — a distributed inverse solution (<strong>eLORETA</strong>)
      projected onto the <em>fsaverage</em> template cortex and parcellated with the Destrieux atlas
      (<em>aparc.a2009s</em>, ~150 ROIs), estimating the cortical generators of the scalp signal.</li>
  </ul>

  <h3>The binning architecture: turning a wave into numbers</h3>
  <p>To turn each continuous waveform into features, the ~2&nbsp;s epoch (0&ndash;2&nbsp;s post-S1) is
  divided into non-overlapping <strong>62.5&nbsp;ms (1/16&nbsp;s) time bins</strong>, and every feature is
  computed per electrode (or source ROI), per bin, and — for spectral power — per band. One feature might
  be "the CSD-transformed potential at the vertex (Cz) averaged between 0.75 and 0.81&nbsp;s." Across all
  electrodes, bins, bands and ROIs this yields about <strong>25,000 candidate features per trial.</strong>
  (Section 6 describes how the model trims this down to the informative handful.)</p>

  {fig("fig_binning.png", "How a continuous waveform becomes features. The grey line is the per-bin CNV signal at the vertex; the coloured bars are the per-bin summaries — each bar is one feature. The model uses 62.5 ms bins (finer than the 0.25 s bins drawn here for clarity).", 4)}

  <p>Below are two examples built directly from the project's own data, averaged across all 30
  participants.</p>

  {fig("fig_cnv_waveform.png", "Grand-average CSD-transformed potential at the vertex (Cz), averaged over participants. The slow negative-going deflection across the foreperiod is the CNV. The straight (blue) and diagonal (red) conditions overlap almost completely: the class-discriminative variance is small relative to within- and between-subject variance, which is what makes single-trial decoding hard.", 5)}

  {fig("fig_psd_bands.png", "Single-trial band-limited spectral power (Morlet) at Cz, grand-averaged. The 1/f-dominated profile (delta ≫ gamma) is typical of EEG. The two conditions are near-identical at the univariate level — consistent with discriminative information residing in distributed, higher-order feature combinations rather than any single band-power contrast.", 6)}

  <div class="callout">
    <span class="lab">Why this matters</span>
    <p>Notice that in Figures 5 and 6 the two conditions look nearly identical. That is the central
    challenge: there is no single univariate "tell" — no one electrode, latency, or band separates
    straight from diagonal. The model's job is to find a faint, distributed pattern spread across
    thousands of features — which is precisely the kind of needle-in-a-haystack task that XGBoost,
    combined with the feature-selection steps in Section 6, is designed for.</p>
  </div>

  <!-- 5 -->
  <h2 id="pooling">5 &nbsp;Borrowing strength across people: partial pooling</h2>
  <p>Every person's brain is a little different, so the safest approach is to train a separate model
  for each participant using only their own trials. But there is a catch: with only ~80 trials per
  person, a flexible model can easily <strong>"memorise" the training trials</strong> instead of
  learning a general pattern — a problem called <strong>overfitting</strong>. It then looks great on
  the data it has seen and falls apart on new data.</p>

  <p><strong>Partial pooling</strong> is the fix. The idea is to let each person's model also learn
  from <em>everyone else's</em> trials, while still being tested only on that person. There are three
  options on a spectrum:</p>

  {fig("fig_pooling.png", "Three ways to use the group's data when predicting one person (marked with a star). Per-participant uses only that person's data (prone to overfitting). Full pooling uses everyone <em>except</em> them. Partial pooling — the approach we use — trains on the target person <em>plus</em> everyone else, then tests on the target. It keeps each person's individuality while borrowing the stability of the crowd.", 7)}

  <p><strong>What is combined, and how:</strong> each participant's trials are rows in a big table
  (with the ~25,000 feature columns described above). Partial pooling simply <strong>stacks everyone's
  rows into one large training table</strong> — about <strong>2,400 trials</strong> instead of ~80 —
  so the model sees the broad, shared pattern of how preparation differs between straight and diagonal
  steps. The held-out trials being scored still belong only to the one person we are predicting, so the
  test stays fair. The result is a model that is anchored by the group but still tuned to the
  individual.</p>

  <div class="callout key">
    <span class="lab">What partial pooling bought us</span>
    <p>Pooling's biggest benefit is <strong>honesty</strong>. Before pooling, the models showed a large
    gap between their optimistic internal scores and their true performance on unseen data — the
    signature of overfitting. After pooling, that gap essentially <strong>disappears</strong> (it even
    reverses slightly: see Section 7), while the headline accuracy holds up or modestly improves. In
    short, pooling makes the reported numbers trustworthy.</p>
  </div>

  <!-- 6 -->
  <h2 id="arch">6 &nbsp;How the model is built and tested, step by step</h2>
  <p>Two ideas run through the whole pipeline: <strong>(a)</strong> never let the model peek at the
  data it will be judged on, and <strong>(b)</strong> ruthlessly trim ~25,000 raw features down to the
  few dozen that genuinely matter.</p>

  <h3>An honest test: nested cross-validation</h3>
  <p>To measure performance fairly, we use <strong>cross-validation</strong>: the trials are split into
  five parts, and each part takes a turn as the "unseen test" while the model learns from the other
  four. We then do this in a <strong>nested</strong> way — an outer split that is used <em>only</em> for
  the final scoring, and an inner split used for all the tuning decisions. This separation is what stops
  the model from flattering itself.</p>

  {fig("fig_nestedcv.png", "Nested cross-validation. The outer split sets aside a test block that is used only to score the finished model. All the choices — which features to keep, which settings to use — are made on a separate inner split of the remaining data. Because the test block is never used to make any choice, the final score is not inflated.", 8)}

  <h3>The feature funnel</h3>
  <p>Inside <em>each</em> training fold (so that no test data leaks in), the ~25,000 features pass
  through a sequence of filters, each removing features that are redundant or unhelpful, like a funnel
  narrowing to the essentials:</p>

  {fig("fig_pipeline.png", "The feature funnel, applied fresh inside every training fold. It starts from ~25,000 candidate features and ends with roughly 120 that survive every filter. “SHAP pruning” (second-to-last step) is explained below.", 9)}

  <table>
    <thead><tr><th>Stage</th><th>What it does, in plain terms</th></tr></thead>
    <tbody>
      <tr><td><strong>Correlation filter</strong></td><td>Drops features that say almost the same thing as another feature, to avoid wasteful duplication.</td></tr>
      <tr><td><strong>ANOVA screen</strong></td><td>A quick statistical test keeps only the ~500 features most related to step type and discards the obvious non-starters.</td></tr>
      <tr><td><strong>Stability selection</strong></td><td>Re-runs the selection on many random sub-samples of the data and keeps only features that <em>repeatedly</em> prove useful — not ones that looked good by luck once.</td></tr>
      <tr><td><strong>Gain pruning</strong></td><td>Trains an XGBoost model and drops any feature the trees never actually used to make a split.</td></tr>
      <tr><td><strong>SHAP pruning</strong></td><td>Measures how much each remaining feature really influenced the predictions and removes the least-influential 20%. (More on SHAP below.)</td></tr>
    </tbody>
  </table>

  <h3>What is "SHAP"?</h3>
  <p><strong>SHAP</strong> is a principled way to answer "<em>how much did each feature contribute to
  this prediction?</em>" It comes from game theory: it treats the features like players on a team and
  fairly divides the credit for the final prediction among them. Features that consistently earn little
  credit are pruned away. The pay-off is twofold — a leaner, less overfit model, and a ranked list of
  which brain measurements actually drive the decision, which is scientifically interesting in its own
  right.</p>

  <h3>Tuning the XGBoost settings</h3>
  <p>Finally, XGBoost has dials — how deep each tree can grow, how fast it learns, how strongly it is
  penalised for complexity. We search across many combinations (efficiently, using a method that quickly
  discards the weak ones) and keep the best, judged only on the inner tuning data. After the funnel, each
  final per-person model rests on roughly <strong>120 features</strong>.</p>

  <!-- 7 -->
  <h2 id="results">7 &nbsp;Current results</h2>
  <p>The numbers below come from the most recent <strong>full-cohort run</strong>: all 30 participants,
  partial pooling, the complete ~25,000-feature set, and the full funnel above. Performance is reported
  mainly as <strong>AUC</strong>, with accuracy alongside.</p>

  <div class="callout">
    <span class="lab">AUC vs. accuracy — what's the difference?</span>
    <p><strong>Accuracy</strong> is the simplest score: the percentage of trials put in the right box,
    using one fixed cut-off (e.g. "call it diagonal if the model is more than 50% sure"). It is easy to
    read but can mislead — if one step type were more common, a lazy model could score well just by
    always guessing that one.</p>
    <p><strong>AUC</strong> (the Area Under the ROC Curve) is a fuller summary. Rather than fixing a
    cut-off, it asks: <em>if you pick one real straight trial and one real diagonal trial at random, how
    often does the model give the diagonal one the higher score?</em> 1.00 means always, 0.50 means it
    is guessing. Because it sweeps across every possible cut-off, AUC is not fooled by an uneven mix of
    classes and rewards the model for <em>ranking</em> trials correctly even when it is unsure — which is
    why it is our headline number.</p>
  </div>

  <div class="cards">
    <div class="card"><div class="big">0.714</div><div class="lab">average AUC across the 30 participants</div></div>
    <div class="card"><div class="big">67%</div><div class="lab">average accuracy (straight vs diagonal)</div></div>
    <div class="card"><div class="big">0.47&ndash;1.00</div><div class="lab">range across individuals</div></div>
  </div>

  <p>A <strong>confusion matrix</strong> is the simplest way to see <em>where</em> that accuracy comes
  from. It cross-tabulates what actually happened against what the model predicted: the green diagonal
  counts the correct calls, and the red off-diagonal counts the mistakes.</p>

  {fig("fig_confusion.png", "Confusion matrix for the whole cohort, pooling all 2,384 predictions. Rows are what the step actually was; columns are what the model predicted. Green cells (the diagonal) are correct calls; red cells are mistakes. The two step types are read about equally well &mdash; 66% of straight and 68% of diagonal steps correct &mdash; so the model is genuinely telling them apart rather than leaning toward one answer.", 10)}

  <p>The errors are <strong>balanced</strong>: the model is roughly as good at spotting straight steps
  (66% correct) as diagonal ones (68% correct), with no strong bias toward either. The next chart breaks
  the same results down by participant.</p>

  {fig("fig_auc_bars.png", "Each participant's own model, scored on their held-out trials and sorted from weakest to strongest. The dashed line is chance (0.50); the solid line is the group average (0.714). Most people (green) are read well above chance; a handful sit near chance, and one falls just below — a reminder that this works far better for some brains than others.", 11)}

  <p>Two things stand out. First, <strong>the model works, on average, clearly above chance</strong>:
  19 of 30 participants reach an AUC of 0.70 or higher, and several are read almost perfectly. Second,
  there is <strong>wide individual variation</strong> — a few participants sit near the chance line.
  Understanding why some brains are so much more "readable" than others is an open and interesting
  question.</p>

  <div class="callout key">
    <span class="lab">The honesty check</span>
    <p>A model that is overfitting scores higher on its own tuning data than on truly unseen data. Here
    the opposite holds: the average score on the <strong>unseen</strong> test trials (0.714) is actually
    <em>higher</em> than the internal tuning score (0.654). The gap between "optimistic internal" and
    "real-world" performance has not just closed — it has slightly reversed. This is the strongest sign
    that <strong>the 0.71 figure is real and not inflated</strong>, and it is the direct pay-off of
    partial pooling.</p>
  </div>

  <div class="callout warn">
    <span class="lab">How to read this honestly</span>
    <p>The group average (0.71) is the stable, trustworthy headline. <strong>Individual</strong> scores
    rest on small test sets (~16 trials per fold), so a single person's value — including the perfect
    1.00 — is noisier and should be read as "high" rather than as an exact figure. Separately, earlier
    controlled comparisons that deliberately used a <em>trimmed</em> feature set (for speed) landed
    around 0.60&ndash;0.64; those experiments were what confirmed the overfitting-gap collapse described
    above. The full-feature run reported here is the project's current best honest estimate.</p>
  </div>

  <!-- 8 -->
  <h2 id="features-informative">8 &nbsp;Which features carry the signal?</h2>
  <p>Because the selection step (Section 6) runs inside all 150 training folds, we can simply tally
  <strong>how often each feature is retained</strong>. Features kept in every fold are the most robustly
  informative, and the pattern across them is physiologically revealing — and reassuringly consistent
  with the CNV literature.</p>

  {fig("fig_feat_informativeness.png", "Feature informativeness, measured as how often each feature is retained across the 150 training folds. (a) By type, the discriminative signal is carried overwhelmingly by temporal-slope features, with a smaller theta-power contribution; the binned amplitude and eLORETA source features, though available, are essentially never retained. (b) By timing, informativeness is concentrated in the early foreperiod, peaking ~0.2 s after the direction cue.", 12)}

  <p>Two patterns stand out. <strong>By feature type</strong>, the discriminative signal is carried
  almost entirely by the <strong>temporal-slope</strong> features — the instantaneous rate of change of
  the potential — with a secondary contribution from <strong>spectral power</strong>, predominantly in
  the <strong>theta</strong> band (4&ndash;8&nbsp;Hz). The per-bin amplitude and the source-space
  features were rarely selected. <strong>By timing</strong>, informativeness is sharply concentrated in
  the <strong>early foreperiod (~0.1&ndash;0.35&nbsp;s post-cue)</strong>, peaking near 0.2&nbsp;s, rather
  than in the terminal CNV. <strong>Spatially</strong>, the retained features concentrate over
  <strong>central and fronto-central sensorimotor cortex</strong> (the C, FC and CP electrode rows —
  Cz, C1&ndash;C6, FC1&ndash;FC4, CP4; strongest on the right, TP8/C6/C4). Taken together, the decoder
  is keying on an <strong>early, sensorimotor, rate-of-change signal with a fronto-central theta
  component</strong> — a physiologically sensible correlate of motor-plan formation, and a result of
  interest in its own right.</p>

  <p>To make these features concrete, the matrix below shows <strong>example single-trial
  topographies</strong> — different feature types (amplitude, frequency-band power, and eLORETA source),
  different participants, conditions and time bins. They are deliberately raw single trials:
  individually noisy, which is exactly why the selection funnel and cross-subject pooling matter.</p>

  {fig("fig_feature_topomaps.png", "Example single-trial feature topographies. The top-down scalp maps show amplitude (CSD) and band-power features at one electrode set; the two brain views are top-down glass-brains of the eLORETA source features (one dot per cortical region). Each panel is one real trial, labelled underneath with its feature, participant, condition (straight vs diagonal) and time bin, and scaled to its own range (red = higher, blue = lower). Single trials are noisy by nature — the patterns become reliable only after the selection and pooling described above.", 13)}

  <!-- 9 -->
  <h2 id="next">9 &nbsp;Summary and next steps</h2>
  <p>In plain terms: <strong>the brain's preparation signal does carry a readable trace of which way a
  person is about to step</strong>, and a carefully-built, honestly-tested XGBoost model can pick it up
  — on average around 0.71 (with 0.50 being chance), and much higher in the most readable individuals.
  Partial pooling across participants was the key step that made these numbers trustworthy rather than
  optimistic. Encouragingly, the model also points to a clear, interpretable substrate — an early,
  sensorimotor, rate-of-change signal — rather than an opaque pattern.</p>

  <div class="callout key">
    <span class="lab">Worth publishing</span>
    <p>We think there is a <strong>paper</strong> in this. An honest, fully cross-validated decoding of
    step direction from the CNV — together with the finding that the signal is carried by early
    sensorimotor slope features — would make a nice contribution to the literature.</p>
  </div>

  <p><strong>Where this can go next:</strong></p>
  <ul>
    <li><strong>Understanding individual differences</strong> — why some participants are read almost
      perfectly and others barely above chance.</li>
    <li><strong>Localising the early signal</strong> — pinning down the cortical source of the
      informative early-foreperiod slope features more precisely (the current template-based source
      features added little and could be improved).</li>
    <li><strong>Alternative models</strong> — among the models we have evaluated, <strong>XGBoost has
      been the most promising</strong>: it gave the strongest results on small subsets of data and with
      much shorter training times, and it pairs that with a sophisticated, carefully-staged model
      architecture. The other candidates are modern neural-network models (compact "CNN/EEGNet"
      deep-learning networks designed for EEG); these are worth exploring as a next step, but may not
      ultimately improve on the current model.</li>
  </ul>

  <hr class="soft"/>

  <!-- glossary -->
  <h2 id="glossary">Plain-language glossary</h2>
  <dl class="gloss">
    <dt>EEG</dt><dd>Electroencephalography — recording the brain's electrical activity with sensors on the scalp.</dd>
    <dt>CNV</dt><dd>Contingent Negative Variation — a slow brain wave that builds up while a person prepares for an expected action.</dd>
    <dt>Epoch / trial</dt><dd>One ~2-second recording from a single attempt, across all sensors.</dd>
    <dt>Feature</dt><dd>A single number summarising one aspect of a trial (e.g. the average signal at one sensor in one time slice).</dd>
    <dt>Classification</dt><dd>Predicting which category something belongs to — here, "straight" vs "diagonal."</dd>
    <dt>AUC</dt><dd>A performance score from 0.50 (random guessing) to 1.00 (perfect). It measures how well the model ranks a true "diagonal" above a true "straight." 0.70 is a solidly useful model on hard biological data.</dd>
    <dt>Overfitting</dt><dd>When a model memorises its training data instead of learning a general rule, so it looks good in practice runs but fails on new data.</dd>
    <dt>Cross-validation</dt><dd>Repeatedly holding out part of the data as an unseen test to estimate real-world performance fairly.</dd>
    <dt>Partial pooling</dt><dd>Training each person's model on their own data plus everyone else's, to borrow the stability of the group while staying personalised.</dd>
    <dt>XGBoost</dt><dd>The model used here: hundreds of small decision trees added together, each correcting the previous ones' mistakes.</dd>
    <dt>SHAP</dt><dd>A fair-credit method for measuring how much each feature contributed to a prediction; used here to prune weak features and to interpret the model.</dd>
  </dl>

  <p class="foot">Figures 1&ndash;3 and 7&ndash;9 are explanatory schematics. Figures 4&ndash;6 and 10&ndash;13 are
  generated directly from this project's data and results (latest full-cohort XGBoost run, 30 participants,
  partial pooling). Performance figures: average AUC 0.714, average accuracy 67%. Prepared for a
  non-specialist audience; a glossary is provided above. Bracketed yellow fields on the title page are
  placeholders to be filled in.</p>

</div>
</body>
</html>"""

OUT.write_text(HTML, encoding="utf-8")
kb = OUT.stat().st_size / 1024
print(f"HTML written: {OUT}  ({kb:.0f} KB)")
