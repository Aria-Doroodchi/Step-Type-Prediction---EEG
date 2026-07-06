"""Hand-drawn schematic figures for the supervisor report (no data dependency):
  fig_timeline      - trial timeline (direction cue -> CNV window -> go cue)
  fig_xgb_tree      - one decision tree splitting on continuous EEG features
  fig_xgb_boosting  - additive gradient boosting (tree1+...+treeN -> prob)
  fig_pooling       - per-participant vs partial vs full pooling
  fig_pipeline      - the per-fold feature funnel + nested CV
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

OUT = Path(r"C:/Users/Ali D/Documents/ML") / "outputs" / "reports" / "supervisor_2026-06-25" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 200, "font.family": "DejaVu Sans"})

C_ONE, C_TWO = "#1f77b4", "#d62728"
BLUE, GREEN, ORANGE, PURPLE, GREY = "#1b3a6b", "#2f7d4f", "#e08a1e", "#6a51a3", "#777777"

def box(ax, x, y, w, h, text, fc, ec=None, tc="white", fs=11, weight="bold", round_=0.02):
    ec = ec or fc
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.01,rounding_size={round_}",
                                fc=fc, ec=ec, lw=1.5))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", color=tc,
            fontsize=fs, weight=weight, wrap=True)

def arrow(ax, x1, y1, x2, y2, color="#444", lw=2.0, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=16, color=color, lw=lw))

# ================= TIMELINE =================
fig, ax = plt.subplots(figsize=(9, 2.8)); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 3)
ax.add_patch(Rectangle((0.3, 1.3), 9.4, 0.12, color="#333"))
for xx, lab, col in [(1.2, "Direction\ncue", BLUE), (8.2, "“Go”\ncue", GREEN)]:
    ax.add_patch(FancyArrowPatch((xx, 2.4), (xx, 1.45), arrowstyle="-|>", mutation_scale=14, color=col, lw=2.2))
    ax.text(xx, 2.55, lab, ha="center", va="bottom", fontsize=11, weight="bold", color=col)
# CNV window shading
ax.add_patch(Rectangle((1.2, 0.55), 7.0, 1.4, color=ORANGE, alpha=0.16))
ax.text(4.7, 0.78, "CNV window  (the brain 'preparing')", ha="center", fontsize=11.5,
        weight="bold", color="#a85a00")
ax.annotate("", xy=(8.2, 0.62), xytext=(1.2, 0.62),
            arrowprops=dict(arrowstyle="<->", color="#a85a00", lw=1.5))
ax.text(4.7, 1.62, "≈ 2 seconds", ha="center", fontsize=10, color="#333")
ax.text(0.3, 0.2, "The model only looks at this preparation window — before any movement happens.",
        fontsize=9.5, color="#555")
ax.set_title("A single trial: predicting the step before the 'go'", fontsize=13, weight="bold")
fig.tight_layout(); fig.savefig(OUT / "fig_timeline.png", bbox_inches="tight"); plt.close(fig)
print("timeline ok")

# ================= ONE DECISION TREE =================
fig, ax = plt.subplots(figsize=(8.5, 5.2)); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 6.2)
# nodes: (x,y,text,color)
box(ax, 3.7, 5.2, 2.6, 0.8, "Cz signal at 1.2 s\n< −1.8 ?", BLUE, fs=10.5)
box(ax, 1.2, 3.4, 2.6, 0.8, "Alpha power\n< 0.4 ?", BLUE, fs=10.5)
box(ax, 6.2, 3.4, 2.6, 0.8, "FCz slope\n< 0.2 ?", BLUE, fs=10.5)
# leaves
leaves = [(0.2, 1.6, "Diagonal\n0.82", C_TWO), (2.5, 1.6, "Straight\n0.31", C_ONE),
          (5.2, 1.6, "Straight\n0.27", C_ONE), (7.5, 1.6, "Diagonal\n0.71", C_TWO)]
for x, y, t, c in leaves:
    box(ax, x, y, 2.1, 0.8, t, c, fs=10.5, round_=0.06)
# edges with yes/no
def edge(x1, y1, x2, y2, lab):
    arrow(ax, x1, y1, x2, y2, color="#555", lw=1.8)
    ax.text((x1+x2)/2 + (0.25 if lab=='no' else -0.25), (y1+y2)/2 + 0.05, lab,
            fontsize=9, color="#333", style="italic")
edge(4.6, 5.2, 2.5, 4.2, "yes"); edge(5.4, 5.2, 7.5, 4.2, "no")
edge(2.1, 3.4, 1.25, 2.4, "yes"); edge(2.9, 3.4, 3.55, 2.4, "no")
edge(7.1, 3.4, 6.25, 2.4, "yes"); edge(7.9, 3.4, 8.55, 2.4, "no")
ax.text(5, 0.5, "Each box asks a yes/no question about one numeric feature. "
        "Follow the answers down to a leaf → a probability of 'diagonal'.",
        ha="center", fontsize=9.5, color="#555")
ax.set_title("One decision tree: a chain of yes/no questions about the signal",
             fontsize=13, weight="bold")
fig.tight_layout(); fig.savefig(OUT / "fig_xgb_tree.png", bbox_inches="tight"); plt.close(fig)
print("tree ok")

# ================= GRADIENT BOOSTING (additive) =================
fig, ax = plt.subplots(figsize=(9.6, 4.2)); ax.axis("off")
ax.set_xlim(0, 12.4); ax.set_ylim(0, 5)
def minitree(cx, cy, s=0.55, label="", correction=""):
    ax.add_patch(FancyBboxPatch((cx-0.95, cy-0.75), 1.9, 1.5, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc="#eef3f8", ec=BLUE, lw=1.4))
    # tiny tree glyph
    ax.plot([cx, cx-0.4], [cy+0.45, cy-0.05], color=BLUE, lw=1.6)
    ax.plot([cx, cx+0.4], [cy+0.45, cy-0.05], color=BLUE, lw=1.6)
    ax.plot([cx-0.4, cx-0.62], [cy-0.05, cy-0.5], color=BLUE, lw=1.4)
    ax.plot([cx-0.4, cx-0.18], [cy-0.05, cy-0.5], color=BLUE, lw=1.4)
    ax.plot([cx+0.4, cx+0.18], [cy-0.05, cy-0.5], color=BLUE, lw=1.4)
    ax.plot([cx+0.4, cx+0.62], [cy-0.05, cy-0.5], color=BLUE, lw=1.4)
    for dx in (-0.62,-0.18,0.18,0.62):
        ax.add_patch(Circle((cx+dx, cy-0.55), 0.07, color=BLUE))
    ax.add_patch(Circle((cx, cy+0.5), 0.08, color=BLUE))
    ax.text(cx, cy-0.95, label, ha="center", fontsize=9.5, weight="bold", color=BLUE)
    if correction:
        ax.text(cx, cy+0.95, correction, ha="center", fontsize=8.5, color=ORANGE, style="italic")
xs = [1.3, 4.0, 6.7]; labs = ["Tree 1", "Tree 2", "Tree 3"]
corr = ["first guess", "fixes Tree 1's\nmistakes", "fixes what's\nstill wrong"]
for x, l, c in zip(xs, labs, corr):
    minitree(x, 2.7, label=l, correction=c)
for i in range(len(xs)-1):
    ax.text((xs[i]+xs[i+1])/2, 2.7, "+", ha="center", va="center", fontsize=22, color="#444", weight="bold")
ax.text((xs[-1]+9.4)/2, 2.7, "+ … ", ha="center", va="center", fontsize=16, color="#444")
arrow(ax, 9.4, 2.7, 10.05, 2.7, color="#444", lw=2.2)
box(ax, 10.05, 1.95, 2.05, 1.5, "Add up,\nturn into\na probability", GREEN, fs=10, round_=0.06)
ax.text(5.6, 0.55, "Hundreds of small trees are added in sequence; each new tree corrects the running total. "
        "The summed score is turned into a probability.",
        ha="center", fontsize=9.5, color="#555")
ax.set_title("Gradient boosting: many small trees, each fixing the last one's errors",
             fontsize=13, weight="bold")
fig.tight_layout(); fig.savefig(OUT / "fig_xgb_boosting.png", bbox_inches="tight"); plt.close(fig)
print("boosting ok")

# ================= POOLING =================
fig, axs = plt.subplots(1, 3, figsize=(10.5, 4.0))
people = list("ABCDEFG")
target = "D"
def draw_panel(ax, title, train_logic, subtitle, hi=True):
    ax.axis("off"); ax.set_xlim(0, 7); ax.set_ylim(0, 7)
    ax.set_title(title, fontsize=12.5, weight="bold")
    for i, p in enumerate(people):
        y = 6.2 - i*0.82
        is_t = (p == target)
        train = train_logic(p)
        fc = GREEN if train else "#e9e9e9"
        ax.add_patch(Rectangle((1.6, y-0.3), 3.2, 0.62, fc=fc, ec="#999", lw=1))
        ax.text(3.2, y, f"Participant {p}" + ("  ★" if is_t else ""), ha="center", va="center",
                fontsize=9.5, color="#222" if train else "#888",
                weight="bold" if is_t else "normal")
    ax.text(3.2, 0.35, subtitle, ha="center", fontsize=8.6, color="#555")
    ax.text(0.2, 6.7, "", fontsize=8)
draw_panel(axs[0], "Per-participant", lambda p: p == target,
           "Train on ★'s own data only\n(~80 trials → easy to over-fit)")
draw_panel(axs[1], "Partial pooling", lambda p: True,
           "Train on ★ + everyone else\n(test still on ★). The chosen recipe.")
draw_panel(axs[2], "Full pooling", lambda p: p != target,
           "Train on everyone except ★\n(no ★ data at all)")
# legend
fig.text(0.5, 0.005, "★ = participant being predicted   ·   green = data used for training   ·   grey = not used",
         ha="center", fontsize=9, color="#444")
fig.suptitle("Three ways to use the group's data", fontsize=13.5, weight="bold", y=1.02)
fig.tight_layout(); fig.savefig(OUT / "fig_pooling.png", bbox_inches="tight"); plt.close(fig)
print("pooling ok")

# ================= PIPELINE / FUNNEL =================
fig, ax = plt.subplots(figsize=(9.5, 5.6)); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
steps = [
    ("≈ 25,000 candidate features\n(amplitude · slopes · power · source)", BLUE, ""),
    ("Drop near-duplicate features\n(correlation filter)", PURPLE, ""),
    ("Keep the 500 most-related\nto step type (ANOVA screen)", PURPLE, ""),
    ("Stability selection:\nkeep only consistently-useful ones", PURPLE, "≈ 40–150"),
    ("Train XGBoost; drop features\nit never uses (gain prune)", ORANGE, ""),
    ("SHAP prune:\ndrop the least-influential 20%", ORANGE, ""),
    ("Final XGBoost model\n+ tuned settings", GREEN, ""),
]
n = len(steps); y0 = 9.2; dy = 1.28; w = 5.4; h = 0.95; x = 2.3
for i, (txt, col, tag) in enumerate(steps):
    y = y0 - i*dy
    box(ax, x, y - h/2, w, h, txt, col, fs=10, round_=0.04)
    if tag:
        ax.text(x + w + 0.25, y, tag + (" features" if tag != "" and "≈" not in tag else ""),
                fontsize=8.5, color="#555", va="center")
    if i < n-1:
        arrow(ax, x + w/2, y - h/2, x + w/2, y - dy + h/2, color="#666", lw=1.8)
# funnel shape hint
ax.text(0.2, 5.0, "fewer and\nfewer\nfeatures →", fontsize=9.5, color="#888", rotation=0, va="center")
ax.set_title("The feature funnel (run separately inside every training fold)", fontsize=12.5, weight="bold")
fig.tight_layout(); fig.savefig(OUT / "fig_pipeline.png", bbox_inches="tight"); plt.close(fig)
print("pipeline ok")

# ================= NESTED CV =================
fig, ax = plt.subplots(figsize=(9.6, 4.4)); ax.axis("off")
ax.set_xlim(0, 11); ax.set_ylim(0, 6)
ax.set_title("Nested cross-validation: an honest test that never peeks", fontsize=12.5, weight="bold")
bw, bh, step, x0 = 1.2, 0.7, 1.4, 3.2
# outer: 5 blocks (4th = TEST)
ax.text(3.0, 4.7, "Outer split\n(scoring)", fontsize=10.5, weight="bold", color=BLUE,
        ha="right", va="center")
for i in range(5):
    x = x0 + i*step
    fc = ORANGE if i == 3 else "#cfe0f2"
    ax.add_patch(Rectangle((x, 4.35), bw, bh, fc=fc, ec="#888"))
    ax.text(x+bw/2, 4.7, "TEST" if i == 3 else "train", ha="center", va="center",
            fontsize=9.5, weight="bold" if i == 3 else "normal",
            color="white" if i == 3 else "#333")
ax.text(x0 + 3*step + bw/2, 4.18, "scored on unseen data", ha="center", fontsize=8.2, color="#a85a00")
# arrow: the TRAIN blocks get split again -> inner row
arrow(ax, x0 + step + bw/2, 4.35, x0 + step + bw/2, 2.55, color="#999", lw=1.6, style="-|>")
ax.text(x0 + step + bw/2 + 0.25, 3.45, "the train blocks\nare split again", fontsize=8.3,
        color="#777", va="center", style="italic")
# inner: 3 blocks (3rd = val)
ax.text(3.0, 2.2, "Inner split\n(tuning)", fontsize=10.5, weight="bold", color=PURPLE,
        ha="right", va="center")
for i in range(3):
    x = x0 + i*step
    fc = "#cdb6ec" if i == 2 else "#efe7fb"
    ax.add_patch(Rectangle((x, 1.85), bw, bh, fc=fc, ec="#888"))
    ax.text(x+bw/2, 2.2, "val" if i == 2 else "train", ha="center", va="center", fontsize=9.5, color="#333")
ax.text(x0 + 3*step, 2.2, "← pick features & settings here,\n    using the training portion only",
        fontsize=9, color="#555", va="center")
ax.text(0.2, 0.55, "Repeated for every fold and every participant. The TEST block is never used to make any "
        "choice — so the\nreported score is not inflated.", fontsize=9.3, color="#555")
fig.tight_layout(); fig.savefig(OUT / "fig_nestedcv.png", bbox_inches="tight"); plt.close(fig)
print("nestedcv ok")
print("ALL SCHEMATICS DONE ->", OUT)
