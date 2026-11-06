"""Figure 1 – revised architecture: dedicated risk model; unevaluated components shaded grey."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8.2})
W, H = 6.7, 8.9
fig = plt.figure(figsize=(W, H), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

EVAL, NOTEVAL, EDGE, INK = "#ffffff", "#d9d9d6", "#3a3a37", "#1f1f1d"

def box(x, y, w, h, text, grey=False, bold_first=True, fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.06",
                                fc=NOTEVAL if grey else EVAL, ec=EDGE, lw=0.9))
    lines = text.split("\n")
    n = len(lines); step = 0.165
    y0 = y + h / 2 + (n - 1) * step / 2
    for k, l in enumerate(lines):
        ax.text(x + w / 2, y0 - k * step, l, ha="center", va="center", color=INK, fontsize=fs,
                fontweight="bold" if (k == 0 and bold_first) else "normal")
    return (x, y, w, h)

def arrow(x1, y1, x2, y2, text=None, tx=None, ty=None, ha="left"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=9))
    if text:
        ax.text(tx if tx is not None else x1 + 0.06, ty if ty is not None else (y1 + y2) / 2, text,
                ha=ha, va="center", fontsize=7.2, color=INK, style="italic")

def line(xs, ys):
    ax.plot(xs, ys, color=EDGE, lw=0.9, solid_capstyle="butt")

# ---- top: interaction space
top = box(0.95, 7.95, 4.8, 0.62, "Real-world interaction space\nLong-term care · daily conversation · non-intrusive sensing")

# ---- inputs
xs = [1.75, 3.35, 4.95]
labels = ["Audio (speech)", "Visual (face)", "Text (utterances)"]
for x, l in zip(xs, labels):
    arrow(x, 7.95, x, 7.32)
    ax.text(x + 0.06, 7.64, l, ha="left", va="center", fontsize=7.2, style="italic", color=INK)

# ---- perception layer container
ax.add_patch(Rectangle((0.55, 5.02), 5.6, 2.3, fc="none", ec=EDGE, lw=1.0, ls=(0, (4, 2.5))))
ax.text(0.68, 7.18, "Multimodal affective perception layer", ha="left", va="center", fontsize=8.2, fontweight="bold", color=INK)
bw, bh, by = 1.5, 0.66, 6.3
box(xs[0] - bw / 2, by, bw, bh, "Audio emotion encoder\neGeMAPS + SVM\n(CREMA-D)", fs=7.6)
box(xs[1] - bw / 2, by, bw, bh, "Visual emotion encoder\nface detection + CNN\n(prototype only)", grey=True, fs=7.6)
box(xs[2] - bw / 2, by, bw, bh, "Text emotion encoder\nchar n-gram + LR\n(SMP2020-EWECT)", fs=7.6)
# merge to affective state
line([xs[0], xs[0]], [by, 6.05]); line([xs[1], xs[1]], [by, 6.05]); line([xs[2], xs[2]], [by, 6.05])
line([xs[0], xs[2]], [6.05, 6.05])
arrow(2.55, 6.05, 2.55, 5.82)
ax.text(2.62, 5.94, "emotion probabilities p(a), p(v), p(t)", ha="left", va="center", fontsize=7.0, style="italic", color=INK)
box(0.8, 5.17, 3.5, 0.65, "Affective state encoding\naggregation over time: mean ē, variability σ², entropy", fs=7.6)

# ---- dedicated risk model (right)
risk = box(4.5, 5.17, 1.5, 0.65, "Risk features\nacoustic + linguistic\n(eGeMAPS, char n-grams)", fs=7.2)

# raw audio + text features to the risk model (dashed), from the interaction space
ax.plot([5.75, 6.27, 6.27], [8.12, 8.12, 5.5], color=EDGE, lw=0.9, ls=(0, (3, 2)))
arrow(6.27, 5.5, 6.0, 5.5)
ax.text(6.355, 6.75, "raw audio + text", rotation=270, ha="center", va="center", fontsize=6.8, style="italic", color=INK)
box(3.3, 4.12, 2.75, 0.62, "Dedicated depression risk model\ntemporally aggregated, validated vs. SDS", fs=7.6)
arrow(5.27, 5.17, 5.27, 4.74)

# ---- outputs into interface
box(0.65, 4.12, 2.35, 0.62, "Affective profile\nē, σ², modality confidence", fs=7.6)
arrow(1.8, 5.17, 1.8, 4.74)

iface = box(0.95, 2.95, 4.8, 0.72, "Affective state interface (structured, auditable)\nrisk estimate + evidence window | affective profile + confidence", grey=True, fs=7.6)
arrow(1.8, 4.12, 1.8, 3.67, "affective profile", 1.86, 3.9)
arrow(4.67, 4.12, 4.67, 3.67, "risk estimate", 4.73, 3.9)

# ---- generative and rule-based components
llm = box(0.65, 1.62, 2.75, 0.8, "LLM-based care agent\nemotion-conditioned dialogue:\ntone, content, explanation", grey=True, fs=7.6)
rule = box(3.75, 1.62, 2.3, 0.8, "Rule-based escalation\noutside the LLM;\nclinician-reviewed thresholds", grey=True, fs=7.6)
arrow(2.0, 2.95, 2.0, 2.42)
arrow(4.9, 2.95, 4.9, 2.42)

care = box(0.95, 0.55, 4.8, 0.62, "Care actions & feedback loop\nsupportive dialogue · caregiver notification · clinical follow-up", grey=True, fs=7.6)
arrow(2.0, 1.62, 2.0, 1.17)
arrow(4.9, 1.62, 4.9, 1.17)

# ---- feedback loops
line([0.95, 0.3, 0.3], [0.86, 0.86, 8.26]); arrow(0.3, 8.26, 0.95, 8.26)
ax.text(0.22, 4.5, "supportive dialogue", rotation=90, ha="center", va="center", fontsize=7.2, style="italic", color=INK)
line([5.75, 6.42, 6.42], [0.86, 0.86, 8.26]); arrow(6.42, 8.26, 5.75, 8.26)
ax.text(6.5, 4.5, "continuous affective monitoring", rotation=270, ha="center", va="center", fontsize=7.2, style="italic", color=INK)

# ---- legend
ax.add_patch(Rectangle((1.3, 0.12), 0.25, 0.16, fc=EVAL, ec=EDGE, lw=0.8))
ax.text(1.62, 0.2, "evaluated on public data in this study", va="center", fontsize=7.2, color=INK)
ax.add_patch(Rectangle((3.85, 0.12), 0.25, 0.16, fc=NOTEVAL, ec=EDGE, lw=0.8))
ax.text(4.17, 0.2, "specified, not evaluated", va="center", fontsize=7.2, color=INK)

for ext in ("png", "pdf"):
    fig.savefig(f"figures/Figure1.{ext}", dpi=300)
