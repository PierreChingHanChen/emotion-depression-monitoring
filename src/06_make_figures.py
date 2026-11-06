"""Figure 2 (ROC curves) and Figure 3 (group differences in emotion probabilities).

Usage: python src/06_make_figures.py --results results --out figures
"""
import argparse
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common import cv_probs, load_derived, participant_matrix  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#8a8a86",
                     "axes.labelcolor": "#2b2b29", "xtick.color": "#55554f", "ytick.color": "#55554f"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="data/derived/eatd_affective_states.csv")
    ap.add_argument("--egemaps", default="data/derived/eatd_egemaps.csv")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load_derived(a.states, a.egemaps)

    specs = [("Audio emotion state (6)", "A", "mean", "#2a78d6", "-"),
             ("Text emotion state (6)", "T", "mean", "#eb6834", "--"),
             ("Audio + text emotion state, mean + SD (24)", "AT", "meansd", "#1baf7a", "-."),
             ("Raw eGeMAPS acoustic features (88)", "G", "mean", "#eda100", ":")]
    fig, ax = plt.subplots(figsize=(4.6, 4.4), dpi=300)
    ax.plot([0, 1], [0, 1], color="#b4b3ad", lw=1)
    for name, mods, agg, color, ls in specs:
        X, lab = participant_matrix(df, mods, agg)
        y = lab.y.values
        p = cv_probs(X, y)
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=color, lw=2, ls=ls, label=f"{name}: AUC {roc_auc_score(y, p):.2f}")
    ax.set_xlabel("1 − specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#ecebe7", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=6.6, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"Figure2.{ext}"), dpi=300)

    E3 = pd.read_csv(os.path.join(a.results, "E3_affective_profiles.csv"))
    E3["d"] = E3.mean_dep - E3.mean_ctrl
    cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f19a8a", "#c0392b"])
    vmax = np.abs(E3.d).max()
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), dpi=300, gridspec_kw={"wspace": 0.35})
    names = {"audio": ("Audio emotion probabilities", ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]),
             "text": ("Text emotion probabilities", ["Anger", "Fear", "Happy", "Neutral", "Sad", "Surprise"])}
    rows = ["positive", "neutral", "negative"]
    for ax, mod in zip(axs, ["audio", "text"]):
        S = E3[E3.modality == mod]
        D = S.pivot(index="condition", columns="emotion", values="d").reindex(rows)
        P = S.pivot(index="condition", columns="emotion", values="p").reindex(rows)
        im = ax.imshow(D.values, cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
        for i in range(3):
            for j in range(6):
                v = D.values[i, j]
                ax.text(j, i, f"{v:+.2f}" + ("*" if P.values[i, j] < 0.05 else ""), ha="center", va="center",
                        fontsize=7, color="#ffffff" if abs(v) > 0.6 * vmax else "#2b2b29")
        ax.set_xticks(range(6), names[mod][1], rotation=30, ha="right")
        ax.set_yticks(range(3), ["Positive Q", "Neutral Q", "Negative Q"])
        ax.set_title(names[mod][0], fontsize=9, color="#2b2b29")
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
    cb.set_label("Depressed − control (mean probability)", fontsize=7)
    cb.outline.set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"Figure3.{ext}"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
