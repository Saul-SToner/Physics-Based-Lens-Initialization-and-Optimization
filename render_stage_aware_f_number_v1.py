"""Render compact, publication-safe diagnostics for the Stage-Aware experiment."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ml_stage_aware_f_number_100_v1"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none"})


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    p = pd.DataFrame(read("stage_predictions.json"))
    colors = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}
    rng = np.random.default_rng(20261118)
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    for rule in (1, 2, 3):
        g = p[p.rule == rule]
        x = (1 - g.entered_aperture.to_numpy()) + rng.uniform(-.08, .08, len(g))
        ax.scatter(x, g.stage_probability_entered, s=22, alpha=.72,
                   color=colors[rule], label=f"Rule {rule}", edgecolors="none")
    ax.axhline(.5, color="#555555", lw=1, ls="--", label="0.5 threshold")
    ax.set(xticks=[0, 1], xticklabels=["Entered aperture\n(n=289)", "Did not enter\n(n=11)"],
           ylabel="OOF probability: entered aperture", ylim=(-.04, 1.04))
    ax.legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.01))
    ax.set_title("Stage status is separable, but the minority class is very small", loc="left", fontweight="bold")
    save(fig, "stage_probability")

    models = ["rf9_leaf3", "rf9_leaf1", "stage_aware_hard", "stage_aware_soft"]
    labels = ["RF\nleaf=3", "RF\nleaf=1", "Stage-aware\nhard", "Stage-aware\nsoft"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=True)
    for ax, subset, title in zip(axes, [~p.is_f500, p.is_f500], ["Non-500 endpoints (n=289)", "F/#=500 endpoints (n=11)"]):
        vals = [np.abs(p.loc[subset, m] - p.loc[subset, "final_f_number"]) for m in models]
        ax.boxplot(vals, positions=np.arange(4), widths=.55, showfliers=False,
                   medianprops={"color": "#111111", "linewidth": 1.3},
                   boxprops={"color": "#777777"}, whiskerprops={"color": "#777777"},
                   capprops={"color": "#777777"})
        for i, v in enumerate(vals):
            ax.scatter(i + rng.uniform(-.13, .13, len(v)), v, s=10, alpha=.35, color="#0072B2")
        ax.set(xticks=np.arange(4), xticklabels=labels, title=title, yscale="symlog", ylim=(-1, 700))
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Absolute F/# error (symlog scale)")
    fig.suptitle("Stage gating reduces F/#=500 error but creates costly false gates", x=.08, ha="left", fontweight="bold")
    save(fig, "regression_error_comparison")


if __name__ == "__main__":
    main()
