"""Render OOF error/decision diagnostics and paired bootstrap uncertainty."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ml_v2_oof_error_decision_analysis"
COLORS = {"Rule 1": "#0072B2", "Rule 2": "#D55E00", "Rule 3": "#009E73",
          "Predicted": "#CC79A7", "Oracle": "#222222"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none", "axes.unicode_minus": False})


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    Image.open(OUT / f"{stem}.png").convert("L").save(OUT / f"{stem}_grayscale_qa.png")


def ecdf(values):
    x = np.sort(np.asarray(values))
    return x, np.arange(1, len(x) + 1) / len(x)


def main():
    rows = pd.DataFrame(read("analysis_rows.json"))
    decisions = pd.DataFrame(read("decision_by_prescription.json"))
    rng = np.random.default_rng(20261118)
    delta = decisions.actual_gap_selected.to_numpy() - decisions.actual_gap_rule_3.to_numpy()
    boots = np.asarray([np.mean(rng.choice(delta, len(delta), replace=True)) for _ in range(20000)])
    regret = decisions.gap_regret.to_numpy()
    regret_boots = np.asarray([np.mean(rng.choice(regret, len(regret), replace=True)) for _ in range(20000)])
    uncertainty = {"paired_comparison": "predicted target-gap policy minus fixed Rule 3",
                   "mean_delta": float(delta.mean()),
                   "bootstrap_95_percentile_ci": list(map(float, np.quantile(boots, [.025, .975]))),
                   "prescriptions_better": int(np.sum(delta < -1e-12)),
                   "prescriptions_equal": int(np.sum(np.abs(delta) <= 1e-12)),
                   "prescriptions_worse": int(np.sum(delta > 1e-12)),
                   "mean_oracle_regret": float(regret.mean()),
                   "oracle_regret_bootstrap_95_percentile_ci": list(map(float, np.quantile(regret_boots, [.025, .975]))),
                   "bootstrap_unit": "prescription", "bootstrap_replicates": 20000,
                   "interpretation": "Exploratory paired uncertainty on the same OOF prescriptions; not an independent confirmation."}
    (OUT / "decision_uncertainty.json").write_text(json.dumps(uncertainty, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.25), constrained_layout=True)
    for label, mask, color, marker in (("F/#=500 (n=31)", rows.is_f500, "#D55E00", "o"),
                                        ("Non-500 (n=569)", ~rows.is_f500, "#0072B2", "s")):
        x, y = ecdf(rows.loc[mask, "f_abs_error"])
        axes[0].step(x, y, where="post", color=color, label=label, lw=1.7)
        axes[0].scatter(x[::max(1, len(x)//12)], y[::max(1, len(y)//12)], color=color, marker=marker, s=14)
    axes[0].set(xscale="symlog", xlabel="Absolute F/# error (symlog)", ylabel="Cumulative fraction",
                title="a  F/# tail is dominated by 500 endpoints", ylim=(0, 1.03))
    axes[0].legend(frameon=False, loc="lower right")
    values = [rows.loc[rows.rule == r, "field_abs_error"].to_numpy() for r in (1, 2, 3)]
    bp = axes[1].boxplot(values, showfliers=False, widths=.55, patch_artist=False,
                         medianprops={"color": "#111111", "linewidth": 1.2})
    for i, (r, vals) in enumerate(zip((1, 2, 3), values), 1):
        axes[1].scatter(i + rng.uniform(-.12, .12, len(vals)), vals, s=8, alpha=.28,
                        color=COLORS[f"Rule {r}"], label=f"Rule {r}")
    axes[1].set(yscale="symlog", xticks=[1, 2, 3], xticklabels=["Rule 1", "Rule 2", "Rule 3"],
                ylabel="Absolute field error (deg, symlog)", title="b  Field error is concentrated in Rule 1")
    save(fig, "error_mechanisms")

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.25), constrained_layout=True)
    policies = [("Rule 1", decisions.actual_gap_rule_1), ("Rule 2", decisions.actual_gap_rule_2),
                ("Rule 3", decisions.actual_gap_rule_3), ("Predicted", decisions.actual_gap_selected),
                ("Oracle", decisions.actual_gap_oracle)]
    styles = {"Rule 1": "--", "Rule 2": ":", "Rule 3": "-.", "Predicted": "-", "Oracle": "-"}
    for label, vals in policies:
        x, y = ecdf(vals)
        axes[0].step(x, y, where="post", color=COLORS[label], ls=styles[label], lw=1.6, label=label)
    axes[0].set(xscale="symlog", xlabel="Actual normalized target gap (lower is better)",
                ylabel="Cumulative fraction", title="a  OOF selection modestly improves policy value", ylim=(0, 1.03))
    axes[0].legend(frameon=False, ncol=2, loc="lower right")
    order = np.argsort(delta)
    axes[1].scatter(np.arange(len(delta)), delta[order], s=12,
                    c=np.where(delta[order] < -1e-12, "#009E73", np.where(delta[order] > 1e-12, "#D55E00", "#999999")),
                    alpha=.8)
    axes[1].axhline(0, color="#222222", lw=1)
    axes[1].set(yscale="symlog", xlabel="Prescriptions sorted by paired difference",
                ylabel="Gap difference: predicted - fixed Rule 3",
                title="b  Benefit occurs in 22 cases; harm in 12")
    save(fig, "decision_feasibility")


if __name__ == "__main__":
    main()
