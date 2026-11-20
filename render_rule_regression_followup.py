"""Render audited P0-P2 follow-up figures and a consolidated report."""

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import mean_absolute_error

SKILL = Path("C:/Users/L2791/.codex/skills/scipilot-figure-skill/scripts")
sys.path.insert(0, str(SKILL))
from setup_style import setup_style
from visual_qa import audit_layout

OUT = Path("ml_rule_regression_followup_100_v1")
BASE = Path("ml_rule_regression_100_v1")
COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}
MARKERS = {1: "o", 2: "s", 3: "^"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(fig, name, qa):
    fig.set_constrained_layout_pads(w_pad=0.13, h_pad=0.13, wspace=0.09, hspace=0.09)
    fig.canvas.draw()
    qa[name] = audit_layout(fig)
    fig.savefig(OUT / f"{name}.png", dpi=300)
    image = Image.open(OUT / f"{name}.png").convert("RGB")
    image.thumbnail((1600, 1100))
    image.save(OUT / f"{name}_preview.png")
    image.convert("L").save(OUT / f"{name}_gray.png")
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def main():
    f500 = read(OUT / "f500_audit.json")
    errors = read(OUT / "f_number_error_audit.json")
    baseline = pd.DataFrame(read(OUT / "per_rule_baseline_comparison.json"))
    log_compare = pd.DataFrame(read(OUT / "log_target_comparison.json"))
    log_predictions = pd.DataFrame(read(OUT / "log_target_predictions.json"))
    sensitivity = pd.DataFrame(read(OUT / "rf_sensitivity_metrics.json"))
    sensitivity_predictions = pd.DataFrame(read(OUT / "rf_sensitivity_predictions.json"))
    direct_predictions = pd.DataFrame(read(BASE / "predictions.json"))

    setup_style(journal="general", lang="zh", use_sciplots=False)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "svg.fonttype": "none"})
    qa = {}

    direct_f = direct_predictions[(direct_predictions.target == "final_f_number") &
                                  (direct_predictions.model == "random_forest") &
                                  (direct_predictions.feature_set == "raw9")].copy()
    direct_f["absolute_error"] = (direct_f.predicted - direct_f.actual).abs()
    direct_f["is_f500"] = direct_f.actual == 500
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), layout="constrained")
    rng = np.random.default_rng(20261118)
    for rule in (1, 2, 3):
        group = direct_f[direct_f.rule == rule]
        jitter = rng.uniform(-0.16, 0.16, len(group))
        axes[0].scatter(np.full(len(group), rule) + jitter, group.absolute_error,
                        s=20, alpha=.65, color=COLORS[rule], marker=MARKERS[rule])
    special = direct_f[direct_f.is_f500]
    axes[0].scatter(np.full(len(special), 2), special.absolute_error, s=62,
                    facecolors="none", edgecolors="black", marker="o", linewidths=1.1,
                    label="F/#=500")
    axes[0].set(xticks=[1, 2, 3], xticklabels=["Rule 1", "Rule 2", "Rule 3"],
                ylabel="F/# 绝对误差（对数轴）")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False)
    for rule in (1, 2, 3):
        group = direct_f[direct_f.rule == rule]
        axes[1].scatter(group.case_id, group.absolute_error, s=21, alpha=.68,
                        color=COLORS[rule], marker=MARKERS[rule], label=f"Rule {rule}")
    axes[1].scatter(special.case_id, special.absolute_error, s=65, facecolors="none",
                    edgecolors="black", linewidths=1.1)
    axes[1].set(xlabel="处方 case_id", ylabel="F/# 绝对误差（对数轴）")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False, ncol=3)
    for ax in axes:
        ax.grid(alpha=.18)
    save(fig, "01_f_number_error_distribution", qa)

    best_models = baseline[((baseline.model == "random_forest") & (baseline.feature_set == "raw9")) |
                           ((baseline.model == "ridge") &
                            (baseline.feature_set == "raw9_plus_physics6"))].copy()
    best_models["label"] = np.where(best_models.model == "random_forest", "RF / 9", "Ridge / 9+6")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), layout="constrained")
    for ax, target, title in zip(axes, ("final_f_number", "final_field_deg"),
                                 ("最终 F/#", "最终最大视场")):
        subset = best_models[best_models.target == target]
        for index, label in enumerate(("Ridge / 9+6", "RF / 9")):
            group = subset[subset.label == label].sort_values("rule")
            offset = (index - .5) * .12
            ax.scatter(group.rule + offset, group.delta_mae_candidate_minus_baseline,
                       s=42, marker=("o" if index == 0 else "s"),
                       color=("#E69F00" if index == 0 else "#0072B2"), label=label)
        ax.axhline(0, color="black", lw=.8)
        ax.set(xticks=[1, 2, 3], xticklabels=["Rule 1", "Rule 2", "Rule 3"],
               ylabel="ΔMAE：模型 - 仅规则均值", title=title)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False)
    save(fig, "02_per_rule_baseline_gain", qa)

    f_log = log_compare[log_compare.target == "final_f_number"].copy()
    f_log["label"] = f_log.apply(lambda row: ("Ridge" if row.model == "ridge" else "RF") +
                                 (" / 9" if row.feature_set == "raw9" else " / 9+6"), axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), layout="constrained")
    x = np.arange(len(f_log))
    for ax, metric_name, title in ((axes[0], "mae", "F/# MAE"),
                                   (axes[1], "rmse", "F/# RMSE")):
        direct = f_log[f"direct_{metric_name}"].to_numpy()
        transformed = f_log[f"log_target_{metric_name}"].to_numpy()
        for index in range(len(x)):
            ax.plot([index - .08, index + .08], [direct[index], transformed[index]],
                    color="#777777", lw=.8)
        ax.scatter(x - .08, direct, s=38, color="#999999", marker="o", label="原始目标")
        ax.scatter(x + .08, transformed, s=38, color="#0072B2", marker="s", label="log 目标")
        ax.set_xticks(x, f_log.label, rotation=22, ha="right")
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False)
    save(fig, "03_log_target_tradeoff", qa)

    valid = sensitivity[(sensitivity.split == "validation") &
                        (sensitivity.scope == "overall")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), layout="constrained")
    depth_values = [4, 8, None]
    depth_labels = ["4", "8", "不限"]
    leaf_values = [1, 3, 8]
    for ax, target, title in zip(axes, ("final_f_number", "final_field_deg"),
                                 ("最终 F/#：OOF MAE", "最大视场：OOF MAE / °")):
        matrix = np.zeros((len(depth_values), len(leaf_values)))
        for i, depth in enumerate(depth_values):
            for j, leaf in enumerate(leaf_values):
                match = valid[(valid.target == target) & (valid.min_samples_leaf == leaf)]
                match = match[match.max_depth.isna()] if depth is None else match[match.max_depth == depth]
                matrix[i, j] = match.iloc[0].mae
        image = ax.pcolormesh(np.arange(4) - 0.5, np.arange(4) - 0.5, matrix,
                              cmap="viridis_r", shading="flat")
        ax.set_aspect("auto")
        ax.invert_yaxis()
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                midpoint = (matrix.min() + matrix.max()) / 2
                ax.text(j, i, f"{matrix[i, j]:.3f}" if target == "final_field_deg" else
                        f"{matrix[i, j]:.2f}", ha="center", va="center",
                        color="white" if matrix[i, j] > midpoint else "black")
        ax.set(xticks=range(3), xticklabels=leaf_values,
               yticks=range(3), yticklabels=depth_labels,
               xlabel="min_samples_leaf", ylabel="max_depth", title=title)
        fig.colorbar(image, ax=ax, label="MAE")
    save(fig, "04_rf_sensitivity", qa)

    (OUT / "figure_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")

    # Fold-wise comparison for sensitivity; descriptive only, not model selection.
    fold_sensitivity = []
    for target in ("final_f_number", "final_field_deg"):
        subset = sensitivity_predictions[sensitivity_predictions.target == target]
        current = subset[subset.max_depth.isna() & (subset.min_samples_leaf == 3)]
        leaf1 = subset[subset.max_depth.isna() & (subset.min_samples_leaf == 1)]
        for fold in range(1, 6):
            a = current[current.fold == fold]
            b = leaf1[leaf1.fold == fold]
            fold_sensitivity.append({"target": target, "fold": fold,
                                     "current_leaf3_mae": float(mean_absolute_error(a.actual, a.predicted)),
                                     "leaf1_mae": float(mean_absolute_error(b.actual, b.predicted))})
    (OUT / "rf_leaf1_vs_leaf3_by_fold.json").write_text(
        json.dumps(fold_sensitivity, indent=2) + "\n", encoding="utf-8")

    rf9 = baseline[(baseline.model == "random_forest") & (baseline.feature_set == "raw9")]
    lines = [
        "# 第一版回归后续核查：P0–P2", "",
        "## 总结", "",
        "F/#=500 不是失败哨兵，也不是 OpticStudio 求解器返回的特殊值。它是 V1 明确设置的初始 F/#。11 个相关终点全部属于 Rule 2：视场阶段先停止，孔径阶段没有启动，因此 F/# 保持初值。处方文件经过保存重载，并通过当前几何和三视场 MTF 检查；从光学验收看它们有效，从学习目标看则是优化策略造成的边界截断值。", "",
        "最佳 RF/9 的 F/# 误差高度集中在 Rule 2 和这 11 个边界样本。log 目标降低了典型样本 MAE，却进一步低估 F/#=500，使总体 RMSE 与 R²变差。RF 结果对叶节点最小样本数有中等敏感性：`min_samples_leaf=1` 在当前五折上优于 3，但这属于同一验证集上的敏感性观察，不能作为无偏调参结论。", "",
        "## P0：F/#=500 来源", "",
        "| 判定项 | 结果 |", "|---|---|",
        "| 特殊失败/缺失哨兵 | 否 |",
        "| OpticStudio 显式上界 | 否 |",
        "| V1 初始连续化边界 | 是，`START_F=500` |",
        "| 当前几何和 MTF 验收有效 | 是，11/11 |",
        "| 独立优化得到的孔径最优值 | 否，Rule 2 未启动孔径阶段 |", "",
        "11 套分别为 case 5、11、25、41、44、57、66、78、85、92、95。最终视场为 2.827°–4.964°；最小几何余量均为正，三视场 MTF 均通过。每组结果哈希、停止原因、MTF 余量和处方重载核查见 `f500_audit.json`。", "",
        "这类值同时具有两层含义：它是一个真实可用的已验收光学终点，但其数值由规则控制流截断，不表示该处方的最佳 F/# 恰好为 500。后续建模可考虑把‘孔径阶段是否启动/边界状态’作为单独过程建模；本轮保留原始标签，不改变数据。", "",
        "## P0：F/# 误差分布", "",
        "| 子集 | n | MAE | RMSE | 绝对误差占比 | 平方误差占比 |", "|---|---:|---:|---:|---:|---:|",
        f"| F/#=500 | {errors['f500_subset']['n']} | {errors['f500_subset']['mae']:.2f} | {errors['f500_subset']['rmse']:.2f} | {errors['f500_subset']['share_total_absolute_error']:.1%} | {errors['f500_subset']['share_total_squared_error']:.1%} |",
        f"| 非 500 | {errors['non_f500']['n']} | {errors['non_f500']['mae']:.2f} | {errors['non_f500']['rmse']:.2f} | — | — |", "",
        "| 规则 | MAE | 中位绝对误差 | P90 绝对误差 | 最大误差 | 绝对误差占比 | 平方误差占比 |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in errors["by_rule"]:
        lines.append(f"| Rule {row['rule']} | {row['mae']:.2f} | {row['median_absolute_error']:.2f} | {row['p90_absolute_error']:.2f} | {row['maximum_absolute_error']:.2f} | {row['share_total_absolute_error']:.1%} | {row['share_total_squared_error']:.1%} |")
    lines += ["", "全体绝对误差中位数为 6.74，P90 为 70.22，最大值为 365.23。误差最大的七行均为 Rule 2 的 F/#=500；case 92、85、41 的绝对误差分别为 365.23、355.76、334.80。非 500 子集 R²为负，原因是模型仍受训练折中的 500 边界混合影响，而移除边界后剩余目标方差显著缩小；这不是建议删除 500 的依据。", "",
              "![F/# 误差分布](01_f_number_error_distribution.png)", "",
              "## P1：逐规则基线比较", "",
              "下表比较 RF/9 与训练折内逐规则均值。负 ΔMAE/ΔRMSE、正 ΔR²表示模型更好。", "",
              "| 输出 | 规则 | 基线 MAE | RF MAE | ΔMAE | MAE 优胜折 | 基线 RMSE | RF RMSE | ΔR² | R²优胜折 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, row in rf9.sort_values(["target", "rule"]).iterrows():
        target = "F/#" if row.target == "final_f_number" else "视场"
        lines.append(f"| {target} | {int(row.rule)} | {row.baseline_mae:.3f} | {row.candidate_mae:.3f} | {row.delta_mae_candidate_minus_baseline:.3f} | {int(row.folds_better_mae)}/5 | {row.baseline_rmse:.3f} | {row.candidate_rmse:.3f} | {row.delta_r2_candidate_minus_baseline:.3f} | {int(row.folds_better_r2)}/5 |")
    lines += ["", "RF/9 在六个‘输出 × 规则’组合中均降低 pooled MAE；Rule 2、3 的 F/# 三项指标均为 5/5 折优胜。Rule 1 F/# 为 4/5，视场的 RMSE/R²为 4/5。由此可确认模型学到了处方内信息，而非只复现规则平均值。Rule 2/3 视场本身接近常数，绝对提升较小，解释时仍应看逐规则 R²和真实变化范围。Ridge 与两套特征的完整比较保存在 `per_rule_baseline_comparison.json`。", "",
              "![逐规则基线增益](02_per_rule_baseline_gain.png)", "",
              "## P1：log 目标回归", "",
              "所有模型在训练折对目标取自然对数，预测后指数变换回原始尺度；以下 MAE、RMSE、R²仍在原始 F/# 尺度计算。", "",
              "| 模型 | 特征 | 原始 MAE | log MAE | 原始 RMSE | log RMSE | 原始 R² | log R² |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, row in log_compare[log_compare.target == "final_f_number"].iterrows():
        lines.append(f"| {row.model} | {row.feature_set} | {row.direct_mae:.2f} | {row.log_target_mae:.2f} | {row.direct_rmse:.2f} | {row.log_target_rmse:.2f} | {row.direct_r2:.3f} | {row.log_target_r2:.3f} |")
    lines += ["", "RF/9 的 MAE 从 25.62 降至 22.17，但 RMSE 从 59.27 升至 69.83，R²从 0.567 降至 0.399。非 500 样本 MAE 从 17.38 降至 10.08，而 500 样本 MAE 从 241.98 升至 339.77。log 回归确实改善主体分布，但更强地压缩尾部，不能替换原始目标模型；它更适合作为稳健误差偏好的候选或双阶段模型的一部分。视场没有长尾问题，log 目标的四个配置均未改善总体原始尺度指标。", "",
              "![log 目标权衡](03_log_target_tradeoff.png)", "",
              "## P2：Random Forest 参数敏感性", "",
              "固定原始 9 参数、500 棵树、`max_features=sqrt`，只改变 `max_depth ∈ {4,8,不限}` 和 `min_samples_leaf ∈ {1,3,8}`。当前模型为深度不限、叶节点最少 3。", "",
              "| 输出 | 配置 | MAE | RMSE | R² |", "|---|---|---:|---:|---:|"]
    valid = sensitivity[(sensitivity.split == "validation") & (sensitivity.scope == "overall")]
    for target in ("final_f_number", "final_field_deg"):
        for _, row in valid[valid.target == target].sort_values("mae").iterrows():
            depth = "不限" if pd.isna(row.max_depth) else str(int(row.max_depth))
            label = "F/#" if target == "final_f_number" else "视场"
            lines.append(f"| {label} | depth={depth}, leaf={int(row.min_samples_leaf)} | {row.mae:.3f} | {row.rmse:.3f} | {row.r2:.3f} |")
    lines += ["", "深度从不限限制到 8 的影响很小；深度 4 明显欠拟合。叶节点从 3 降到 1 后，F/# MAE 25.62 → 23.16、R² 0.567 → 0.602；视场 MAE 0.365 → 0.342、R² 0.815 → 0.827，并且两个输出的 MAE 都在五折中 5/5 优于当前 leaf=3。leaf=8 明显退化。", "",
              "这说明当前结论受叶节点平滑强度影响，但模型家族的总体结论没有反转：RF 仍优于规则均值，F/# 尾部仍是主要困难。因为参数敏感性使用了同一五折结果，leaf=1 不能被报告为无偏‘最优参数’；应冻结候选后用未见的第二批数据一次性验证。", "",
              "![RF 参数敏感性](04_rf_sensitivity.png)", "",
              "## 文件与限制", "",
              "本轮没有读取第二批数据、没有运行新的 Zemax 优化，也没有修改第一版结果。`f500_audit.json` 保存逐组物理验收证据；`f_number_error_audit.json` 保存误差贡献；log 和 RF 敏感性的逐行 OOF 预测及指标均保存在同目录。参数敏感性属于模型分析，不构成已完成调参。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"figure_qa": qa, "report": str(OUT / "report.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
