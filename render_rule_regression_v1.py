"""Create audited tables, plots, and report for the V1 regression experiment."""

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import sklearn

SKILL = Path("C:/Users/L2791/.codex/skills/scipilot-figure-skill/scripts")
sys.path.insert(0, str(SKILL))
from profile_data import profile_data
from setup_style import setup_style
from visual_qa import audit_layout


OUT = Path("ml_rule_regression_100_v1")
COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}
MARKERS = {1: "o", 2: "s", 3: "^"}
CONFIGS = [
    ("dummy_mean", "rule_only", "Dummy 均值"),
    ("rule_mean", "rule_only", "仅规则均值"),
    ("ridge", "raw9", "Ridge / 9"),
    ("ridge", "raw9_plus_physics6", "Ridge / 9+6"),
    ("random_forest", "raw9", "RF / 9"),
    ("random_forest", "raw9_plus_physics6", "RF / 9+6"),
]


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def key(model, feature_set):
    return model + "__" + feature_set


def fmt(value):
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def save(fig, name, qa):
    fig.set_constrained_layout_pads(w_pad=0.12, h_pad=0.12, wspace=0.08, hspace=0.08)
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
    predictions = pd.DataFrame(read("predictions.json"))
    metrics = pd.DataFrame(read("metrics.json"))
    fold_metrics = pd.DataFrame(read("fold_metrics.json"))
    comparisons = pd.DataFrame(read("baseline_comparisons.json"))
    pairwise = pd.DataFrame(read("pairwise_rule_differences.json"))
    manifest = read("dataset_manifest.json")

    # Skill-required profile of the actual analysis table, with IDs excluded from correlation.
    feature_rows = json.loads(Path("feature_engineering_100_v1/features.json").read_text())
    outcomes = json.loads(Path("feature_engineering_100_v1/outcomes.json").read_text())
    by_case = {row["case_id"]: row["features"] for row in feature_rows}
    profile_frame = pd.DataFrame([{
        "case_id": str(row["case_id"]), "rule": str(row["rule"]),
        **{name: by_case[row["case_id"]][name] for name in manifest["raw9"] + manifest["physics6_added"]},
        "final_f_number": row["final_working_f_number"],
        "final_field_deg": row["final_max_field_deg"],
    } for row in outcomes])
    profile = profile_data(profile_frame.drop(columns="case_id"), group_cols=["rule"])
    (OUT / "data_profile.json").write_text(json.dumps(profile, indent=2, default=str) + "\n", encoding="utf-8")

    setup_style(journal="general", lang="zh", use_sciplots=False)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "svg.fonttype": "none"})
    qa = {}

    target_meta = {
        "final_f_number": ("最终 F/#", "预测 F/#"),
        "final_field_deg": ("最终最大视场 / °", "预测最大视场 / °"),
    }
    for target, (actual_label, predicted_label) in target_meta.items():
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0), layout="constrained")
        for ax, (model, feature_set, label) in zip(axes.flat, CONFIGS):
            data = predictions[(predictions.target == target) &
                               (predictions.model == model) &
                               (predictions.feature_set == feature_set)]
            for rule in (1, 2, 3):
                group = data[data.rule == rule]
                ax.scatter(group.actual, group.predicted, s=18, alpha=0.62,
                           color=COLORS[rule], marker=MARKERS[rule], label=f"Rule {rule}")
            low = min(data.actual.min(), data.predicted.min())
            high = max(data.actual.max(), data.predicted.max())
            pad = max((high - low) * 0.04, 0.05)
            ax.plot([low - pad, high + pad], [low - pad, high + pad],
                    color="black", lw=0.8, ls="--")
            ax.set_xlim(data.actual.min() - pad, data.actual.max() + pad)
            ax.set_ylim(low - pad, high + pad)
            row = metrics[(metrics.target == target) & (metrics.model == model) &
                          (metrics.feature_set == feature_set) &
                          (metrics.split == "validation") & (metrics.scope == "overall")].iloc[0]
            ax.set_title(f"{label}\nMAE={fmt(row.mae)}, R²={row.r2:.3f}")
            ax.grid(alpha=0.18)
        for ax in axes[:, 0]:
            ax.set_ylabel(predicted_label)
        for ax in axes[1, :]:
            ax.set_xlabel(actual_label)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
        save(fig, f"01_predictions_{target}", qa)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), layout="constrained")
    x = np.arange(len(CONFIGS))
    for row_index, target in enumerate(target_meta):
        for col_index, metric_name in enumerate(("mae", "rmse")):
            ax = axes[row_index, col_index]
            for split, color, marker, offset, label in (
                ("train", "#999999", "o", -0.07, "训练"),
                ("validation", "#0072B2", "s", 0.07, "验证 OOF"),
            ):
                values = []
                for model, feature_set, _ in CONFIGS:
                    item = metrics[(metrics.target == target) & (metrics.model == model) &
                                   (metrics.feature_set == feature_set) &
                                   (metrics.split == split) & (metrics.scope == "overall")].iloc[0]
                    values.append(item[metric_name])
                ax.scatter(x + offset, values, color=color, marker=marker, s=34, label=label, zorder=3)
            for index, (model, feature_set, _) in enumerate(CONFIGS):
                vals = []
                for split in ("train", "validation"):
                    item = metrics[(metrics.target == target) & (metrics.model == model) &
                                   (metrics.feature_set == feature_set) &
                                   (metrics.split == split) & (metrics.scope == "overall")].iloc[0]
                    vals.append(item[metric_name])
                ax.plot([index - 0.07, index + 0.07], vals, color="#666666", lw=0.8)
            ax.set_xticks(x, [label for _, _, label in CONFIGS], rotation=25, ha="right")
            ax.set_ylabel(("F/#" if target == "final_f_number" else "视场 / °") +
                          (" MAE" if metric_name == "mae" else " RMSE"))
            ax.grid(axis="y", alpha=0.2)
            if row_index == 0 and col_index == 0:
                ax.legend(frameon=False)
    save(fig, "02_train_validation_errors", qa)

    # Best endpoint model by pooled OOF MAE: RF/raw9 for both targets.
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6), layout="constrained")
    best = predictions[(predictions.model == "random_forest") &
                       (predictions.feature_set == "raw9")]
    for row_index, target in enumerate(target_meta):
        table = best[best.target == target].pivot(index="case_id", columns="rule",
                                                  values=["actual", "predicted"])
        for col_index, (a, b) in enumerate(combinations((1, 2, 3), 2)):
            ax = axes[row_index, col_index]
            actual = table["actual", a] - table["actual", b]
            predicted = table["predicted", a] - table["predicted", b]
            ax.scatter(actual, predicted, s=18, alpha=0.65, color="#0072B2")
            low = min(actual.min(), predicted.min())
            high = max(actual.max(), predicted.max())
            pad = max((high - low) * 0.04, 0.02)
            ax.plot([low - pad, high + pad], [low - pad, high + pad],
                    color="black", lw=0.8, ls="--")
            ax.axhline(0, color="#777777", lw=0.5)
            ax.axvline(0, color="#777777", lw=0.5)
            ax.set_title(f"Rule {a} - Rule {b}")
            unit = "F/#" if target == "final_f_number" else "视场 / °"
            ax.set_xlabel(f"真实 {unit} 规则差")
            ax.set_ylabel(f"预测 {unit} 规则差")
            ax.grid(alpha=0.18)
    save(fig, "03_pairwise_rule_differences", qa)

    (OUT / "figure_qa.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    valid = metrics[(metrics.split == "validation") & (metrics.scope == "overall")]
    per_rule = metrics[(metrics.split == "validation") & (metrics.scope != "overall")]
    lines = [
        "# Zemax 优化规则条件化回归：第一版实验", "",
        "## 结论", "",
        "第一批 100 套处方已经包含可预测信号。Random Forest 使用原始 9 参数时，两项输出都稳定优于仅按规则编号取训练折均值的基线；但性能还不足以把模型当作高精度 Zemax 替代器。F/# 的困难主要集中在 Rule 2 的长尾极值，视场总体 R² 又部分来自三条规则之间本来就很大的均值差异。", "",
        "物理派生特征对 Ridge 明显有帮助，对 Random Forest 没有带来总体改善；因此不能笼统地说派生特征有效。Ridge 还产生了负 F/#、负视场及超出合理范围的外推，当前更适合解释线性增量，而不适合直接部署。", "",
        "## 数据与验证设计", "",
        "只读取 `feature_engineering_100_v1`：100 个处方 × 3 条规则，共 300 行。输入只含优化前参数和规则编号；输出为最终 F/#、最大视场。第二批数据未读取，历史结果未修改，也未生成最优规则标签。", "",
        "原始特征为 5 个采样参数和论文映射得到的 4 个参数。扩展组增加 6 个无量纲量：间距/|EFL|、两片功率比与功率平衡、初始空气间隔/|EFL|、物距/|EFL|、像距/|EFL|。规则采用 one-hot 条件展开并与各特征相乘，使同一模型可学习规则特异的斜率。", "",
        "采用固定随机种子的五折 GroupKFold，每折验证 20 套完整处方（60 行），同一处方的三条规则始终位于同一折。标准化仅在 Ridge 的训练折拟合。Random Forest 固定 500 棵树、叶节点最少 3 个样本、`max_features=sqrt`；未调参。", "",
        "除标准 Dummy 均值外，加入了训练折内逐规则均值基线。后者才是‘仅根据规则编号预测’的直接比较对象。", "",
        "## 总体 OOF 指标", "",
        "| 输出 | 模型 | 特征 | MAE | RMSE | R² |", "|---|---|---|---:|---:|---:|",
    ]
    for target in target_meta:
        for model, feature_set, label in CONFIGS:
            row = valid[(valid.target == target) & (valid.model == model) &
                        (valid.feature_set == feature_set)].iloc[0]
            lines.append(f"| {target_meta[target][0]} | {label} | {feature_set} | {fmt(row.mae)} | {fmt(row.rmse)} | {row.r2:.3f} |")
    lines += ["", "OOF 表示每个处方只由未见过该处方的训练折模型预测一次。", "",
        "各预测散点面板均画出 y=x 参考线；为了完整显示 Ridge 的非物理外推，各面板纵轴独立取值，比较模型时应以标注指标和点到参考线的距离为准。", "",
        "![F/# 预测散点](01_predictions_final_f_number.png)", "",
              "![视场预测散点](01_predictions_final_field_deg.png)", "",
              "## 逐规则 OOF 指标", "",
              "| 输出 | 模型 | 特征 | 规则 | MAE | RMSE | R² |", "|---|---|---|---:|---:|---:|---:|"]
    for target in target_meta:
        for model, feature_set, label in CONFIGS:
            for rule in (1, 2, 3):
                row = per_rule[(per_rule.target == target) & (per_rule.model == model) &
                               (per_rule.feature_set == feature_set) &
                               (per_rule.scope == f"rule_{rule}")].iloc[0]
                lines.append(f"| {target_meta[target][0]} | {label} | {feature_set} | {rule} | {fmt(row.mae)} | {fmt(row.rmse)} | {row.r2:.3f} |")
    lines += ["", "Rule 2 的 F/# 最难：最佳 RF/9 的 MAE 为 54.68、RMSE 为 98.49。Rule 1 的视场最难：最佳 RF/9 的 MAE 为 0.900°；Rule 2/3 多数终点接近 5°，所以绝对误差小，但这不代表模型掌握了丰富的组内变化。", "",
              "## 训练与验证误差", "",
              "| 输出 | 模型 | 特征 | 训练 MAE | 验证 MAE | 训练 RMSE | 验证 RMSE |", "|---|---|---|---:|---:|---:|---:|"]
    for target in target_meta:
        for model, feature_set, label in CONFIGS:
            train = metrics[(metrics.target == target) & (metrics.model == model) &
                            (metrics.feature_set == feature_set) & (metrics.split == "train") &
                            (metrics.scope == "overall")].iloc[0]
            val = valid[(valid.target == target) & (valid.model == model) &
                        (valid.feature_set == feature_set)].iloc[0]
            lines.append(f"| {target_meta[target][0]} | {label} | {feature_set} | {fmt(train.mae)} | {fmt(val.mae)} | {fmt(train.rmse)} | {fmt(val.rmse)} |")
    lines += ["", "训练指标把五个训练折的预测合并计算；同一记录会在四个训练折中出现，因此只用于诊断拟合差距，不能当作独立泛化证据。RF 的 F/# 训练—验证差距较明显，物理扩展后的验证 RMSE反而增大，提示有限样本下的方差问题。", "",
              "![训练与验证误差](02_train_validation_errors.png)", "",
              "## 是否优于仅规则编号基线", "",
              "| 输出 | 模型 | 特征 | pooled ΔMAE | MAE 优胜折 | pooled ΔRMSE | RMSE 优胜折 | pooled ΔR² | R² 优胜折 |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, row in comparisons.sort_values(["target", "model", "feature_set"]).iterrows():
        lines.append(f"| {target_meta[row.target][0]} | {row.model} | {row.feature_set} | {row.pooled_delta_mae_candidate_minus_rule_baseline:.3f} | {int(row.folds_better_mae)}/5 | {row.pooled_delta_rmse_candidate_minus_rule_baseline:.3f} | {int(row.folds_better_rmse)}/5 | {row.pooled_delta_r2_candidate_minus_rule_baseline:.3f} | {int(row.folds_better_r2)}/5 |")
    lines += ["", "负 ΔMAE/ΔRMSE、正 ΔR² 表示优于规则均值基线。RF/9 对 F/# 的五折 MAE、RMSE、R² 均为 5/5 优胜；对视场为 MAE 5/5、RMSE 和 R² 4/5。由此可以说特征带来了可重复的交叉验证增量，但五折训练集合相互重叠，不能把 5/5 当作独立统计显著性检验。", "",
              "## 物理派生特征效果", "",
              "Ridge：F/# MAE 39.11 → 31.22、R² 0.126 → 0.365；视场 MAE 0.415 → 0.372，但 RMSE 和 R²没有改善。RF：F/# MAE 25.62 → 25.73、RMSE 59.27 → 63.46；视场几乎不变。因此 V1 中派生特征主要帮助线性模型表达非线性尺度关系，没有提高最好的树模型。", "",
              "## 规则间预测差异", "",
              "下表使用总体 MAE 最低的 RF/9。方向一致率只对真实差值严格不为零的处方计算；不人为设置工程平局阈值。", "",
              "| 输出 | 规则差 | 真实严格平局 | 差值 MAE | 差值 RMSE | 差值 R² | 差值相关 | 非平局方向一致率 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    best_pairwise = pairwise[(pairwise.model == "random_forest") & (pairwise.feature_set == "raw9")]
    for _, row in best_pairwise.iterrows():
        lines.append(f"| {target_meta[row.target][0]} | {row.rule_pair} | {int(row.actual_exact_ties)} | {fmt(row.delta_mae)} | {fmt(row.delta_rmse)} | {row.delta_r2:.3f} | {row.pearson_delta:.3f} | {row.direction_agreement_non_ties:.3f} |")
    lines += ["", "Rule 1 与 Rule 2/3 的视场方向容易识别，因为真实非平局处方的方向高度一致。Rule 2 与 Rule 3 的差异稀疏：F/# 有 86 个严格平局，视场有 92 个严格平局；方向一致率基于仅 14 或 8 个非平局样本，不能据此宣称模型能稳定区分这两条规则。", "",
              "![规则差预测](03_pairwise_rule_differences.png)", "",
              "## 误差与适用边界", "",
              "- F/# 分布有强长尾，Rule 2 含 F/#=500 的终点；RMSE 对这些极值高度敏感。RF/9 是当前总体最佳，但 F/# MAE 25.62 仍较大。", "- Ridge OOF 产生了负 F/#、负视场和超过观测范围的预测，说明无约束线性外推不满足物理边界。表中保留原始预测，没有裁剪来美化指标。", "- 视场总体 R² 约 0.815，但逐规则看 Rule 2/3 的变化范围很窄；总体得分包含规则均值差带来的容易部分。", "- 100 套处方适合做第一版可行性验证，不足以进行可靠超参数选择、概率校准或部署结论。第二批数据应继续保持完全未见，作为后续一次性外部测试集。", "", "## 可复现文件", "",
              "`predictions.json` 保存全部 OOF 预测，`metrics.json` 与 `fold_metrics.json` 保存总体、逐规则及逐折指标，`baseline_comparisons.json` 保存基线增量，`pairwise_rule_differences.json` 保存规则差比较。`fold_assignments.json` 固定处方分组，`provenance.json` 保存输入哈希和随机种子。", "",
              f"运行环境：scikit-learn {sklearn.__version__}、NumPy {np.__version__}、pandas {pd.__version__}。图表程序自检结果见 `figure_qa.json`。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "environment.json").write_text(json.dumps({
        "scikit_learn": sklearn.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure_qa": qa, "report": str(OUT / "report.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
