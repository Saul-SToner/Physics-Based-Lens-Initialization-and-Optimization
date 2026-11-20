"""Verify nine completed P1 groups and write Markdown tables and cost totals."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    batch = json.loads((args.output/"batch.json").read_text())
    rows = batch["groups"]
    assert sorted((row["case_id"],row["rule"]) for row in rows) == [
        (case,rule) for case in (1,2,3) for rule in (1,2,3)
    ]
    assert not batch["historical_files_changed"]
    manifest = json.loads((args.output/"run_manifest.json").read_text())
    assert all(sha256((root/path).read_bytes()).hexdigest()==value
               for path,value in manifest["historical_sha256_before"].items())
    table = [
        "# P1 三套随机处方联调",
        "",
        "按生成器 id 作为 case_id，选取 1、2、3。每组使用新 OpticStudio 会话，"
        "从对应共轭恢复处方启动。每步请求 50 个 DLS 循环。",
        "",
        "| Case | Rule | 最后接受 F/# | 最大视场 ° | MTF min（3 视场） | RMS µm（3 视场） | 尝试 | 请求循环 | DLS 秒 | 组耗时秒 |",
        "|---|---|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    residuals = [
        "| Case | Rule | 物像 B mm | 放大率误差 | 瞳共轭 B mm | 最小几何余量 mm | 停止原因 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    totals = {}
    for row in rows:
        case,rule = row["case_id"],row["rule"]
        disk = json.loads((args.output/f"case_{case:03d}"/f"rule_{rule}"/"result.json").read_text())
        assert disk==row
        assert row["audit"]["case_identity_verified"]
        assert row["cost"]["requested_dls_cycles"]==50*len(row["steps"])
        assert all(step["case_id"]==case and step["rule"]==rule for step in row["steps"])
        final = row["final_accepted"]
        cost = row["cost"]
        if final is not None:
            assert row["audit"]["saved_final_verified"]
            assert final["accepted"] and len(final["fields"])==3
            mtf = "/".join(f'{min(f["mtf_tangential"],f["mtf_sagittal"]):.5f}' for f in final["fields"])
            rms = "/".join(f'{f["rms_spot_radius"]:.3f}' for f in final["fields"])
            aperture,field = f'{final["working_f_number"]:.3f}',f'{final["max_field_deg"]:.3f}'
            clearance = min(final["clearances_mm"].values())-0.001
            residuals.append(
                f'| {case} | {rule} | {final["object_image_b_mm"]:.3e} | '
                f'{final["magnification_error_at_image"]:.3e} | {final["exit_pupil_b_mm"]:.3e} | '
                f'{clearance:.6f} | {json.dumps(row["stop_reasons"])} |'
            )
        else:
            aperture=field=mtf=rms="—"
            residuals.append(f'| {case} | {rule} | — | — | — | — | initial_rejected |')
        table.append(
            f'| {case} | {rule} | {aperture} | {field} | {mtf} | {rms} | '
            f'{cost["attempts"]} | {cost["requested_dls_cycles"]} | '
            f'{cost["dls_seconds"]:.2f} | {cost["wall_seconds"]:.2f} |'
        )
        total=totals.setdefault(str(rule),dict(groups=0,attempts=0,requested_dls_cycles=0,dls_seconds=0.0,wall_seconds=0.0))
        total["groups"]+=1
        for key in ("attempts","requested_dls_cycles","dls_seconds","wall_seconds"):
            total[key]+=cost[key]
    cost_report={
        "batch":batch["cost"],"by_rule":totals,
        "completed_dls_cycles":None,
        "cycle_accounting":"50 requested cycles per completed optimization call; the API exposes no completed-cycle counter",
        "historical_files_verified":len(manifest["historical_sha256_before"]),
        "targets_reached":sum(row["target_reached"] for row in rows),
    }
    (args.output/"cost.json").write_text(json.dumps(cost_report,indent=2)+"\n",encoding="utf-8")
    table += ["", "MTF min 为该视场子午与弧矢方向中的较低值。"
              "三视场依次为 0、0.707 倍和 1 倍最大视场，在各自规定频率评估。", ""]+residuals
    table += ["","几何余量为所有有效间隔/厚度中最小值减去 0.001 mm 硬下限。"
              "共轭与放大率沿用冻结流程的软约束，接受条件为几何及 MTF 达标。",
              "", "## 计算成本", "",
              f'累计请求 DLS 循环 {batch["cost"]["requested_dls_cycles"]}，'
              f'DLS 用时 {batch["cost"]["dls_seconds"]:.2f} 秒，'
              f'含 API 会话的批次总耗时 {batch["cost"]["wall_seconds_including_api_sessions"]:.2f} 秒。',
              "", "实际完成循环数未由该 API 暴露。请求循环包含被拒绝的尝试。"
              "逐组耗时包含评价函数构建、DLS、分析和回退，批次耗时另包含建模及 API 启停。",
              "", "## 联调审计", "",
              "- 原脚本中的 PMAG、EXPP 和瞳面残差样例常量已在 P1 引擎中参数化。",
              "- 生成器 id 显式映射为 case_id，组内每条步骤均记录 case_id、rule 和 step。",
              "- 每组新建 API 会话，并检查种子共轭、几何、表面数量、视场数量和实际评价函数目标。",
              "- 每个已接受终点从磁盘重新加载，核对共轭残差与日志一致。",
              "- 历史处方、日志与冻结文件前后校验一致。",
              "- 修复规则 2 第二阶段停止原因被覆盖的问题，本批 Case 2 / Rule 2 的停止字段已校正，修正前后值保存在 audit.json。",
              "- API 会话切换附近出现原生 FRU__delta_init 诊断信息；九组均完成调用并通过终点重载检查，原始信息保存在 audit.json。",
              "- 启动检查曾因目标值末位舍入差异中断，随后采用 1e-12 容差。"
              "未执行 DLS 的启动尝试位于 p1_random_3，完整批次位于本目录。",
              "", "成本为本批单次运行的实测值，尚未重复计时。",
              ""]
    (args.output/"summary.md").write_text("\n".join(table),encoding="utf-8")
    print(json.dumps(cost_report,indent=2))


if __name__=="__main__":
    main()
