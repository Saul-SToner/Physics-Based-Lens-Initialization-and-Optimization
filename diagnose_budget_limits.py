"""Read-only replay of V1 adaptive continuation termination."""
from pathlib import Path
import json, math, hashlib
from collections import Counter
root=Path(__file__).resolve().parent
out=root/'budget_diagnosis_100_v1'; out.mkdir(exist_ok=False)
batch=root/'background_runs/batch_300_20261118'
state=json.loads((batch/'state.json').read_text()); labels=json.loads((root/'feature_engineering_100_v1/labels.json').read_text()); failed={r['case_id'] for r in labels if r['target_success_mask']==0}
sources=[root/'run_p1.py',root/'p1_engine.py',batch/'state.json']; rows=[]; errors=[]
for key,job in sorted(state['jobs'].items()):
    path=batch/job['attempt']/'experiment/result.json'; sources.append(path)
    raw=path.read_bytes(); result=json.loads(raw)
    assert hashlib.sha256(raw).hexdigest()==job['acceptance']['result_sha256']
    current={'aperture':500.,'field':1.}; factors={'aperture':.707,'field':1.414}; target={'aperture':20.,'field':5.}; rejects=Counter(); terminals={}; lastaccept=None; trace=[]
    for i,s in enumerate(result['steps']):
        dim=s['stage']; before=dict(factors)
        if dim!='initial':
            proposed=current[dim]*factors[dim]; expected=max(20.,proposed) if dim=='aperture' else min(5.,proposed)
            actual=s['working_f_number'] if dim=='aperture' else s['max_field_deg']
            if not math.isclose(expected,actual,rel_tol=1e-12,abs_tol=1e-12): errors.append([key,i,'candidate_mismatch'])
        if s['accepted']:
            current={'aperture':s['working_f_number'],'field':s['max_field_deg']}; lastaccept=i
        elif dim!='initial':
            factors[dim]=math.sqrt(factors[dim]); rejects[dim]+=1
            if (dim=='aperture' and factors[dim]>.99) or (dim=='field' and factors[dim]<1.01): terminals[dim]={'step':i+1,'factor_after':factors[dim],'rejections':rejects[dim]}
        trace.append({'step':i+1,'stage':dim,'accepted':s['accepted'],'factor_before':before,'factor_after':dict(factors),'f_number':s['working_f_number'],'field_deg':s['max_field_deg'],'rejection_reasons':s['rejection_reasons']})
    for dim,reason in result['stop_reasons'].items():
        if reason=='step_limit' and dim not in terminals: errors.append([key,dim,'unexplained_stop'])
    tail=result['steps'][lastaccept+1:] if lastaccept is not None else result['steps']
    accepted=[s for s in result['steps'] if s['accepted']]
    rows.append({'case_id':job['case_id'],'rule':job['rule'],'all_rules_failed':job['case_id'] in failed,'stop_reasons':result['stop_reasons'],'threshold_crossings':terminals,'rejected_steps_by_dimension':dict(rejects),'total_steps':len(result['steps']),'accepted_steps':len(accepted),'trailing_rejections_after_last_accepted':len(tail),'tail_geometry_rejected':sum(not s['geometry_valid'] for s in tail),'tail_mtf_rejected':sum(any(not f['meets_threshold'] for f in s.get('fields',[])) for s in tail),'final_f_number':current['aperture'],'final_field_deg':current['field'],'trajectory':trace})
assert not errors,errors
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
summary={'groups_replayed':len(rows),'replay_errors':errors,'step_limit_group_count':sum(bool(r['threshold_crossings']) for r in rows),'step_limit_dimension_count':sum(len(r['threshold_crossings']) for r in rows),'failed_78_rule_counts':{str(rule):{'n':len(sub:=[r for r in rows if r['all_rules_failed'] and r['rule']==rule]),'threshold_stop_groups':sum(bool(r['threshold_crossings']) for r in sub),'total_steps_range':[min(r['total_steps'] for r in sub),max(r['total_steps'] for r in sub)],'rejection_count_at_crossing':dict(Counter(str(v['rejections']) for r in sub for v in r['threshold_crossings'].values()))} for rule in (1,2,3)},'factors_after_six_rejections':{'aperture':.707**(1/64),'field':1.414**(1/64)},'interpretation':'step_limit is an adaptive multiplicative step-size threshold; no stage total-step cap exists. Existing logs do not establish whether 50 DLS cycles per attempt are adequate.'}
# Deterministic small experiment candidates, with reasons; no optimization launched here.
metrics=json.loads((root/'rule_discriminability_100_v1/group_metrics.json').read_text()); paired=json.loads((root/'rule_discriminability_100_v1/paired_metrics.json').read_text())
near=min(failed,key=lambda cid:min(max(r['f_number']/20-1,1-r['field_deg']/5) for r in metrics if r['case_id']==cid))
different=sorted(p['case_id'] for p in paired if p['a']==2 and p['b']==3 and p['endpoint_relation']!='endpoint_tie' and p['case_id'] in failed)
selected=[]
for cid in [near]+different:
    if cid not in selected: selected.append(cid)
    if len(selected)==3: break
plan={'status':'prepared_not_launched','case_ids':selected,'selection':'nearest common target among all-failed cases, then smallest case IDs with differing Rule 2/3 endpoints; three distinct cases','rules':[1,2,3],'single_factor_design':'Increase DLS budget per candidate while keeping .99/1.01 progression thresholds fixed. Do not call this a stage-step-budget experiment. Verify supported SDK cycle modes before choosing execution implementation.','separate_experiment':'Relaxing .99/1.01 is a continuation-resolution experiment, must be reported separately from DLS budget.','preserve':['V1 source and results','merit function','geometry constraints','success standard','2 DLS cores','independent case sessions'],'baseline':'Use matching V1 data for historical comparison, and repeat 50-cycle controls under the new run environment before attributing timing or small numerical differences.'}
for name,value in [('analysis.json',summary),('trajectories.json',rows),('experiment_plan.json',plan),('provenance.json',hashes)]: (out/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
report=['# V1 停止机制及预算假设核查','','此前将 step_limit 解释为阶段步数上限不准确。本轮逐步重放 300 组运行，所有候选规格和停止条件均与日志一致，未发现阶段总步数限制。','', 'run_p1.py 在拒绝后对对应推进因子开平方。F/# 因子从 0.707 出发，超过 0.99 停止；视场因子从 1.414 出发，低于 1.01 停止。第六次累计拒绝后分别变为 %.9f、%.9f，触发 step_limit。成功步骤不会重置该因子，因此六次拒绝不必连续。'%(.707**(1/64),1.414**(1/64)),'','这与单次 DLS 请求的 50 循环是两个独立机制。日志缺少每次迭代的 merit 轨迹和实际完成循环数，无法仅据 step_limit 证明 DLS 预算不足，也不能证明物理不可实现。','', '| 规则 | 全失败处方数 | 发生推进因子阈值停止 | 总尝试步数范围 |','|---|---:|---:|---:|']
for rule,s in summary['failed_78_rule_counts'].items(): report.append(f"| {rule} | {s['n']} | {s['threshold_stop_groups']} | {s['total_steps_range']} |")
report+=['','## 正确的后续对照','','先进行单次候选处方的 DLS 预算敏感性实验，保持 0.99/1.01 推进阈值不变。需要先核查 SDK 支持的固定循环模式；若只支持 50 循环，则额外调用不能伪称单次固定 100/200 循环，须明确记录调用次数及潜在重新初始化差异。','',f'建议首轮处方：{selected}。第一套是全失败处方中最接近共同规格目标的样本，其余为 Rule 2/3 规格不一致的确定性选样。三条规则均应运行并设置同环境 50 循环控制。','', '另行研究减小最小推进幅度停止阈值（例如更接近 1），这是推进分辨率敏感性，不应与 DLS 预算变化混为一项。当前只完成诊断与实验准备，尚未启动新的 OpticStudio 优化。','','原始历史结果均只读；完整逐步重放见 trajectories.json。']
(out/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
print(json.dumps({'summary':summary,'selected_cases':selected},indent=2))
