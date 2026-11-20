"""Summarize budget-call experiment without changing V1 success semantics."""
from pathlib import Path
import json,hashlib
from collections import Counter
from itertools import combinations
from budget_batch import read,validate_group
root=Path(__file__).resolve().parent
run=root/'background_runs/budget_52_5_11_v1'
state=read(run/'state.json')
if state['status']!='completed' or state['completed']!=27: raise SystemExit('Batch not complete: '+str(state.get('completed'))+'/27 '+state['status'])
out=root/'budget_comparison_52_5_11_v1'; out.mkdir(exist_ok=False)
rows=[]; sources={}; original=read(root/'background_runs/batch_300_20261118/state.json')
for key,j in state['jobs'].items():
    d=run/j['attempt']; validation=validate_group(d,j['case_id'],j['rule'])
    assert validation['result_sha256']==j['acceptance']['result_sha256']
    path=d/'experiment/result.json'; result=read(path); final=result['final_accepted']; n=j['dls_calls']
    assert result['settings']['dls_calls_per_candidate']==n
    assert all(s['dls_calls']==n and len(s['dls_call_results'])==n and all(c['dls_requested_cycles']==50 for c in s['dls_call_results']) for s in result['steps'])
    assert result['cost']['requested_dls_cycles']==50*n*len(result['steps'])
    assert result['cost']['dls_calls']==n*len(result['steps'])
    for s in result['steps']:
        assert all(c['working_f_number']==s['working_f_number'] and c['max_field_deg']==s['max_field_deg'] for c in s['dls_call_results'])
        assert s['accepted']==s['dls_call_results'][-1]['accepted']
    row={'case_id':j['case_id'],'rule':j['rule'],'calls_per_candidate':n,'target_reached':result['target_reached'],'has_accepted_endpoint':final is not None,'stop_reasons':result['stop_reasons'],'threshold_stopped':any(v=='step_limit' for v in result['stop_reasons'].values()),'threshold_stopped_dimensions':sum(v=='step_limit' for v in result['stop_reasons'].values()),'cost':result['cost'],'f_number':final['working_f_number'] if final else None,'field_deg':final['max_field_deg'] if final else None,'fields':final.get('fields') if final else None,'geometry_valid':final['geometry_valid'] if final else None,'conjugate_residuals':{k:final[k] for k in ('object_image_b_mm','magnification_error_at_image','exit_pupil_b_mm')} if final else None,'within_candidate_call_transitions':dict(Counter(str([s['dls_call_results'][0]['accepted'],s['dls_call_results'][-1]['accepted']]) for s in result['steps'])),'within_candidate_merit_first_to_last':[{'step':s['step'],'stage':s['stage'],'first_merit':s['dls_call_results'][0]['final_merit'],'last_merit':s['final_merit']} for s in result['steps']]}
    rows.append(row); sources[str(path.relative_to(root))]=validation['result_sha256']
summary={}
for n in (1,2,4):
    sub=[r for r in rows if r['calls_per_candidate']==n]
    summary[str(n)]={'groups':len(sub),'target_reached':sum(r['target_reached'] for r in sub),'threshold_stopped':sum(r['threshold_stopped'] for r in sub),'threshold_stopped_dimensions':sum(r['threshold_stopped_dimensions'] for r in sub),'no_accepted_endpoint':sum(not r['has_accepted_endpoint'] for r in sub),'requested_cycles':sum(r['cost']['requested_dls_cycles'] for r in sub),'dls_seconds':sum(r['cost']['dls_seconds'] for r in sub),'group_seconds':sum(r['cost']['wall_seconds'] for r in sub)}
comparisons=[]
for c in (52,5,11):
    for rule in (1,2,3):
        for na,nb in combinations((1,2,4),2):
            a=next(r for r in rows if (r['case_id'],r['rule'],r['calls_per_candidate'])==(c,rule,na)); b=next(r for r in rows if (r['case_id'],r['rule'],r['calls_per_candidate'])==(c,rule,nb))
            same=a['f_number'] is not None and b['f_number'] is not None and abs(a['f_number']-b['f_number'])<=1e-9 and abs(a['field_deg']-b['field_deg'])<=1e-9
            if not a['has_accepted_endpoint'] or not b['has_accepted_endpoint']: relation='missing_endpoint'
            else:
                fa,fb=a['f_number'],b['f_number']; va,vb=a['field_deg'],b['field_deg']
                if same: relation='numerical_tie'
                elif fb<=fa+1e-9 and vb>=va-1e-9: relation='higher_budget_dominates'
                elif fa<=fb+1e-9 and va>=vb-1e-9: relation='lower_budget_dominates'
                else: relation='tradeoff'
            comparisons.append({'case_id':c,'rule':rule,'calls_a':na,'calls_b':nb,'endpoint_relation':relation,'delta_f_number':b['f_number']-a['f_number'] if a['has_accepted_endpoint'] and b['has_accepted_endpoint'] else None,'delta_field_deg':b['field_deg']-a['field_deg'] if a['has_accepted_endpoint'] and b['has_accepted_endpoint'] else None,'same_specs':same,'delta_edge_mtf_min':min(b['fields'][-1]['mtf_tangential'],b['fields'][-1]['mtf_sagittal'])-min(a['fields'][-1]['mtf_tangential'],a['fields'][-1]['mtf_sagittal']) if same else None,'threshold_stop_removed':a['threshold_stopped'] and not b['threshold_stopped'],'new_success':not a['target_reached'] and b['target_reached']})
# Same-environment one-call regression against previous V1 endpoints.
regression=[]
for r in rows:
    if r['calls_per_candidate']!=1: continue
    j=original['jobs'][f"case_{r['case_id']:03d}_rule_{r['rule']}"]; path=root/'background_runs/batch_300_20261118'/j['attempt']/'experiment/result.json'; old=read(path); f=old['final_accepted']
    sources[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    regression.append({'case_id':r['case_id'],'rule':r['rule'],'same_target_reached':r['target_reached']==old['target_reached'],'same_stop_reasons':r['stop_reasons']==old['stop_reasons'],'delta_f_number':r['f_number']-f['working_f_number'] if r['f_number'] is not None else None,'delta_field_deg':r['field_deg']-f['max_field_deg'] if r['field_deg'] is not None else None,'max_mtf_difference':max(abs(x[k]-y[k]) for x,y in zip(r['fields'],f['fields']) for k in ('mtf_tangential','mtf_sagittal')) if r['fields'] else None,'max_rms_difference_um':max(abs(x['rms_spot_radius']-y['rms_spot_radius']) for x,y in zip(r['fields'],f['fields'])) if r['fields'] else None})
for name,v in [('groups.json',rows),('summary.json',summary),('paired_comparisons.json',comparisons),('one_call_regression.json',regression),('source_sha256.json',sources)]: (out/name).write_text(json.dumps(v,indent=2)+'\n')
lines=['# 处方 52、5、11：DLS 调用预算对照','','同一候选依次调用 1、2、4 次固定 50 循环 DLS，每次 2 核。调用间保留当前处方；每次调用沿用 V1 评价函数构建，完成全部调用后按最终一次结果验收。推进因子及 0.99/1.01 停止条件不变。额外调用含优化器重新调用影响，不等价于单次固定 100/200 循环。每组独立会话，从相同初始处方重建。','', '| 每候选调用数 | 达标 / 9 | 推进阈值停止 / 9 | 停止方向数 | 请求循环预算 | DLS 进程秒 |','|---|---:|---:|---:|---:|---:|']
for n,s in summary.items(): lines.append(f"| {n} | {s['target_reached']} | {s['threshold_stopped']} | {s['threshold_stopped_dimensions']} | {s['requested_cycles']} | {s['dls_seconds']:.2f} |")
lines+=['','| 处方 | 规则 | 调用数 | 最终 F/# | 最大视场 / ° | 达标 | 推进阈值停止 |','|---|---|---:|---:|---:|---|---|']
for r in sorted(rows,key=lambda r:(r['case_id'],r['rule'],r['calls_per_candidate'])): lines.append(f"| {r['case_id']} | {r['rule']} | {r['calls_per_candidate']} | {r['f_number']} | {r['field_deg']} | {r['target_reached']} | {r['threshold_stopped']} |")
lines+=['','## 配对规格变化','','规格支配使用 F/# 越小、视场越大，数值容差各 1e-9；不另设工程明显性阈值。成像质量只在相同最终规格比较，完整三视场 MTF/RMS 保存在 groups.json。','']
for na,nb in combinations((1,2,4),2): lines.append(f"- {na} → {nb} 调用：{dict(Counter(c['endpoint_relation'] for c in comparisons if c['calls_a']==na and c['calls_b']==nb))}")
base=summary['1']; conclusions=['','## 两个问题的直接回答','']
for n in (2,4):
    sub=[c for c in comparisons if c['calls_a']==1 and c['calls_b']==n]
    counts=dict(Counter(c['endpoint_relation'] for c in sub))
    maxf=max((abs(c['delta_f_number']) for c in sub if c['delta_f_number'] is not None),default=0)
    maxv=max((abs(c['delta_field_deg']) for c in sub if c['delta_field_deg'] is not None),default=0)
    conclusions.append(f"- 1 → {n} 次调用：规格比较 {counts}；最大绝对 F/# 变化 {maxf:.6g}，最大视场变化 {maxv:.6g}°；新增成功 {sum(c['new_success'] for c in sub)} 组。完整幅度用于判断实际影响，不将超过数值容差直接等同于工程明显改善。")
    conclusions.append(f"- 推进幅度停止比例：{base['threshold_stopped']}/9 → {summary[str(n)]['threshold_stopped']}/9；原停止被解除 {sum(c['threshold_stop_removed'] for c in sub)} 组。小样本单次结果不等于总体概率降低。")
lines+=conclusions
lines+=['','## 限制','','仅三套定向选样、每个配置单次运行；结论适用于本次处方，不可直接推广到全部 100 套。预算增加可能改变局部优化路径，终点不保证单调改善。真实完成循环数仍由 API 缺失，所报循环数为请求预算。时间为进程耗时而非并行日历耗时。one_call_regression.json 核查新一轮控制组与历史 V1 的差异。']
(out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'summary':summary,'relations':{f'{na}_to_{nb}':dict(Counter(c['endpoint_relation'] for c in comparisons if c['calls_a']==na and c['calls_b']==nb)) for na,nb in combinations((1,2,4),2)},'regression':regression},indent=2))
