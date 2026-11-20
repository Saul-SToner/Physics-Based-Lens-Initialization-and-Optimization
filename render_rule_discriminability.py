"""Render descriptive rule analysis; sources and scientific results are read only."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0,str(Path('C:/Users/L2791/.codex/skills/scipilot-figure-skill/scripts')))
from setup_style import setup_style
from visual_qa import audit_layout
from profile_data import profile_data
out=Path('rule_discriminability_100_v1')
a=json.loads((out/'analysis.json').read_text()); rows=json.loads((out/'group_metrics.json').read_text()); df=pd.DataFrame(rows)
pairs=json.loads((out/'paired_metrics.json').read_text())
profile=profile_data(df[['rule','f_number','field_deg','mtf_min_0','mtf_min_1','mtf_min_2','rms_0_um','rms_1_um','rms_2_um','dls_seconds','group_seconds','requested_cycles']].assign(rule=lambda d:d.rule.astype(str)),group_cols=['rule'])
(out/'data_profile.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
setup_style(journal='general',lang='zh',use_sciplots=False)
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'svg.fonttype':'none'})
colors=['#0072B2','#D55E00','#009E73']; styles=['-','--',':']; markers=['o','s','^']; qa={}
def save(fig,name):
    fig.set_constrained_layout_pads(w_pad=0.16, h_pad=0.16, wspace=0.12, hspace=0.12); fig.canvas.draw(); qa[name]=audit_layout(fig)
    fig.savefig(out/(name+'.png'),dpi=300)
    im=Image.open(out/(name+'.png')).convert('RGB'); im.thumbnail((1400,1100)); im.save(out/(name+'_preview.png'))
    im.convert('L').save(out/(name+'_gray.png'))
    fig.savefig(out/(name+'.svg'))
    plt.close(fig)
def ecdf(ax,key,label,log=False):
    for r,c,ls in zip((1,2,3),colors,styles):
        x=np.sort(df.loc[df.rule==r,key]); ax.step(x,np.arange(1,101)/100,where='post',color=c,ls=ls,label=f'Rule {r}',lw=1.8)
    ax.set(xlabel=label,ylabel='累计处方比例'); ax.grid(alpha=.2)
    if log:
        from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter
        ax.set_xscale('log'); ax.xaxis.set_major_locator(FixedLocator([20,50,100,200,500])); ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
fig,axs=plt.subplots(2,3,figsize=(12,7),layout='constrained')
for ax,(key,label,log) in zip(axs.flat,[('f_number','最终 F/#（对数轴）',True),('field_deg','最终最大视场 / °',False),('mtf_min_2','边缘视场 MTF：min(T,S)',False),('rms_2_um','边缘视场 RMS / μm',False),('group_seconds','组内实际耗时 / s',False),('requested_cycles','请求 DLS 循环预算',False)]): ecdf(ax,key,label,log)
axs[0,0].legend(); save(fig,'01_distributions')
fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for r,c,m in zip((1,2,3),colors,markers):
    d=df[(df.rule==r)&df.all_rules_failed]; axs[0].scatter(d.aperture_progress_fraction,d.field_progress_fraction,s=27,c=c,marker=m,alpha=.65,label=f'Rule {r}')
axs[0].scatter([1],[1],marker='*',s=130,color='black',label='共同规格目标'); axs[0].set(xlim=(0,1.05),ylim=(0,1.05),xlabel='20 / 最终 F/#（规格进度代理）',ylabel='最终最大视场 / 5°'); axs[0].legend(fontsize=9)
bottom=np.zeros(3)
for k,label,c,h in [('field_only','仅视场不足','#999999','//'),('aperture_only','仅 F/# 不足','#DDDDDD','..'),('both','两项均不足','#444444','xx')]:
    vals=np.array([a['failed_78'][str(r)]['deficit_types'].get(k,0) for r in (1,2,3)])
    axs[1].bar(['Rule 1','Rule 2','Rule 3'],vals,bottom=bottom,color=c,hatch=h,edgecolor='white',label=label)
    for i,v in enumerate(vals):
        if v: axs[1].text(i,bottom[i]+v/2,str(v),ha='center',va='center',color='white' if k=='both' else 'black',bbox=dict(facecolor=c,edgecolor='none',pad=1.2))
    bottom+=vals
axs[1].set(ylabel='处方数',ylim=(0,96)); axs[1].legend(loc='upper center',ncol=1,fontsize=9)
save(fig,'02_failed_target_gaps')
fig,axs=plt.subplots(3,4,figsize=(13,7),layout='constrained')
for i,(ra,rb) in enumerate([(1,2),(1,3),(2,3)]):
    p=[p for p in pairs if p['a']==ra and p['b']==rb]
    vals=[[-x['a_minus_b_f_number'] for x in p],[x['a_minus_b_field_deg'] for x in p],[x['a_minus_b_mtf_edge'] for x in p if x['same_specs']],[-np.log2(x['dls_time_ratio_a_over_b']) for x in p]]
    for j,v in enumerate(vals):
        ax=axs[i,j]; jitter=np.random.default_rng(20261118).uniform(-.28,.28,len(v)); ax.scatter(v,jitter,s=13,color=colors[i],alpha=.65); ax.axvline(0,color='black',lw=.7); ax.set(yticks=[],ylim=(-.38,.38)); ax.grid(axis='x',alpha=.2)
        if j==0: ax.set_ylabel(f'Rule {ra} vs {rb}')
        ax.set_title(f'n={len(v)}',fontsize=9)
        if i==2: ax.set_xlabel(['F/# B − A','视场 A − B / °','边缘 MTF A − B\n（同规格子集）','log2(耗时 B / A)'][j])
for j in range(4):
    lim=max(abs(x) for ax in axs[:,j] for x in ax.get_xlim());
    for ax in axs[:,j]: ax.set_xlim(-lim,lim)
save(fig,'03_paired_comparisons')
fig,axs=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
for r,c,m in zip((1,2,3),colors,markers):
    d=df[df.rule==r]; axs[0].scatter(d.requested_cycles,d.dls_seconds,c=c,marker=m,s=20,alpha=.6,label=f'Rule {r}')
    x=np.sort(d.seconds_per_requested_cycle); axs[1].step(x,np.arange(1,101)/100,where='post',color=c,ls=styles[r-1],label=f'Rule {r}')
axs[0].set(xlabel='请求 DLS 循环预算',ylabel='实测 DLS 调用耗时 / s'); axs[0].legend(); axs[1].set(xlabel='DLS 秒数 / 请求循环预算',ylabel='累计处方比例')
for ax in axs: ax.grid(alpha=.2)
save(fig,'04_cost_vs_budget')
(out/'figure_qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
# Markdown tables keep all field metrics and each pair's exact/numerical ties.
def f(x): return f'{x:.5g}'
def interval(s): return f"{f(s['median'])} [{f(s['q25'])}, {f(s['q75'])}]"
lines=['# 前 100 套处方：三条规则可区分性分析','',
'结论：存在明确的规格取舍和少数处方的路径差异，足以支持连续、多目标或集合形式的比较；现有结果不支持为每套处方稳定指定唯一的三分类赢家。Rule 1 与另外两条的差别较强，Rule 2 与 Rule 3 大部分处方相同，少数处方的差别很大。','',
'## 数据与比较口径','',
'100 套独立初始处方，每套三条规则，共 300 个最终接受终点。沿用冻结的成功标签与全部 V1 设置，无新增优化、训练或历史结果改写。共同目标为 F/#=20、最大视场=5°，同时满足现有三视场 MTF 和几何验收条件。','',
'下表为中位数 [第 25, 75 百分位]，完整最小值、最大值、均值、标准差和 5/95 百分位见 analysis.json。MTF 使用各视场 min(T,S)，完整 T、S 值保留在 group_metrics.json。RMS 单位为 μm。三视场随终点最大视场改变，MTF 采样频率随终点改变，因此跨不同规格的 MTF/RMS 分布仅供描述；成像质量的规则优劣只在同一处方、同 F/#、同视场及同采样频率子集比较。','',
'数值平局沿用先前回归核查精度：F/# 与视场 1e-9、MTF 1e-6、RMS 0.001 μm；采样频率匹配精度 1e-9 lp/mm。它们用于识别数值差异，不是工程显著性阈值，也没有改变成功标准。同时报告严格相等数量。单次计时不设置任意“时间平局”阈值。','',
'## 终点与成本分布','', '| 指标 | Rule 1 | Rule 2 | Rule 3 |','|---|---:|---:|---:|']
for key,label in [('f_number','F/#'),('field_deg','最大视场 / °')]+[(f'mtf_min_{i}',f'视场 {i+1} MTF min(T,S)') for i in range(3)]+[(f'rms_{i}_um',f'视场 {i+1} RMS / μm') for i in range(3)]+[('dls_seconds','DLS 调用耗时 / s'),('group_seconds','组内实际耗时 / s'),('requested_cycles','请求循环预算')]:
    lines.append('| '+label+' | '+' | '.join(interval(a['distributions'][str(r)][key]) for r in (1,2,3))+' |')
lines+=['','![终点分布](01_distributions.png)','','## 78 套全部未达标处方','','所有最终接受终点均通过当时的成像和几何验收；“未达标”主要指扩展规格没有达到共同目标。最后一次拒绝的原因只是终止附近的直接障碍，不是对整条失败路径的唯一因果解释。几何原因按组统计，曲面明细可重叠。','', '| 规则 | 仅 F/# 不足 | 仅视场不足 | 两项不足 | F/# 相对超额中位数 | 视场相对缺口中位数 | 末次 MTF 拒绝 | 末次几何拒绝 |','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in (1,2,3):
    z=a['failed_78'][str(r)]; t=z['deficit_types']; lines.append(f"| Rule {r} | {t.get('aperture_only',0)} | {t.get('field_only',0)} | {t.get('both',0)} | {z['f_number_excess_ratio']['median']:.1%} | {z['field_shortfall_fraction']['median']:.1%} | {z['last_rejection_types'].get('mtf',0)} | {z['last_rejection_types'].get('geometry',0)} |")
lines+=['','F/# 相对超额=(F/#−20)/20；视场相对缺口=(5−视场)/5。两项分别保留，不引入任意加权距离。20/F/# 仅是规格进度代理，不能解释为真实孔径比，因为 EFL 可能变化。逐组缺口、停止原因和几何曲面违规见 failed_78_details.json。','','![未达标处方目标缺口](02_failed_target_gaps.png)','','## 逐处方配对比较','','下面 A/B 顺序为表中两条规则；规格 Pareto 仅使用较低 F/#、较大视场两个目标。MTF/RMS 没有混入规格 Pareto。','', '| A vs B | A 支配 | B 支配 | 规格平局 | 规格取舍 | 同规格成像比较 n |','|---|---:|---:|---:|---:|---:|']
for key,z in a['pairwise'].items():
    t=z['endpoint_relations']; lines.append(f"| {key.replace('_vs_',' vs ')} | {t.get('a_dominates',0)} | {t.get('b_dominates',0)} | {t.get('endpoint_tie',0)} | {t.get('tradeoff',0)} | {z['matched_specs_n']} |")
lines+=['','| A vs B | 指标 | n | A 较优 | 数值平局 | B 较优 | 严格相等 | A−B 中位数 [Q25,Q75] |','|---|---|---:|---:|---:|---:|---:|---:|']
for key,z in a['pairwise'].items():
    for metric,v in z['metrics'].items(): lines.append(f"| {key.replace('_vs_',' vs ')} | {metric} | {v['n']} | {v['a_better']} | {v['numerical_ties']} | {v['b_better']} | {v['exact_ties']} | {interval(v['a_minus_b'])} |")
lines+=['','图中正值表示 A 在对应指标更优；耗时表示本次观测更快，不能解释为跨运行稳定优势。不同指标的优劣方向在配对表中已分别处理。','','![配对差异](03_paired_comparisons.png)','','Rule 2 与 Rule 3：86 套规格平局；Rule 3 在 8 套规格支配 Rule 2，Rule 2 在 1 套支配 Rule 3，另有 5 套发生规格取舍。13 套 Rule 3 的 F/# 更小，其中一部分 Rule 2 停在 F/#=500，导致总体均值差距很大。86 套同规格处方中，边缘 MTF 的差值中位数约 −1.76e-8，边缘 RMS 差值中位数约 3.73e-6 μm；数值差异正负均有，不能据此宣称统一的成像优胜规则。','','## 成功、Pareto 与平局集合','','成功率：Rule 1 为 18%，Rule 2、3 均为 22%。18 套三条全成功，4 套仅 Rule 2、3 成功（case 1、15、49、84），78 套全部未成功。没有唯一成功规则。','', '| 集合口径 | 集合：处方数 |','|---|---|']
for key,z in a['sets'].items(): lines.append('| '+key+' | '+'；'.join(f'{k}: {v}' for k,v in z.items())+' |')
lines+=['','严格规格 Pareto 有 98 套包含多条规则，仅 2 套唯一为 Rule 3。数值容差口径使一套原本由末位浮点差异决定的集合发生改变，结果同时保留。22 套有成功规则的处方中，按请求循环预算只有 1 套唯一为 Rule 2，其余 21 套仍有并列。成本集合不等价于实测耗时集合。','','## requested cycles 能否代表真实计算成本','',f"300 组 requested cycles 合计 {a['cost']['completed_attempt_requested_cycles']:,}；实际完成循环数 300/300 均缺失（API 未提供计数）。预算与实测 DLS 调用耗时 Pearson={a['cost']['pearson_requested_vs_dls_seconds']:.3f}，Spearman={a['cost']['spearman_requested_vs_dls_seconds']:.3f}，可作粗略工作量代理，不能称为实际迭代次数或精确耗时。",'',
'同为 800 请求循环的 79 组，DLS 耗时 138.68–316.73 s，相差 2.28 倍。处方复杂度、每次迭代耗时、提前收敛、运行时并发竞争与会话时期均可能影响成本；现有日志不能把它们各自的贡献完全分离。13 组较早会话的秒/请求循环中位数 0.211，后续 287 组为 0.251，计时有明显运行环境混杂。','',
f"最终采用的 300 组累计 DLS 调用时间 {a['cost']['completed_attempt_dls_seconds']/3600:.3f} 进程小时；另有 case 2 / Rule 3 的一次已记录重试消耗 1050 请求循环、230.25 s。全部可见记录合计 {a['cost']['all_observed_requested_cycles']:,} 请求循环和 {a['cost']['all_observed_dls_seconds']/3600:.3f} DLS 进程小时。这是并发任务耗时之和，不是日历总耗时；没有完整日志的中断工作可能未计入。",'',
'组内实际耗时包含该组优化流程，但不包含 Ray 排队、OpticStudio/API 会话启动和全部失败重试。若未来需要真实端到端成本，应另记调度、会话启动、组执行、重试及总体完成时间；本次没有重新运行优化。','','![成本与预算](04_cost_vs_budget.png)','','## 对可区分性的判断','','1. 规格空间有强信号：Rule 1 与 Rule 2/3 的主要差异是孔径与视场取舍，失败处方同样提供有效信息；不能将 78 套全失败简单视作无信息样本。','2. Rule 2/3 的信号稀疏但存在：86 套规格相同，14 套不同，且部分差异幅度很大。可以研究哪些初始条件触发路径差异；100 套仅提供 14 个这种配对差异样本，尚不足以确认对新处方的稳定预测能力。','3. 同规格成像差异整体较小且方向混合，现有单次运行不能将微小差值与优化数值波动完全区分。没有新增工程显著性标准。','4. 耗时存在观测差异，但并发环境与运行时期混杂；请求循环只能作为预算指标。当前数据适合保留连续表现、Pareto 和平局集合，不能制造唯一三分类真值。','','## 审计与文件','','所有原始输入、当前采用结果及相关历史状态在读取前后通过 SHA-256 核查；分组标识和结果摘要哈希一致。独立检查 300 组、每套恰好 Rule 1/2/3、所有配对计数、失败类型计数、成功集合与既有标签一致。provenance.json 保存源文件哈希，analysis.json 保存全部统计，group_metrics.json / paired_metrics.json 保存逐组与逐对结果。']
(out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'figure_qa':qa,'report':str(out/'report.md')},ensure_ascii=False))


from collections import Counter
lines=['','## 停止原因补充核对','','78 套全失败处方的日志均涉及 step_limit（阶段步数上限），不能解读成已证明物理不可实现。末次 MTF/几何拒绝说明上限附近的受阻表现，未证明耗尽更大预算仍必然失败。','', '| 规则 | aperture / field 停止原因 | 处方数 |','|---|---|---:|']
for rule in (1,2,3):
    counts=Counter((r['stop_reasons'].get('aperture'),r['stop_reasons'].get('field')) for r in rows if r['rule']==rule and r['all_rules_failed'])
    for (x,y),n in counts.items(): lines.append(f'| Rule {rule} | {x} / {y} | {n} |')
lines+=['','逐处方成功集合与 Pareto 集合见 case_sets.json；独立计数和 307 个源文件哈希核查见 verification.json。']
with (out/'report.md').open('a',encoding='utf-8') as stream: stream.write('\n'.join(lines)+'\n')
