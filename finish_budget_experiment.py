"""Finish analysis as part of the launched budget experiment, then notify."""
from pathlib import Path
import sys,subprocess,time,json,traceback
from budget_batch import read,save
from notify_pushplus import deliver_terminal
root=Path(__file__).resolve().parent; run=root/'background_runs/budget_52_5_11_v1'
while True:
    state=read(run/'state.json')
    if state['status'] in ('stopped','failed'):
        save(run/'analysis_status.json',{'status':'not_run','reason':state.get('reason'),'completed':state['completed']}); break
    if state['status']=='completed':
        try:
            process=subprocess.run([sys.executable,'-B',str(root/'summarize_budget_comparison.py')],cwd=root,capture_output=True,text=True)
            (run/'analysis.log').write_text(process.stdout+'\n'+process.stderr,encoding='utf-8')
            if process.returncode: raise RuntimeError('Summary failed; see analysis.log')
            out=root/'budget_comparison_52_5_11_v1'; summary=read(out/'summary.json'); pairs=read(out/'paired_comparisons.json')
            content='预算对照 27/27 已验收并完成分析。\n'
            for n in (1,2,4):
                s=summary[str(n)]; content+=f"每候选 {n} 次：成功 {s['target_reached']}/9，推进阈值停止 {s['threshold_stopped']}/9。\n"
            for n in (2,4):
                p=[x for x in pairs if x['calls_a']==1 and x['calls_b']==n]
                content+=f"1→{n} 次：高预算规格支配 {sum(x['endpoint_relation']=='higher_budget_dominates' for x in p)}/9，规格平局 {sum(x['endpoint_relation']=='numerical_tie' for x in p)}/9。\n"
            content+='报告：'+str(out/'report.md')
            receipt=deliver_terminal(out,{'status':'completed','event_id':state['event_id']+'_analysis','completed':27,'total':27,'notification_title':'Zemax DLS预算对照分析完成','reason':content})
            save(run/'analysis_status.json',{'status':'completed','report':str(out/'report.md'),'notification_accepted':receipt.get('accepted')})
        except Exception:
            save(run/'analysis_status.json',{'status':'failed','error':traceback.format_exc()})
        break
    time.sleep(15)
