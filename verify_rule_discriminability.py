import json,hashlib
from pathlib import Path
from collections import Counter
p=Path('rule_discriminability_100_v1'); a=json.loads((p/'analysis.json').read_text()); rows=json.loads((p/'group_metrics.json').read_text()); pairs=json.loads((p/'paired_metrics.json').read_text()); provenance=json.loads((p/'provenance.json').read_text())
assert len(rows)==300 and len({(r['case_id'],r['rule']) for r in rows})==300
assert set(r['case_id'] for r in rows)==set(range(1,101))
assert all({r['rule'] for r in rows if r['case_id']==i}=={1,2,3} for i in range(1,101))
assert sum(r['all_rules_failed'] for r in rows)==234
for pair,z in a['pairwise'].items():
    assert sum(z['endpoint_relations'].values())==100
    for metric,s in z['metrics'].items(): assert s['a_better']+s['numerical_ties']+s['b_better']==s['n']
for r in ('1','2','3'): assert sum(a['failed_78'][r]['deficit_types'].values())==78
for file,digest in provenance['sources_sha256'].items(): assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
labels=json.loads(Path('feature_engineering_100_v1/labels.json').read_text())
for label in labels:
    case=[r for r in rows if r['case_id']==label['case_id']]
    assert sorted(r['rule'] for r in case if r['target_reached'])==label['successful_rules']
stop={str(rule):dict(Counter(json.dumps(r['stop_reasons'],sort_keys=True) for r in rows if r['rule']==rule and r['all_rules_failed'])) for rule in (1,2,3)}
sets=[]
for label in labels:
    case=[r for r in rows if r['case_id']==label['case_id']]
    sets.append({**label,'failure_details_by_rule':{str(r['rule']):{'deficit_type':r['deficit_type'],'stop_reasons':r['stop_reasons']} for r in case}})
(p/'case_sets.json').write_text(json.dumps(sets,indent=2),encoding='utf-8')
verification={'groups':300,'paired_rows':len(pairs),'source_hashes_verified':len(provenance['sources_sha256']),'source_files_unchanged':True,'case_rule_and_success_consistency':'pass','count_reconciliation':'pass','failed_cases_stop_reason_counts':stop}
(p/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print(json.dumps(verification))
