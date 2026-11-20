"""Isolated extra-DLS-call experiment. Never edits the V1 implementation."""
from copy import deepcopy

def chain_attempt(base, calls, *args, **kwargs):
    if calls not in (1,2,4): raise ValueError('Expected 1, 2, or 4 calls')
    # Same candidate and current optimized prescription on each call. No restore,
    # continuation update, or acceptance decision occurs between calls.
    history=[base(*args,**kwargs) for _ in range(calls)]
    result=deepcopy(history[-1])
    result['dls_call_results']=history
    result['dls_calls']=calls
    result['dls_requested_cycles']=sum(r['dls_requested_cycles'] for r in history)
    result['dls_completed_cycles']=(sum(r['dls_completed_cycles'] for r in history)
        if all(r['dls_completed_cycles'] is not None for r in history) else None)
    result['dls_seconds']=sum(r['dls_seconds'] for r in history)
    result['attempt_wall_seconds']=sum(r['attempt_wall_seconds'] for r in history)
    return result

def install(calls):
    import run_p1
    base=run_p1.attempt
    def repeated(*args,**kwargs):
        return chain_attempt(base,calls,*args,**kwargs)
    run_p1.attempt=repeated
    writer=run_p1.write_json
    def persist(path,value):
        if 'steps' in value and 'settings' in value:
            value['settings'].update(dls_calls_per_candidate=calls,
                fixed_cycles_per_call=50,dls_cycles_per_step=50*calls,
                budget_experiment='sequential fixed-50 DLS calls; original continuation unchanged')
            value['cost']['dls_calls']=sum(r['dls_calls'] for r in value['steps'])
        return writer(path,value)
    run_p1.write_json=persist
