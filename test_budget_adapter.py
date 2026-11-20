import unittest
from budget_adapter import chain_attempt
class BudgetTests(unittest.TestCase):
    def test_aggregation_and_last_call_acceptance(self):
        calls=[]
        def base(*args,**kw):
            calls.append(len(calls)+1)
            return dict(dls_requested_cycles=50,dls_completed_cycles=None,dls_seconds=2,attempt_wall_seconds=3,accepted=len(calls)==1,final_merit=1/len(calls))
        r=chain_attempt(base,4)
        self.assertEqual(calls,[1,2,3,4]); self.assertEqual(r['dls_requested_cycles'],200)
        self.assertEqual(r['dls_seconds'],8); self.assertEqual(r['attempt_wall_seconds'],12)
        self.assertFalse(r['accepted']); self.assertEqual(r['final_merit'],.25)
        self.assertIsNone(r['dls_completed_cycles'])
    def test_one_call_preserves_endpoint(self):
        def base(): return dict(dls_requested_cycles=50,dls_completed_cycles=7,dls_seconds=2,attempt_wall_seconds=3,accepted=True,final_merit=.1)
        r=chain_attempt(base,1)
        for k,v in base().items(): self.assertEqual(r[k],v)
if __name__=='__main__': unittest.main()
