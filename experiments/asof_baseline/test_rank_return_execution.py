import unittest
import numpy as np
import pandas as pd
from run_rank_return import make_decisions


class RankReturnExecutionTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"TRADEDATE": pd.to_datetime(["2024-01-01"]*3),
                             "ticker": list("ABC"), "rank_score": [3.,2.,1.],
                             "expected_return": [-.01,.001,.01]})

    def test_cost_threshold_and_sunk_entry_cost(self):
        d=make_decisions(self.frame(),"rank_cash",10)
        self.assertEqual(d.can_enter.tolist(),[False,False,True])
        self.assertEqual(d.can_hold.tolist(),[False,True,True])
        self.assertEqual(d.prediction.tolist(),[3.,2.,1.])

    def test_absolute_ordering_and_all_in_control(self):
        d=make_decisions(self.frame(),"return_always",25)
        self.assertEqual(d.prediction.tolist(),[-.01,.001,.01])
        self.assertTrue(d.can_enter.all() and d.can_hold.all())

    def test_outcome_availability_cannot_change_decisions(self):
        p=self.frame().assign(y_return5=np.nan,complete_labels=False)
        a=make_decisions(p,"rank_cash",10)
        p["y_return5"]=100.;p["complete_labels"]=True
        pd.testing.assert_frame_equal(a,make_decisions(p,"rank_cash",10))


if __name__=="__main__":
    unittest.main()
