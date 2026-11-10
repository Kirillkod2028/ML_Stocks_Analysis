"""Chronology and return-unit checks for the rank/absolute comparison."""
import unittest

import numpy as np
import pandas as pd

from compare_rank_return import inner_split, market_features, net_expected_return, tune_and_fit


class RankReturnChecks(unittest.TestCase):
    def test_market_features_have_prefix_and_future_perturbation_invariance(self):
        dates=pd.bdate_range("2023-01-02",periods=100)
        close=100*np.exp(np.cumsum(np.sin(np.arange(100))*.01+.0001))
        raw=pd.DataFrame({"TRADEDATE":dates,"CLOSE":close})
        full=market_features(raw,dates)
        prefix=market_features(raw.iloc[:60],dates[:60])
        pd.testing.assert_frame_equal(prefix,full.iloc[:60])
        changed=raw.copy();changed.loc[60:,"CLOSE"]*=100
        pd.testing.assert_frame_equal(full.iloc[:60],market_features(changed,dates).iloc[:60])
        self.assertAlmostEqual(full.iloc[40].market_R5,close[40]/close[35]-1)

    def test_missing_market_price_is_not_filled(self):
        dates=pd.bdate_range("2023-01-02",periods=30)
        raw=pd.DataFrame({"TRADEDATE":dates,"CLOSE":np.arange(30)+100.})
        raw.loc[25,"CLOSE"]=np.nan
        out=market_features(raw,dates)
        self.assertTrue(np.isnan(out.iloc[25].market_R5))
        self.assertTrue(np.isnan(out.iloc[26].market_sigma20))

    def test_net_return_accounts_for_cash_budgeted_entry_cost(self):
        c=.001
        self.assertAlmostEqual(net_expected_return(.02,10),1.02*(1-c)/(1+c)-1)
        self.assertAlmostEqual(net_expected_return((1+c)/(1-c)-1,10),0)
        self.assertLess(net_expected_return(0.,10),0.)
        with self.assertRaises(ValueError):net_expected_return(0.,-1)

    def test_inner_split_purges_overlapping_label_windows(self):
        dates=pd.bdate_range("2023-01-02",periods=120)
        rows=pd.DataFrame({"TRADEDATE":dates,"label_end":dates+pd.offsets.BDay(5)})
        fit,val=inner_split(rows)
        self.assertLess(fit.label_end.max(),val.TRADEDATE.min())
        self.assertEqual(val.TRADEDATE.nunique(),63)

    def test_absolute_forecast_does_not_require_test_labels(self):
        dates=pd.bdate_range("2023-01-02",periods=120)
        train=pd.DataFrame({"TRADEDATE":dates,"label_end":dates+pd.offsets.BDay(5),
                            "ticker":"A","x":np.sin(np.arange(120)),"y_return5":.01})
        test=pd.DataFrame({"x":[0.,1.],"y_return5":[np.nan,np.nan]})
        pred,_,_=tune_and_fit(train,test,["x"],"absolute")
        np.testing.assert_allclose(pred,[.01,.01])


if __name__=="__main__":unittest.main()
