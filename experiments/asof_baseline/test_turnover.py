"""Execution and causality checks for retention-based turnover experiments."""
import unittest

import numpy as np
import pandas as pd

from turnover import simulate_retention, equal_phase_composite


class TurnoverChecks(unittest.TestCase):
    def setUp(self):
        self.cal = pd.bdate_range("2024-01-01", periods=12)

    def quotes(self, tickers="ABCDEF"):
        return pd.DataFrame([dict(TRADEDATE=day, ticker=ticker, OPEN=100., CLOSE=100.)
                             for day in self.cal for ticker in tickers])

    def signals(self, orders):
        return pd.DataFrame([dict(TRADEDATE=self.cal[i], ticker=ticker, prediction=float(len(order)-rank))
                             for i, order in orders.items() for rank, ticker in enumerate(order)])

    def simulate(self, p, q=None, **kwargs):
        return simulate_retention(p, self.quotes() if q is None else q, self.cal,
                                  end_date=self.cal[-1], **kwargs)

    def test_exact_fees_flat_retained_shares_and_no_borrow(self):
        p = self.signals({i:"A" for i in range(10)})
        d,t,s = self.simulate(p, self.quotes("A"), max_positions=1, enter_rank=1,
                              retain_rank=1, side_bps=25.)
        expected = (1.-.0025)/(1.+.0025)
        self.assertAlmostEqual(d.nav.iloc[-1], expected)
        self.assertEqual(t.action.tolist(), ["BUY", "FINAL_SELL"])
        self.assertAlmostEqual(s["total_cost_per_initial_nav"], 1.-expected)
        self.assertAlmostEqual(s["max_drawdown"], expected-1.)
        self.assertTrue(d.cash.ge(-1e-12).all())
        self.assertTrue(d.exposure.le(1.+1e-12).all())

    def test_entry_next_open_and_fixed_terminal_close(self):
        q = self.quotes("A")
        q.loc[q.TRADEDATE >= self.cal[1], ["OPEN","CLOSE"]] = 200.
        q.loc[q.TRADEDATE == self.cal[-1], "CLOSE"] = 220.
        d,t,s = self.simulate(self.signals({0:"A"}), q, max_positions=1, enter_rank=1, retain_rank=1)
        self.assertAlmostEqual(s["realized_price_return"], .1)
        self.assertEqual(t.TRADEDATE.tolist(), [self.cal[1], self.cal[-1]])

    def test_rank_buffer_retains_old_shares_without_weight_reset(self):
        p = self.signals({0:"ABCDEF", 1:"FBCDEA", 2:"FBCDEA"})
        q = self.quotes()
        q.loc[q.ticker.eq("B") & q.TRADEDATE.ge(self.cal[2]), ["OPEN","CLOSE"]] = 200.
        d, buffered, s = self.simulate(p,q,retain_rank=10)
        _, unbuffered, _ = self.simulate(p,q,retain_rank=5)
        self.assertEqual(set(buffered.loc[buffered.action.eq("BUY"), "ticker"]), set("ABCDE"))
        self.assertEqual(int(buffered.action.eq("BUY").sum()),5)
        self.assertEqual(int(unbuffered.action.eq("BUY").sum()),6)
        self.assertEqual(unbuffered.loc[unbuffered.action.eq("SELL"), "ticker"].tolist(),["A"])
        b = buffered.loc[buffered.ticker.eq("B")]
        self.assertAlmostEqual(b.shares.iloc[0],b.shares.iloc[-1])
        self.assertAlmostEqual(s["realized_price_return"],.2)

    def test_missing_buy_open_has_no_future_based_replacement(self):
        p=self.signals({0:"ABCDEF"})
        p["complete_labels"]=[False,True,True,True,True,True]
        q=self.quotes()
        q.loc[q.ticker.eq("A") & q.TRADEDATE.eq(self.cal[1]),"OPEN"]=np.nan
        d,t,s=self.simulate(p,q)
        self.assertEqual(set(t.loc[t.action.eq("BUY"),"ticker"]),set("BCDE"))
        self.assertNotIn("F",t.ticker.tolist())
        self.assertAlmostEqual(d.loc[d.TRADEDATE.eq(self.cal[1]),"cash"].iloc[0],.2)
        self.assertEqual(s["missing_open_buys"],1)
        p["complete_labels"]=True
        d2,t2,_=self.simulate(p,q)
        pd.testing.assert_frame_equal(d,d2)
        pd.testing.assert_frame_equal(t,t2)

    def test_missing_signal_sells_at_next_available_open_without_slot_leverage(self):
        p=self.signals({0:"A",1:"B"})
        q=self.quotes("AB")
        q.loc[q.ticker.eq("A") & q.TRADEDATE.isin(self.cal[2:4]),["OPEN","CLOSE"]]=np.nan
        q.loc[q.ticker.eq("A") & q.TRADEDATE.ge(self.cal[4]),["OPEN","CLOSE"]]=80.
        d,t,s=self.simulate(p,q,max_positions=1,enter_rank=1,retain_rank=1)
        self.assertEqual(t.action.tolist(),["BUY","DEFER_SELL","DEFER_SELL","SELL"])
        self.assertEqual(t.TRADEDATE.iloc[-1],self.cal[4])
        self.assertTrue(d.holdings.le(1).all())
        self.assertTrue(d.cash.ge(-1e-12).all())
        self.assertAlmostEqual(s["realized_price_return"],-.2)

    def test_same_open_sell_funds_buy_with_no_use_of_close(self):
        p=self.signals({0:"A",1:"B"})
        q=self.quotes("AB")
        q.loc[q.ticker.eq("A") & q.TRADEDATE.ge(self.cal[2]),["OPEN","CLOSE"]]=120.
        q.loc[q.ticker.eq("B"),["OPEN","CLOSE"]]=60.
        q.loc[q.ticker.eq("A") & q.TRADEDATE.eq(self.cal[2]),"CLOSE"]=1000.
        d,t,s=self.simulate(p,q,max_positions=1,enter_rank=1,retain_rank=1)
        day=t.loc[t.TRADEDATE.eq(self.cal[2])]
        self.assertEqual(day.action.tolist(),["SELL","BUY"])
        np.testing.assert_allclose(day.notional,[1.2,1.2])
        self.assertAlmostEqual(s["realized_price_return"],.2)

    def test_unresolved_terminal_position_is_not_realized(self):
        p=self.signals({0:"A"})
        q=self.quotes("A")
        q.loc[q.TRADEDATE.ge(self.cal[5]),["OPEN","CLOSE"]]=np.nan
        d,t,s=self.simulate(p,q,max_positions=1,enter_rank=1,retain_rank=1)
        self.assertEqual(s["status"],"unresolved_positions")
        self.assertIsNone(s["realized_price_return"])
        self.assertIsNone(s["annualized_price_return"])
        self.assertEqual(s["unresolved_tickers"],["A"])

    def test_continuity_quote_can_exit_but_cannot_open_old_receipt(self):
        q = self.quotes("ABCDEF")
        q["entry_allowed"] = True
        q.loc[q.ticker.eq("A") & q.TRADEDATE.ge(self.cal[2]), "entry_allowed"] = False
        # A is already held before conversion; its synthetic quote remains an
        # available exit, then a later high score cannot repurchase the receipt.
        p = self.signals({0:"ABCDEF", 1:"FBCDEA", 2:"ABCDEF"})
        d,t,s = self.simulate(p,q,retain_rank=5)
        a = t.loc[t.ticker.eq("A")]
        self.assertEqual(a.action.tolist(), ["BUY", "SELL", "SKIP_BUY_INELIGIBLE"])
        self.assertEqual(s["ineligible_buys"], 1)
        self.assertAlmostEqual(d.loc[d.TRADEDATE.eq(self.cal[3]), "cash"].iloc[0], .2)
        self.assertAlmostEqual(s["realized_price_return"], 0.)

    def test_missing_eligibility_cannot_be_interpreted_as_permission(self):
        for invalid in (None, np.nan, "False"):
            q = self.quotes("A")
            q["entry_allowed"] = pd.Series([True]*len(q), dtype=object)
            q.loc[1,"entry_allowed"] = invalid
            with self.assertRaisesRegex(ValueError, "entry_allowed"):
                self.simulate(self.signals({0:"A"}),q)

    def test_same_ticker_custody_assumption_applies_only_to_old_holdings(self):
        event = dict(predecessor="A", successor="A", last_predecessor_session=str(self.cal[2].date()),
                     first_tradable_date=str(self.cal[6].date()), eligibility="credited local custody")
        q = self.quotes("A")
        q.loc[q.TRADEDATE.isin(self.cal[3:6]), ["OPEN","CLOSE"]] = np.nan
        _,_,old = self.simulate(self.signals({0:"A"}),q,same_ticker_events=[event])
        _,_,new = self.simulate(self.signals({6:"A"}),q,same_ticker_events=[event])
        self.assertEqual(old["same_ticker_conversion_cohorts"], 1)
        self.assertEqual(old["same_ticker_conversion_uses"][0]["entry_dates"], [str(self.cal[1].date())])
        self.assertEqual(new["same_ticker_conversion_cohorts"], 0)
        self.assertEqual(new["same_ticker_conversion_uses"], [])

    def test_phase_uses_calendar_not_available_signal_count(self):
        p=self.signals({0:"A",2:"A",3:"A",4:"A",5:"A",7:"A"})
        d,t,s=self.simulate(p,self.quotes("A"),rebalance_every=5,phase=1,
                            max_positions=1,enter_rank=1,retain_rank=1)
        self.assertTrue(t.empty) # calendar days1,6 had no decision, not shifted
        self.assertAlmostEqual(d.nav.iloc[-1],1.)
        self.assertEqual(s["phase_origin"],str(self.cal[0].date()))

    def test_future_signals_and_quotes_do_not_change_prefix(self):
        p=self.signals({i:"ABCDEF" for i in range(10)})
        q=self.quotes()
        d,t,_=self.simulate(p,q)
        changed=p.copy()
        changed.loc[changed.TRADEDATE.ge(self.cal[6]),"prediction"] *= -1
        q2=q.copy()
        q2.loc[q2.TRADEDATE.ge(self.cal[6]),["OPEN","CLOSE"]] *= 1.8
        d2,t2,_=self.simulate(changed,q2)
        pd.testing.assert_frame_equal(d.loc[d.TRADEDATE.lt(self.cal[6])],
                                      d2.loc[d2.TRADEDATE.lt(self.cal[6])])
        pd.testing.assert_frame_equal(t.loc[t.TRADEDATE.lt(self.cal[6])],
                                      t2.loc[t2.TRADEDATE.lt(self.cal[6])])

    def test_composite_uses_every_phase_equal_initial_capital(self):
        p=self.signals({i:"ABCDEF" for i in range(10)})
        q=self.quotes()
        q["OPEN"] *= q.TRADEDATE.map({d:1+i*.01 for i,d in enumerate(self.cal)})
        q["CLOSE"]=q.OPEN
        phases=[self.simulate(p,q,rebalance_every=5,phase=phase,side_bps=10.)
                for phase in range(5)]
        d,t,s=equal_phase_composite(phases)
        np.testing.assert_allclose(d.nav,np.mean([x[0].nav for x in phases],axis=0))
        self.assertAlmostEqual(s["total_cost_per_initial_nav"],
                               np.mean([x[2]["total_cost_per_initial_nav"] for x in phases]))
        self.assertEqual(s["phase"],"equal_composite")
        self.assertEqual(d.TRADEDATE.iloc[0],self.cal[0])


if __name__ == "__main__":
    unittest.main()
