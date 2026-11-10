"""Accounting tests that would detect executable-portfolio optimism."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from portfolio import simulate, quotes_from_raw


class PortfolioChecks(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.bdate_range("2024-01-01", periods=12)

    def quotes(self, tickers=("A",)):
        return pd.DataFrame([dict(TRADEDATE=d, ticker=t, OPEN=100., CLOSE=100.)
                             for d in self.calendar for t in tickers])

    def signals(self, days=(0,), tickers=("A",)):
        return pd.DataFrame([dict(TRADEDATE=self.calendar[i], ticker=t,
                                  prediction=float(len(tickers)-j))
                             for i in days for j, t in enumerate(tickers)])

    def test_flat_price_exact_two_sided_cost_and_no_borrowing(self):
        d, t, s = simulate(self.signals(), self.quotes(), self.calendar, side_bps=25.)
        expected = .8 + .2 * (1.-.0025)/(1.+.0025)
        self.assertAlmostEqual(d.nav.iloc[-1], expected)
        self.assertGreaterEqual(d.cash.min(), -1e-12)
        self.assertAlmostEqual(s["realized_price_return"], expected-1.)
        self.assertAlmostEqual(s["total_cost_per_initial_nav"], 1.-expected)
        self.assertAlmostEqual(s["max_drawdown"], expected-1.)
        self.assertEqual(t.action.tolist(), ["BUY", "SELL"])

    def test_entry_uses_next_open_and_exit_fifth_session_close(self):
        q = self.quotes()
        q.loc[q.TRADEDATE >= self.calendar[1], ["OPEN", "CLOSE"]] = 200.
        q.loc[q.TRADEDATE == self.calendar[5], "CLOSE"] = 220.
        d, t, s = simulate(self.signals(), q, self.calendar)
        self.assertAlmostEqual(d.nav.iloc[-1], 1.02)
        self.assertEqual(t.TRADEDATE.tolist(), [self.calendar[1], self.calendar[5]])
        self.assertEqual(s["sessions"], 6)

    def test_suspension_retains_position_and_does_not_fund_same_morning(self):
        q = self.quotes()
        q.loc[q.TRADEDATE == self.calendar[5], ["OPEN", "CLOSE"]] = np.nan
        q.loc[q.TRADEDATE == self.calendar[6], "CLOSE"] = 110.
        d, t, s = simulate(self.signals(days=(0, 5)), q, self.calendar)
        self.assertEqual(t.action.tolist(), ["BUY", "SKIP_NO_CASH", "SELL"])
        self.assertEqual(s["delayed_exits"], 1)
        self.assertEqual(s["stale_nav_days"], 1)
        self.assertEqual(s["skipped_cashless_sleeves"], 1)
        self.assertAlmostEqual(d.nav.iloc[-1], 1.02)

    def test_missing_entry_is_not_replaced_using_future_information(self):
        tickers = tuple("ABCDEF")
        p = self.signals(tickers=tickers)
        p["complete_labels"] = [False, True, True, True, True, True]
        q = self.quotes(tickers)
        q.loc[(q.TRADEDATE == self.calendar[1]) & q.ticker.eq("A"), "OPEN"] = np.nan
        d, t, s = simulate(p, q, self.calendar)
        self.assertEqual(s["skipped_open_orders"], 1)
        self.assertEqual(set(t.loc[t.action.eq("BUY"), "ticker"]), set("BCDE"))
        self.assertNotIn("F", t.ticker.tolist())
        self.assertAlmostEqual(d.nav.iloc[-1], 1.)
        p["complete_labels"] = True
        d2, t2, s2 = simulate(p, q, self.calendar)
        pd.testing.assert_frame_equal(d, d2)
        pd.testing.assert_frame_equal(t, t2)

    def test_unresolved_exit_cannot_be_reported_as_realized_return(self):
        q = self.quotes()
        q.loc[q.TRADEDATE >= self.calendar[5], ["OPEN", "CLOSE"]] = np.nan
        d, t, s = simulate(self.signals(), q, self.calendar)
        self.assertEqual(s["status"], "unresolved_positions")
        self.assertIsNone(s["realized_price_return"])
        self.assertIsNone(s["annualized_price_return"])
        self.assertEqual(s["unresolved_positions"], 1)

    def test_continuity_quote_values_old_holding_without_new_entry(self):
        q = self.quotes(tickers=("A",))
        q["entry_allowed"] = q.TRADEDATE.lt(self.calendar[5])
        q.loc[q.TRADEDATE.ge(self.calendar[5]), ["OPEN","CLOSE"]] = 110.
        d,t,s = simulate(self.signals(days=(0,5)),q,self.calendar)
        self.assertEqual(t.action.tolist(), ["BUY", "SELL", "SKIP_BUY_INELIGIBLE"])
        self.assertAlmostEqual(s["realized_price_return"], .02)
        self.assertEqual(s["skipped_ineligible_orders"], 1)

    def test_ineligible_entry_does_not_promote_lower_ranked_stock(self):
        tickers=tuple("ABCDEF")
        q = self.quotes(tickers)
        q["entry_allowed"] = q.ticker.ne("A")
        d,t,s = simulate(self.signals(tickers=tickers),q,self.calendar)
        self.assertEqual(set(t.loc[t.action.eq("BUY"), "ticker"]), set("BCDE"))
        self.assertNotIn("F", t.ticker.tolist())
        self.assertAlmostEqual(s["realized_price_return"], 0.)
        self.assertEqual(s["skipped_ineligible_orders"], 1)

    def test_missing_eligibility_fails_closed(self):
        q = self.quotes()
        q["entry_allowed"] = pd.Series([True]*len(q), dtype=object)
        q.loc[1,"entry_allowed"] = None
        with self.assertRaisesRegex(ValueError, "entry_allowed"):
            simulate(self.signals(),q,self.calendar)

    def test_same_ticker_conversion_labels_old_receipt_cohort_only(self):
        event = dict(predecessor="A", successor="A", last_predecessor_session=str(self.calendar[2].date()),
                     first_tradable_date=str(self.calendar[6].date()), eligibility="credited local custody")
        q = self.quotes()
        q.loc[q.TRADEDATE.isin(self.calendar[3:6]), ["OPEN","CLOSE"]] = np.nan
        _,_,s = simulate(self.signals(days=(0,6)),q,self.calendar,same_ticker_events=[event])
        self.assertEqual(s["same_ticker_conversion_cohorts"], 1)
        self.assertEqual(s["same_ticker_conversion_uses"][0]["entry_dates"], [str(self.calendar[1].date())])
        self.assertEqual(s["delayed_exits"], 1)

    def test_suspension_spanning_weeks_keeps_shares_until_observed_exit(self):
        self.calendar = pd.bdate_range("2024-01-01", periods=22)
        q = self.quotes()
        q.loc[(q.TRADEDATE >= self.calendar[5]) &
              (q.TRADEDATE < self.calendar[14]), ["OPEN", "CLOSE"]] = np.nan
        q.loc[q.TRADEDATE >= self.calendar[14], ["OPEN", "CLOSE"]] = 80.
        d, t, s = simulate(self.signals(days=(0, 5, 10)), q, self.calendar)
        self.assertEqual(s["skipped_cashless_sleeves"], 2)
        self.assertEqual(s["stale_nav_days"], 9)
        self.assertEqual(s["delayed_exits"], 1)
        self.assertEqual(t.loc[t.action.eq("SELL"), "TRADEDATE"].iloc[0], self.calendar[14])
        self.assertAlmostEqual(s["realized_price_return"], -.04)

    def test_suspended_position_does_not_lock_returned_cash_from_other_stock(self):
        q = self.quotes(tickers=("A", "B"))
        suspended = (q.ticker.eq("A") & q.TRADEDATE.ge(self.calendar[5]) &
                     q.TRADEDATE.lt(self.calendar[8]))
        q.loc[suspended, ["OPEN", "CLOSE"]] = np.nan
        q.loc[q.ticker.eq("A") & q.TRADEDATE.ge(self.calendar[8]), ["OPEN", "CLOSE"]] = 80.
        d, t, s = simulate(self.signals(days=(0, 5), tickers=("A", "B")),
                           q, self.calendar)
        self.assertEqual(s["partial_recycles"], 1)
        self.assertEqual(s["skipped_cashless_sleeves"], 0)
        self.assertEqual(len(t.loc[t.action.eq("BUY") & t.ticker.eq("B")]), 2)
        morning = t.loc[t.action.eq("BUY") & t.TRADEDATE.eq(self.calendar[6])]
        self.assertEqual(morning.ticker.tolist(), ["B"])
        self.assertAlmostEqual(morning.notional.sum(), .05)
        old_exit = t.loc[t.action.eq("SELL") & t.ticker.eq("A")]
        self.assertEqual(old_exit.TRADEDATE.tolist(), [self.calendar[8]])
        self.assertAlmostEqual(old_exit.notional.sum(), .08)
        self.assertGreaterEqual(d.cash.min(), -1e-12)
        self.assertTrue(d.exposure.le(1. + 1e-12).all())
        self.assertAlmostEqual(s["realized_price_return"], -.02)

    def test_horizon_must_fit_calendar_and_holdout_cutoff(self):
        with self.assertRaisesRegex(ValueError, "planned exit"):
            simulate(self.signals(days=(8,)), self.quotes(), self.calendar)
        with self.assertRaisesRegex(ValueError, "planned exit"):
            simulate(self.signals(), self.quotes(), self.calendar,
                     end_date=self.calendar[4])

    def test_raw_quote_loader_detects_large_jump_after_suspension(self):
        with TemporaryDirectory() as directory:
            raw = Path(directory)
            market = pd.DataFrame({"TRADEDATE": self.calendar, "OPEN": 100.,
                                   "HIGH": 100., "LOW": 100., "CLOSE": 100.})
            market.to_csv(raw / "IMOEX.csv", index=False)
            stock = market.copy()
            stock.loc[1:3, ["OPEN", "HIGH", "LOW", "CLOSE"]] = np.nan
            stock.loc[4:, ["OPEN", "HIGH", "LOW", "CLOSE"]] = 300.
            stock.to_csv(raw / "A.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Unexplained >70%"):
                quotes_from_raw(raw, ["A"])

    def test_five_sleeves_fully_invest_without_leverage(self):
        d, t, s = simulate(self.signals(days=range(5)), self.quotes(), self.calendar)
        self.assertTrue(np.allclose(d.nav, 1.))
        self.assertGreaterEqual(d.cash.min(), -1e-12)
        self.assertEqual(int(t.action.eq("BUY").sum()), 5)
        self.assertEqual(int(t.action.eq("SELL").sum()), 5)
        self.assertAlmostEqual(s["total_notional_per_initial_nav"], 2.)


if __name__ == "__main__":
    unittest.main()
