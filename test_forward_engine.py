"""
Unit + integration tests for forward_test.py (no network needed).
Run: python test_forward_engine.py
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

TMP = tempfile.mkdtemp(prefix="fwdtest_")
os.environ["FORWARD_DIR"] = os.path.join(TMP, "forward")

import forward_test as ft  # noqa: E402  (after env var is set)


def make_prices(n=260, spy_trend="up", end="2025-06-30"):
    """Deterministic synthetic price panel for SPY/IEF/GLD/SHY."""
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=n)
    base = {"SPY": 500.0, "IEF": 95.0, "GLD": 200.0, "SHY": 84.0}
    spy_drift = {"up": 0.0008, "down": -0.002}[spy_trend]
    data = {}
    for t, p0 in base.items():
        d = 0.0001 if t != "SPY" else spy_drift
        data[t] = p0 * np.exp(d * np.arange(n))
    return pd.DataFrame(data, index=idx)


class TestSignal(unittest.TestCase):
    def test_risk_on_when_spy_above_ma(self):
        px = make_prices(spy_trend="up")
        regime, price, ma = ft.compute_signal(px, px.index[-1])
        self.assertEqual(regime, "RISK_ON")
        self.assertGreater(price, ma)

    def test_risk_off_when_spy_below_ma(self):
        px = make_prices(n=300, spy_trend="down")
        regime, price, ma = ft.compute_signal(px, px.index[-1])
        self.assertEqual(regime, "RISK_OFF")
        self.assertLess(price, ma)

    def test_target_weights(self):
        self.assertIn("SPY", ft.target_weights("RISK_ON"))
        self.assertIn("SHY", ft.target_weights("RISK_OFF"))
        self.assertNotIn("SPY", ft.target_weights("RISK_OFF"))
        self.assertAlmostEqual(sum(ft.target_weights("RISK_ON").values()), 1.0)
        self.assertAlmostEqual(sum(ft.target_weights("RISK_OFF").values()), 1.0)


class TestMonthEnd(unittest.TestCase):
    def test_last_trading_day_detection(self):
        px = make_prices(end="2025-06-30")
        self.assertTrue(ft.is_last_trading_day_of_month(px.index, px.index[-1]))
        self.assertFalse(ft.is_last_trading_day_of_month(px.index, px.index[-5]))


class TestTradePlanning(unittest.TestCase):
    def test_tolerance_band_blocks_small_drift(self):
        state = {"cash": 0.0, "shares": {"SPY": 11.0, "IEF": 36.8, "GLD": 5.0}}
        row = pd.Series({"SPY": 500.0, "IEF": 95.0, "GLD": 200.0, "SHY": 84.0})
        # weights ~ 55/35/10 by construction
        nav = sum(state["shares"][t] * row[t] for t in state["shares"])
        tw = {"SPY": 0.55, "IEF": 0.35, "GLD": 0.10}
        trades, drift, nav2 = ft.plan_trades(state, row, tw)
        self.assertEqual(trades, [])
        self.assertLessEqual(drift, ft.CONFIG["tolerance"])
        self.assertAlmostEqual(nav, nav2)

    def test_rebalance_trades_sum_and_costs(self):
        state = {"cash": 0.0, "shares": {"SPY": 20.0}}  # 100% SPY
        row = pd.Series({"SPY": 500.0, "IEF": 100.0, "GLD": 200.0, "SHY": 80.0})
        tw = {"SHY": 0.55, "IEF": 0.35, "GLD": 0.10}
        trades, drift, nav = ft.plan_trades(state, row, tw)
        self.assertGreater(len(trades), 0)
        total_cost = sum(t["cost"] for t in trades)
        self.assertAlmostEqual(total_cost,
                               sum(abs(t["value"]) for t in trades) * 0.001)
        # after trades, values should match targets
        new_state = {"cash": state["cash"], "shares": dict(state["shares"])}
        ft.apply_trades(new_state, trades, pd.Timestamp("2025-06-30"), "TEST")
        for t, w in tw.items():
            val = new_state["shares"].get(t, 0.0) * row[t]
            self.assertAlmostEqual(val / nav, w, places=2)


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.dir = os.path.join(TMP, "e2e")
        os.makedirs(self.dir, exist_ok=True)
        # point module file paths at isolated dir
        self._old = (ft.FORWARD_DIR, ft.STATE_FILE, ft.NAV_FILE, ft.TRADES_FILE,
                     ft.SIGNALS_FILE, ft.RUNS_FILE, ft.REPORT_FILE, ft.SUMMARY_FILE)
        d = os.path.join(self.dir, "forward")
        ft.FORWARD_DIR = d
        ft.STATE_FILE = os.path.join(d, "state.json")
        ft.NAV_FILE = os.path.join(d, "nav_history.csv")
        ft.TRADES_FILE = os.path.join(d, "trades.csv")
        ft.SIGNALS_FILE = os.path.join(d, "signals.csv")
        ft.RUNS_FILE = os.path.join(d, "runs.csv")
        ft.REPORT_FILE = os.path.join(d, "report.html")
        ft.SUMMARY_FILE = os.path.join(d, "summary.json")
        self._orig_load = ft.load_prices

    def tearDown(self):
        ft.load_prices = self._orig_load
        (ft.FORWARD_DIR, ft.STATE_FILE, ft.NAV_FILE, ft.TRADES_FILE,
         ft.SIGNALS_FILE, ft.RUNS_FILE, ft.REPORT_FILE, ft.SUMMARY_FILE) = self._old

    def test_bootstrap_idempotency_and_regime_switch(self):
        px1 = make_prices(spy_trend="up", end="2025-06-30")  # month end, risk-on
        ft.load_prices = lambda end_date=None: px1

        st = ft.run()
        self.assertEqual(st["experiment_id"], ft.CONFIG["experiment_id"])
        self.assertIn("SPY", st["shares"])
        self.assertNotIn("SHY", st["shares"])
        nav1 = st["nav"]
        self.assertAlmostEqual(nav1, 10_000 - st["total_costs"], places=2)
        self.assertLess(st["cash"], 50)  # nearly fully invested

        # idempotency: second run same date -> no new nav row, no trades
        n_nav_before = len(ft.read_nav_history())
        st2 = ft.run()
        self.assertEqual(len(ft.read_nav_history()), n_nav_before)
        trades_df = pd.read_csv(ft.TRADES_FILE)
        self.assertTrue((trades_df["reason"] == "INITIAL_ALLOCATION").all())

        # regime switch at next month end: SPY crashes below MA200
        px2_up = make_prices(spy_trend="up", end="2025-06-30")
        crash = px2_up.copy()
        july = pd.bdate_range("2025-07-01", "2025-07-31")
        crash_part = pd.DataFrame(
            {t: crash[t].iloc[-1] * (1 + np.random.default_rng(1).normal(
                -0.02 if t == "SPY" else 0.0, 0.005, len(july))).cumprod()
             for t in crash.columns}, index=july)
        px2 = pd.concat([crash, crash_part])
        # force SPY well below its MA200
        px2["SPY"] = px2["SPY"] * np.concatenate(
            [np.ones(len(crash)), np.linspace(1.0, 0.80, len(july))])
        ft.load_prices = lambda end_date=None: px2

        st3 = ft.run()
        self.assertIn("SHY", st3["shares"])
        self.assertLess(st3["shares"].get("SPY", 0.0) * px2["SPY"].iloc[-1] /
                        st3["nav"], 0.05)
        trades_df = pd.read_csv(ft.TRADES_FILE)
        self.assertTrue((trades_df["reason"] == "MONTH_END_REBALANCE").any())
        # accounting identity: NAV must equal cash + holdings marked to market
        last_row = px2.iloc[-1]
        recomputed = st3["cash"] + sum(st3["shares"].get(t, 0.0) * last_row[t]
                                       for t in last_row.index)
        self.assertAlmostEqual(st3["nav"], recomputed, places=6)
        # defensive rotation must beat the SPY crash
        self.assertGreater(st3["nav"] / 10_000 - 1,
                           st3["spy_bench_nav"] / 10_000 - 1)

        # run mid-month: no trading
        px3 = px2.iloc[:-10]  # mid July
        ft.load_prices = lambda end_date=None: px3
        st4 = ft.run()
        trades_df2 = pd.read_csv(ft.TRADES_FILE)
        self.assertEqual(len(trades_df2), len(trades_df))

        # report + summary exist
        self.assertTrue(os.path.exists(ft.REPORT_FILE))
        with open(ft.SUMMARY_FILE, "r", encoding="utf-8") as f:
            summary = json.load(f)
        self.assertIn("nav", summary)


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)