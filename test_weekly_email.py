"""
Tests for send_weekly_email.py.
Run: python -m pytest test_weekly_email.py -v

Mutation contract:
  test_email_uses_static_benchmark_and_verdict MUST FAIL on the pre-fix
  code that used spy_bench_nav as primary benchmark and omitted verdict.
  Verified by running against unmodified send_weekly_email.py.
"""

import json
import os
import shutil
import tempfile
import unittest

import pandas as pd

TMP = tempfile.mkdtemp(prefix="emailtest_")


def _write_fixture(d, *, nav=11000, bench_nav=10500, spy_bench_nav=12000,
                   after_tax_nav=10800, bench_after_tax_nav=10400,
                   verdict="NOT_READY", risk_off_cycles=0, years_elapsed=0.5,
                   include_v2_fields=True):
    """Create fake forward/ data with widely separated benchmark values."""
    os.makedirs(d, exist_ok=True)
    state = {
        "experiment_id": "test-fixture",
        "start_date": "2026-01-01", "start_capital": 10000.0,
        "cash": -10.0, "shares": {"SPY": 10.0, "IEF": 30.0, "GLD": 3.0},
        "total_costs": 10.0, "spy_start_price": 500.0,
        "last_run_date": "2026-06-30", "last_trade_month": "2026-06",
        "nav": nav, "spy_bench_nav": spy_bench_nav,
        "benchmark": {"nav": bench_nav, "after_tax_nav": bench_after_tax_nav},
        "after_tax_nav": after_tax_nav,
        "tax_lots": {}, "loss_carryforward": 0, "total_realized_tax": 0,
        "risk_off_cycles": [], "current_risk_off_entry": None,
    }
    with open(os.path.join(d, "state.json"), "w") as f:
        json.dump(state, f)

    summary = {
        "experiment_id": "test-fixture", "asof": "2026-06-30",
        "status": "HOLD", "regime": "RISK_ON",
        "spy_price": 600.0, "ma200": 550.0,
        "nav": nav, "total_return_pct": (nav / 10000 - 1) * 100,
        "bench_return_pct": (bench_nav / 10000 - 1) * 100,
        "spy_bench_return_pct": (spy_bench_nav / 10000 - 1) * 100,
        "n_trades": 0, "holdings": state["shares"],
        "cash": -10.0,
    }
    if include_v2_fields:
        summary.update({
            "bench_nav": bench_nav,
            "bench_after_tax_nav": bench_after_tax_nav,
            "after_tax_nav": after_tax_nav,
            "verdict": verdict,
            "risk_off_cycles_completed": risk_off_cycles,
            "years_elapsed": years_elapsed,
            "horizon_met": False,
        })
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(summary, f)

    nav_csv = pd.DataFrame([
        {"date": "2026-01-01", "nav": 9990, "after_tax_nav": 9980,
         "bench_nav": 9990, "bench_after_tax_nav": 9980,
         "spy_bench_nav": 10000, "regime": "RISK_ON"},
        {"date": "2026-06-30", "nav": nav, "after_tax_nav": after_tax_nav,
         "bench_nav": bench_nav, "bench_after_tax_nav": bench_after_tax_nav,
         "spy_bench_nav": spy_bench_nav, "regime": "RISK_ON"},
    ])
    nav_csv.to_csv(os.path.join(d, "nav_history.csv"), index=False)

    trades = pd.DataFrame(columns=["date", "ticker", "value", "reason"])
    trades.to_csv(os.path.join(d, "trades.csv"), index=False)

    signals = pd.DataFrame([
        {"date": "2026-06-30", "regime": "RISK_ON", "spy_price": 600, "ma200": 550},
    ])
    signals.to_csv(os.path.join(d, "signals.csv"), index=False)


class TestWeeklyEmailContent(unittest.TestCase):
    def setUp(self):
        self.dir = os.path.join(TMP, self._testMethodName)
        _write_fixture(self.dir)
        import send_weekly_email as swe
        self._mod = swe
        self._old_fwd = swe.FORWARD_DIR
        swe.FORWARD_DIR = self.dir

    def tearDown(self):
        self._mod.FORWARD_DIR = self._old_fwd

    def test_email_uses_static_benchmark_and_verdict(self):
        """
        Primary benchmark must be bench_nav (static 55/35/10), not spy_bench_nav.
        Verdict must appear in the email.

        With the fixture (nav=11000, bench_nav=10500, spy_bench_nav=12000):
          correct gap = +10% - 5% = +5.00%
          wrong gap   = +10% - 20% = -10.00%

        Old code computes the wrong gap and omits verdict → FAILS.
        """
        html, _ = self._mod.build_html()
        self.assertIn("NOT_READY", html, "verdict must appear in email")
        self.assertIn("+5.00%", html,
                      "primary gap must be vs static benchmark (+5%), not SPY")
        self.assertNotIn("פער מול SPY", html,
                         "SPY must not be labeled as the primary benchmark")

    def test_subject_line_uses_static_benchmark(self):
        """Subject line must reference static benchmark gap, not SPY."""
        html, total_ret = self._mod.build_html()
        subject = self._mod._build_subject(total_ret)
        self.assertNotIn("-10.00", subject,
                         "subject must not show SPY gap as headline")

    def test_experiment_progress_shown(self):
        """RISK_OFF cycle count and time horizon must appear."""
        html, _ = self._mod.build_html()
        self.assertIn("0/2", html, "risk_off cycles must show 0 of 2")
        self.assertIn("0.5", html, "years elapsed must appear")

    def test_missing_v2_fields_show_dash(self):
        """If summary.json lacks v2 fields, display '—' instead of crashing."""
        _write_fixture(self.dir, include_v2_fields=False)
        html, _ = self._mod.build_html()
        self.assertIn("—", html, "missing fields must render as dash")
        self.assertIsInstance(html, str)


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
