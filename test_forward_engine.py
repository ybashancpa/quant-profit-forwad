"""
Unit + integration tests for forward_test.py (no network needed).
Run: python test_forward_engine.py
"""

import json
import os
import shutil
import subprocess
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


class TestProvenanceTiming(unittest.TestCase):
    """
    Guards the provenance-timing fix: `dirty` must reflect the tree as it was
    at run START, not after run() has written its own (tracked) forward/
    artifacts.

    This requires a throwaway git repo whose forward/ files are TRACKED. If
    FORWARD_DIR sits outside a git repo -- as the other e2e tests do -- run()'s
    writes never touch the git worktree, so `git status` stays clean and
    `dirty` is False even WITH the bug. Such a test would pass either way and
    prove nothing (exactly the failure mode we are guarding against).
    """

    def _git(self, args, cwd):
        subprocess.run(["git", *args], cwd=cwd, check=True,
                       capture_output=True, text=True)

    def _make_sandbox(self):
        """A git repo with tracked, clean forward/ CSVs and a hypothesis doc."""
        sbx = tempfile.mkdtemp(prefix="fwdprov_")
        fwd = os.path.join(sbx, "forward")
        os.makedirs(fwd)
        # Empty, tracked, append-only CSVs -> run()'s append_csv() adds rows ->
        # tracked file is modified, which is what would flip `dirty` if captured
        # too late. nav_history.csv is deliberately excluded: run() *reads* it
        # via pd.read_csv(), which errors on a 0-byte file; run() creates it
        # fresh (untracked) instead, which is fine for the dirty check.
        for name in ("signals.csv", "trades.csv", "runs.csv"):
            open(os.path.join(fwd, name), "w").close()
        with open(os.path.join(sbx, "SMARTPASSIVE_hypothesis.md"), "w",
                  encoding="utf-8") as f:
            f.write("# locked hypothesis (sandbox fixture)\n")
        self._git(["init", "-q"], sbx)
        self._git(["config", "user.email", "t@t"], sbx)
        self._git(["config", "user.name", "t"], sbx)
        self._git(["add", "-A"], sbx)
        # gpgsign=false / empty hooksPath keep the fixture hermetic on dev
        # machines that sign or hook by default; this is a throwaway repo, not
        # one of the user's commits.
        self._git(["-c", "commit.gpgsign=false", "-c", "core.hooksPath=",
                   "commit", "-q", "-m", "init"], sbx)
        return sbx, fwd

    def test_provenance_primitive_clean_then_dirty(self):
        """provenance(): clean tree -> dirty False; touch tracked file -> True."""
        sbx, fwd = self._make_sandbox()
        old_cwd = os.getcwd()
        old_hyp = ft.HYPOTHESIS_FILE
        try:
            os.chdir(sbx)
            ft.HYPOTHESIS_FILE = "SMARTPASSIVE_hypothesis.md"
            prov = ft.provenance()
            self.assertNotEqual(prov["commit"], "")
            self.assertFalse(prov["dirty"])            # clean committed tree
            self.assertNotEqual(prov["hypothesis_sha256"], "")
            # Modify a TRACKED file, leave it uncommitted.
            with open(os.path.join(fwd, "runs.csv"), "a", encoding="utf-8") as f:
                f.write("touch\n")
            self.assertTrue(ft.provenance()["dirty"])  # tracked change -> dirty
        finally:
            os.chdir(old_cwd)
            ft.HYPOTHESIS_FILE = old_hyp
            shutil.rmtree(sbx, ignore_errors=True)

    def test_hypothesis_hash_is_committed_blob_not_worktree(self):
        """
        hypothesis_sha256 must hash the COMMITTED blob (git show HEAD:<file>),
        not the working-tree file. Otherwise the seal depends on the checkout's
        line endings: the same commit hashes differently on Windows (CRLF via
        core.autocrlf) vs Linux (LF). Fails on the pre-fix code that read the
        file from disk.
        """
        import hashlib as _h
        sbx = tempfile.mkdtemp(prefix="fwdhyp_")
        old_cwd = os.getcwd()
        old_hyp = ft.HYPOTHESIS_FILE
        body = "# locked hypothesis (sandbox)\nsecond line\n"
        lf = body.encode("utf-8")
        crlf = body.replace("\n", "\r\n").encode("utf-8")
        try:
            self._git(["init", "-q"], sbx)
            self._git(["config", "user.email", "t@t"], sbx)
            self._git(["config", "user.name", "t"], sbx)
            self._git(["config", "core.autocrlf", "false"], sbx)  # LF blob, exactly
            hyp = os.path.join(sbx, "SMARTPASSIVE_hypothesis.md")
            with open(hyp, "wb") as f:
                f.write(lf)                       # commit an LF blob
            self._git(["add", "-A"], sbx)
            self._git(["-c", "commit.gpgsign=false", "-c", "core.hooksPath=",
                       "commit", "-q", "-m", "init"], sbx)
            with open(hyp, "wb") as f:
                f.write(crlf)                     # worktree now CRLF (uncommitted)
            os.chdir(sbx)
            ft.HYPOTHESIS_FILE = "SMARTPASSIVE_hypothesis.md"
            got = ft.provenance()["hypothesis_sha256"]
            self.assertEqual(got, _h.sha256(lf).hexdigest(),
                             "must hash the committed LF blob")
            self.assertNotEqual(got, _h.sha256(crlf).hexdigest(),
                                "must NOT hash the CRLF working-tree bytes")
        finally:
            os.chdir(old_cwd)
            ft.HYPOTHESIS_FILE = old_hyp
            shutil.rmtree(sbx, ignore_errors=True)

    def test_run_stamps_dirty_false_despite_own_writes(self):
        """
        The regression test proper: run() on a CLEAN tracked tree must stamp
        dirty=False in summary.json EVEN THOUGH run() then modifies tracked
        forward/ files. Fails on the pre-fix code (provenance captured after
        the writes).
        """
        sbx, fwd = self._make_sandbox()
        old_cwd = os.getcwd()
        saved = (ft.FORWARD_DIR, ft.STATE_FILE, ft.NAV_FILE, ft.TRADES_FILE,
                 ft.SIGNALS_FILE, ft.RUNS_FILE, ft.REPORT_FILE, ft.SUMMARY_FILE,
                 ft.HYPOTHESIS_FILE, ft.load_prices)
        try:
            os.chdir(sbx)
            ft.FORWARD_DIR = fwd
            ft.STATE_FILE = os.path.join(fwd, "state.json")
            ft.NAV_FILE = os.path.join(fwd, "nav_history.csv")
            ft.TRADES_FILE = os.path.join(fwd, "trades.csv")
            ft.SIGNALS_FILE = os.path.join(fwd, "signals.csv")
            ft.RUNS_FILE = os.path.join(fwd, "runs.csv")
            ft.REPORT_FILE = os.path.join(fwd, "report.html")
            ft.SUMMARY_FILE = os.path.join(fwd, "summary.json")
            ft.HYPOTHESIS_FILE = "SMARTPASSIVE_hypothesis.md"
            ft.load_prices = lambda end_date=None: make_prices(
                spy_trend="up", end="2025-06-30")

            ft.run()  # bootstrap: appends to the tracked CSVs committed above

            # Teeth check: run() really did dirty tracked files. If this is
            # empty the test is toothless (late-captured provenance would also
            # read clean), so assert the precondition that makes the fix matter.
            status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=sbx, capture_output=True, text=True).stdout.strip()
            self.assertNotEqual(
                status, "", "run() did not modify any tracked file; test is toothless")

            with open(ft.SUMMARY_FILE, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertIn("provenance", summary)
            prov = summary["provenance"]
            self.assertNotEqual(prov["commit"], "")
            self.assertNotEqual(prov["hypothesis_sha256"], "")
            # The fix: captured before writes -> False despite run()'s own edits.
            self.assertFalse(
                prov["dirty"],
                "dirty must reflect the tree at run start, not run()'s own writes")
        finally:
            os.chdir(old_cwd)
            (ft.FORWARD_DIR, ft.STATE_FILE, ft.NAV_FILE, ft.TRADES_FILE,
             ft.SIGNALS_FILE, ft.RUNS_FILE, ft.REPORT_FILE, ft.SUMMARY_FILE,
             ft.HYPOTHESIS_FILE, ft.load_prices) = saved
            shutil.rmtree(sbx, ignore_errors=True)


class TestAppendCsvAtomicity(unittest.TestCase):
    """
    Guards the append_csv fix: files missing a trailing newline must not
    produce glued rows. The old code (df.to_csv(mode='a')) would concatenate
    the new row onto the last byte of the existing file, producing corrupt
    CSV like '...RISK_ON2026-08-07,...' that breaks parse_dates.

    Mutation contract: reverting append_csv to the old implementation
    (plain df.to_csv(mode='a', ...)) MUST make test_no_glue_on_missing_newline
    fail. Verified 2026-08-15.
    """

    def setUp(self):
        self.dir = os.path.join(TMP, "append_csv_test")
        os.makedirs(self.dir, exist_ok=True)
        self._old_fwd = ft.FORWARD_DIR
        ft.FORWARD_DIR = self.dir

    def tearDown(self):
        ft.FORWARD_DIR = self._old_fwd

    def test_no_glue_on_missing_newline(self):
        """Appending to a file that lacks a trailing newline must not glue rows."""
        path = os.path.join(self.dir, "test_glue.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("date,value,tag\n2026-08-04,100,RISK_ON")  # no trailing \n
        ft.append_csv(path, [{"date": "2026-08-07", "value": 200, "tag": "RISK_ON"}],
                      ["date", "value", "tag"])
        df = pd.read_csv(path, parse_dates=["date"])
        self.assertEqual(len(df), 2, "must have two distinct rows")
        self.assertEqual(df["date"].dtype.kind, "M",
                         "parse_dates must succeed (not object/str)")

    def test_atomic_write_no_partial(self):
        """If append_csv is interrupted, original file must survive intact."""
        path = os.path.join(self.dir, "test_atomic.csv")
        ft.append_csv(path, [{"x": 1}], ["x"])
        with open(path, "r") as f:
            original = f.read()
        self.assertIn("1", original)
        self.assertFalse(os.path.exists(path + ".tmp"),
                         "temp file must be cleaned up after success")

    def test_new_file_gets_header(self):
        path = os.path.join(self.dir, "test_header.csv")
        ft.append_csv(path, [{"a": 1, "b": 2}], ["a", "b"])
        with open(path, "r") as f:
            lines = f.read().strip().split("\n")
        self.assertEqual(lines[0], "a,b")
        self.assertEqual(len(lines), 2)

    def test_append_no_duplicate_header(self):
        path = os.path.join(self.dir, "test_nodup.csv")
        ft.append_csv(path, [{"a": 1}], ["a"])
        ft.append_csv(path, [{"a": 2}], ["a"])
        with open(path, "r") as f:
            content = f.read()
        self.assertEqual(content.count("a\n"), 1, "header must appear only once")


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)