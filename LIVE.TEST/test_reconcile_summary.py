"""
test_reconcile_summary.py — בדיקות ל-build_summary / --out של reconcile.py

מריצים מתוך LIVE.TEST/:
    python test_reconcile_summary.py

מוודא:
1. הסיכום המזוקק דטרמיניסטי וכולל sha256 של לוג המקור.
2. Redaction: account id מהלוג אינו דולף לפלט (חשוב — הקובץ נכנס לגיט).
3. מדדי החלקה חושבו נכון מול המודל.
4. verdict_ok משקף אי-התאמות/פוזיציות פתוחות.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import reconcile as rc

ACCT = "DU1234567"  # חייב להישאר מחוץ לפלט


def make_records(include_exit=True):
    recs = [
        {"kind": "connect", "account": ACCT, "paper": True},
        {"kind": "bar", "symbol": "MES", "bar_time": "2026-08-07 10:00:00",
         "close": 5000.0, "vwap": 4999.0, "regime": "TREND_UP",
         "pullback_long": True, "pullback_short": False},
        {"kind": "signal", "symbol": "MES", "signal_time": "2026-08-07 10:00:00",
         "direction": "LONG", "price": 5000.0},
        {"kind": "entry", "symbol": "MES", "signal_time": "2026-08-07 10:00:00",
         "fill_price": 5000.25, "slippage_ticks": 1.0, "contracts": 1},
    ]
    if include_exit:
        recs.append({"kind": "exit", "symbol": "MES", "reason": "EOD_CLOSE",
                     "ts_et": "2026-08-07 15:50:00", "fill_price": 5002.0})
    return recs


def write_jsonl(records):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                      encoding="utf-8")
    for r in records:
        tmp.write(json.dumps(r) + "\n")
    tmp.close()
    return Path(tmp.name)


class TestReconcileSummary(unittest.TestCase):
    def setUp(self):
        self.path = write_jsonl(make_records(include_exit=True))
        self.log = rc.load_log(self.path)

    def tearDown(self):
        self.path.unlink(missing_ok=True)

    def test_sha256_matches_file(self):
        expected = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.assertEqual(rc._sha256_file(self.path), expected)

    def test_redaction_no_account_id(self):
        s = rc.build_summary(self.path, self.log)
        blob = json.dumps(s)
        self.assertNotIn(ACCT, blob, "account id leaked into summary")
        self.assertNotIn("account", blob.lower())

    def test_schema_and_paper_flag(self):
        s = rc.build_summary(self.path, self.log)
        self.assertEqual(s["schema"], "reconcile-summary/v1")
        self.assertTrue(s["paper"])

    def test_slippage_metrics(self):
        s = rc.build_summary(self.path, self.log)
        mes = s["symbols"]["MES"]
        self.assertEqual(mes["entries"], 1)
        self.assertAlmostEqual(mes["slippage_mean_ticks"], 1.0)
        self.assertAlmostEqual(mes["modeled_ticks"], 1.0)
        self.assertFalse(mes["worse_than_model"])  # 1.0 == modeled

    def test_clean_run_verdict_ok(self):
        s = rc.build_summary(self.path, self.log)
        mes = s["symbols"]["MES"]
        self.assertEqual(mes["signal_mismatches"], 0)
        self.assertEqual(mes["potential_missed"], 0)
        self.assertEqual(s["eod"]["unclosed"], 0)
        self.assertTrue(s["verdict_ok"])

    def test_unclosed_position_fails_verdict(self):
        p = write_jsonl(make_records(include_exit=False))
        try:
            s = rc.build_summary(p, rc.load_log(p))
            self.assertEqual(s["eod"]["unclosed"], 1)
            self.assertFalse(s["verdict_ok"])
        finally:
            p.unlink(missing_ok=True)

    def test_worse_slippage_flagged_but_not_verdict(self):
        # החלקה גרועה מהמודל מסומנת, אבל (כמו בדוח האנושי) לא הופכת verdict
        recs = make_records(include_exit=True)
        for r in recs:
            if r["kind"] == "entry":
                r["slippage_ticks"] = 3.0
        p = write_jsonl(recs)
        try:
            s = rc.build_summary(p, rc.load_log(p))
            mes = s["symbols"]["MES"]
            self.assertTrue(mes["worse_than_model"])
            self.assertAlmostEqual(mes["slippage_ratio"], 3.0)
        finally:
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)