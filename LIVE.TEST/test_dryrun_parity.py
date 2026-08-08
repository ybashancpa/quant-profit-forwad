"""
test_dryrun_parity.py — שקילות dry-run מול backtest על אותם נרות

הבדיקה החזקה ביותר בתשתית: מריצה את מסלול ה-dry-run (דרך
LiveTrader.process_closed_bar האמיתי) על snapshot הנתונים של H1,
ומשווה עסקה מול עסקה מול פלט ה-Backtester.

אם שני המסלולים מייצרים אותן כניסות, אותם מחירי מילוי ואותן
יציאות — זו הוכחה שהם מיישמים את אותה לוגיקה. זה בדיוק מה
ש-reconcile.py מנסה לאמת בלייב, רק כאן דטרמיניסטית: בלי Gateway,
בלי שוק, בלי החלקה אמיתית.

אחרי שהבדיקה הזו עוברת, השאלה היחידה שנשארת ב-Paper היא ההחלקה.

שימוש:
    python test_dryrun_parity.py [--symbols MYM M2K]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from backtest import Backtester
from config import RiskConfig, StrategyConfig
from live_trader import LiveTrader
from screener import MICROS, to_instrument
from strategy import MomentumPullbackStrategy

DATA_DIR = Path(__file__).parent / "results_h1" / "data"
EPS = 1e-6


# ══════════════════════════════════════════════════════════════
def run_backtest(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    inst = to_instrument(next(m for m in MICROS if m.symbol == symbol))
    bt = Backtester(inst, MomentumPullbackStrategy(StrategyConfig()),
                    RiskConfig(account_size=5000.0), 5000.0)
    bt.run(df)
    return bt.results()


def run_dryrun_replay(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    מזרים את הנרות ל-LiveTrader אמיתי במצב dry-run, אירוע אחרי אירוע,
    בדיוק כפי ש-IB היה שולח אותם (נר סגור + נר נבנה).
    """
    args = argparse.Namespace(
        symbols=[symbol], capital=5000.0, risk=0.01,
        dry_run=True, live=False, host="127.0.0.1", port=4002,
        client_id=99,
    )
    trader = LiveTrader([symbol], args)

    n = len(df)
    # נר מלאכותי "נבנה" אחרי הנר האחרון, כדי שהאחרון ייחשב סגור
    tail = df.iloc[[-1]].copy()
    tail.index = tail.index + pd.Timedelta(minutes=5)
    extended = pd.concat([df, tail])

    for i in range(1, n + 1):
        # df עד נר i (כולל) = הנר הסגור האחרון הוא i-1, והנר ה"נבנה"
        # הוא i — בדיוק המצב בלייב כשנר חדש נפתח.
        chunk = extended.iloc[: i + 1]
        trader.process_closed_bar(symbol, chunk)

    # ── איסוף עסקאות מהלוג (entry/exit מדומים) ──
    entries, exits = [], []
    with open(trader.logger.path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("simulated"):
                continue
            if rec["kind"] == "entry":
                entries.append(rec)
            elif rec["kind"] == "exit":
                exits.append(rec)
    trader.logger.close()

    if len(entries) != len(exits):
        print(f"  ⚠ {symbol}: {len(entries)} כניסות מול {len(exits)} יציאות "
              f"(ייתכן פוזיציה פתוחה בסוף הנתונים)")

    rows = []
    for e, x in zip(entries, exits):
        rows.append({
            "signal_time": e["signal_time"],
            "entry_time": e["entry_time"],
            "direction": e.get("direction", ""),
            "entry_price": e["fill_price"],
            "initial_stop": e["stop"],
            "target_price": e["target"],
            "contracts": e["contracts"],
            "exit_time": x["exit_time"],
            "exit_price": x["fill_price"],
            "exit_reason": x["reason"],
            "net_pnl": x["net_pnl"],
            "r_multiple": x["r_multiple"],
        })
    sim = pd.DataFrame(rows)

    # כיוון לא נרשם ב-entry הלוג? נשלים מהלוג המקורי אם חסר
    if not sim.empty and (sim.direction == "").any():
        pend = []
        with open(trader.logger.path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("kind") == "pending_entry":
                    pend.append(rec["direction"])
        if len(pend) == len(sim):
            sim["direction"] = pend

    return sim


# ══════════════════════════════════════════════════════════════
def compare(symbol: str, bt: pd.DataFrame, sim: pd.DataFrame) -> int:
    """מחזיר מספר אי-התאמות"""
    problems = 0

    if len(bt) != len(sim):
        print(f"  ✗ {symbol}: מספר עסקאות שונה — בקטסט {len(bt)}, "
              f"סימולציה {len(sim)}")
        problems += 1
        k = min(len(bt), len(sim))
    else:
        k = len(bt)
        print(f"  ✓ {symbol}: {k} עסקאות בשני המסלולים")

    bt = bt.reset_index(drop=True)
    sim = sim.reset_index(drop=True)

    for i in range(k):
        b, s = bt.iloc[i], sim.iloc[i]
        diffs = []

        if str(b.signal_time) != str(s.signal_time):
            diffs.append(f"signal_time: {b.signal_time} ≠ {s.signal_time}")
        if str(b.entry_time) != str(s.entry_time):
            diffs.append(f"entry_time: {b.entry_time} ≠ {s.entry_time}")
        if str(b.direction) != str(s.direction):
            diffs.append(f"direction: {b.direction} ≠ {s.direction}")
        if abs(b.entry_price - s.entry_price) > EPS:
            diffs.append(f"entry_price: {b.entry_price} ≠ {s.entry_price}")
        if abs(b.initial_stop - s.initial_stop) > EPS:
            diffs.append(f"initial_stop: {b.initial_stop} ≠ {s.initial_stop}")
        if abs(b.target_price - s.target_price) > EPS:
            diffs.append(f"target_price: {b.target_price} ≠ {s.target_price}")
        if int(b.contracts) != int(s.contracts):
            diffs.append(f"contracts: {b.contracts} ≠ {s.contracts}")
        if str(b.exit_time) != str(s.exit_time):
            diffs.append(f"exit_time: {b.exit_time} ≠ {s.exit_time}")
        if abs(b.exit_price - s.exit_price) > EPS:
            diffs.append(f"exit_price: {b.exit_price} ≠ {s.exit_price}")
        if str(b.exit_reason) != str(s.exit_reason):
            diffs.append(f"exit_reason: '{b.exit_reason}' ≠ '{s.exit_reason}'")
        if abs(b.net_pnl - s.net_pnl) > EPS:
            diffs.append(f"net_pnl: {b.net_pnl:.6f} ≠ {s.net_pnl:.6f}")
        if abs(b.r_multiple - s.r_multiple) > EPS:
            diffs.append(f"r_multiple: {b.r_multiple:.6f} ≠ {s.r_multiple:.6f}")

        if diffs:
            problems += 1
            print(f"  ✗ {symbol} עסקה #{i+1} ({b.signal_time}):")
            for d in diffs:
                print(f"      {d}")
            if problems > 5:
                print("  … עוצר אחרי 5 אי-התאמות")
                break

    if problems == 0 and k > 0:
        print(f"  ✓ {symbol}: כל {k} העסקאות זהות לחלוטין "
              f"(כניסות, מילויים, יציאות, P&L, R)")
    return problems


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["MYM", "M2K"])
    args = p.parse_args()

    print("═" * 66)
    print("  בדיקת שקילות: dry-run (LiveTrader) מול Backtester")
    print("═" * 66)

    total_problems = 0
    total_trades = 0

    for sym in args.symbols:
        snap = DATA_DIR / f"{sym}_5m_snapshot.parquet"
        if not snap.exists():
            print(f"  ✗ {sym}: אין snapshot ({snap}). הרץ קודם test_h1.py")
            total_problems += 1
            continue

        df = pd.read_parquet(snap)
        print(f"\n── {sym}: {len(df)} נרות, "
              f"{df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d} ──")

        bt = run_backtest(sym, df)
        sim = run_dryrun_replay(sym, df)
        total_trades += len(bt)
        total_problems += compare(sym, bt, sim)

    print("\n" + "═" * 66)
    if total_problems == 0 and total_trades > 0:
        print(f"  ✓ שקילות מלאה: {total_trades} עסקאות, 0 אי-התאמות.")
        print("    מסלול ה-dry-run מיישם בדיוק את אותה לוגיקה כמו הבקטסט.")
        print("    השאלה היחידה שנשארת ב-Paper היא ההחלקה.")
    elif total_trades == 0:
        print("  ✗ לא נוצרו עסקאות — אין מה להשוות")
        total_problems = 1
    else:
        print(f"  ✗ נמצאו {total_problems} אי-התאמות. לתקן לפני כל שימוש בלייב.")
    print("═" * 66)
    return 1 if total_problems else 0


if __name__ == "__main__":
    sys.exit(main())