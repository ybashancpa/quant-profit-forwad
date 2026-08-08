"""
main.py — נקודת הכניסה להרצת בקטסט

שימוש:
    python main.py                      # בקטסט על נתונים אמיתיים (yfinance)
    python main.py --source ibkr        # נתוני MNQ אמיתיים מ-TWS
    python main.py --synthetic          # בדיקת תקינות בלבד
    python main.py --sweep              # סריקת פרמטרים
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

import indicators as ind
from backtest import Backtester
from config import INSTRUMENTS, RiskConfig, StrategyConfig
from data import get_data, synthetic_data
from screener import MICROS, download, rth_only, to_instrument
from strategy import MomentumPullbackStrategy


def load(symbol: str, args) -> pd.DataFrame:
    """טוען נתונים וממיר לגרף הביצוע"""
    if args.synthetic:
        print(f"\n▸ {symbol}: נתונים סינתטיים (בדיקת תקינות בלבד)")
        return synthetic_data(n_days=args.days, seed=args.seed)

    print(f"\n▸ {symbol}: טוען נתונים...")

    if args.source == "ibkr":
        return get_data(symbol, interval="3 mins", days=args.days, source="ibkr")

    # ‼️ סימבול חוזה אמיתי (NQ=F), לא פרוקסי ETF.
    #    פרוקסי נותן סקאלת מחיר שגויה: QQQ ב-$715 עם מכפיל של MNQ
    #    מייצר רווח/הפסד קטן פי ~41 מהמציאות.
    spec = next((m for m in MICROS if m.symbol == symbol), None)
    if spec is None:
        raise ValueError(f"{symbol} לא מוגדר ב-MICROS")

    df = download(spec, "5m", args.days)
    if df is None:
        raise RuntimeError(f"לא התקבלו נתונים עבור {spec.yahoo}")

    # ‼️ סינון לשעות המסחר — בלי זה ה-ATR מדולל בסשן הלילה
    if spec.category != "crypto":
        df = rth_only(df)

    print(f"  ✓ {spec.yahoo}: {len(df):,} נרות (RTH) | "
          f"{df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}")
    return df


def _instrument(symbol: str):
    """מחפש ב-MICROS (15 חוזים) ונופל ל-INSTRUMENTS אם צריך"""
    spec = next((m for m in MICROS if m.symbol == symbol), None)
    if spec is not None:
        return to_instrument(spec)
    if symbol in INSTRUMENTS:
        return INSTRUMENTS[symbol]
    raise KeyError(f"{symbol} לא מוגדר לא ב-MICROS ולא ב-INSTRUMENTS")


def run_one(symbol: str, args, strat_cfg: StrategyConfig, risk_cfg: RiskConfig):
    inst = _instrument(symbol)
    df = load(symbol, args)

    if len(df) < 500:
        print(f"  ⚠ רק {len(df)} נרות — מעט מדי לבקטסט משמעותי")

    strat = MomentumPullbackStrategy(strat_cfg)
    bt = Backtester(inst, strat, risk_cfg, args.capital)
    bt.run(df)
    bt.print_report()
    return bt


def sweep(symbol: str, args, risk_cfg: RiskConfig):
    """סריקת פרמטרים — מוצא לאיזה מכפיל סטופ יש בכלל עסקאות"""
    inst = _instrument(symbol)
    df = load(symbol, args)

    print(f"\n{'='*70}")
    print(f"  סריקת פרמטרים — {symbol}")
    print(f"{'='*70}")
    print(f"  {'stop×ATR':<10}{'ADX':<7}{'עסקאות':>8}{'הצלחה':>9}{'PF':>7}{'נטו':>11}{'ממוצע R':>10}")
    print("  " + "-" * 64)

    results = []
    for stop_m in [0.75, 1.0, 1.25, 1.5]:
        for adx_t in [20.0, 25.0, 30.0]:
            cfg = StrategyConfig(
                stop_atr_mult=stop_m,
                adx_entry_threshold=adx_t,
                adx_exit_threshold=adx_t - 5,
            )
            bt = Backtester(inst, MomentumPullbackStrategy(cfg), risk_cfg, args.capital)
            bt.run(df)
            s = bt.stats()
            n = s.get("עסקאות", 0)
            if n == 0:
                print(f"  {stop_m:<10}{adx_t:<7.0f}{'0':>8}{'—':>9}{'—':>7}{'—':>11}{'—':>10}")
                continue
            print(f"  {stop_m:<10}{adx_t:<7.0f}{n:>8}{s['אחוז הצלחה']:>9}"
                  f"{s['Profit Factor']:>7}{s['רווח נקי']:>11}{s['ממוצע R']:>10}")
            results.append((stop_m, adx_t, s))

    print("=" * 70)
    return results


def main():
    p = argparse.ArgumentParser(description="MNQ Momentum Intraday Backtester")
    p.add_argument("--symbols", nargs="+", default=["MYM", "M2K"],
                   help="ברירת מחדל: המכשירים שעברו את מסנן התקציב")
    p.add_argument("--source", default="yfinance", choices=["yfinance", "ibkr", "auto"])
    p.add_argument("--days", type=int, default=59)
    p.add_argument("--capital", type=float, default=5000.0)
    p.add_argument("--risk", type=float, default=0.01, help="סיכון לעסקה (0.01 = 1%%)")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    risk_cfg = RiskConfig(account_size=args.capital, risk_per_trade_pct=args.risk)
    strat_cfg = StrategyConfig()

    print("╔" + "═" * 60 + "╗")
    print("║" + "  MNQ Momentum Pullback — Intraday Backtest".ljust(60) + "║")
    print("╚" + "═" * 60 + "╝")
    print(f"  הון: ${args.capital:,.0f} | סיכון לעסקה: {args.risk*100:.1f}% "
          f"(${args.capital*args.risk:.0f})")
    print(f"  מקור נתונים: {args.source}")

    if args.sweep:
        for s in args.symbols:
            sweep(s, args, risk_cfg)
        return

    total = 0.0
    for s in args.symbols:
        try:
            bt = run_one(s, args, strat_cfg, risk_cfg)
            total += bt.equity - args.capital
        except Exception as e:
            print(f"  ✗ {s} נכשל: {type(e).__name__}: {e}")

    print(f"\n  סה\"כ על פני {len(args.symbols)} מכשירים: ${total:,.2f}")
    print("  ⚠ תוצאות בקטסט אינן תחזית. אין ערובה שהתנהגות העבר תחזור.")


if __name__ == "__main__":
    main()
