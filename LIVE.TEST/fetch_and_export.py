"""
fetch_and_export.py — הורדת נתונים וייצוא לקובץ

מריצים את זה אצלך במחשב (יש לך גישה לרשת, לי אין).
הפלט: קובץ CSV אחד לכל מכשיר בתיקיית ./export/
את הקבצים האלה אפשר להעלות לשיחה כדי שאכייל עליהם.

שימוש:
    python fetch_and_export.py                    # yfinance (פרוקסי ETF)
    python fetch_and_export.py --source ibkr      # MNQ אמיתי, דורש TWS פעיל
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import indicators as ind
from config import DATA_PROXY, INSTRUMENTS
from data import fetch_ibkr, fetch_yfinance

OUT = Path("./export")
OUT.mkdir(exist_ok=True)


def export(symbol: str, source: str, days: int, to_3min: bool):
    print(f"\n{'─'*55}")
    print(f"▸ {symbol}")

    if source == "ibkr":
        df = fetch_ibkr(symbol, bar_size="3 mins", duration=f"{days} D")
        tag = "MNQ_real"
    else:
        df = fetch_yfinance(symbol, interval="5m", days=days)
        tag = f"proxy_{DATA_PROXY.get(symbol, symbol)}"

    if to_3min and source != "ibkr":
        print("  ⚠ יאהו לא מספק נרות 3 דק'. משאיר 5 דק'.")

    # תקציר שיעזור לכייל בלי להעלות את כל הקובץ
    a = ind.atr(df, 14).dropna()
    inst = INSTRUMENTS[symbol]

    print(f"  נרות: {len(df):,} | ימים: {df.index.normalize().nunique()}")
    print(f"  טווח: {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}")
    print(f"  מחיר: {df.close.min():,.2f} – {df.close.max():,.2f}")
    print(f"  ATR חציוני: {a.median():.2f} נק'")
    print(f"  ATR אחוזון 25/75: {a.quantile(.25):.2f} / {a.quantile(.75):.2f}")
    print(f"  → סיכון לחוזה בסטופ 1.5×ATR: "
          f"${a.median()*1.5*inst.multiplier:,.0f}")

    path = OUT / f"{symbol}_{tag}.csv"
    df.to_csv(path)
    print(f"  ✓ נשמר: {path}  ({path.stat().st_size/1024:.0f} KB)")
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["MNQ", "MES", "M6E"])
    p.add_argument("--source", default="yfinance", choices=["yfinance", "ibkr"])
    p.add_argument("--days", type=int, default=59)
    p.add_argument("--to-3min", action="store_true")
    args = p.parse_args()

    print("═" * 55)
    print("  הורדת נתונים לכיול")
    print("═" * 55)

    paths = []
    for s in args.symbols:
        try:
            paths.append(export(s, args.source, args.days, args.to_3min))
        except Exception as e:
            print(f"  ✗ {s}: {type(e).__name__}: {e}")

    print(f"\n{'═'*55}")
    print(f"  {len(paths)} קבצים ב-./export/")
    print("  העלה אותם לשיחה כדי שאכייל את הפרמטרים.")
    print("═" * 55)


if __name__ == "__main__":
    main()
