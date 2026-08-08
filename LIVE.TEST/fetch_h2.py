"""
fetch_h2.py — הורדת נתוני M2K (RTY=F) לבדיקת H2

מוריד נרות 5 דקות מ-yahoo, מסנן RTH, ושומר snapshot ל-parquet.
הרצה חוזרת טוענת מה-snapshot ולא מהרשת.

שימוש:
    python fetch_h2.py                     # הורדה ראשונית (yfinance)
    python fetch_h2.py --refresh           # כפה הורדה חדשה
    python fetch_h2.py --source ibkr --years 3   # (עתידי — דורש IBKR)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from h2_config import (
    DATA_DIR, ET, LOCKED_YAHOO, OUT_DIR, SNAPSHOT_PATH,
    assert_locked,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_yfinance(days: int = 59) -> pd.DataFrame | None:
    """מוריד נרות 5 דקות מ-yahoo עבור RTY=F."""
    import yfinance as yf

    end = datetime.now()
    start = end - timedelta(days=min(days, 59))

    print(f"  מוריד {LOCKED_YAHOO} | {start:%Y-%m-%d} → {end:%Y-%m-%d} | 5m")
    try:
        df = yf.download(LOCKED_YAHOO, start=start, end=end,
                         interval="5m", progress=False, auto_adjust=False)
    except Exception as e:
        print(f"  ✗ שגיאת הורדה: {type(e).__name__}: {e}")
        return None

    if df is None or df.empty:
        print("  ✗ לא התקבלו נתונים")
        return None

    # yfinance עשוי להחזיר MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]

    need = ["open", "high", "low", "close", "volume"]
    if any(c not in df.columns for c in need):
        print(f"  ✗ עמודות חסרות: {set(need) - set(df.columns)}")
        return None

    df = df[need].dropna(subset=["close"])

    # המרת אזור זמן
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    df.index.name = "datetime"

    return df.sort_index()


def rth_only(df: pd.DataFrame) -> pd.DataFrame:
    """סינון ל-RTH בלבד: 9:30–16:00 ET."""
    from datetime import time as _t
    return df.between_time(_t(9, 30), _t(16, 0))


def main():
    p = argparse.ArgumentParser(description="הורדת נתוני H2")
    p.add_argument("--source", choices=["yfinance", "ibkr"], default="yfinance")
    p.add_argument("--symbol", default="M2K")
    p.add_argument("--days", type=int, default=59)
    p.add_argument("--years", type=int, default=3, help="עבור IBKR בלבד")
    p.add_argument("--refresh", action="store_true",
                   help="כפה הורדה חדשה במקום snapshot")
    args = p.parse_args()

    # ── assert על הגדרות נעולות לפני כל פעולה ──
    assert_locked()
    assert args.symbol == "M2K", f"מכשיר חייב להיות M2K, התקבל {args.symbol}"

    OUT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    if args.source == "ibkr":
        print("  ⚠ מקור IBKR טרם מומש. יש להשתמש ב-yfinance בשלב זה.")
        print("    רכישת נתוני IBKR מותנית בתוצאת הבדיקה הנוכחית.")
        sys.exit(1)

    # ── snapshot קיים? ──
    if SNAPSHOT_PATH.exists() and not args.refresh:
        print(f"  [snapshot] נטען מקובץ שמור: {SNAPSHOT_PATH.name}")
        print(f"  hash: {sha256_file(SNAPSHOT_PATH)[:16]}…")
        df = pd.read_parquet(SNAPSHOT_PATH)
        days = df.index.normalize().nunique()
        print(f"  {len(df):,} נרות | {days} ימי מסחר | "
              f"{df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}")
        return

    # ── הורדה חדשה ──
    df = download_yfinance(args.days)
    if df is None:
        sys.exit(1)

    df = rth_only(df)

    # שמירה
    df.to_parquet(SNAPSHOT_PATH)
    h = sha256_file(SNAPSHOT_PATH)
    days = df.index.normalize().nunique()

    print(f"  ✓ נשמר: {SNAPSHOT_PATH}")
    print(f"  hash: {h[:16]}…")
    print(f"  {len(df):,} נרות 5m (RTH) | {days} ימי מסחר | "
          f"{df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}")


if __name__ == "__main__":
    main()