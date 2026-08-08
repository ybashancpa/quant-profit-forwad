"""
data.py — שליפת נתונים היסטוריים

שני מקורות:
  1. yfinance — לפיתוח מהיר. מוגבל: 60 יום אחורה בנרות תוך-יומיים.
     נשתמש בפרוקסי ETF (QQQ במקום MNQ) כי יאהו לא מספק חוזי מיקרו כמו שצריך.
  2. IBKR — למקור האמת. נרות אמיתיים של MNQ, כולל סשן לילה.
     דורש TWS/Gateway פעיל.

⚠️ ההבדל בין QQQ ל-MNQ אינו זניח: שעות מסחר שונות, גאפים שונים,
   נפח שונה, ולכן VWAP שונה. QQQ טוב ללטש את הלוגיקה,
   אבל את הפרמטרים הסופיים חייבים לכייל על נתוני MNQ אמיתיים.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from config import ET, DATA_PROXY, INSTRUMENTS

CACHE_DIR = Path(os.getenv("MNQ_CACHE", "./data_cache"))
CACHE_DIR.mkdir(exist_ok=True, parents=True)

_STANDARD_COLS = ["open", "high", "low", "close", "volume"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """מנרמל שמות עמודות ואזור זמן לפורמט אחיד"""
    df = df.copy()

    # MultiIndex columns (yfinance מחזיר כזה לפעמים)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    rename = {"adj_close": "adj_close", "vol": "volume"}
    df = df.rename(columns=rename)

    missing = [c for c in _STANDARD_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"חסרות עמודות: {missing}. יש: {list(df.columns)}")

    df = df[_STANDARD_COLS]

    # אזור זמן -> ET
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    df.index.name = "datetime"

    return df.sort_index().dropna(subset=["close"])


# ══════════════════════════════════════════════════════════════
# מקור 1: yfinance (פיתוח)
# ══════════════════════════════════════════════════════════════
def fetch_yfinance(
    symbol: str,
    interval: str = "5m",
    days: int = 59,
    use_proxy: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    שליפה מ-yfinance.

    מגבלות יאהו:
      1m  -> 7 ימים אחורה בלבד
      2m/5m/15m/30m/60m -> 60 ימים אחורה
      1d  -> ללא הגבלה
    """
    import yfinance as yf

    ticker = DATA_PROXY.get(symbol, symbol) if use_proxy else symbol

    cache_file = CACHE_DIR / f"{ticker}_{interval}_{days}d.parquet"
    if use_cache and cache_file.exists():
        age_h = (datetime.now().timestamp() - cache_file.stat().st_mtime) / 3600
        if age_h < 12:
            print(f"  [cache] {ticker} {interval}")
            return pd.read_parquet(cache_file)

    max_days = 7 if interval == "1m" else 59
    days = min(days, max_days)

    end = datetime.now()
    start = end - timedelta(days=days)

    print(f"  [yfinance] מוריד {ticker} @ {interval}, {days} ימים...")
    df = yf.download(
        ticker, start=start, end=end, interval=interval,
        progress=False, auto_adjust=False, prepost=False,
    )

    if df.empty:
        raise RuntimeError(
            f"לא התקבלו נתונים עבור {ticker}. "
            "בדוק חיבור רשת או נסה סימבול/interval אחר."
        )

    df = _normalize(df)

    if use_cache:
        df.to_parquet(cache_file)

    print(f"  ✓ {len(df):,} נרות | {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}")
    return df


# ══════════════════════════════════════════════════════════════
# מקור 2: IBKR (ייצור)
# ══════════════════════════════════════════════════════════════
def fetch_ibkr(
    symbol: str,
    bar_size: str = "3 mins",
    duration: str = "30 D",
    host: str = "127.0.0.1",
    port: int = 7497,          # 7497 = TWS paper | 7496 = TWS live
                               # 4002 = Gateway paper | 4001 = Gateway live
    client_id: int = 11,
    what_to_show: str = "TRADES",
    use_rth: bool = True,
) -> pd.DataFrame:
    """
    שליפת נרות היסטוריים מ-IBKR. דורש TWS או IB Gateway פעיל.

    מגבלת קצב: IBKR חוסם יותר מ-60 בקשות היסטוריות ב-10 דקות.
    אל תריץ בלולאה בלי המתנה.
    """
    try:
        from ib_async import IB, Future, util
    except ImportError:
        raise ImportError("התקן: pip install ib_async")

    inst = INSTRUMENTS[symbol]

    ib = IB()
    ib.connect(host, port, clientId=client_id, timeout=15)

    try:
        # חוזה רציף — IBKR בוחר אוטומטית את החוזה הפעיל
        contract = Future(
            symbol=inst.symbol,
            exchange=inst.exchange,
            currency=inst.currency,
        )
        details = ib.reqContractDetails(contract)
        if not details:
            raise RuntimeError(f"לא נמצא חוזה עבור {symbol}")

        # בוחרים את התפוגה הקרובה ביותר שעדיין פעילה
        contracts = sorted(
            [d.contract for d in details],
            key=lambda c: c.lastTradeDateOrContractMonth,
        )
        active = contracts[0]
        print(f"  [IBKR] חוזה: {active.localSymbol} (תפוגה {active.lastTradeDateOrContractMonth})")

        bars = ib.reqHistoricalData(
            active,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=2,          # UTC epoch
        )

        if not bars:
            raise RuntimeError("IBKR החזיר 0 נרות. בדוק הרשאות market data.")

        df = util.df(bars)
        df = df.set_index("date")
        df = _normalize(df)

        print(f"  ✓ {len(df):,} נרות | {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}")
        return df

    finally:
        ib.disconnect()


# ══════════════════════════════════════════════════════════════
# ממשק אחיד
# ══════════════════════════════════════════════════════════════
def get_data(
    symbol: str,
    interval: str = "5m",
    days: int = 59,
    source: str = "auto",
) -> pd.DataFrame:
    """
    source: 'yfinance' | 'ibkr' | 'auto'
    'auto' מנסה IBKR ונופל ל-yfinance אם אין חיבור.
    """
    if source == "ibkr":
        return fetch_ibkr(symbol, bar_size=interval.replace("m", " mins"))

    if source == "auto":
        try:
            return fetch_ibkr(symbol, bar_size=interval.replace("m", " mins"))
        except Exception as e:
            print(f"  ⚠ IBKR לא זמין ({type(e).__name__}), עובר ל-yfinance")

    return fetch_yfinance(symbol, interval=interval, days=days)


# ══════════════════════════════════════════════════════════════
# מחולל נתונים סינתטיים — לבדיקת תקינות הקוד בלבד
# ══════════════════════════════════════════════════════════════
def synthetic_data(
    n_days: int = 40,
    bars_per_day: int = 130,      # 6.5 שעות / 3 דקות
    start_price: float = 20000.0,
    seed: int = 42,
    trend_strength: float = 0.35,
) -> pd.DataFrame:
    """
    מייצר נתונים סינתטיים עם מגמות תוך-יומיות אמיתיות.
    ‼️ לבדיקת תקינות קוד בלבד. תוצאות בקטסט על נתונים אלה חסרות משמעות.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    frames = []
    price = start_price
    day = pd.Timestamp("2025-01-02", tz=ET)

    for d in range(n_days):
        while day.weekday() >= 5:
            day += pd.Timedelta(days=1)

        # כל יום מקבל "אופי" — מגמתי חזק, מגמתי חלש, או מדשדש
        regime = rng.choice(["trend_up", "trend_down", "chop"], p=[0.3, 0.3, 0.4])
        drift = {"trend_up": trend_strength, "trend_down": -trend_strength, "chop": 0.0}[regime]
        vol = start_price * 0.0006

        idx = pd.date_range(
            day.replace(hour=9, minute=33), periods=bars_per_day, freq="3min", tz=ET
        )

        rets = rng.normal(drift * vol / 20, vol, bars_per_day)
        closes = price * np.exp(np.cumsum(rets) / price * price / price)
        closes = price + np.cumsum(rets)

        opens = np.concatenate([[price], closes[:-1]])
        noise = np.abs(rng.normal(0, vol * 0.7, bars_per_day))
        highs = np.maximum(opens, closes) + noise
        lows = np.minimum(opens, closes) - np.abs(rng.normal(0, vol * 0.7, bars_per_day))
        volumes = rng.integers(300, 3000, bars_per_day).astype(float)

        frames.append(pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volumes,
        }, index=idx))

        price = closes[-1] * (1 + rng.normal(0, 0.002))   # גאפ לילה
        day += pd.Timedelta(days=1)

    df = pd.concat(frames)
    df.index.name = "datetime"
    return df


if __name__ == "__main__":
    print("בדיקת מחולל סינתטי:")
    d = synthetic_data(n_days=5)
    print(d.head())
    print(f"\n{len(d)} נרות, {d.index.normalize().nunique()} ימים")
