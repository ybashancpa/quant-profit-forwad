"""
indicators.py — אינדיקטורים טכניים

עיקרון מנחה: כל אינדיקטור מחושב **ללא הצצה לעתיד** (no look-ahead).
ערך האינדיקטור בשורה i מבוסס אך ורק על נתונים עד שורה i כולל.
זו הנקודה שבה רוב הבקטסטים משקרים לעצמם.
"""

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════
# EMA
# ══════════════════════════════════════════════════════════════
def ema(series: pd.Series, period: int) -> pd.Series:
    """ממוצע נע אקספוננציאלי"""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


# ══════════════════════════════════════════════════════════════
# ATR (Wilder)
# ══════════════════════════════════════════════════════════════
def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range — צריך עמודות high/low/close"""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range בשיטת Wilder (RMA), לא SMA.
    זה מה ש-TradingView ו-IBKR משתמשים בו — חשוב להתאמה.
    """
    tr = true_range(df)
    # Wilder smoothing == EMA עם alpha = 1/period
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


# ══════════════════════════════════════════════════════════════
# ADX (Wilder) — מדד כוח מגמה
# ══════════════════════════════════════════════════════════════
def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    מחזיר DataFrame עם עמודות: plus_di, minus_di, adx

    לוגיקה:
      +DM = תנועה כלפי מעלה, אם היא גדולה מהתנועה כלפי מטה
      -DM = תנועה כלפי מטה, אם היא גדולה מהתנועה כלפי מעלה
      DX  = |+DI - -DI| / (+DI + -DI) * 100
      ADX = ממוצע Wilder של DX

    ADX לא אומר כיוון — רק כוח. 25+ = מגמה חזקה.
    """
    high, low = df["high"], df["low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    tr = true_range(df)

    # Wilder smoothing
    alpha = 1 / period
    atr_w = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_dm_s = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    minus_dm_s = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    # הגנה מחלוקה באפס
    atr_safe = atr_w.replace(0, np.nan)

    plus_di = 100 * plus_dm_s / atr_safe
    minus_di = 100 * minus_dm_s / atr_safe

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum

    adx_line = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    return pd.DataFrame({
        "plus_di": plus_di,
        "minus_di": minus_di,
        "adx": adx_line,
    }, index=df.index)


# ══════════════════════════════════════════════════════════════
# VWAP — מעוגן לתחילת סשן
# ══════════════════════════════════════════════════════════════
def session_vwap(df: pd.DataFrame, session_col: str | None = None) -> pd.Series:
    """
    VWAP מעוגן לסשן (מתאפס כל יום מסחר).

    זו נקודה קריטית: VWAP מתגלגל על פני ימים הוא חסר משמעות.
    ה-VWAP חייב להתאפס בפתיחת כל יום — זה מה שהופך אותו ל"קו המשווה" היומי.

    צריך עמודות: high, low, close, volume
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3

    if session_col and session_col in df.columns:
        groups = df[session_col]
    else:
        # קיבוץ לפי תאריך (בהנחה שהאינדקס הוא DatetimeIndex בשעון הבורסה)
        groups = df.index.normalize()

    pv = typical * df["volume"]
    cum_pv = pv.groupby(groups).cumsum()
    cum_vol = df["volume"].groupby(groups).cumsum()

    vwap = cum_pv / cum_vol.replace(0, np.nan)
    return vwap.ffill()


def vwap_bands(df: pd.DataFrame, vwap: pd.Series, n_std: float = 1.0) -> pd.DataFrame:
    """רצועות סטיית תקן סביב ה-VWAP (שימושי לזיהוי מתיחות יתר)"""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    groups = df.index.normalize()

    dev_sq = ((typical - vwap) ** 2) * df["volume"]
    cum_dev = dev_sq.groupby(groups).cumsum()
    cum_vol = df["volume"].groupby(groups).cumsum().replace(0, np.nan)

    std = np.sqrt(cum_dev / cum_vol)
    return pd.DataFrame({
        "vwap_upper": vwap + n_std * std,
        "vwap_lower": vwap - n_std * std,
        "vwap_std": std,
    }, index=df.index)


# ══════════════════════════════════════════════════════════════
# עזר: מיזוג טווחי זמן ללא look-ahead
# ══════════════════════════════════════════════════════════════
def align_higher_timeframe(
    exec_df: pd.DataFrame,
    htf_series: pd.Series,
    name: str,
) -> pd.Series:
    """
    ממפה סדרה מטווח זמן גבוה (15 דק') לגרף הביצוע (3 דק') — בלי look-ahead.

    ‼️ הנקודה הקריטית ‼️
    נר 15 דקות שנפתח ב-10:00 נסגר ב-10:15. הערך שלו זמין לנו רק ב-10:15.
    אם נשתמש בו ב-10:03 — אנחנו קוראים את העתיד, והבקטסט יראה נהדר
    ובלייב יקרוס. זה בדיוק הפער בין backtest ל-live שכבר נשרפת עליו.

    הפתרון: לדחוף את הערך קדימה בנר אחד, ואז reindex עם ffill.
    """
    shifted = htf_series.shift(1)              # זמין רק אחרי סגירת הנר
    aligned = shifted.reindex(
        exec_df.index.union(shifted.index)
    ).ffill().reindex(exec_df.index)
    aligned.name = name
    return aligned


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """המרת נרות לטווח זמן גבוה יותר"""
    out = df.resample(rule, label="right", closed="right").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    return out.dropna(subset=["close"])
