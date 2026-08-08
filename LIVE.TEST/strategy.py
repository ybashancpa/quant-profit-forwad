"""
strategy.py — Momentum Pullback Strategy

הלוגיקה בדיוק כפי שהוגדרה:

  שכבה 1 — כיוון (VWAP, גרף ביצוע):
      מחיר > VWAP  →  מותר Long בלבד
      מחיר < VWAP  →  מותר Short בלבד

  שכבה 2 — כוח (ADX על 15 דק'):
      ADX > 25  →  המגמה חזקה מספיק, מותר להיכנס
      ADX < 20  →  המגמה קרסה, יוצאים מפוזיציה קיימת

  שכבה 3 — הקשר (EMA 20/50 שעתי):
      EMA20 > EMA50  →  הטיה יומית חיובית
      חייב להסכים עם ה-VWAP, אחרת אין עסקה

  טריגר — Pullback:
      המחיר נוגע ב-EMA20 של גרף ה-3 דק'
      + באותו רגע ADX(15m) עדיין > 25
      + נר היפוך שמאשר שהנסיגה נגמרה

  יציאות:
      סטופ ראשוני:  1.5 × ATR
      יעד:          2R (סגירה חלקית)
      Trailing:     מ-1R, Chandelier 2 × ATR
      Regime exit:  ADX < 20 או חציית VWAP נגד הפוזיציה
      Time exit:    15:50 ET — סגירה כפויה
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

import indicators as ind
from config import StrategyConfig


class Direction(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


class Regime(Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    CHOPPY = "CHOPPY"          # ADX נמוך — אין מגמה
    CONFLICTED = "CONFLICTED"  # VWAP וה-1H לא מסכימים


@dataclass
class Signal:
    timestamp: pd.Timestamp
    direction: Direction
    entry_price: float
    stop_price: float
    target_price: float
    atr: float
    regime: Regime
    adx: float
    reason: str

    @property
    def risk_points(self) -> float:
        return abs(self.entry_price - self.stop_price)


class MomentumPullbackStrategy:

    def __init__(self, cfg: StrategyConfig | None = None):
        self.cfg = cfg or StrategyConfig()

    # ══════════════════════════════════════════════════════════
    # הכנת נתונים — כל האינדיקטורים בטבלה אחת
    # ══════════════════════════════════════════════════════════
    def prepare(self, df_exec: pd.DataFrame) -> pd.DataFrame:
        """
        מקבל גרף ביצוע (3 דק'), מחשב את כל האינדיקטורים
        בכל שלושת טווחי הזמן ומיישר אותם — בלי look-ahead.
        """
        c = self.cfg
        df = df_exec.copy()

        # ── שכבה 1: VWAP על גרף הביצוע ──
        df["vwap"] = ind.session_vwap(df)

        # ── טריגר: EMA20 + ATR על גרף הביצוע ──
        df["ema_trigger"] = ind.ema(df["close"], c.ema_trigger)
        df["atr"] = ind.atr(df, c.atr_period)

        # ── שכבה 2: ADX על 15 דקות ──
        df_15 = ind.resample_ohlcv(df, c.filter_timeframe)
        adx_15 = ind.adx(df_15, c.adx_period)
        df["adx_15m"] = ind.align_higher_timeframe(df, adx_15["adx"], "adx_15m")
        df["plus_di_15m"] = ind.align_higher_timeframe(df, adx_15["plus_di"], "plus_di_15m")
        df["minus_di_15m"] = ind.align_higher_timeframe(df, adx_15["minus_di"], "minus_di_15m")

        # ── שכבה 3: EMA 20/50 שעתי ──
        df_1h = ind.resample_ohlcv(df, c.context_timeframe)
        ema_f = ind.ema(df_1h["close"], c.ema_fast_context)
        ema_s = ind.ema(df_1h["close"], c.ema_slow_context)
        df["ema20_1h"] = ind.align_higher_timeframe(df, ema_f, "ema20_1h")
        df["ema50_1h"] = ind.align_higher_timeframe(df, ema_s, "ema50_1h")

        # ── משטר שוק ──
        df["regime"] = self._classify_regime(df)

        # ── טריגר pullback ──
        df["pullback_long"] = self._detect_pullback(df, Direction.LONG)
        df["pullback_short"] = self._detect_pullback(df, Direction.SHORT)

        return df

    # ══════════════════════════════════════════════════════════
    # סיווג משטר שוק — שילוב שלוש השכבות
    # ══════════════════════════════════════════════════════════
    def _classify_regime(self, df: pd.DataFrame) -> pd.Series:
        c = self.cfg

        buf = df["vwap"] * c.vwap_buffer_pct
        above_vwap = df["close"] > (df["vwap"] + buf)
        below_vwap = df["close"] < (df["vwap"] - buf)

        strong = df["adx_15m"] > c.adx_entry_threshold

        ctx_up = df["ema20_1h"] > df["ema50_1h"]
        ctx_dn = df["ema20_1h"] < df["ema50_1h"]

        regime = pd.Series(Regime.CHOPPY, index=df.index, dtype=object)

        if c.require_context_alignment:
            long_ok = above_vwap & strong & ctx_up
            short_ok = below_vwap & strong & ctx_dn
            # VWAP אומר כיוון אחד, השעתי אומר אחר → אין עסקה
            conflict = strong & ((above_vwap & ctx_dn) | (below_vwap & ctx_up))
        else:
            long_ok = above_vwap & strong
            short_ok = below_vwap & strong
            conflict = pd.Series(False, index=df.index)

        regime[conflict] = Regime.CONFLICTED
        regime[long_ok] = Regime.TREND_UP
        regime[short_ok] = Regime.TREND_DOWN

        # שורות בלי מספיק היסטוריה לאינדיקטורים
        warmup = df[["adx_15m", "ema50_1h", "atr", "ema_trigger"]].isna().any(axis=1)
        regime[warmup] = Regime.CHOPPY

        return regime

    # ══════════════════════════════════════════════════════════
    # זיהוי Pullback
    # ══════════════════════════════════════════════════════════
    def _detect_pullback(self, df: pd.DataFrame, direction: Direction) -> pd.Series:
        """
        Long:
          1. הנר נגע ב-EMA20 מלמעלה (low ירד עד/מתחת ל-EMA20 + טולרנס)
          2. הסגירה חזרה מעל ה-EMA20 — הנסיגה נגמרה (נר היפוך)
          3. הסגירה מעל הפתיחה — לחץ קונים חזר

        הטולרנס נמדד ב-ATR ולא באחוזים, כדי שיתאים אוטומטית
        לתנודתיות של כל מכשיר. 0.15 ATR ב-MNQ ≈ 2-4 נקודות.
        """
        c = self.cfg
        tol = df["atr"] * c.pullback_touch_tolerance_atr
        e = df["ema_trigger"]

        if direction is Direction.LONG:
            touched = df["low"] <= (e + tol)
            recovered = df["close"] > e
            if c.require_reversal_bar:
                reversal = df["close"] > df["open"]
                return touched & recovered & reversal
            return touched & recovered

        touched = df["high"] >= (e - tol)
        recovered = df["close"] < e
        if c.require_reversal_bar:
            reversal = df["close"] < df["open"]
            return touched & recovered & reversal
        return touched & recovered

    # ══════════════════════════════════════════════════════════
    # יצירת סיגנל כניסה עבור נר בודד
    # ══════════════════════════════════════════════════════════
    def generate_signal(self, row: pd.Series, ts: pd.Timestamp) -> Signal | None:
        c = self.cfg

        if pd.isna(row.get("atr")) or row["atr"] <= 0:
            return None

        # חלון זמן
        t = ts.time()
        if t < c.session_start or t >= c.no_new_entries_after:
            return None

        regime = row["regime"]
        atr_v = float(row["atr"])
        price = float(row["close"])

        # ── LONG ──
        if regime is Regime.TREND_UP and row["pullback_long"]:
            # אישור נוסף: DI חיובי מוביל
            if row["plus_di_15m"] <= row["minus_di_15m"]:
                return None
            stop = price - c.stop_atr_mult * atr_v
            risk = price - stop
            return Signal(
                timestamp=ts, direction=Direction.LONG,
                entry_price=price, stop_price=stop,
                target_price=price + c.target_r_multiple * risk,
                atr=atr_v, regime=regime, adx=float(row["adx_15m"]),
                reason=f"Pullback→EMA20 | ADX15m={row['adx_15m']:.1f} | מעל VWAP",
            )

        # ── SHORT ──
        if regime is Regime.TREND_DOWN and row["pullback_short"]:
            if row["minus_di_15m"] <= row["plus_di_15m"]:
                return None
            stop = price + c.stop_atr_mult * atr_v
            risk = stop - price
            return Signal(
                timestamp=ts, direction=Direction.SHORT,
                entry_price=price, stop_price=stop,
                target_price=price - c.target_r_multiple * risk,
                atr=atr_v, regime=regime, adx=float(row["adx_15m"]),
                reason=f"Pullback→EMA20 | ADX15m={row['adx_15m']:.1f} | מתחת VWAP",
            )

        return None

    # ══════════════════════════════════════════════════════════
    # בדיקת יציאה מסיבת משטר שוק
    # ══════════════════════════════════════════════════════════
    def should_exit_regime(self, row: pd.Series, direction: Direction) -> tuple[bool, str]:
        """יציאה כשהתנאים שהצדיקו את הכניסה כבר לא מתקיימים"""
        c = self.cfg

        adx_v = row.get("adx_15m")
        if pd.notna(adx_v) and adx_v < c.adx_exit_threshold:
            return True, f"ADX קרס ל-{adx_v:.1f}"

        vwap_v = row.get("vwap")
        if pd.notna(vwap_v):
            if direction is Direction.LONG and row["close"] < vwap_v:
                return True, "המחיר חצה מתחת ל-VWAP"
            if direction is Direction.SHORT and row["close"] > vwap_v:
                return True, "המחיר חצה מעל ל-VWAP"

        return False, ""
