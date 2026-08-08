"""
config.py — הגדרות מרכזיות למערכת המסחר
כל פרמטר שניתן לכוונון נמצא כאן. אין magic numbers בשאר הקוד.
"""

from dataclasses import dataclass, field
from datetime import time
from zoneinfo import ZoneInfo

# ══════════════════════════════════════════════════════════════
# אזורי זמן
# ══════════════════════════════════════════════════════════════
ET = ZoneInfo("America/New_York")      # שעון הבורסה
IL = ZoneInfo("Asia/Jerusalem")        # שעון מקומי (לדוחות)


# ══════════════════════════════════════════════════════════════
# מפרט מכשירים
# ══════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Instrument:
    symbol: str              # סימבול IBKR
    name: str
    multiplier: float        # שווי נקודה בדולר
    tick_size: float         # גודל טיק מינימלי
    exchange: str
    currency: str = "USD"
    commission_rt: float = 1.60   # עמלה round-trip לחוזה (IBKR ~0.85 + NFA/exchange ~0.55 לכיוון)
    slippage_ticks: float = 1.0   # הנחת החלקה בביצוע, בטיקים, לכל צד

    @property
    def tick_value(self) -> float:
        """שווי טיק בודד בדולר"""
        return self.tick_size * self.multiplier

    def points_to_usd(self, points: float, contracts: int = 1) -> float:
        return points * self.multiplier * contracts

    def round_to_tick(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size


INSTRUMENTS = {
    "MNQ": Instrument(
        symbol="MNQ", name="Micro E-mini Nasdaq-100",
        multiplier=2.0, tick_size=0.25, exchange="CME",
        # 0.25 נק' = $0.50 לטיק
    ),
    "MES": Instrument(
        symbol="MES", name="Micro E-mini S&P 500",
        multiplier=5.0, tick_size=0.25, exchange="CME",
        # 0.25 נק' = $1.25 לטיק
    ),
    "M6E": Instrument(
        symbol="M6E", name="Micro EUR/USD",
        multiplier=12500.0, tick_size=0.0001, exchange="CME",
        commission_rt=1.30,
        # 0.0001 = $1.25 לטיק
    ),
    "MCL": Instrument(
        symbol="MCL", name="Micro WTI Crude Oil",
        multiplier=100.0, tick_size=0.01, exchange="NYMEX",
        # 0.01 = $1.00 לטיק
    ),
    "M2K": Instrument(
        symbol="M2K", name="Micro Russell 2000",
        multiplier=5.0, tick_size=0.10, exchange="CME",
    ),
}

# פרוקסי ETF לשליפת נתונים היסטוריים בשלב הפיתוח
# (יאהו לא מספק נתוני אינטרה-דיי אמינים לחוזי מיקרו)
DATA_PROXY = {
    "MNQ": "QQQ",
    "MES": "SPY",
    "M6E": "FXE",
    "MCL": "USO",
    "M2K": "IWM",
}


# ══════════════════════════════════════════════════════════════
# הגדרות חשבון וניהול סיכון
# ══════════════════════════════════════════════════════════════
@dataclass
class RiskConfig:
    account_size: float = 5000.0

    # סיכון לעסקה בודדת (1% = $50 בחשבון של $5,000)
    risk_per_trade_pct: float = 0.01

    # מספר עסקאות מקסימלי במקביל (מגבלת מרג'ין!)
    max_concurrent_positions: int = 2

    # חוזים מקסימליים לכל מכשיר
    max_contracts_per_instrument: int = 2

    # הפסד יומי מצטבר שמפסיק את המסחר עד למחרת
    daily_loss_limit_pct: float = 0.03    # 3% = $150

    # רצף הפסדים שמפסיק את המסחר
    max_consecutive_losses: int = 3

    # דרישת מרג'ין תוך-יומי משוערת לחוזה (IBKR, אינדיקטיבי — לאמת מול TWS!)
    intraday_margin: dict = field(default_factory=lambda: {
        "MNQ": 2400.0, "MES": 1800.0, "M6E": 400.0,
        "MCL": 1200.0, "M2K": 800.0,
    })

    @property
    def risk_dollars(self) -> float:
        return self.account_size * self.risk_per_trade_pct

    @property
    def daily_loss_dollars(self) -> float:
        return self.account_size * self.daily_loss_limit_pct


# ══════════════════════════════════════════════════════════════
# פרמטרי אסטרטגיה — Momentum Pullback
# ══════════════════════════════════════════════════════════════
@dataclass
class StrategyConfig:
    # --- טווחי זמן ---
    exec_timeframe: str = "3min"      # גרף ביצוע (טריגר)
    filter_timeframe: str = "15min"   # גרף פילטר (ADX)
    context_timeframe: str = "1h"     # גרף הקשר (EMA 20/50)

    # --- שכבה 1: כיוון (VWAP) ---
    vwap_buffer_pct: float = 0.0005   # אזור נייטרלי סביב VWAP (0.05%) — מונע רעש בצמוד לקו

    # --- שכבה 2: כוח מגמה (ADX על 15 דק') ---
    adx_period: int = 14
    adx_entry_threshold: float = 25.0  # מעל זה = מגמה חזקה, מותר להיכנס
    adx_exit_threshold: float = 20.0   # מתחת לזה = המגמה קרסה, יוצאים

    # --- שכבה 3: הקשר (EMA שעתי) ---
    ema_fast_context: int = 20
    ema_slow_context: int = 50
    require_context_alignment: bool = True  # לדרוש שגם ה-1H יתמוך בכיוון

    # --- טריגר: Pullback ל-EMA20 בגרף הביצוע ---
    ema_trigger: int = 20
    pullback_touch_tolerance_atr: float = 0.15  # "נגיעה" = בתוך 0.15 ATR מה-EMA
    require_reversal_bar: bool = True   # לדרוש נר היפוך שמאשר שהנסיגה נגמרה

    # --- יציאות ---
    atr_period: int = 14
    stop_atr_mult: float = 1.5          # סטופ ראשוני = 1.5 × ATR
    target_r_multiple: float = 2.0      # יעד ראשון = 2R
    partial_exit_pct: float = 0.5       # לסגור חצי ביעד הראשון (אם יש 2+ חוזים)
    # ── Trailing stop ──
    # ‼️ הפרמטרים חייבים לקיים:  trail_after_r > trail_atr_mult / stop_atr_mult
    #    אחרת ברגע ההדלקה הסטופ הנגרר נמצא *מתחת* לנקודת הכניסה,
    #    והמנגנון חונק כל עסקה לפני שהיא מגיעה ליעד.
    #
    #    הגרסה הקודמת (1.0R / 2.0×ATR עם סטופ 1.5×ATR) הפרה את התנאי:
    #      1R = 1.5 ATR  |  מרחק trail = 2.0 ATR = 1.33R
    #      → בהדלקה ב-1.0R הסטופ נחת על -0.33R
    #    התוצאה: עסקה אחת מתוך 22 הגיעה ליעד.
    trail_after_r: float = 1.5          # מדליקים מאוחר יותר
    trail_atr_mult: float = 1.0         # ומרחק צמוד יותר
    #    כעת:  1.5R - (1.0/1.5)R = +0.83R נעולים ברגע ההדלקה

    # --- חלון זמן מסחר (שעון ET) ---
    session_start: time = time(9, 45)   # לא נכנסים ב-15 הדק' הראשונות (רעש פתיחה)
    no_new_entries_after: time = time(15, 15)
    hard_close: time = time(15, 50)     # סגירה כפויה של כל הפוזיציות

    # --- מגבלות ---
    max_trades_per_day_per_instrument: int = 3
    min_bars_between_trades: int = 5    # לא להיכנס מיד אחרי יציאה


    def __post_init__(self):
        """מוודא שה-trailing לא סותר את עצמו"""
        locked_r = self.trail_after_r - (self.trail_atr_mult / self.stop_atr_mult)
        if locked_r <= 0:
            raise ValueError(
                f"trailing סותר את עצמו: בהדלקה ב-{self.trail_after_r}R "
                f"הסטופ ינחת על {locked_r:+.2f}R. "
                f"דרוש trail_after_r > trail_atr_mult/stop_atr_mult "
                f"(= {self.trail_atr_mult/self.stop_atr_mult:.2f})"
            )
        self.locked_r_at_trail = locked_r


@dataclass
class BacktestConfig:
    start_date: str = "2024-01-01"
    end_date: str | None = None
    instruments: list = field(default_factory=lambda: ["MNQ", "MES", "M6E"])
    initial_capital: float = 5000.0
    verbose: bool = True


# מופעים ברירת מחדל
RISK = RiskConfig()
STRAT = StrategyConfig()
