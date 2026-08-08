"""
risk.py — ניהול סיכון וגודל פוזיציה

כלל הברזל: הסיכון לעסקה נקבע **לפני** הכניסה, לפי מרחק הסטופ.
לא "כמה חוזים אני רוצה" אלא "כמה חוזים מותרים לי בהינתן הסטופ הזה".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from config import Instrument, RiskConfig


@dataclass
class SizingResult:
    contracts: int
    risk_dollars: float          # סיכון בפועל בדולרים
    risk_points: float
    margin_required: float
    rejected_reason: str = ""

    @property
    def ok(self) -> bool:
        return self.contracts > 0


def size_position(
    instrument: Instrument,
    entry: float,
    stop: float,
    account_equity: float,
    cfg: RiskConfig,
    open_margin_used: float = 0.0,
) -> SizingResult:
    """
    חישוב גודל פוזיציה.

    contracts = floor( (equity × risk%) / (stop_points × multiplier) )

    דוגמה MNQ:
      equity $5,000, risk 1% = $50
      ATR(3m) = 12 נק' → סטופ 1.5×ATR = 18 נק'
      סיכון לחוזה = 18 × $2 = $36
      → floor(50/36) = 1 חוזה, סיכון בפועל $36 (0.72%)
    """
    risk_points = abs(entry - stop)
    if risk_points <= 0:
        return SizingResult(0, 0, 0, 0, "מרחק סטופ אפס")

    risk_budget = account_equity * cfg.risk_per_trade_pct
    risk_per_contract = risk_points * instrument.multiplier

    if risk_per_contract > risk_budget:
        return SizingResult(
            0, 0, risk_points, 0,
            f"הסטופ רחב מדי: ${risk_per_contract:.0f} לחוזה > תקציב ${risk_budget:.0f}",
        )

    contracts = int(risk_budget // risk_per_contract)
    contracts = min(contracts, cfg.max_contracts_per_instrument)

    if contracts < 1:
        return SizingResult(0, 0, risk_points, 0, "פחות מחוזה אחד")

    # בדיקת מרג'ין
    margin_per = cfg.intraday_margin.get(instrument.symbol, 2500.0)
    margin_needed = margin_per * contracts

    while contracts > 0 and (open_margin_used + margin_needed) > account_equity:
        contracts -= 1
        margin_needed = margin_per * contracts

    if contracts < 1:
        return SizingResult(
            0, 0, risk_points, 0,
            f"אין מרג'ין: דרוש ${margin_per:.0f}, פנוי ${account_equity - open_margin_used:.0f}",
        )

    return SizingResult(
        contracts=contracts,
        risk_dollars=risk_per_contract * contracts,
        risk_points=risk_points,
        margin_required=margin_needed,
    )


# ══════════════════════════════════════════════════════════════
# מפסק זרם — עוצר מסחר במצבים מסוכנים
# ══════════════════════════════════════════════════════════════
@dataclass
class CircuitBreaker:
    cfg: RiskConfig
    current_day: date | None = None
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    # ‼️ שונה מ-`halted` ל-`risk_halted`: זהו מצב סיכון ידוע
    # (RISK_HALT במדיניות HALT_POLICY.md) — לא כניסות חדשות,
    # אבל ניהול פוזיציות קיימות נמשך. אין לבלבל עם SAFETY halt.
    risk_halted: bool = False
    risk_halt_reason: str = ""
    trades_today: dict = field(default_factory=dict)

    def new_day(self, d: date):
        if self.current_day != d:
            self.current_day = d
            self.daily_pnl = 0.0
            self.risk_halted = False
            self.risk_halt_reason = ""
            self.trades_today = {}
            # רצף ההפסדים מתאפס ביום חדש.
            # ‼️ בגרסה קודמת הוא לא התאפס, אבל `halted` כן —
            # מה שיצר מצב של עסקה אחת ביום ואז השבתה, לנצח.
            self.consecutive_losses = 0

    def record_trade(self, symbol: str, pnl: float):
        self.daily_pnl += pnl
        self.trades_today[symbol] = self.trades_today.get(symbol, 0) + 1

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        if self.daily_pnl <= -self.cfg.daily_loss_dollars:
            self.risk_halted = True
            self.risk_halt_reason = (
                f"מגבלת הפסד יומי: ${self.daily_pnl:.0f} "
                f"(מגבלה ${-self.cfg.daily_loss_dollars:.0f})"
            )
        elif self.consecutive_losses >= self.cfg.max_consecutive_losses:
            self.risk_halted = True
            self.risk_halt_reason = f"{self.consecutive_losses} הפסדים ברצף"

    def can_trade(self, symbol: str, max_per_day: int) -> tuple[bool, str]:
        if self.risk_halted:
            return False, self.risk_halt_reason
        if self.trades_today.get(symbol, 0) >= max_per_day:
            return False, f"מכסת עסקאות יומית ל-{symbol}"
        return True, ""
