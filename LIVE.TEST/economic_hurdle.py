"""
economic_hurdle.py — שלב 0 בפרוטוקול המחקר

לפני מסמך השערה, לפני נתונים, לפני קוד.

השאלה: **האם הרעיון הזה יכול לעבוד**, בהנחה שהאפקט קיים בדיוק
בגודל שהספרות מדווחת? אם התשובה לא — הרעיון נדחה כאן, בעשר דקות,
במקום אחרי חודשיים של בנייה.

זה חישוב, לא מדידה. הוא אינו סובל מגודל מדגם.

────────────────────────────────────────────────────────────────
מקור הכלי: H2. שם המבחן שהכריע היה חישוב שאפשר היה להריץ
לפני שנכתבה שורת קוד אחת. הכלי הזה הופך אותו לשלב קבוע.
────────────────────────────────────────────────────────────────

שימוש:
    python economic_hurdle.py --preset h2          # שחזור H2
    python economic_hurdle.py --preset list        # כל התרחישים השמורים
    python economic_hurdle.py --sharpe-annual 1.0 --sigma-pct 0.337 \
        --price 3011 --multiplier 5 --tick 0.10 \
        --commission 1.52 --slippage-ticks 2 --trades-per-year 250

הערה: הכלי בודק כדאיות כלכלית בלבד. הוא אינו בודק התאמה לתקציב
הסיכון (מרג'ין, גודל פוזיציה, סטופ). יש לבדוק זאת בנפרד.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field


@dataclass
class Idea:
    name: str
    # ── קלט ראשי: Sharpe ──
    # אפשר לספק sharpe_annual (מהספרות) או sharpe_per_trade (מהמסמך הנעול).
    # אם שניהם מסופקים, sharpe_per_trade גובר.
    sharpe_annual: float | None = None
    sharpe_per_trade_input: float | None = None

    # ── קלט ראשי: σ ──
    # אפשר לספק sigma_pct + price + multiplier, או sigma_dollars ישירות.
    # אם sigma_dollars מסופק, הוא גובר.
    sigma_pct: float | None = None
    sigma_dollars_input: float | None = None
    price: float | None = None
    multiplier: float | None = None

    # ── עלויות ותדירות ──
    tick: float = 0.0
    commission: float = 0.0
    slippage_ticks: float = 0.0
    trades_per_year: float = 250.0

    # ── חשבון ──
    account: float = 5000.0
    risk_free_pct: float = 3.5
    contracts: int = 1
    source: str = ""

    def __post_init__(self):
        # ── ולידציה ──
        if self.sharpe_annual is None and self.sharpe_per_trade_input is None:
            raise ValueError("נדרש sharpe_annual או sharpe_per_trade_input")
        if self.sigma_dollars_input is None:
            if self.sigma_pct is None or self.price is None or self.multiplier is None:
                raise ValueError("נדרש sigma_dollars_input, או (sigma_pct + price + multiplier)")
        if self.trades_per_year <= 0:
            raise ValueError("trades_per_year חייב להיות > 0")
        if self.account <= 0:
            raise ValueError("account חייב להיות > 0")
        if self.contracts < 1:
            raise ValueError("contracts חייב להיות >= 1")
        if self.tick < 0:
            raise ValueError("tick חייב להיות >= 0")
        if self.commission < 0:
            raise ValueError("commission חייב להיות >= 0")
        if self.slippage_ticks < 0:
            raise ValueError("slippage_ticks חייב להיות >= 0")
        if self.risk_free_pct < 0:
            raise ValueError("risk_free_pct חייב להיות >= 0")

    # ── גדלים נגזרים ──
    @property
    def sharpe_per_trade(self) -> float:
        if self.sharpe_per_trade_input is not None:
            return self.sharpe_per_trade_input
        return self.sharpe_annual / math.sqrt(self.trades_per_year)

    @property
    def sigma_dollars(self) -> float:
        if self.sigma_dollars_input is not None:
            return self.sigma_dollars_input * self.contracts
        return self.sigma_pct / 100 * self.price * self.multiplier * self.contracts

    @property
    def tick_value(self) -> float:
        return self.tick * (self.multiplier or 0) * self.contracts

    @property
    def expected_gross(self) -> float:
        return self.sharpe_per_trade * self.sigma_dollars

    @property
    def slippage_cost(self) -> float:
        return self.slippage_ticks * self.tick_value

    @property
    def commission_cost(self) -> float:
        return self.commission * self.contracts

    @property
    def net_per_trade(self) -> float:
        return self.expected_gross - self.commission_cost - self.slippage_cost

    @property
    def annual_net(self) -> float:
        return self.net_per_trade * self.trades_per_year

    @property
    def annual_pct(self) -> float:
        return self.annual_net / self.account * 100

    @property
    def hurdle_dollars(self) -> float:
        """התשואה חסרת הסיכון — מה שצריך להכות"""
        return self.account * self.risk_free_pct / 100

    @property
    def passes(self) -> bool:
        # >= ולא > — עקבי עם H2 שבה "מתחת ל-3.5%" = FAIL
        return self.annual_net >= self.hurdle_dollars

    # ── ספים ──
    @property
    def breakeven_ticks(self) -> float:
        """החלקה שבה הרווח מתאפס (לפני חסר-סיכון)"""
        if self.tick_value <= 0:
            return float("inf")
        return (self.expected_gross - self.commission_cost) / self.tick_value

    @property
    def hurdle_ticks(self) -> float:
        """החלקה שבה מפסיקים להכות את חסר-הסיכון"""
        if self.tick_value <= 0:
            return float("inf")
        per_trade_hurdle = self.hurdle_dollars / self.trades_per_year
        return (self.expected_gross - self.commission_cost
                - per_trade_hurdle) / self.tick_value

    @property
    def required_sharpe_annual(self) -> float:
        """איזה Sharpe שנתי היה נדרש כדי לעבור"""
        if self.sigma_dollars <= 0:
            return float("inf")
        per_trade_hurdle = self.hurdle_dollars / self.trades_per_year
        need = self.commission_cost + self.slippage_cost + per_trade_hurdle
        return need / self.sigma_dollars * math.sqrt(self.trades_per_year)

    @property
    def required_effect_multiple(self) -> float:
        """פי כמה האפקט צריך להיות גדול יותר. <1 = יש מרווח"""
        s = self.sharpe_per_trade
        if s <= 0:
            return float("inf")
        return self.required_sharpe_annual / (s * math.sqrt(self.trades_per_year))

    # ── הספק ──
    def trades_needed(self, power: float = 2.8) -> float:
        """עסקאות לזיהוי האפקט (power=2.8 ≈ 80% ב-α=0.05)"""
        s = self.sharpe_per_trade
        return (power / s) ** 2 if s > 0 else float("inf")

    @property
    def years_needed(self) -> float:
        return self.trades_needed() / self.trades_per_year


# ══════════════════════════════════════════════════════════════
def report(idea: Idea) -> bool:
    w = 64
    print("═" * w)
    print(f"  {idea.name}")
    if idea.source:
        print(f"  מקור גודל האפקט: {idea.source}")
    print("═" * w)

    print(f"\n  {'הנחות':.<34}")
    if idea.sharpe_per_trade_input is not None:
        print(f"    Sharpe לעסקה (נעול){'':.<11} {idea.sharpe_per_trade:.4f}")
    else:
        print(f"    Sharpe שנתי (מהספרות){'':.<8} {idea.sharpe_annual:.2f}")
        print(f"    → Sharpe לעסקה{'':.<15} {idea.sharpe_per_trade:.4f}")
    if idea.sigma_dollars_input is not None:
        print(f"    σ לעסקה (ישיר){'':.<15} ${idea.sigma_dollars:,.2f}")
    else:
        print(f"    σ לעסקה{'':.<22} {idea.sigma_pct:.3f}%  =  ${idea.sigma_dollars:,.2f}")
    print(f"    עסקאות בשנה{'':.<18} {idea.trades_per_year:,.0f}")
    print(f"    חוזים{'':.<24} {idea.contracts}")

    print(f"\n  {'החשבון':.<34}")
    print(f"    תוחלת גולמית{'':.<17} ${idea.expected_gross:>8.4f}")
    print(f"    − עמלה{'':.<23} ${-idea.commission_cost:>8.4f}")
    print(f"    − החלקה ({idea.slippage_ticks:.1f} טיקים){'':.<10} ${-idea.slippage_cost:>8.4f}")
    print(f"    {'':.<29} {'─'*9}")
    print(f"    = נטו לעסקה{'':.<18} ${idea.net_per_trade:>+8.4f}")
    print(f"    × {idea.trades_per_year:,.0f}{'':.<25} ${idea.annual_net:>+8.1f}/שנה")
    print(f"    על חשבון ${idea.account:,.0f}{'':.<12} {idea.annual_pct:>+8.2f}%")

    print(f"\n  {'מול הרף':.<34}")
    print(f"    תשואה חסרת סיכון{'':.<13} ${idea.hurdle_dollars:>8.1f}  ({idea.risk_free_pct}%)")
    print(f"    התוצאה{'':.<23} ${idea.annual_net:>+8.1f}")
    diff = idea.annual_net - idea.hurdle_dollars
    print(f"    הפרש{'':.<25} ${diff:>+8.1f}")

    print(f"\n  {'ספים':.<34}")
    print(f"    החלקה לרווחיות אפס{'':.<11} {idea.breakeven_ticks:>8.2f} טיקים")
    print(f"    החלקה להכאת חסר-סיכון{'':.<8} {idea.hurdle_ticks:>8.2f} טיקים")
    print(f"    Sharpe שנתי נדרש{'':.<13} {idea.required_sharpe_annual:>8.2f}")
    print(f"    פי כמה מהמדווח{'':.<15} {idea.required_effect_multiple:>8.2f}x")

    print(f"\n  {'הספק (למידע בלבד)':.<34}")
    tn = idea.trades_needed()
    print(f"    עסקאות לזיהוי האפקט{'':.<10} {tn:>8,.0f}" if tn < 1e9 else
          f"    עסקאות לזיהוי האפקט{'':.<10} {'∞':>8}")
    print(f"    שנים בקצב הזה{'':.<16} {idea.years_needed:>8.1f}" if idea.years_needed < 1e6 else
          f"    שנים בקצב הזה{'':.<16} {'∞':>8}")

    print("\n" + "─" * w)
    if idea.passes:
        print(f"  ✓ CANDIDATE — עובר את הרף ב-${diff:+,.0f}")
        print(f"    עבר כדאיות כלכלית. המשך למסנן תקציב, מנגנון והספק.")
        print(f"    ‼️ אין לרכוש נתונים עד מעבר כל המסננים.")
        print(f"\n    ‼️ הכלי בודק כדאיות כלכלית בלבד. הוא *אינו* בודק")
        print(f"       התאמה לתקציב הסיכון. בדוק בנפרד:")
        if idea.price and idea.multiplier:
            notional = idea.price * idea.multiplier * idea.contracts
            print(f"       נומינלי לפוזיציה: ${notional:,.0f}")
            print(f"       = {notional/idea.account:.1f}x גודל החשבון")
        if idea.hurdle_ticks < 1.0:
            print(f"    ⚠️ אבל המרווח דק: ההחלקה חייבת להיות מתחת")
            print(f"       ל-{idea.hurdle_ticks:.2f} טיקים. למדוד לפני שבונים.")
    else:
        print(f"  ✗ REJECT — חסר ${-diff:,.0f} לשנה")
        print(f"    האפקט צריך להיות גדול פי {idea.required_effect_multiple:.2f} ממה שפורסם.")
        print(f"    אין לכתוב מסמך השערה, לרכוש נתונים או לכתוב קוד.")
    print("─" * w)
    return idea.passes


# ══════════════════════════════════════════════════════════════
# תרחישים שמורים — מהמחקר שכבר בוצע
# ══════════════════════════════════════════════════════════════
PRESETS = {
    "h2": Idea(
        name="H2 — מומנטום חלון סגירה, M2K (שחזור מדויק)",
        # שחזור מדויק של test_h2.py: Sharpe לעסקה 0.063, σ_dollars=$35.8301
        sharpe_per_trade_input=0.063, sigma_dollars_input=35.8301,
        price=2988.2, multiplier=5.0, tick=0.10,
        commission=1.52, slippage_ticks=2, trades_per_year=250,
        source="Baltussen et al. (2021) JFE; Gao et al. (2018) JFE",
    ),
    "h2_mnq": Idea(
        name="H2 על MNQ — לבדיקה, לא נעול",
        sharpe_annual=1.0, sigma_pct=0.337, price=29539, multiplier=2.0,
        tick=0.25, commission=1.56, slippage_ticks=2, trades_per_year=250,
        source="σ מ-QQQ (=נאסד\"ק, תקף). עובר כלכלית; מסנן תקציב לא מומש בכלי זה",
    ),
    # ‼️ MES הוסר: σ של S&P בחלון הסגירה לא נמדד.
    #    שימוש ב-σ של נאסד"ק (0.337%) היה שגוי — SPY מראה כמחצית
    #    מהתנודתיות של QQQ. למדוד לפני שמריצים.
    "h2_25k": Idea(
        name="H2 על M2K בחשבון $25,000 (4 חוזים)",
        # σ של M2K (ראסל), לא נאסד"ק!
        sharpe_annual=1.0, sigma_pct=0.23806, price=3011, multiplier=5.0,
        tick=0.10, commission=1.52, slippage_ticks=2, trades_per_year=250,
        account=25000, contracts=4,
        source="בדיקת רגישות לגודל חשבון. σ מ-M2K (לא נאסד\"ק)",
    ),
}


def main():
    p = argparse.ArgumentParser(description="Economic hurdle — שלב 0")
    p.add_argument("--preset", help="שם תרחיש שמור, או 'list'")
    p.add_argument("--name", default="רעיון חדש")
    p.add_argument("--sharpe-annual", type=float)
    p.add_argument("--sharpe-per-trade", type=float)
    p.add_argument("--sigma-pct", type=float)
    p.add_argument("--sigma-dollars", type=float)
    p.add_argument("--price", type=float)
    p.add_argument("--multiplier", type=float)
    p.add_argument("--tick", type=float, default=0.0)
    p.add_argument("--commission", type=float, default=1.52)
    p.add_argument("--slippage-ticks", type=float, default=2.0)
    p.add_argument("--trades-per-year", type=float, default=250)
    p.add_argument("--account", type=float, default=5000)
    p.add_argument("--contracts", type=int, default=1)
    p.add_argument("--risk-free", type=float, default=3.5)
    a = p.parse_args()

    if a.preset == "list":
        print("תרחישים שמורים:\n")
        for k, v in PRESETS.items():
            print(f"  {k:<10} {v.name}")
        return

    if a.preset:
        if a.preset not in PRESETS:
            print(f"✗ לא קיים: {a.preset}. נסה --preset list")
            return
        report(PRESETS[a.preset])
        return

    # ── קלט ידני ──
    has_sharpe = a.sharpe_annual is not None or a.sharpe_per_trade is not None
    has_sigma = a.sigma_dollars is not None or (
        a.sigma_pct is not None and a.price is not None and a.multiplier is not None)

    if not has_sharpe:
        p.error("נדרש --sharpe-annual או --sharpe-per-trade")
    if not has_sigma:
        p.error("נדרש --sigma-dollars, או (--sigma-pct + --price + --multiplier)")

    report(Idea(
        name=a.name,
        sharpe_annual=a.sharpe_annual,
        sharpe_per_trade_input=a.sharpe_per_trade,
        sigma_pct=a.sigma_pct,
        sigma_dollars_input=a.sigma_dollars,
        price=a.price, multiplier=a.multiplier, tick=a.tick,
        commission=a.commission, slippage_ticks=a.slippage_ticks,
        trades_per_year=a.trades_per_year, account=a.account,
        contracts=a.contracts, risk_free_pct=a.risk_free,
    ))


if __name__ == "__main__":
    main()