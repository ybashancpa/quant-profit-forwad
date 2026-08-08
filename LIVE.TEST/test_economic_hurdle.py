"""
test_economic_hurdle.py — בדיקות יחידה לכלי Economic Hurdle

מאמת:
  1. שחזור מדויק של H2 מול summary.json
  2. ולידציה של קלטים
  3. חישובי ספים (breakeven, hurdle_ticks, required_sharpe)
  4. תנאי גבול (>= ולא >)
  5. מקרי קצה (אפס, שלילי)

שימוש:
    python test_economic_hurdle.py
"""

from __future__ import annotations

import math
import sys

from economic_hurdle import Idea


def test_h2_exact_reproduction():
    """
    שחזור מדויק של H2 מול results_h2/summary.json:
    - Sharpe לעסקה: 0.063
    - sigma_dollars: 35.8301
    - עמלה: 1.52, החלקה: 2 טיקים × $0.50 = $1.00
    - נטו לעסקה: 0.063 × 35.8301 − 1.52 − 1.00 = −0.2627
    - שנתי: −0.2627 × 250 = −65.68
    """
    # price/multiplier לא בשימוש כאן — sigma_dollars_input גובר.
    # הם נשמרים רק כדי שהדוח יציג נומינלי אם יידרש.
    idea = Idea(
        name="H2 exact",
        sharpe_per_trade_input=0.063,
        sigma_dollars_input=35.8301,
        price=2988.2, multiplier=5.0, tick=0.10,
        commission=1.52, slippage_ticks=2, trades_per_year=250,
    )

    expected_gross = 0.063 * 35.8301  # = 2.2573
    assert abs(idea.expected_gross - expected_gross) < 0.001, \
        f"expected_gross: {idea.expected_gross} != {expected_gross}"

    expected_net = expected_gross - 1.52 - 1.00  # = -0.2627
    assert abs(idea.net_per_trade - expected_net) < 0.001, \
        f"net_per_trade: {idea.net_per_trade} != {expected_net}"

    expected_annual = expected_net * 250  # = -65.68
    assert abs(idea.annual_net - expected_annual) < 0.1, \
        f"annual_net: {idea.annual_net} != {expected_annual}"

    assert not idea.passes, "H2 חייב להיכשל (FAIL)"
    print("  ✓ שחזור H2 מדויק: net=$-65.68, REJECT")


def test_h2_zero_slippage():
    """
    באפס החלקה: $2.2573 − $1.52 = +$0.7373 לעסקה
    שנתי: $184.3 = 3.69% — מעל הסף $175
    """
    idea = Idea(
        name="H2 zero slip",
        sharpe_per_trade_input=0.063,
        sigma_dollars_input=35.8301,
        price=2988.2, multiplier=5.0, tick=0.10,
        commission=1.52, slippage_ticks=0, trades_per_year=250,
    )
    assert idea.passes, "באפס החלקה H2 עובר (בקושי)"
    assert abs(idea.annual_net - 184.3) < 1.0
    print("  ✓ H2 באפס החלקה: +$184/שנה, PASS (בקושי)")


def test_hurdle_ticks():
    """
    hurdle_ticks = (expected − commission − rf_per_trade) / tick_value
    = (2.2573 − 1.52 − 0.70) / 0.50 = 0.0746 ≈ 0.07
    """
    idea = Idea(
        name="H2 hurdle ticks",
        sharpe_per_trade_input=0.063,
        sigma_dollars_input=35.8301,
        price=2988.2, multiplier=5.0, tick=0.10,
        commission=1.52, slippage_ticks=2, trades_per_year=250,
    )
    expected_hurdle_ticks = (2.2573 - 1.52 - 0.70) / 0.50
    assert abs(idea.hurdle_ticks - expected_hurdle_ticks) < 0.01, \
        f"hurdle_ticks: {idea.hurdle_ticks} != {expected_hurdle_ticks}"
    print(f"  ✓ hurdle_ticks = {idea.hurdle_ticks:.2f} (צפוי ~0.07)")


def test_breakeven_ticks():
    """
    breakeven_ticks = (expected − commission) / tick_value
    = (2.2573 − 1.52) / 0.50 = 1.4746 ≈ 1.47
    """
    idea = Idea(
        name="H2 breakeven",
        sharpe_per_trade_input=0.063,
        sigma_dollars_input=35.8301,
        price=2988.2, multiplier=5.0, tick=0.10,
        commission=1.52, slippage_ticks=2, trades_per_year=250,
    )
    expected_be = (2.2573 - 1.52) / 0.50
    assert abs(idea.breakeven_ticks - expected_be) < 0.01, \
        f"breakeven_ticks: {idea.breakeven_ticks} != {expected_be}"
    print(f"  ✓ breakeven_ticks = {idea.breakeven_ticks:.2f} (צפוי ~1.47)")


def test_required_sharpe():
    """
    Sharpe שנתי נדרש בתרחיש 2 טיקים:
    need = 1.52 + 1.00 + 0.70 = 3.22
    required_per_trade = 3.22 / 35.8301 = 0.0899
    required_annual = 0.0899 × √250 = 1.42
    """
    idea = Idea(
        name="H2 required sharpe",
        sharpe_per_trade_input=0.063,
        sigma_dollars_input=35.8301,
        price=2988.2, multiplier=5.0, tick=0.10,
        commission=1.52, slippage_ticks=2, trades_per_year=250,
    )
    expected_req = (1.52 + 1.00 + 0.70) / 35.8301 * math.sqrt(250)
    assert abs(idea.required_sharpe_annual - expected_req) < 0.01, \
        f"required_sharpe: {idea.required_sharpe_annual} != {expected_req}"
    print(f"  ✓ required_sharpe_annual = {idea.required_sharpe_annual:.2f} (צפוי ~1.42)")


def test_equality_passes():
    """תנאי גבול: annual_net == hurdle_dollars → PASS (>=)"""
    # נבנה רעיון שעובר בדיוק את הסף
    # hurdle = 5000 × 0.035 = 175
    # צריך net_per_trade × 250 = 175 → net_per_trade = 0.70
    # expected_gross = 0.70 + commission + slippage = 0.70 + 0 + 0 = 0.70
    # sharpe_per_trade × sigma_dollars = 0.70
    # נבחר sigma_dollars = 100, sharpe_per_trade = 0.007
    idea = Idea(
        name="Equality test",
        sharpe_per_trade_input=0.007,
        sigma_dollars_input=100.0,
        price=100, multiplier=1.0, tick=0.01,
        commission=0.0, slippage_ticks=0, trades_per_year=250,
    )
    assert abs(idea.annual_net - 175.0) < 0.01, \
        f"annual_net: {idea.annual_net} != 175"
    assert idea.passes, "שוויון בדיוק לסף חייב לעבור (>=)"
    print("  ✓ תנאי גבול: annual_net == hurdle → PASS")


def test_validation_errors():
    """ולידציה: קלטים לא חוקיים חייבים לזרוק ValueError"""
    errors = 0

    # trades_per_year = 0
    try:
        Idea(name="bad", sharpe_annual=1.0, sigma_pct=0.1, price=100,
             multiplier=1, trades_per_year=0)
        print("  ✗ trades_per_year=0 לא זרק שגיאה")
    except ValueError:
        errors += 1

    # account = 0
    try:
        Idea(name="bad", sharpe_annual=1.0, sigma_pct=0.1, price=100,
             multiplier=1, account=0)
        print("  ✗ account=0 לא זרק שגיאה")
    except ValueError:
        errors += 1

    # חסר sharpe
    try:
        Idea(name="bad", sigma_pct=0.1, price=100, multiplier=1)
        print("  ✗ חסר sharpe לא זרק שגיאה")
    except ValueError:
        errors += 1

    # חסר sigma
    try:
        Idea(name="bad", sharpe_annual=1.0)
        print("  ✗ חסר sigma לא זרק שגיאה")
    except ValueError:
        errors += 1

    # commission שלילי
    try:
        Idea(name="bad", sharpe_annual=1.0, sigma_pct=0.1, price=100,
             multiplier=1, commission=-1)
        print("  ✗ commission<0 לא זרק שגיאה")
    except ValueError:
        errors += 1

    assert errors == 5, f"ציפיתי ל-5 שגיאות ולידציה, קיבלתי {errors}"
    print("  ✓ ולידציה: 5 מקרים לא חוקיים נזרקו כראוי")


def test_sigma_dollars_scaling():
    """sigma_dollars_input × contracts"""
    idea1 = Idea(name="1 contract", sharpe_per_trade_input=0.063,
                 sigma_dollars_input=35.83, price=100, multiplier=1,
                 contracts=1)
    idea4 = Idea(name="4 contracts", sharpe_per_trade_input=0.063,
                 sigma_dollars_input=35.83, price=100, multiplier=1,
                 contracts=4)
    assert abs(idea4.sigma_dollars - 4 * idea1.sigma_dollars) < 0.01
    print("  ✓ sigma_dollars scales with contracts")


def test_sharpe_per_trade_from_annual():
    """sharpe_annual → sharpe_per_trade = annual / √trades"""
    idea = Idea(name="conversion", sharpe_annual=1.0, sigma_pct=0.1,
                price=100, multiplier=1, trades_per_year=250)
    expected = 1.0 / math.sqrt(250)
    assert abs(idea.sharpe_per_trade - expected) < 1e-10
    print("  ✓ sharpe_annual → sharpe_per_trade conversion")


def test_sharpe_per_trade_input_overrides():
    """sharpe_per_trade_input גובר על sharpe_annual"""
    idea = Idea(name="override", sharpe_annual=999.0,
                sharpe_per_trade_input=0.063, sigma_pct=0.1,
                price=100, multiplier=1)
    assert idea.sharpe_per_trade == 0.063
    print("  ✓ sharpe_per_trade_input overrides sharpe_annual")


def main():
    print("═" * 60)
    print("  Economic Hurdle — Unit Tests")
    print("═" * 60)

    tests = [
        ("שחזור H2 מדויק", test_h2_exact_reproduction),
        ("H2 באפס החלקה", test_h2_zero_slippage),
        ("hurdle_ticks", test_hurdle_ticks),
        ("breakeven_ticks", test_breakeven_ticks),
        ("required_sharpe_annual", test_required_sharpe),
        ("תנאי גבול (>=", test_equality_passes),
        ("ולידציה", test_validation_errors),
        ("sigma_dollars scaling", test_sigma_dollars_scaling),
        ("sharpe conversion", test_sharpe_per_trade_from_annual),
        ("sharpe override", test_sharpe_per_trade_input_overrides),
    ]

    failures = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as e:
            print(f"  ✗ {name}: {e}")
            failures += 1
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            failures += 1

    print("═" * 60)
    if failures == 0:
        print(f"  ✓ כל {len(tests)} הבדיקות עברו")
    else:
        print(f"  ✗ {failures} כשלים מתוך {len(tests)}")
        sys.exit(1)


if __name__ == "__main__":
    main()