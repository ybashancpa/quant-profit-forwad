"""
test_orders.py — בדיקות unit ל-bracket builder ולמצבי כשל

רץ בלי Gateway, בלי חיבור, בלי הזמנות.
מטרה: לוודא שהבאג שהפיל את מנגנון ההגנה (parentId שבור)
לא יכול לחזור, ושה-invariants תופסים כל מבנה לא חוקי.

שימוש:
    python test_orders.py
"""

from __future__ import annotations

import sys

from ib_async import LimitOrder, MarketOrder, StopOrder

from orders import Bracket, BracketError, build_market_bracket, validate_bracket


class IdGen:
    """מדמה את ib.client.getReqId — מונה עולה"""

    def __init__(self, start=100):
        self.n = start

    def __call__(self):
        self.n += 1
        return self.n


PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} {detail}")


def expect_raises(name: str, fn):
    global PASS, FAIL
    try:
        fn()
        FAIL += 1
        print(f"  ✗ {name} — ציפינו ל-BracketError אבל לא נזרק")
    except BracketError:
        PASS += 1
        print(f"  ✓ {name}")
    except Exception as e:
        FAIL += 1
        print(f"  ✗ {name} — נזרק {type(e).__name__} במקום BracketError")


def main():
    print("═" * 64)
    print("  test_orders.py — bracket builder invariants")
    print("═" * 64)

    # ── 1. בנייה תקינה BUY ──
    print("\n[1] בנייה תקינה — BUY")
    b = build_market_bracket("BUY", 2, 105.0, 99.5, IdGen())
    check("parent הוא MKT", b.parent.orderType == "MKT")
    check("target הוא LMT", b.take_profit.orderType == "LMT")
    check("stop הוא STP", b.stop_loss.orderType == "STP")
    check("IDs חיוביים", all(o.orderId > 0 for o in b.orders))
    check("IDs ייחודיים", len({o.orderId for o in b.orders}) == 3)
    check("target.parentId → parent", b.take_profit.parentId == b.parent_id)
    check("stop.parentId → parent", b.stop_loss.parentId == b.parent_id)
    check("ילדים בצד הנגדי (SELL)",
          b.take_profit.action == "SELL" and b.stop_loss.action == "SELL")
    check("כמות זהה בכולם",
          all(o.totalQuantity == 2 for o in b.orders))
    check("transmit = (F,F,T)",
          (b.parent.transmit, b.take_profit.transmit, b.stop_loss.transmit)
          == (False, False, True))
    check("סדר שידור: parent→target→stop",
          [o.orderId for o in b.orders]
          == [b.parent_id, b.target_id, b.stop_id])
    check("מחיר יעד נשמר", abs(b.take_profit.lmtPrice - 105.0) < 1e-9)
    check("מחיר סטופ נשמר", abs(b.stop_loss.auxPrice - 99.5) < 1e-9)

    # ── 2. בנייה תקינה SELL ──
    print("\n[2] בנייה תקינה — SELL")
    b2 = build_market_bracket("SELL", 1, 95.0, 102.0, IdGen())
    check("ילדים בצד הנגדי (BUY)",
          b2.take_profit.action == "BUY" and b2.stop_loss.action == "BUY")
    check("קישורי parentId", b2.take_profit.parentId == b2.parent_id
          and b2.stop_loss.parentId == b2.parent_id)

    # ── 3. השוואה לבאג המקורי ──
    print("\n[3] רגרסיה: הבאג המקורי (הורה עם orderId=0)")
    # סימולציה של מה שהקוד הישן יצר: הורה חדש בלי orderId
    broken_parent = MarketOrder("BUY", 1)           # orderId=0
    tp = LimitOrder("SELL", 1, 105.0, orderId=2, parentId=999)
    sl = StopOrder("SELL", 1, 99.0, orderId=3, parentId=999)
    broken = Bracket(broken_parent, tp, sl)
    try:
        validate_bracket(broken, "BUY", 1, 105.0, 99.0)
        check("מבנה שבור נתפס", False, "validate_bracket לא זיהה את הבאג")
    except BracketError:
        check("מבנה שבור נתפס (orderId=0 / parentId זר)", True)

    # ── 4. קלטים לא חוקיים ──
    print("\n[4] קלטים לא חוקיים נדחים")
    expect_raises("action לא חוקי",
                  lambda: build_market_bracket("HOLD", 1, 105, 99, IdGen()))
    expect_raises("quantity=0",
                  lambda: build_market_bracket("BUY", 0, 105, 99, IdGen()))
    expect_raises("BUY עם סטופ מעל יעד",
                  lambda: build_market_bracket("BUY", 1, 100, 101, IdGen()))
    expect_raises("SELL עם סטופ מתחת ליעד",
                  lambda: build_market_bracket("SELL", 1, 100, 99, IdGen()))
    expect_raises("מחיר שלילי",
                  lambda: build_market_bracket("BUY", 1, -5, 99, IdGen()))

    # ── 5. הקצאת IDs כושלת ──
    print("\n[5] הקצאת orderId כושלת → BracketError")
    def broken_gen():
        raise RuntimeError("client disconnected")
    expect_raises("next_order_id נכשל",
                  lambda: build_market_bracket("BUY", 1, 105, 99, broken_gen))

    # ── 6. IDs לא ייחודיים ──
    print("\n[6] IDs כפולים נדחים")
    dup = iter([7, 7, 8])
    expect_raises("שני IDs זהים",
                  lambda: build_market_bracket("BUY", 1, 105, 99, lambda: next(dup)))

    # ── 7. mutation: כל invariant בנפרד ──
    print("\n[7] mutation tests — כל invariant תופס שיבוש")
    base = build_market_bracket("BUY", 1, 105.0, 99.0, IdGen())

    def mutated(**kw):
        m = build_market_bracket("BUY", 1, 105.0, 99.0, IdGen())
        for k, v in kw.items():
            setattr(m.parent, k, v) if hasattr(m.parent, k) else None
        return m

    m1 = build_market_bracket("BUY", 1, 105.0, 99.0, IdGen())
    m1.take_profit.parentId = 0
    expect_raises("target.parentId=0", lambda: validate_bracket(m1, "BUY", 1, 105.0, 99.0))

    m2 = build_market_bracket("BUY", 1, 105.0, 99.0, IdGen())
    m2.stop_loss.parentId = m2.target_id  # מצביע לאח במקום להורה
    expect_raises("stop.parentId → target", lambda: validate_bracket(m2, "BUY", 1, 105.0, 99.0))

    m3 = build_market_bracket("BUY", 1, 105.0, 99.0, IdGen())
    m3.parent.transmit = True  # שידור מוקדם של ההורה
    expect_raises("parent.transmit=True", lambda: validate_bracket(m3, "BUY", 1, 105.0, 99.0))

    m4 = build_market_bracket("BUY", 1, 105.0, 99.0, IdGen())
    m4.stop_loss.transmit = False  # אף אחד לא משדר
    expect_raises("stop.transmit=False", lambda: validate_bracket(m4, "BUY", 1, 105.0, 99.0))

    m5 = build_market_bracket("BUY", 1, 105.0, 99.0, IdGen())
    m5.stop_loss.totalQuantity = 5  # כמות לא תואמת
    expect_raises("כמות ילד שונה", lambda: validate_bracket(m5, "BUY", 1, 105.0, 99.0))

    m6 = build_market_bracket("BUY", 1, 105.0, 99.0, IdGen())
    m6.take_profit.action = "BUY"  # ילד באותו צד
    expect_raises("ילד לא נגדי", lambda: validate_bracket(m6, "BUY", 1, 105.0, 99.0))

    # ── סיכום ──
    print("\n" + "═" * 64)
    print(f"  {PASS} עברו | {FAIL} נכשלו")
    print("═" * 64)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())