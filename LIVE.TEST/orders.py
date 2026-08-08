"""
orders.py — בנייה ואימות של bracket order (Market parent + LMT target + STP stop)

‼️ באג שתוקן כאן ‼️
הגרסה הקודמת קראה ל-IB.bracketOrder() ואז החליפה את bracket[0]
ב-MarketOrder חדש עם orderId=0. הילדים נשארו מצביעים (parentId)
על הורה שלא קיים — הסטופ והיעד נדחו או התייתמו. בדיוק מנגנון
ההגנה שאמור לשרוד קריסת תהליך — לא היה נשלח.

התיקון: בנייה מפורשת של שלוש ההזמנות עם IDs ייחודיים שהוקצו מראש,
קשרי parentId נכונים, וסדר שידור parent → target → stop (האחרון
ב-transmit=True משדר את כולם).

הפונקציות כאן טהורות (לא שולחות כלום) ונבדקות ב-test_orders.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from ib_async import LimitOrder, MarketOrder, StopOrder


class BracketError(RuntimeError):
    """כשל בבנייה או באימות bracket — fail closed, לא שולחים."""


@dataclass(frozen=True)
class Bracket:
    parent: MarketOrder
    take_profit: LimitOrder
    stop_loss: StopOrder

    @property
    def orders(self) -> list:
        # סדר השידור המחייב: parent → target → stop
        return [self.parent, self.take_profit, self.stop_loss]

    @property
    def parent_id(self) -> int:
        return self.parent.orderId

    @property
    def target_id(self) -> int:
        return self.take_profit.orderId

    @property
    def stop_id(self) -> int:
        return self.stop_loss.orderId


def build_market_bracket(
    action: str,
    quantity: int,
    take_profit_price: float,
    stop_price: float,
    next_order_id,
) -> Bracket:
    """
    בונה bracket עם Market entry.

    Args:
        action: 'BUY' או 'SELL'
        quantity: מספר חוזים (>= 1)
        take_profit_price: מחיר יעד (LMT בצד הנגדי)
        stop_price: מחיר סטופ (STP בצד הנגדי)
        next_order_id: callable שמחזיר orderId ייחודי (למשל ib.client.getReqId)

    Raises:
        BracketError אם הקלטים לא חוקיים.
    """
    if action not in ("BUY", "SELL"):
        raise BracketError(f"action לא חוקי: {action}")
    if quantity < 1:
        raise BracketError(f"quantity לא חוקית: {quantity}")
    if take_profit_price <= 0 or stop_price <= 0:
        raise BracketError("מחירי יעד/סטופ חייבים להיות חיוביים")
    if action == "BUY" and stop_price >= take_profit_price:
        raise BracketError(
            f"BUY: סטופ ({stop_price}) חייב להיות מתחת ליעד ({take_profit_price})"
        )
    if action == "SELL" and stop_price <= take_profit_price:
        raise BracketError(
            f"SELL: סטופ ({stop_price}) חייב להיות מעל ליעד ({take_profit_price})"
        )

    reverse = "SELL" if action == "BUY" else "BUY"

    try:
        parent_id = int(next_order_id())
        target_id = int(next_order_id())
        stop_id = int(next_order_id())
    except Exception as e:
        raise BracketError(f"הקצאת orderId נכשלה: {e}") from e

    if len({parent_id, target_id, stop_id}) != 3:
        raise BracketError(
            f"orderIds לא ייחודיים: {parent_id}, {target_id}, {stop_id}"
        )

    parent = MarketOrder(
        action, quantity,
        orderId=parent_id,
        transmit=False,
    )
    take_profit = LimitOrder(
        reverse, quantity, take_profit_price,
        orderId=target_id,
        parentId=parent_id,
        transmit=False,
    )
    stop_loss = StopOrder(
        reverse, quantity, stop_price,
        orderId=stop_id,
        parentId=parent_id,
        transmit=True,   # האחרון משדר את כל הקבוצה
    )

    bracket = Bracket(parent, take_profit, stop_loss)
    validate_bracket(bracket, action, quantity, take_profit_price, stop_price)
    return bracket


def validate_bracket(
    bracket: Bracket,
    action: str,
    quantity: int,
    take_profit_price: float,
    stop_price: float,
) -> None:
    """
    Invariants שאם לא מתקיימים — לא שולחים. הרצה לפני placeOrder.
    """
    p, t, s = bracket.parent, bracket.take_profit, bracket.stop_loss
    reverse = "SELL" if action == "BUY" else "BUY"

    checks = [
        (p.orderId > 0 and t.orderId > 0 and s.orderId > 0,
         "כל orderId חייב להיות חיובי"),
        (len({p.orderId, t.orderId, s.orderId}) == 3,
         "orderIds חייבים להיות ייחודיים"),
        (t.parentId == p.orderId,
         f"target.parentId ({t.parentId}) != parent.orderId ({p.orderId})"),
        (s.parentId == p.orderId,
         f"stop.parentId ({s.parentId}) != parent.orderId ({p.orderId})"),
        (p.orderType == "MKT", f"parent צריך להיות MKT, התקבל {p.orderType}"),
        (t.orderType == "LMT", f"target צריך להיות LMT, התקבל {t.orderType}"),
        (s.orderType == "STP", f"stop צריך להיות STP, התקבל {s.orderType}"),
        (p.action == action, f"parent.action {p.action} != {action}"),
        (t.action == reverse and s.action == reverse,
         "ילדים חייבים להיות בצד הנגדי"),
        (p.totalQuantity == quantity and t.totalQuantity == quantity
         and s.totalQuantity == quantity, "כמות חייבת להיות זהה בכל ההזמנות"),
        ((p.transmit, t.transmit, s.transmit) == (False, False, True),
         "transmit חייב להיות (False, False, True) — האחרון משדר"),
        (abs(t.lmtPrice - take_profit_price) < 1e-9, "מחיר יעד לא תואם"),
        (abs(s.auxPrice - stop_price) < 1e-9, "מחיר סטופ לא תואם"),
    ]
    for ok, msg in checks:
        if not ok:
            raise BracketError(f"bracket לא חוקי: {msg}")