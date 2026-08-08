"""
dryrun_sim.py — סימולציית מחזור חיים של פוזיציה ב-dry-run

למה זה קיים
───────────
בלי זה, `--dry-run` בודק רק שסיגנלים נוצרים. הוא לא בודק trailing,
לא יציאת משטר, לא סגירה ב-15:50, ולא שהלוג מכיל מחזור שלם ש-`reconcile.py`
יכול לעבוד עליו.

עם זה, כל מסלול הקוד נבדק — בלי לשלוח אף הזמנה.

מה זה **לא** בודק
──────────────────
החלקה אמיתית, דחיות הזמנות, התנהגות OCA בשרת, מילויים חלקיים.
לאלה נדרש חשבון Paper ששולח הזמנות בפועל.

שקילות עם הבקטסט (נקריטית!)
───────────────────────────
המסלול המדומה חייב לייצר בדיוק את אותן עסקאות כמו Backtester על אותם
נרות. לכן:

  1. מילוי בפתיחת הנר **הבא** (pending), עם אותה החלקת כניסה כמו
     `_slip()` בבקטסט. מילוי במחיר סגירת נר הסיגנל היה אופטימי
     באופן שיטתי (הפער סגירה→פתיחה נוטה להיות בכיוון התנועה
     שיצרה את הסיגנל).
  2. סטופ/יעד מחושבים מחדש ממחיר המילוי, עם `target_r_multiple`
     מהקונפיג — בדיוק כמו backtest.py.
  3. sizing מחושב **בזמן המילוי** ממחיר המילוי והסטופ החדש ומההון
     העדכני — כמו בבקטסט, לא בזמן הסיגנל.
  4. סדר בדיקות היציאה זהה לבקטסט: stop → target → EOD → regime →
     trailing. כשסטופ ויעד נפגעים באותו נר — הסטופ קודם (הנחה
     שמרנית).
  5. יציאת regime/Eוד מדמה החלקת יציאה כמו `_slip(..., is_entry=False)`.

שילוב
─────
ב-`live_trader.py`:

    from dryrun_sim import queue_entry, fill_pending, simulate_exit_check, simulate_exit

    # ב-enter(), אחרי חישוב sizing ולפני שליחת ההזמנה:
    if self.args.dry_run:
        queue_entry(self, symbol, sig, signal_ts)
        return

    # ‼️ ב-process_closed_bar(), אחרי breaker.new_day ולפני בדיקת EOD:
    if self.args.dry_run:
        fill_pending(self, symbol, row, ts)
    # אין return כאן! בבקטסט פוזיציה שמתמלאת בנר N גם מנוהלת
    # (stop/target) מול הנר N. לכן ממשיכים לאותו מסלול ניהול.

    # ב-manage_position(), לפני בדיקת ib.positions():
    if self.args.dry_run:
        hit, why = simulate_exit_check(self, symbol, row)
        if hit:
            self.close_position(symbol, why, row=row, ts=ts)
            return
    else:
        ... בדיקת ib.positions() הקיימת ...

    # ב-close_position(), בראש הפונקציה:
    if self.args.dry_run:
        simulate_exit(self, symbol, reason, row=row, ts=ts)
        return
"""

from __future__ import annotations

from strategy import Direction


# ══════════════════════════════════════════════════════════════
def _slip(inst, price: float, direction: Direction, is_entry: bool) -> float:
    """
    שכפול מדויק של Backtester._slip — החלקה תמיד לרעתך + עיגול לטיק.
    חייב להישאר זהה לבקטסט, אחרת בדיקת השקילות תיכשל.
    """
    slip = inst.slippage_ticks * inst.tick_size
    if is_entry:
        adj = slip if direction is Direction.LONG else -slip
    else:
        adj = -slip if direction is Direction.LONG else slip
    return inst.round_to_tick(price + adj)


# ══════════════════════════════════════════════════════════════
def queue_entry(trader, symbol: str, sig, signal_ts) -> None:
    """
    ‼️ אינו יוצר פוזיציה. רק מציב אותה בהמתנה לנר הבא.
    ה-sizing הסופי יחושב בזמן המילוי (מחיר מילוי + הון עדכני).
    """
    if not hasattr(trader, "_sim_pending"):
        trader._sim_pending = {}
    trader._sim_pending[symbol] = (sig, signal_ts)
    trader.logger.log("pending_entry", symbol=symbol, simulated=True,
                      signal_time=str(signal_ts),
                      signal_price=sig.entry_price,
                      direction=sig.direction.name)
    print(f"     ↳ [dry-run] ממתין למילוי בפתיחת הנר הבא")


def fill_pending(trader, symbol: str, row, ts) -> bool:
    """
    ממלא סיגנל ממתין בפתיחת הנר הנוכחי — כולל החלקת כניסה,
    חישוב סטופ/יעד מחדש, ו-sizing עדכני. בדיוק שלב 1 של הבקטסט.
    מחזיר True אם נוצרה פוזיציה.
    """
    pending = getattr(trader, "_sim_pending", {}).pop(symbol, None)
    if pending is None:
        return False
    sig, signal_ts = pending

    inst = trader.instruments[symbol]

    # החלקת כניסה — אותו חישוב כמו בבקטסט
    fill = _slip(inst, float(row["open"]), sig.direction, is_entry=True)

    # סטופ/יעד מחושבים מחדש ממחיר המילוי (backtest.py, שלב 1).
    # ‼️ ה-sizing מחושב על המחירים ה*לא מעוגלים* — בדיוק כמו בבקטסט,
    #    שמריץ size_position(fill, stop) לפני round_to_tick. עיגול
    #    מוקדם היה משנה את מרחק הסטופ ואת מספר החוזים.
    c = trader.strat_cfg
    risk_pts = sig.risk_points
    if sig.direction is Direction.LONG:
        stop = fill - risk_pts
        target = fill + c.target_r_multiple * risk_pts
    else:
        stop = fill + risk_pts
        target = fill - c.target_r_multiple * risk_pts

    # sizing בזמן המילוי, מהמחירים החדשים ומההון העדכני
    from risk import size_position
    sizing = size_position(inst, fill, stop, trader.equity, trader.risk_cfg)
    if not sizing.ok:
        trader.logger.log("sizing_rejected", symbol=symbol, simulated=True,
                          reason=sizing.rejected_reason)
        print(f"     ↳ [dry-run] נדחה במילוי: {sizing.rejected_reason}")
        return False

    stop = inst.round_to_tick(stop)
    target = inst.round_to_tick(target)

    from live_trader import LivePosition   # ייבוא מושהה — הימנעות ממעגליות
    trader.positions[symbol] = LivePosition(
        symbol=symbol,
        direction=sig.direction,
        contracts=sizing.contracts,
        entry_price=fill,
        initial_stop=stop,
        current_stop=stop,
        target=target,
        signal_time=signal_ts,
        entry_time=ts,
        trail_extreme=fill,
    )

    slip = abs(fill - float(row["open"]))
    trader.logger.log(
        "entry", symbol=symbol, simulated=True,
        signal_time=str(signal_ts), entry_time=str(ts),
        direction=sig.direction.name,
        fill_price=fill, slippage=slip,
        slippage_ticks=slip / inst.tick_size if inst.tick_size else 0.0,
        contracts=sizing.contracts, stop=stop, target=target,
        risk_dollars=sizing.risk_dollars,
    )
    print(f"     ↳ [dry-run] פוזיציה מדומה: {sizing.contracts} חוזים @ {fill:.2f} "
          f"| stop {stop:.2f} | target {target:.2f}")
    return True


# ══════════════════════════════════════════════════════════════
def simulate_exit_check(trader, symbol: str, row) -> tuple[bool, str]:
    """
    מחליף את ה-OCA בשרת: בודק סטופ ויעד מול high/low של הנר.

    ‼️ הסטופ נבדק לפני היעד — אותה הנחה שמרנית כמו ב-backtest.py.
       כששניהם נפגעים באותו נר, בלי נתוני טיק אי אפשר לדעת מי ראשון.
    """
    pos = trader.positions.get(symbol)
    if pos is None:
        return False, ""

    c = trader.strat_cfg
    is_long = pos.direction is Direction.LONG
    low, high = float(row["low"]), float(row["high"])

    if is_long:
        if low <= pos.current_stop:
            return True, "Stop Loss"
        if high >= pos.target:
            return True, f"Target {c.target_r_multiple}R"
    else:
        if high >= pos.current_stop:
            return True, "Stop Loss"
        if low <= pos.target:
            return True, f"Target {c.target_r_multiple}R"

    return False, ""


# ══════════════════════════════════════════════════════════════
def simulate_exit(trader, symbol: str, reason: str, row=None, ts=None) -> None:
    """
    סוגר פוזיציה מדומה ורושם `exit` מלא, כדי ש-reconcile.py ובדיקת
    השקילות יוכלו להשוות עסקה מול עסקה מול הבקטסט.

    מחיר יציאה:
      Stop/Target → מחיר הסטופ/יעד בדיוק (כמו בקטסט, בלי החלקה).
      EOD/Regime  → סגירת הנר עם החלקת יציאה (כמו `_slip` בבקטסט).
    """
    pos = trader.positions.get(symbol)
    if pos is None:
        return

    inst = trader.instruments[symbol]

    if reason.startswith("Stop Loss"):
        fill = pos.current_stop
    elif reason.startswith("Target"):
        fill = pos.target
    elif row is not None:
        fill = _slip(inst, float(row["close"]), pos.direction, is_entry=False)
    else:
        fill = pos.entry_price

    is_long = pos.direction is Direction.LONG
    pts = (fill - pos.entry_price) if is_long else (pos.entry_price - fill)
    gross = inst.points_to_usd(pts, pos.contracts)
    comm = inst.commission_rt * pos.contracts
    net = gross - comm

    risk_pts = abs(pos.entry_price - pos.initial_stop)
    r_mult = (net / (risk_pts * inst.multiplier * pos.contracts)
              if risk_pts > 0 else 0.0)

    trader.equity += net
    trader.breaker.record_trade(symbol, net)
    if hasattr(trader, "bars_since_exit"):
        trader.bars_since_exit[symbol] = 0

    trader.logger.log(
        "exit", symbol=symbol, simulated=True, reason=reason,
        exit_time=str(ts), fill_price=fill, entry_price=pos.entry_price,
        contracts=pos.contracts, gross_pnl=gross, commission=comm,
        net_pnl=net, r_multiple=r_mult, bars_held=pos.bars_held,
        signal_time=str(pos.signal_time),
    )
    print(f"  ● [dry-run] {symbol} יצא @ {fill:.2f} | {reason} | "
          f"${net:+,.2f} ({r_mult:+.2f}R)")

    del trader.positions[symbol]


# ══════════════════════════════════════════════════════════════
def self_test() -> bool:
    """בדיקה עצמאית ללא חיבור — מאמתת את לוגיקת stop/target"""
    import pandas as pd

    class _Cfg:
        target_r_multiple = 2.0

    class _Pos:
        def __init__(self, d, stop, target):
            self.direction, self.current_stop, self.target = d, stop, target

    class _T:
        def __init__(self, pos):
            self.positions = {"X": pos}
            self.strat_cfg = _Cfg()

    cases = [
        # (כיוון, סטופ, יעד, low, high, צפוי)
        (Direction.LONG,  100, 120,  99, 110, "Stop Loss"),
        (Direction.LONG,  100, 120, 105, 121, "Target 2.0R"),
        (Direction.LONG,  100, 120,  99, 121, "Stop Loss"),  # שניהם → סטופ
        (Direction.LONG,  100, 120, 105, 110, ""),
        (Direction.SHORT, 120, 100, 110, 121, "Stop Loss"),
        (Direction.SHORT, 120, 100,  99, 110, "Target 2.0R"),
        (Direction.SHORT, 120, 100,  99, 121, "Stop Loss"),  # שניהם → סטופ
        (Direction.SHORT, 120, 100, 105, 115, ""),
    ]

    ok = True
    for d, stop, tgt, lo, hi, want in cases:
        t = _T(_Pos(d, stop, tgt))
        row = pd.Series({"low": lo, "high": hi})
        hit, why = simulate_exit_check(t, "X", row)
        got = why if hit else ""
        mark = "✓" if got == want else "✗"
        if got != want:
            ok = False
        print(f"  {mark} {d.name:5} stop={stop} tgt={tgt} "
              f"L/H={lo}/{hi} → {got or '(אין יציאה)'}")

    hit, _ = simulate_exit_check(type("E", (), {"positions": {}})(), "X",
                                 pd.Series({"low": 1, "high": 2}))
    print(f"  {'✓' if not hit else '✗'} פוזיציה לא קיימת → אין יציאה")
    return ok and not hit


if __name__ == "__main__":
    print("בדיקת dryrun_sim:")
    print("✓ הכל עבר" if self_test() else "✗ יש כשלים")