"""
live_trader.py — סוחר חי מול IB Gateway

עיקרון מנחה
───────────
הקובץ הזה **אינו מכיל לוגיקת מסחר**. הוא מייבא את `MomentumPullbackStrategy`
וקורא ל-`prepare()` ו-`generate_signal()` — בדיוק אותן פונקציות שהבקטסט קורא
להן. כל שכפול של הלוגיקה כאן היה יוצר את הפער שאנחנו מנסים למדוד.

תפקידו: לנהל חיבור, נרות, הזמנות ולוגים. שום החלטה.

בטיחות
──────
1. הסטופ נשלח כ-bracket order ויושב על שרתי IBKR. אם התהליך קורס
   או החיבור נופל — הסטופ עדיין שם. זו ההגנה החשובה ביותר.
2. ברירת המחדל היא חשבון דמו (פורט 4002). מעבר ללייב דורש --live
   וגם אישור אינטראקטיבי.
3. כל סיגנל, הזמנה ומילוי נכתבים ל-JSONL לפני הפעולה, לא אחריה.

שימוש
─────
    # IB Gateway פתוח, API מאופשר, פורט 4002
    python live_trader.py --symbols MYM M2K
    python live_trader.py --symbols MYM --dry-run    # בלי לשלוח הזמנות
"""

from __future__ import annotations

import argparse
import json
import signal as _signal
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path

import pandas as pd

from config import ET, RiskConfig, StrategyConfig
from risk import CircuitBreaker, size_position
from screener import MICROS, to_instrument
from strategy import Direction, MomentumPullbackStrategy

LOG_DIR = Path("./live_logs")
LOG_DIR.mkdir(exist_ok=True)

# מספר נרות היסטוריים שנטענים בהתחלה כדי לחמם את האינדיקטורים.
# EMA50 שעתי דורש 50 שעות → ~600 נרות של 5 דק'. לוקחים מרווח.
WARMUP_BARS = 900


# ══════════════════════════════════════════════════════════════
# לוגר
# ══════════════════════════════════════════════════════════════
class Logger:
    """כותב JSONL. שורה אחת לאירוע. נכתב לפני הפעולה."""

    def __init__(self, session: str):
        self.path = LOG_DIR / f"live_{session}.jsonl"
        self.f = open(self.path, "a", encoding="utf-8")

    def log(self, kind: str, **data):
        rec = {
            "ts_utc": datetime.utcnow().isoformat(),
            "ts_et": datetime.now(ET).isoformat(),
            "kind": kind,
            **data,
        }
        self.f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self.f.flush()          # ‼️ flush מיידי — קריסה לא תאבד את הרשומה
        return rec

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════
# מצב פוזיציה
# ══════════════════════════════════════════════════════════════
@dataclass
class LivePosition:
    symbol: str
    direction: Direction
    contracts: int
    entry_price: float
    initial_stop: float
    current_stop: float
    target: float
    signal_time: pd.Timestamp
    entry_time: datetime
    parent_id: int | None = None
    stop_id: int | None = None
    target_id: int | None = None
    trail_extreme: float = 0.0
    bars_held: int = 0


# ══════════════════════════════════════════════════════════════
# הסוחר
# ══════════════════════════════════════════════════════════════
class LiveTrader:

    def __init__(self, symbols: list[str], args):
        self.args = args
        self.symbols = symbols
        self.strat_cfg = StrategyConfig()
        self.risk_cfg = RiskConfig(
            account_size=args.capital, risk_per_trade_pct=args.risk
        )
        self.strategy = MomentumPullbackStrategy(self.strat_cfg)
        self.breaker = CircuitBreaker(self.risk_cfg)

        session = datetime.now(ET).strftime("%Y%m%d_%H%M%S")
        self.logger = Logger(session)
        self.session = session

        self.bars: dict[str, pd.DataFrame] = {}
        self.contracts: dict = {}
        self.positions: dict[str, LivePosition] = {}
        self.instruments = {
            s: to_instrument(next(m for m in MICROS if m.symbol == s))
            for s in symbols
        }
        self.ib = None
        self.running = True
        self.equity = args.capital

    # ══════════════════════════════════════════════════════════
    def connect(self):
        from ib_async import IB

        self.ib = IB()
        port = self.args.port
        print(f"▸ מתחבר ל-{self.args.host}:{port} (clientId={self.args.client_id})")
        self.ib.connect(self.args.host, port, clientId=self.args.client_id, timeout=20)

        acct = self.ib.managedAccounts()
        is_paper = bool(acct) and acct[0].startswith("D")
        print(f"  ✓ מחובר | חשבון {acct} | "
              f"{'דמו' if is_paper else '⚠️ חשבון אמיתי'}")

        if not is_paper and not self.args.live:
            raise RuntimeError(
                "זוהה חשבון אמיתי אך --live לא הועבר. מתנתק מטעמי בטיחות."
            )

        self.logger.log("connect", account=acct, paper=is_paper, port=port)

        # ניתוק בלתי צפוי — נרשם ומטופל
        self.ib.disconnectedEvent += self._on_disconnect

    def _on_disconnect(self):
        self.logger.log("disconnect_event", open_positions=list(self.positions))
        print("  ⚠️ החיבור נותק. הסטופים יושבים על שרתי IBKR.")

    # ══════════════════════════════════════════════════════════
    def setup_contracts(self):
        from ib_async import Future

        for s in self.symbols:
            inst = self.instruments[s]
            c = Future(symbol=inst.symbol, exchange=inst.exchange,
                       currency=inst.currency)
            details = self.ib.reqContractDetails(c)
            if not details:
                raise RuntimeError(f"לא נמצא חוזה עבור {s}")
            active = sorted(
                [d.contract for d in details],
                key=lambda x: x.lastTradeDateOrContractMonth,
            )[0]
            self.contracts[s] = active
            print(f"  ✓ {s}: {active.localSymbol} "
                  f"(תפוגה {active.lastTradeDateOrContractMonth})")
            self.logger.log("contract", symbol=s,
                            local=active.localSymbol,
                            expiry=active.lastTradeDateOrContractMonth)

    # ══════════════════════════════════════════════════════════
    def load_history(self):
        """טוען נרות היסטוריים לחימום האינדיקטורים ומפעיל עדכון חי"""
        from ib_async import util

        for s in self.symbols:
            bars = self.ib.reqHistoricalData(
                self.contracts[s],
                endDateTime="",
                durationStr="10 D",
                barSizeSetting="5 mins",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=2,
                keepUpToDate=True,      # ‼️ מפעיל עדכונים חיים
            )
            df = util.df(bars).set_index("date")
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(ET)
            df = df[["open", "high", "low", "close", "volume"]]
            self.bars[s] = df
            print(f"  ✓ {s}: {len(df)} נרות היסטוריים | אחרון {df.index[-1]:%H:%M}")

            bars.updateEvent += lambda b, has_new, sym=s: self._on_bar(sym, b, has_new)

            self.logger.log("history_loaded", symbol=s, bars=len(df),
                            last=str(df.index[-1]))

    # ══════════════════════════════════════════════════════════
    def _on_bar(self, symbol: str, bars, has_new_bar: bool):
        """נקרא בכל עדכון. פועלים רק כשנר נסגר."""
        if not has_new_bar:
            return
        try:
            self.process_bar(symbol, bars)
        except Exception as e:
            self.logger.log("error", symbol=symbol, error=str(e),
                            tb=traceback.format_exc())
            print(f"  ✗ שגיאה ב-{symbol}: {e}")

    # ══════════════════════════════════════════════════════════
    def process_bar(self, symbol: str, bars):
        """
        הלב. נקרא בסגירת כל נר.

        ‼️ הסדר כאן חייב להיות זהה לסדר ב-backtest.py:
           1. עדכון נרות
           2. ניהול פוזיציה קיימת
           3. חיפוש סיגנל חדש
        """
        from ib_async import util

        df = util.df(bars).set_index("date")
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(ET)
        df = df[["open", "high", "low", "close", "volume"]]
        self.bars[symbol] = df

        # הנר האחרון עדיין נבנה — מתעלמים ממנו
        closed = df.iloc[:-1]
        if len(closed) < 100:
            return

        prepared = self.strategy.prepare(closed)
        row = prepared.iloc[-1]
        ts = prepared.index[-1]

        self.breaker.new_day(ts.date())

        self.logger.log(
            "bar", symbol=symbol, bar_time=str(ts),
            close=float(row["close"]), vwap=float(row.get("vwap", float("nan"))),
            atr=float(row.get("atr", float("nan"))),
            adx_15m=float(row.get("adx_15m", float("nan"))),
            regime=str(row.get("regime")),
            pullback_long=bool(row.get("pullback_long", False)),
            pullback_short=bool(row.get("pullback_short", False)),
        )

        # ── סגירה כפויה בסוף היום ──
        if ts.time() >= self.strat_cfg.hard_close:
            if symbol in self.positions:
                self.close_position(symbol, "EOD Close")
            return

        # ── ניהול פוזיציה קיימת ──
        if symbol in self.positions:
            self.manage_position(symbol, row, ts)
            return

        # ── חיפוש סיגנל ──
        allowed, why = self.breaker.can_trade(
            symbol, self.strat_cfg.max_trades_per_day_per_instrument
        )
        if not allowed:
            self.logger.log("blocked", symbol=symbol, reason=why)
            return

        sig = self.strategy.generate_signal(row, ts)
        if sig is None:
            return

        self.logger.log(
            "signal", symbol=symbol, signal_time=str(ts),
            direction=sig.direction.name, price=sig.entry_price,
            stop=sig.stop_price, target=sig.target_price,
            atr=sig.atr, adx=sig.adx, reason=sig.reason,
        )
        print(f"  ⚡ {symbol} {sig.direction.name} @ {sig.entry_price:.2f} "
              f"| stop {sig.stop_price:.2f} | {ts:%H:%M}")

        self.enter(symbol, sig, ts)

    # ══════════════════════════════════════════════════════════
    def enter(self, symbol: str, sig, signal_ts):
        inst = self.instruments[symbol]

        sizing = size_position(inst, sig.entry_price, sig.stop_price,
                               self.equity, self.risk_cfg)
        if not sizing.ok:
            self.logger.log("sizing_rejected", symbol=symbol,
                            reason=sizing.rejected_reason)
            print(f"     ↳ נדחה: {sizing.rejected_reason}")
            return

        if self.args.dry_run:
            self.logger.log("dry_run_entry", symbol=symbol,
                            contracts=sizing.contracts)
            print(f"     ↳ [dry-run] היה נכנס עם {sizing.contracts} חוזים")
            return

        from ib_async import MarketOrder

        action = "BUY" if sig.direction is Direction.LONG else "SELL"
        stop_px = inst.round_to_tick(sig.stop_price)
        tgt_px = inst.round_to_tick(sig.target_price)

        # ‼️ bracket: ההזמנה הראשית + סטופ + יעד ב-OCA.
        #    הסטופ יושב על שרתי IBKR ושורד קריסה של התהליך.
        bracket = self.ib.bracketOrder(
            action, sizing.contracts,
            limitPrice=None, takeProfitPrice=tgt_px, stopLossPrice=stop_px,
        )
        bracket[0] = MarketOrder(action, sizing.contracts)
        bracket[0].transmit = False
        bracket[1].transmit = False
        bracket[2].transmit = True      # האחרון משדר את כולם

        trades = [self.ib.placeOrder(self.contracts[symbol], o) for o in bracket]
        self.ib.sleep(1.5)

        fill = trades[0].orderStatus.avgFillPrice or sig.entry_price

        pos = LivePosition(
            symbol=symbol, direction=sig.direction, contracts=sizing.contracts,
            entry_price=fill, initial_stop=stop_px, current_stop=stop_px,
            target=tgt_px, signal_time=signal_ts, entry_time=datetime.now(ET),
            parent_id=bracket[0].orderId, target_id=bracket[1].orderId,
            stop_id=bracket[2].orderId, trail_extreme=fill,
        )
        self.positions[symbol] = pos

        slip = abs(fill - sig.entry_price)
        self.logger.log(
            "entry", symbol=symbol, signal_time=str(signal_ts),
            signal_price=sig.entry_price, fill_price=fill,
            slippage=slip, slippage_ticks=slip / inst.tick_size,
            contracts=sizing.contracts, stop=stop_px, target=tgt_px,
            risk_dollars=sizing.risk_dollars,
        )
        print(f"     ↳ מולא @ {fill:.2f} | החלקה {slip/inst.tick_size:.1f} טיקים")

    # ══════════════════════════════════════════════════════════
    def manage_position(self, symbol: str, row, ts):
        """
        ‼️ סדר הבדיקות זהה ל-backtest.py.
        הסטופ והיעד מנוהלים על ידי IBKR (OCA).
        כאן מטפלים רק ב-trailing וביציאת משטר.
        """
        pos = self.positions[symbol]
        pos.bars_held += 1
        inst = self.instruments[symbol]
        is_long = pos.direction is Direction.LONG
        c = self.strat_cfg

        # האם ה-OCA כבר סגר את הפוזיציה?
        live_pos = [p for p in self.ib.positions()
                    if p.contract.conId == self.contracts[symbol].conId]
        if not live_pos or live_pos[0].position == 0:
            self.logger.log("position_closed_by_bracket", symbol=symbol)
            print(f"  ● {symbol}: נסגר על ידי סטופ/יעד")
            del self.positions[symbol]
            return

        price = float(row["close"])
        if is_long:
            pos.trail_extreme = max(pos.trail_extreme, float(row["high"]))
        else:
            pos.trail_extreme = min(pos.trail_extreme, float(row["low"]))

        # ── יציאת משטר שוק ──
        should, why = self.strategy.should_exit_regime(row, pos.direction)
        if should:
            self.close_position(symbol, f"Regime: {why}")
            return

        # ── Trailing ──
        risk_pts = abs(pos.entry_price - pos.initial_stop)
        moved = (price - pos.entry_price) if is_long else (pos.entry_price - price)

        if moved >= c.trail_after_r * risk_pts and pd.notna(row.get("atr")):
            atr_v = float(row["atr"])
            if is_long:
                new_stop = pos.trail_extreme - c.trail_atr_mult * atr_v
                better = new_stop > pos.current_stop
            else:
                new_stop = pos.trail_extreme + c.trail_atr_mult * atr_v
                better = new_stop < pos.current_stop

            if better:
                new_stop = inst.round_to_tick(new_stop)
                self.modify_stop(symbol, new_stop)

    # ══════════════════════════════════════════════════════════
    def modify_stop(self, symbol: str, new_stop: float):
        pos = self.positions[symbol]
        old = pos.current_stop

        if self.args.dry_run:
            self.logger.log("dry_run_trail", symbol=symbol,
                            old=old, new=new_stop)
            return

        for t in self.ib.openTrades():
            if t.order.orderId == pos.stop_id:
                t.order.auxPrice = new_stop
                self.ib.placeOrder(t.contract, t.order)
                pos.current_stop = new_stop
                self.logger.log("trail_stop", symbol=symbol,
                                old=old, new=new_stop)
                print(f"  ↗ {symbol}: סטופ הועבר {old:.2f} → {new_stop:.2f}")
                return

        self.logger.log("trail_failed", symbol=symbol,
                        reason="הזמנת הסטופ לא נמצאה")

    # ══════════════════════════════════════════════════════════
    def close_position(self, symbol: str, reason: str):
        pos = self.positions.get(symbol)
        if pos is None:
            return

        if self.args.dry_run:
            self.logger.log("dry_run_exit", symbol=symbol, reason=reason)
            del self.positions[symbol]
            return

        from ib_async import MarketOrder

        # ביטול ה-OCA לפני סגירה ידנית
        for t in self.ib.openTrades():
            if t.order.orderId in (pos.stop_id, pos.target_id):
                self.ib.cancelOrder(t.order)
        self.ib.sleep(0.7)

        action = "SELL" if pos.direction is Direction.LONG else "BUY"
        trade = self.ib.placeOrder(
            self.contracts[symbol], MarketOrder(action, pos.contracts)
        )
        self.ib.sleep(1.5)

        fill = trade.orderStatus.avgFillPrice or 0.0
        inst = self.instruments[symbol]
        pts = ((fill - pos.entry_price) if pos.direction is Direction.LONG
               else (pos.entry_price - fill))
        gross = inst.points_to_usd(pts, pos.contracts)
        comm = inst.commission_rt * pos.contracts
        net = gross - comm

        self.equity += net
        self.breaker.record_trade(symbol, net)

        self.logger.log(
            "exit", symbol=symbol, reason=reason, fill_price=fill,
            entry_price=pos.entry_price, contracts=pos.contracts,
            gross_pnl=gross, commission=comm, net_pnl=net,
            bars_held=pos.bars_held, signal_time=str(pos.signal_time),
        )
        print(f"  ● {symbol} יצא @ {fill:.2f} | {reason} | ${net:+,.2f}")
        del self.positions[symbol]

    # ══════════════════════════════════════════════════════════
    def flatten_all(self):
        for s in list(self.positions):
            self.close_position(s, "Shutdown")

    def run(self):
        _signal.signal(_signal.SIGINT, lambda *a: setattr(self, "running", False))

        print("╔" + "═" * 58 + "╗")
        print("║" + f"  Live Trader | session {self.session}".ljust(58) + "║")
        print("╚" + "═" * 58 + "╝")

        try:
            self.connect()
            self.setup_contracts()
            self.load_history()

            print(f"\n▸ פעיל. מכשירים: {', '.join(self.symbols)}")
            print(f"  סגירה כפויה: {self.strat_cfg.hard_close:%H:%M} ET")
            print(f"  לוג: {self.logger.path}")
            if self.args.dry_run:
                print("  ⚠️ DRY RUN — לא נשלחות הזמנות")
            print("  Ctrl+C ליציאה מסודרת\n")

            while self.running:
                self.ib.sleep(1)
                now = datetime.now(ET).time()
                if now >= time(16, 5):
                    print("\n▸ סוף סשן")
                    break

        except KeyboardInterrupt:
            pass
        except Exception as e:
            self.logger.log("fatal", error=str(e), tb=traceback.format_exc())
            print(f"\n✗ שגיאה קטלנית: {e}")
            traceback.print_exc()
        finally:
            print("\n▸ סוגר פוזיציות...")
            try:
                self.flatten_all()
            except Exception as e:
                print(f"  ⚠️ כשל בסגירה: {e}")
                print("  ‼️ בדוק ידנית ב-TWS/Gateway")
            self.logger.log("session_end", final_equity=self.equity)
            self.logger.close()
            if self.ib and self.ib.isConnected():
                self.ib.disconnect()
            print(f"  לוג נשמר: {self.logger.path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["MYM", "M2K"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4002,
                   help="4002=Gateway דמו | 4001=Gateway לייב | 7497=TWS דמו")
    p.add_argument("--client-id", type=int, default=21)
    p.add_argument("--capital", type=float, default=5000.0)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--dry-run", action="store_true",
                   help="מחשב הכל אך לא שולח הזמנות")
    p.add_argument("--live", action="store_true",
                   help="נדרש כדי לאפשר חשבון אמיתי")
    args = p.parse_args()

    if args.live:
        print("⚠️  מצב לייב — כסף אמיתי.")
        if input("   הקלד LIVE לאישור: ").strip() != "LIVE":
            print("   בוטל.")
            sys.exit(0)

    LiveTrader(args.symbols, args).run()


if __name__ == "__main__":
    main()
