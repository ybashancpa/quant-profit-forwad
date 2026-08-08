"""
backtest.py — מנוע בקטסט מבוסס אירועים

עקרונות שמונעים בקטסט משקר:
  1. כניסה בפתיחת הנר **הבא** אחרי הסיגנל — לא בסגירת נר הסיגנל.
     בלייב אתה לא יכול לקנות במחיר סגירה של נר שרק עכשיו נסגר.
  2. סטופ נבדק לפי low/high של הנר, לא לפי close.
  3. כשגם הסטופ וגם היעד נפגעים באותו נר — מניחים שהסטופ נפגע ראשון
     (הנחה שמרנית; בלי נתוני טיק אי אפשר לדעת).
  4. עמלות והחלקה נגבות בכל צד.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import time

import numpy as np
import pandas as pd

from config import Instrument, RiskConfig, StrategyConfig
from risk import CircuitBreaker, size_position
from strategy import Direction, MomentumPullbackStrategy, Regime, Signal


@dataclass
class Trade:
    symbol: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    # audit בלבד: מועד יצירת הסיגנל (הנר שבו נוצר), לא משפיע על ביצוע/חישוב
    signal_time: pd.Timestamp | None = None
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    contracts: int = 0
    stop_price: float = 0.0
    initial_stop: float = 0.0      # לא משתנה — הבסיס לחישוב R
    target_price: float = 0.0
    exit_reason: str = ""
    gross_pnl: float = 0.0
    commission: float = 0.0
    net_pnl: float = 0.0
    r_multiple: float = 0.0
    mae: float = 0.0          # Maximum Adverse Excursion — כמה ירד נגדך
    mfe: float = 0.0          # Maximum Favorable Excursion — כמה עלה לטובתך
    adx_at_entry: float = 0.0
    bars_held: int = 0


class Backtester:

    def __init__(
        self,
        instrument: Instrument,
        strategy: MomentumPullbackStrategy,
        risk_cfg: RiskConfig,
        initial_capital: float = 5000.0,
    ):
        self.inst = instrument
        self.strat = strategy
        self.risk_cfg = risk_cfg
        self.initial_capital = initial_capital

        self.equity = initial_capital
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple] = []
        self.breaker = CircuitBreaker(risk_cfg)
        self.rejections: dict[str, int] = {}

    # ══════════════════════════════════════════════════════════
    def _slip(self, price: float, direction: Direction, is_entry: bool) -> float:
        """
        החלקה: תמיד לרעתך.
        כניסת Long -> משלם יותר. יציאת Long -> מקבל פחות.
        """
        slip = self.inst.slippage_ticks * self.inst.tick_size
        if is_entry:
            adj = slip if direction is Direction.LONG else -slip
        else:
            adj = -slip if direction is Direction.LONG else slip
        return self.inst.round_to_tick(price + adj)

    def _log_reject(self, reason: str):
        key = reason.split(":")[0][:40]
        self.rejections[key] = self.rejections.get(key, 0) + 1

    # ══════════════════════════════════════════════════════════
    def run(self, df_exec: pd.DataFrame) -> pd.DataFrame:
        c = self.strat.cfg
        df = self.strat.prepare(df_exec)

        position: Trade | None = None
        pos_dir = Direction.FLAT
        pending: Signal | None = None
        bars_since_exit = 999
        trail_extreme = np.nan

        rows = df.itertuples()
        prev_ts = None

        for i, r in enumerate(rows):
            ts: pd.Timestamp = r.Index
            self.breaker.new_day(ts.date())

            # ──────────────────────────────────────────────
            # שלב 1: ביצוע סיגנל ממתין (בפתיחת הנר הנוכחי)
            # ──────────────────────────────────────────────
            if pending is not None and position is None:
                fill = self._slip(r.open, pending.direction, is_entry=True)

                # הסטופ מחושב מחדש ממחיר המילוי האמיתי
                risk_pts = pending.risk_points
                if pending.direction is Direction.LONG:
                    stop = fill - risk_pts
                    target = fill + c.target_r_multiple * risk_pts
                else:
                    stop = fill + risk_pts
                    target = fill - c.target_r_multiple * risk_pts

                sizing = size_position(
                    self.inst, fill, stop, self.equity, self.risk_cfg
                )

                if sizing.ok:
                    position = Trade(
                        symbol=self.inst.symbol,
                        direction=pending.direction.name,
                        entry_time=ts,
                        entry_price=fill,
                        signal_time=pending.timestamp,
                        contracts=sizing.contracts,
                        stop_price=self.inst.round_to_tick(stop),
                        initial_stop=self.inst.round_to_tick(stop),
                        target_price=self.inst.round_to_tick(target),
                        adx_at_entry=pending.adx,
                    )
                    pos_dir = pending.direction
                    trail_extreme = fill
                else:
                    self._log_reject(sizing.rejected_reason)

                pending = None

            # ──────────────────────────────────────────────
            # שלב 2: ניהול פוזיציה פתוחה
            # ──────────────────────────────────────────────
            if position is not None:
                position.bars_held += 1
                is_long = pos_dir is Direction.LONG

                # MAE / MFE
                if is_long:
                    position.mae = min(position.mae, r.low - position.entry_price)
                    position.mfe = max(position.mfe, r.high - position.entry_price)
                    trail_extreme = max(trail_extreme, r.high)
                else:
                    position.mae = min(position.mae, position.entry_price - r.high)
                    position.mfe = max(position.mfe, position.entry_price - r.low)
                    trail_extreme = min(trail_extreme, r.low)

                # ‼️ R נמדד מול הסטופ המקורי, לא הנגרר
                risk_pts = abs(position.entry_price - position.initial_stop)
                exit_price = None
                exit_reason = ""

                # --- א. סטופ (נבדק ראשון — הנחה שמרנית) ---
                if is_long and r.low <= position.stop_price:
                    exit_price = position.stop_price
                    exit_reason = "Stop Loss"
                elif not is_long and r.high >= position.stop_price:
                    exit_price = position.stop_price
                    exit_reason = "Stop Loss"

                # --- ב. יעד ---
                if exit_price is None:
                    if is_long and r.high >= position.target_price:
                        exit_price = position.target_price
                        exit_reason = f"Target {c.target_r_multiple}R"
                    elif not is_long and r.low <= position.target_price:
                        exit_price = position.target_price
                        exit_reason = f"Target {c.target_r_multiple}R"

                # --- ג. סגירה כפויה בסוף היום ---
                if exit_price is None and ts.time() >= c.hard_close:
                    exit_price = self._slip(r.close, pos_dir, is_entry=False)
                    exit_reason = "EOD Close"

                # --- ד. יציאת משטר שוק ---
                if exit_price is None:
                    row_s = pd.Series({
                        "adx_15m": r.adx_15m, "vwap": r.vwap, "close": r.close
                    })
                    should, why = self.strat.should_exit_regime(row_s, pos_dir)
                    if should:
                        exit_price = self._slip(r.close, pos_dir, is_entry=False)
                        exit_reason = f"Regime: {why}"

                # --- ה. Trailing stop (Chandelier) ---
                if exit_price is None and not np.isnan(r.atr):
                    moved = (
                        (r.close - position.entry_price) if is_long
                        else (position.entry_price - r.close)
                    )
                    if moved >= c.trail_after_r * risk_pts:
                        if is_long:
                            new_stop = trail_extreme - c.trail_atr_mult * r.atr
                            if new_stop > position.stop_price:
                                position.stop_price = self.inst.round_to_tick(new_stop)
                        else:
                            new_stop = trail_extreme + c.trail_atr_mult * r.atr
                            if new_stop < position.stop_price:
                                position.stop_price = self.inst.round_to_tick(new_stop)

                # --- סגירת העסקה ---
                if exit_price is not None:
                    position.exit_time = ts
                    position.exit_price = exit_price
                    position.exit_reason = exit_reason or "Unknown"

                    pts = (
                        (exit_price - position.entry_price) if is_long
                        else (position.entry_price - exit_price)
                    )
                    position.gross_pnl = self.inst.points_to_usd(pts, position.contracts)
                    position.commission = self.inst.commission_rt * position.contracts
                    position.net_pnl = position.gross_pnl - position.commission
                    position.r_multiple = (
                        position.net_pnl / (risk_pts * self.inst.multiplier * position.contracts)
                        if risk_pts > 0 else 0.0
                    )
                    position.mae = self.inst.points_to_usd(position.mae, position.contracts)
                    position.mfe = self.inst.points_to_usd(position.mfe, position.contracts)

                    self.equity += position.net_pnl
                    self.breaker.record_trade(self.inst.symbol, position.net_pnl)
                    self.trades.append(position)

                    position = None
                    pos_dir = Direction.FLAT
                    bars_since_exit = 0
                    trail_extreme = np.nan

            # ──────────────────────────────────────────────
            # שלב 3: חיפוש סיגנל חדש
            # ──────────────────────────────────────────────
            bars_since_exit += 1

            if position is None and pending is None:
                allowed, why = self.breaker.can_trade(
                    self.inst.symbol, c.max_trades_per_day_per_instrument
                )
                if allowed and bars_since_exit >= c.min_bars_between_trades:
                    row = pd.Series(r._asdict())
                    sig = self.strat.generate_signal(row, ts)
                    if sig is not None:
                        pending = sig
                elif not allowed:
                    self._log_reject(why)

            self.equity_curve.append((ts, self.equity))

        return self.results()

    # ══════════════════════════════════════════════════════════
    def results(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([asdict(t) for t in self.trades])

    def stats(self) -> dict:
        df = self.results()
        if df.empty:
            return {"trades": 0, "note": "לא נוצרו עסקאות"}

        wins = df[df.net_pnl > 0]
        losses = df[df.net_pnl <= 0]

        eq = pd.Series([e for _, e in self.equity_curve])
        peak = eq.cummax()
        dd = (eq - peak) / peak
        max_dd = dd.min() * 100

        gross_win = wins.net_pnl.sum()
        gross_loss = abs(losses.net_pnl.sum())

        # Sharpe על בסיס עסקאות (לא יומי) — אינדיקטיבי בלבד
        r = df.r_multiple
        sharpe = (r.mean() / r.std() * np.sqrt(len(r))) if r.std() > 0 else 0

        days = df.entry_time.dt.date.nunique()

        return {
            "עסקאות": len(df),
            "ימי מסחר": days,
            "עסקאות ליום": round(len(df) / days, 2) if days else 0,
            "אחוז הצלחה": f"{len(wins) / len(df) * 100:.1f}%",
            "רווח נקי": f"${df.net_pnl.sum():,.2f}",
            "תשואה": f"{df.net_pnl.sum() / self.initial_capital * 100:.2f}%",
            "עמלות": f"${df.commission.sum():,.2f}",
            "Profit Factor": round(gross_win / gross_loss, 2) if gross_loss else float("inf"),
            "ממוצע R": round(r.mean(), 3),
            "רווח ממוצע": f"${wins.net_pnl.mean():,.2f}" if len(wins) else "—",
            "הפסד ממוצע": f"${losses.net_pnl.mean():,.2f}" if len(losses) else "—",
            "Max Drawdown": f"{max_dd:.2f}%",
            "Sharpe (עסקאות)": round(sharpe, 2),
            "הון סופי": f"${self.equity:,.2f}",
            "העסקה הגרועה": f"${df.net_pnl.min():,.2f}",
            "העסקה הטובה": f"${df.net_pnl.max():,.2f}",
        }

    def print_report(self):
        print("\n" + "═" * 62)
        print(f"  דוח בקטסט — {self.inst.symbol} ({self.inst.name})")
        print("═" * 62)

        for k, v in self.stats().items():
            print(f"  {k:.<28} {v}")

        if self.rejections:
            print("\n  סיבות לדחיית כניסה:")
            for k, v in sorted(self.rejections.items(), key=lambda x: -x[1])[:6]:
                print(f"    • {k}: {v}")

        df = self.results()
        if not df.empty:
            print("\n  פילוח לפי סיבת יציאה:")
            g = df.groupby("exit_reason").agg(
                n=("net_pnl", "size"),
                total=("net_pnl", "sum"),
                avg_r=("r_multiple", "mean"),
            ).sort_values("n", ascending=False)
            for reason, row in g.iterrows():
                print(f"    {reason:.<32} {int(row.n):>3} | ${row.total:>9,.0f} | {row.avg_r:+.2f}R")
        print("═" * 62)
