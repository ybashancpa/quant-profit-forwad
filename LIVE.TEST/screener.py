"""
screener.py — סריקת כל 15 חוזי המיקרו

שני מסננים:
  A. מסנן תקציב  — האם סטופ סביר נכנס ב-1% מ-$5,000?
  B. מסנן ביצועים — מה קורה כשמריצים את האסטרטגיה בפועל

מוריד נתוני חוזים ישירות (NQ=F, ES=F...) — לא פרוקסי ETF.
כך רמות המחיר וה-ATR נכונים מלכתחילה, בלי סקיילינג.

שימוש:
    python screener.py                  # סריקה מלאה
    python screener.py --budget-only    # רק מסנן התקציב (מהיר)
"""

from __future__ import annotations

import argparse
import time as _time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import indicators as ind
from backtest import Backtester
from config import ET, Instrument, RiskConfig, StrategyConfig
from strategy import MomentumPullbackStrategy

# ══════════════════════════════════════════════════════════════
# 15 חוזי המיקרו + סימבול יאהו לנתונים היסטוריים
# ══════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class MicroSpec:
    symbol: str
    name: str
    yahoo: str          # סימבול להורדת נתונים
    multiplier: float
    tick_size: float
    exchange: str
    commission_rt: float = 1.60
    category: str = "index"


MICROS = [
    # ── מדדים ──
    MicroSpec("MES", "Micro S&P 500",      "ES=F",    5.0,    0.25,  "CME"),
    MicroSpec("MNQ", "Micro Nasdaq-100",   "NQ=F",    2.0,    0.25,  "CME"),
    MicroSpec("MYM", "Micro Dow Jones",    "YM=F",    0.5,    1.00,  "CBOT"),
    MicroSpec("M2K", "Micro Russell 2000", "RTY=F",   5.0,    0.10,  "CME"),

    # ── סחורות ──
    MicroSpec("MCL", "Micro WTI Crude",    "CL=F",  100.0,    0.01,  "NYMEX", category="commodity"),
    MicroSpec("MGC", "Micro Gold",         "GC=F",   10.0,    0.10,  "COMEX", category="commodity"),
    MicroSpec("SIL", "Micro Silver",       "SI=F", 1000.0,    0.005, "COMEX", category="commodity"),

    # ── קריפטו ──
    MicroSpec("MBT", "Micro Bitcoin",      "BTC-USD", 0.1,    5.00,  "CME", 2.50, "crypto"),
    MicroSpec("MET", "Micro Ether",        "ETH-USD", 0.1,    0.50,  "CME", 2.50, "crypto"),
    MicroSpec("MSL", "Micro Solana",       "SOL-USD",25.0,    0.01,  "CME", 2.50, "crypto"),
    MicroSpec("MXR", "Micro XRP",          "XRP-USD",2500.0,  0.0001,"CME", 2.50, "crypto"),

    # ── מט"ח ──
    MicroSpec("M6E", "Micro EUR/USD",      "6E=F", 12500.0,   0.0001,"CME", 1.30, "fx"),
    MicroSpec("MJY", "Micro USD/JPY",      "6J=F",  1250000.0,0.0000005,"CME",1.30,"fx"),
    MicroSpec("M6B", "Micro GBP/USD",      "6B=F",  6250.0,   0.0001,"CME", 1.30, "fx"),
    MicroSpec("M6A", "Micro AUD/USD",      "6A=F", 10000.0,   0.0001,"CME", 1.30, "fx"),
]


def to_instrument(m: MicroSpec) -> Instrument:
    return Instrument(
        symbol=m.symbol, name=m.name, multiplier=m.multiplier,
        tick_size=m.tick_size, exchange=m.exchange,
        commission_rt=m.commission_rt,
    )


# ══════════════════════════════════════════════════════════════
def download(m: MicroSpec, interval: str, days: int) -> pd.DataFrame | None:
    import yfinance as yf
    from datetime import datetime, timedelta

    end = datetime.now()
    start = end - timedelta(days=min(days, 59))

    try:
        df = yf.download(m.yahoo, start=start, end=end, interval=interval,
                         progress=False, auto_adjust=False)
    except Exception as e:
        print(f"    ✗ {type(e).__name__}")
        return None

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]

    need = ["open", "high", "low", "close", "volume"]
    if any(c not in df.columns for c in need):
        return None
    df = df[need].dropna(subset=["close"])

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    df.index.name = "datetime"
    return df.sort_index()


# ══════════════════════════════════════════════════════════════
# מסנן A — תקציב
# ══════════════════════════════════════════════════════════════
def rth_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    מסנן לשעות המסחר הראשיות בלבד (9:30-16:00 ET).

    ‼️ קריטי ‼️ NQ=F נסחר ~23 שעות. ATV חציוני על פני כל הסשן
    מדלל את התנודתיות בשעות שבהן אתה באמת סוחר — פי ~1.7 ב-NQ.
    בלי הפילטר הזה הסיכון לחוזה מוצג נמוך מדי, וזו טעות
    שמתגלה רק אחרי שהכסף בשוק.
    """
    from datetime import time as _t
    return df.between_time(_t(9, 30), _t(16, 0))


def budget_screen(m: MicroSpec, df: pd.DataFrame, budget: float,
                  stop_mult: float, max_contracts: int = 4,
                  min_stop_ticks: int = 8,
                  min_deploy_pct: float = 0.30) -> dict:
    """
    מחשב את הסיכון לחוזה בודד בסטופ נתון,
    ואת מכפיל הסטופ המקסימלי שעוד נכנס בתקציב.
    """
    # מסננים נר ראשון של כל יום — גאפ לילה מנפח ATR מלאכותית
    first = df.groupby(df.index.normalize()).head(1).index
    d = df.drop(first)

    a = ind.atr(d, 14).dropna()
    if a.empty:
        return {}

    price = float(d["close"].iloc[-1])
    atr_pts = float(a.median())
    atr_pct = atr_pts / price * 100

    risk_per_contract = atr_pts * stop_mult * m.multiplier
    max_stop_mult = budget / (atr_pts * m.multiplier) if atr_pts > 0 else 0

    # כמה חוזים צריך כדי לפרוס את התקציב, וכמה באמת נפרוס
    contracts_needed = budget / risk_per_contract if risk_per_contract > 0 else 1e9
    contracts_used = min(int(contracts_needed), max_contracts)
    deployed = contracts_used * risk_per_contract
    deploy_pct = deployed / budget

    # כמה טיקים הסטופ? פחות מ-8 טיקים = בתוך הרעש
    stop_ticks = (atr_pts * stop_mult) / m.tick_size

    # עמלה כאחוז מהסיכון
    comm_drag = m.commission_rt / budget

    return {
        "price": price,
        "atr_pts": atr_pts,
        "atr_pct": atr_pct,
        "risk_1c": risk_per_contract,
        "max_stop_mult": max_stop_mult,
        "stop_ticks": stop_ticks,
        "comm_drag_r": comm_drag,
        "contracts_needed": contracts_needed,
        "contracts_used": contracts_used,
        "deployed": deployed,
        "deploy_pct": deploy_pct,
        "too_big": risk_per_contract > budget,
        "too_small": deploy_pct < min_deploy_pct,
        "too_noisy": stop_ticks < min_stop_ticks,
        "fits": (risk_per_contract <= budget
                 and deploy_pct >= min_deploy_pct
                 and stop_ticks >= min_stop_ticks),
        "bars": len(df),
        "days": df.index.normalize().nunique(),
    }


# ══════════════════════════════════════════════════════════════
# מסנן B — ביצועים בפועל
# ══════════════════════════════════════════════════════════════
def performance_screen(m: MicroSpec, df: pd.DataFrame, risk_cfg: RiskConfig,
                       stop_mults=(0.75, 1.0, 1.25, 1.5)) -> list[dict]:
    inst = to_instrument(m)
    out = []
    for sm in stop_mults:
        cfg = StrategyConfig(stop_atr_mult=sm)
        bt = Backtester(inst, MomentumPullbackStrategy(cfg), risk_cfg,
                        risk_cfg.account_size)
        try:
            bt.run(df)
        except Exception:
            continue
        d = bt.results()
        if d.empty:
            out.append({"stop_mult": sm, "n": 0})
            continue

        wins = (d.net_pnl > 0).sum()
        n = len(d)
        gw = d.loc[d.net_pnl > 0, "net_pnl"].sum()
        gl = abs(d.loc[d.net_pnl <= 0, "net_pnl"].sum())

        # רווח צפוי לעסקה ב-R — המדד שבאמת קובע
        expectancy = d.r_multiple.mean()

        # שגיאת תקן של אחוז ההצלחה — כמה המספר בכלל אמין
        wr = wins / n
        se = np.sqrt(wr * (1 - wr) / n) if n > 1 else np.nan

        out.append({
            "stop_mult": sm, "n": n,
            "win_rate": wr * 100,
            "win_rate_se": se * 100 if not np.isnan(se) else np.nan,
            "pf": gw / gl if gl > 0 else np.inf,
            "net": d.net_pnl.sum(),
            "expectancy_r": expectancy,
        })
    return out


# ══════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capital", type=float, default=5000.0)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--interval", default="5m")
    p.add_argument("--days", type=int, default=59)
    p.add_argument("--stop-mult", type=float, default=1.5)
    p.add_argument("--budget-only", action="store_true")
    p.add_argument("--all-hours", action="store_true",
                   help="לא לסנן ל-RTH (לא מומלץ למסחר תוך-יומי)")
    args = p.parse_args()

    budget = args.capital * args.risk
    risk_cfg = RiskConfig(account_size=args.capital, risk_per_trade_pct=args.risk)

    print("╔" + "═" * 74 + "╗")
    print("║" + f"  סריקת 15 חוזי מיקרו | הון ${args.capital:,.0f} | סיכון {args.risk*100:.1f}% = ${budget:.0f}".ljust(74) + "║")
    print("╚" + "═" * 74 + "╝")

    data, screens = {}, {}

    print(f"\n▸ מוריד נתונים ({args.interval}, {args.days} ימים)...\n")
    for m in MICROS:
        print(f"  {m.symbol:5} ({m.yahoo:9})", end=" ")
        df = download(m, args.interval, args.days)
        if df is None or len(df) < 200:
            print("✗ אין נתונים מספיקים")
            continue
        if m.category != "crypto" and not args.all_hours:
            df = rth_only(df)
            if len(df) < 200:
                print("✗ מעט מדי נרות ב-RTH")
                continue
        data[m.symbol] = df
        s = budget_screen(m, df, budget, args.stop_mult)
        screens[m.symbol] = s
        print(f"✓ {len(df):>5,} נרות | ATR {s['atr_pct']:.3f}% | סיכון ${s['risk_1c']:>7,.0f}")
        _time.sleep(0.4)   # נימוס כלפי יאהו

    # ── טבלה A ──
    print("\n" + "═" * 76)
    print(f"  מסנן תקציב — סטופ {args.stop_mult}×ATR, תקציב ${budget:.0f} לעסקה")
    print("═" * 76)
    print(f"  {'':5}{'מחיר':>12}{'ATR%':>8}{'סיכון/חוזה':>12}{'חוזים':>8}{'נפרס':>9}{'טיקים':>7}{'':>10}")
    print("  " + "-" * 72)

    rows = []
    for m in MICROS:
        s = screens.get(m.symbol)
        if not s:
            continue
        if s["too_big"]:
            mark = f"✗ גדול {s['risk_1c']/budget:.1f}x"
        elif s["too_noisy"]:
            mark = "✗ רעש"
        elif s["too_small"]:
            mark = "✗ קטן מדי"
        else:
            mark = "✓"
        pr = f"{s['price']:,.4f}" if s["price"] < 10 else f"{s['price']:,.2f}"
        cn = f"{s['contracts_needed']:.0f}" if s['contracts_needed'] < 1000 else ">1k"
        print(f"  {m.symbol:5}{pr:>12}{s['atr_pct']:>7.3f}%"
              f"{'$'+format(s['risk_1c'],',.2f' if s['risk_1c']<10 else ',.0f'):>12}"
              f"{cn:>8}{s['deploy_pct']*100:>8.0f}%{s['stop_ticks']:>7.0f}{mark:>10}")
        rows.append((m, s))

    fits = [(m, s) for m, s in rows if s["fits"]]
    print("\n  עוברים את מסנן התקציב: " + (", ".join(m.symbol for m, _ in fits) or "אף אחד"))

    if args.budget_only:
        return

    # ── טבלה B ──
    print("\n" + "═" * 76)
    print("  מסנן ביצועים — האסטרטגיה בפועל")
    print("═" * 76)
    print(f"  {'':5}{'stop':>6}{'עסקאות':>8}{'הצלחה':>10}{'±שג\"ת':>8}{'PF':>7}{'תוחלת R':>10}{'נטו':>10}")
    print("  " + "-" * 72)

    for m, s in rows:
        res = performance_screen(m, data[m.symbol], risk_cfg)
        for r in res:
            if r["n"] == 0:
                print(f"  {m.symbol:5}{r['stop_mult']:>6}{'0':>8}{'—':>10}")
                continue
            print(f"  {m.symbol:5}{r['stop_mult']:>6}{r['n']:>8}"
                  f"{r['win_rate']:>9.1f}%{r['win_rate_se']:>7.1f}%"
                  f"{r['pf']:>7.2f}{r['expectancy_r']:>10.3f}"
                  f"{'$'+format(r['net'],',.0f'):>10}")
        print()

    print("═" * 76)
    print("  ⚠ 41 ימי מסחר = מדגם קטן. שגיאת התקן בעמודה '±שג\"ת' מראה")
    print("    כמה אחוז ההצלחה עצמו לא יציב. עם 10 עסקאות, ±15% זה נורמלי —")
    print("    כלומר '60% הצלחה' ו-'45% הצלחה' הם אותו מספר סטטיסטית.")
    print("═" * 76)


if __name__ == "__main__":
    main()
