"""
FORWARD PAPER-TRADING ENGINE: SmartPassive 55/35/10 + MA200
============================================================
Locked rules (from Tests 7-13; NO parameter changes allowed):
  Risk-On  (SPY >  MA200): 55% SPY, 35% IEF, 10% GLD
  Risk-Off (SPY <= MA200): 55% SHY, 35% IEF, 10% GLD
  Trade only on the LAST trading day of each month (plus the one-time
  initial allocation). 3% drift tolerance band. 10bps/side paper costs.
  Fractional shares, long-only, no leverage. $10,000 starting capital.

State & audit (append-only, idempotent):
  forward/state.json, nav_history.csv, trades.csv, signals.csv, runs.csv
  forward/report.html, summary.json

Usage:
  python forward_test.py                 # live run (latest close)
  python forward_test.py --date 2026-07-31   # simulate as-of date
  python forward_test.py --force         # rerun same date (no dup trades)
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ============================================================
# LOCKED CONFIGURATION (do not tune during the experiment)
# ============================================================
CONFIG = {
    "experiment_id": "smartpassive-forward-v1",
    "config_hash": "",  # filled below
    "start_capital": 10_000.0,
    "tickers": ["SPY", "IEF", "GLD", "SHY"],
    "risk_on": {"SPY": 0.55, "IEF": 0.35, "GLD": 0.10},
    "risk_off": {"SHY": 0.55, "IEF": 0.35, "GLD": 0.10},
    "ma_lookback": 200,
    "cost_per_side": 0.001,
    "tolerance": 0.03,
    "history_days": 750,
}
CONFIG["config_hash"] = hashlib.sha1(
    json.dumps({k: v for k, v in CONFIG.items() if k != "config_hash"},
               sort_keys=True).encode()).hexdigest()[:12]

FORWARD_DIR = os.environ.get("FORWARD_DIR", "forward")
STATE_FILE = os.path.join(FORWARD_DIR, "state.json")
NAV_FILE = os.path.join(FORWARD_DIR, "nav_history.csv")
TRADES_FILE = os.path.join(FORWARD_DIR, "trades.csv")
SIGNALS_FILE = os.path.join(FORWARD_DIR, "signals.csv")
RUNS_FILE = os.path.join(FORWARD_DIR, "runs.csv")
REPORT_FILE = os.path.join(FORWARD_DIR, "report.html")
SUMMARY_FILE = os.path.join(FORWARD_DIR, "summary.json")


# ============================================================
# DATA
# ============================================================
def load_prices(end_date=None):
    """Download adjusted closes for the 4 tickers (~2y), truncate at end_date."""
    import yfinance as yf
    start = (pd.Timestamp.today() - pd.Timedelta(days=CONFIG["history_days"])
             ).strftime("%Y-%m-%d")
    data = yf.download(CONFIG["tickers"], start=start, auto_adjust=True,
                       progress=False, threads=False)
    if isinstance(data.columns, pd.MultiIndex):
        px = data["Close"][CONFIG["tickers"]]
    else:
        px = data[["Close"]]
    px = px.dropna()
    if end_date is not None:
        px = px.loc[:pd.Timestamp(end_date)]
    if len(px) < CONFIG["ma_lookback"] + 5:
        raise RuntimeError(f"Insufficient price history: {len(px)} rows")
    return px


# ============================================================
# SIGNAL + TARGETS (pure functions, unit-testable)
# ============================================================
def compute_signal(prices, asof):
    """MA200 regime signal at `asof` (point-in-time)."""
    spy = prices["SPY"].loc[:asof]
    if len(spy) < CONFIG["ma_lookback"]:
        raise RuntimeError("Not enough SPY history for MA200")
    ma = spy.rolling(CONFIG["ma_lookback"]).mean().iloc[-1]
    price = spy.iloc[-1]
    regime = "RISK_ON" if price > ma else "RISK_OFF"
    return regime, float(price), float(ma)


def target_weights(regime):
    return dict(CONFIG["risk_on"] if regime == "RISK_ON" else CONFIG["risk_off"])


def is_last_trading_day_of_month(index, date):
    month_days = index[(index.year == date.year) & (index.month == date.month)]
    return len(month_days) > 0 and date == month_days.max()


def plan_trades(state, prices_row, target_w):
    """
    Return list of trade dicts needed to reach target weights.
    Applies the tolerance band on current-vs-target weight drift.
    """
    holdings = state["shares"]
    cash = state["cash"]
    nav = cash + sum(holdings.get(t, 0.0) * prices_row[t] for t in prices_row.index)
    current_w = {t: holdings.get(t, 0.0) * prices_row[t] / nav
                 for t in prices_row.index}
    drift = max(abs(current_w.get(t, 0.0) - target_w.get(t, 0.0))
                for t in set(list(current_w) + list(target_w)))
    if drift <= CONFIG["tolerance"]:
        return [], drift, nav
    trades = []
    for t in prices_row.index:
        target_value = target_w.get(t, 0.0) * nav
        current_value = holdings.get(t, 0.0) * prices_row[t]
        delta_value = target_value - current_value
        if abs(delta_value) >= 1.0:  # ignore sub-$1 adjustments
            trades.append({
                "ticker": t,
                "price": float(prices_row[t]),
                "value": float(delta_value),
                "shares": float(delta_value / prices_row[t]),
                "cost": abs(delta_value) * CONFIG["cost_per_side"],
            })
    return trades, drift, nav


def apply_trades(state, trades, date, reason):
    """Mutate state: execute paper trades, record rows."""
    rows = []
    for tr in trades:
        t = tr["ticker"]
        state["shares"][t] = state["shares"].get(t, 0.0) + tr["shares"]
        state["cash"] -= tr["value"] + tr["cost"]
        state["total_costs"] = state.get("total_costs", 0.0) + tr["cost"]
        rows.append({
            "date": str(date.date()), "ticker": t, "reason": reason,
            "price": round(tr["price"], 4), "shares": round(tr["shares"], 6),
            "value": round(tr["value"], 2), "cost": round(tr["cost"], 4),
        })
    # drop dust
    state["shares"] = {k: v for k, v in state["shares"].items() if abs(v) > 1e-9}
    return rows


# ============================================================
# STATE + APPEND-ONLY LOGS
# ============================================================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_state(state):
    os.makedirs(FORWARD_DIR, exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def append_csv(path, rows, columns):
    if not rows:
        return
    os.makedirs(FORWARD_DIR, exist_ok=True)
    df = pd.DataFrame(rows, columns=columns)
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=write_header, index=False)


def read_nav_history():
    if os.path.exists(NAV_FILE):
        return pd.read_csv(NAV_FILE, parse_dates=["date"])
    return pd.DataFrame(columns=["date", "nav", "spy_bench_nav", "regime"])


# ============================================================
# REPORT
# ============================================================
def write_report(state, nav_df, last_signal):
    start_nav = CONFIG["start_capital"]
    nav = state["nav"]
    total_ret = (nav / start_nav - 1) * 100
    days = (pd.Timestamp(state["last_run_date"]) -
            pd.Timestamp(state["start_date"])).days
    bench_ret = (state["spy_bench_nav"] / start_nav - 1) * 100
    cummax = nav_df["nav"].cummax()
    dd = ((nav_df["nav"] - cummax) / cummax).min() * 100 if len(nav_df) else 0.0

    holdings = "".join(
        f"<tr><td>{t}</td><td>{v:.4f}</td></tr>"
        for t, v in sorted(state["shares"].items()))

    svg = ""
    if len(nav_df) > 1:
        w, h = 760, 220
        x = np.linspace(0, w, len(nav_df))
        lo, hi = nav_df["nav"].min(), nav_df["nav"].max()
        rng = (hi - lo) or 1.0
        y = h - (nav_df["nav"].values - lo) / rng * (h - 20) - 10
        yb = h - (nav_df["spy_bench_nav"].values - lo) / rng * (h - 20) - 10
        pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
        ptsb = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, yb))
        svg = (f'<svg width="{w}" height="{h}" style="border:1px solid #ccc">'
               f'<polyline points="{ptsb}" fill="none" stroke="#999" stroke-width="1.5"/>'
               f'<polyline points="{pts}" fill="none" stroke="#0a7" stroke-width="2"/>'
               f'</svg><p style="color:#666">green=portfolio, gray=SPY benchmark</p>')

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Forward Test Report - {CONFIG['experiment_id']}</title>
<style>body{{font-family:Arial;margin:24px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 10px}}</style></head><body>
<h1>SmartPassive Forward Paper Test</h1>
<p>Experiment: <b>{CONFIG['experiment_id']}</b> (config hash {CONFIG['config_hash']})</p>
<table>
<tr><th>Start date</th><td>{state['start_date']}</td></tr>
<tr><th>Last run</th><td>{state['last_run_date']}</td></tr>
<tr><th>Days elapsed</th><td>{days}</td></tr>
<tr><th>NAV</th><td>${nav:,.2f}</td></tr>
<tr><th>Total return</th><td>{total_ret:+.2f}%</td></tr>
<tr><th>SPY benchmark return</th><td>{bench_ret:+.2f}%</td></tr>
<tr><th>Max drawdown</th><td>{dd:.2f}%</td></tr>
<tr><th>Current regime</th><td>{last_signal['regime']} (SPY {last_signal['spy_price']:.2f} vs MA200 {last_signal['ma200']:.2f})</td></tr>
<tr><th>Cash</th><td>${state['cash']:,.2f}</td></tr>
<tr><th>Total costs paid</th><td>${state.get('total_costs',0):,.2f}</td></tr>
</table>
<h2>Holdings</h2><table><tr><th>Ticker</th><th>Shares</th></tr>{holdings}</table>
<h2>NAV history</h2>{svg}
</body></html>"""
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(html)


# ============================================================
# MAIN RUN
# ============================================================
def run(asof=None, force=False):
    prices = load_prices(end_date=asof)
    latest = prices.index[-1]

    state = load_state()
    if state is not None and not force:
        if pd.Timestamp(state["last_run_date"]) >= latest:
            print(f"[NO_ACTION] already processed {state['last_run_date']} "
                  f"(latest close {latest.date()}). Use --force to re-mark.")
            append_csv(RUNS_FILE, [{
                "run_ts": datetime.now().isoformat(timespec="seconds"),
                "asof": str(latest.date()), "status": "NO_ACTION_ALREADY_RUN",
                "trades": 0, "nav": state["nav"]}],
                ["run_ts", "asof", "status", "trades", "nav"])
            return state

    regime, spy_price, ma = compute_signal(prices, latest)
    signal_row = {"date": str(latest.date()), "regime": regime,
                  "spy_price": round(spy_price, 4), "ma200": round(ma, 4)}
    append_csv(SIGNALS_FILE, [signal_row],
               ["date", "regime", "spy_price", "ma200"])

    trade_rows = []
    status = "HOLD"

    if state is None:
        # -------- bootstrap: initial allocation --------
        state = {
            "experiment_id": CONFIG["experiment_id"],
            "config_hash": CONFIG["config_hash"],
            "start_date": str(latest.date()),
            "start_capital": CONFIG["start_capital"],
            "cash": CONFIG["start_capital"],
            "shares": {},
            "total_costs": 0.0,
            "spy_start_price": spy_price,
            "last_run_date": str(latest.date()),
            "last_trade_month": None,
            "nav": CONFIG["start_capital"],
            "spy_bench_nav": CONFIG["start_capital"],
        }
        tw = target_weights(regime)
        row0 = prices.loc[latest]
        trades, drift, nav = plan_trades(state, row0, tw)
        trade_rows = apply_trades(state, trades, latest, "INITIAL_ALLOCATION")
        state["last_trade_month"] = f"{latest.year}-{latest.month:02d}"
        status = "INITIAL_ALLOCATION"
    else:
        month_key = f"{latest.year}-{latest.month:02d}"
        if (is_last_trading_day_of_month(prices.index, latest)
                and state.get("last_trade_month") != month_key):
            tw = target_weights(regime)
            row0 = prices.loc[latest]
            trades, drift, nav = plan_trades(state, row0, tw)
            if trades:
                trade_rows = apply_trades(state, trades, latest, "MONTH_END_REBALANCE")
                state["last_trade_month"] = month_key
                status = "REBALANCED"
            else:
                status = f"MONTH_END_NO_TRADE(drift={drift*100:.2f}%)"
        else:
            status = "HOLD(not month-end)"

    # -------- mark-to-market --------
    row0 = prices.loc[latest]
    nav = state["cash"] + sum(state["shares"].get(t, 0.0) * row0[t]
                              for t in row0.index)
    state["nav"] = float(nav)
    state["spy_bench_nav"] = float(
        CONFIG["start_capital"] * spy_price / state["spy_start_price"])
    state["last_run_date"] = str(latest.date())

    nav_df = read_nav_history()
    if len(nav_df) == 0 or nav_df["date"].max() < latest:
        append_csv(NAV_FILE, [{
            "date": str(latest.date()), "nav": round(nav, 2),
            "spy_bench_nav": round(state["spy_bench_nav"], 2),
            "regime": regime}],
            ["date", "nav", "spy_bench_nav", "regime"])
        nav_df = read_nav_history()

    append_csv(TRADES_FILE, trade_rows,
               ["date", "ticker", "reason", "price", "shares", "value", "cost"])
    append_csv(RUNS_FILE, [{
        "run_ts": datetime.now().isoformat(timespec="seconds"),
        "asof": str(latest.date()), "status": status,
        "trades": len(trade_rows), "nav": round(nav, 2)}],
        ["run_ts", "asof", "status", "trades", "nav"])

    save_state(state)
    write_report(state, nav_df, signal_row)

    summary = {
        "experiment_id": CONFIG["experiment_id"],
        "asof": str(latest.date()),
        "status": status,
        "regime": regime,
        "spy_price": spy_price,
        "ma200": ma,
        "nav": round(nav, 2),
        "total_return_pct": round((nav / CONFIG["start_capital"] - 1) * 100, 3),
        "spy_bench_return_pct": round(
            (state["spy_bench_nav"] / CONFIG["start_capital"] - 1) * 100, 3),
        "n_trades": len(trade_rows),
        "holdings": state["shares"],
        "cash": round(state["cash"], 2),
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[{status}] asof={latest.date()} regime={regime} "
          f"(SPY {spy_price:.2f} vs MA200 {ma:.2f})")
    print(f"  NAV=${nav:,.2f} ({summary['total_return_pct']:+.2f}%) | "
          f"SPY bench {summary['spy_bench_return_pct']:+.2f}% | "
          f"trades={len(trade_rows)}")
    for tr in trade_rows:
        print(f"    {tr['ticker']}: {'BUY' if tr['value']>0 else 'SELL'} "
              f"${abs(tr['value']):,.2f} @ {tr['price']:.2f}")
    print(f"  report: {REPORT_FILE}")
    return state


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="simulate as-of YYYY-MM-DD")
    ap.add_argument("--force", action="store_true",
                    help="re-mark even if date already processed")
    args = ap.parse_args()
    try:
        run(asof=args.date, force=args.force)
    except Exception as e:
        append_csv(RUNS_FILE, [{
            "run_ts": datetime.now().isoformat(timespec="seconds"),
            "asof": args.date or "latest", "status": f"ERROR: {e}",
            "trades": 0, "nav": ""}],
            ["run_ts", "asof", "status", "trades", "nav"])
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)