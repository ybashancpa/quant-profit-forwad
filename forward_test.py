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
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ============================================================
# LOCKED CONFIGURATION (do not tune during the experiment)
# ============================================================
CONFIG = {
    "experiment_id": "smartpassive-forward-v2",
    "config_hash": "",  # filled below
    "start_capital": 10_000.0,
    "tickers": ["SPY", "IEF", "GLD", "SHY"],
    "risk_on": {"SPY": 0.55, "IEF": 0.35, "GLD": 0.10},
    "risk_off": {"SHY": 0.55, "IEF": 0.35, "GLD": 0.10},
    "static_benchmark": {"SPY": 0.55, "IEF": 0.35, "GLD": 0.10},
    "ma_lookback": 200,
    "cost_per_side": 0.001,
    "tolerance": 0.03,
    "history_days": 750,
    "tax_rate": 0.25,
    "min_years": 3,
    "min_risk_off_cycles": 2,
    "hard_stop_date": "2034-08-04",
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
HYPOTHESIS_FILE = os.environ.get("HYPOTHESIS_FILE", "SMARTPASSIVE_hypothesis.md")


# ============================================================
# PROVENANCE STAMPING
# ============================================================
def provenance():
    """
    Git provenance stamp for every report: commit SHA, dirty flag, and the
    sha256 of the locked hypothesis document. Added after the v1/v2 mix-up so
    each artifact can be traced to the exact code+config that produced it.

    `dirty` reflects only modifications to TRACKED files (--untracked-files=no).
    Untracked build/data artifacts must not flip the flag, otherwise it becomes
    permanent noise that everyone ignores (the usual way warnings die). A true
    `dirty` therefore means the code that ran differs from the committed code.

    `hypothesis_sha256` hashes the COMMITTED blob (git show HEAD:<file>), not
    the working-tree file. A preregistration seal must be a property of the
    content, not of the checkout: reading from disk hashes line-ending-converted
    bytes, so the same commit yields different hashes on Windows (CRLF via
    core.autocrlf) vs Linux (LF). The blob is the one canonical representation,
    identical on every platform. This does NOT change the CI value: on Linux the
    worktree already equals the blob, so the stamped hash is unchanged run-to-run
    (it only fixes off-CI/Windows verification). If the file is not in HEAD
    (uncommitted), there is no canonical seal yet and the field is left empty.
    """
    prov = {"commit": "", "dirty": None, "hypothesis_sha256": ""}
    try:
        prov["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        prov["dirty"] = bool(porcelain)
    except Exception:
        prov["commit"] = ""
        prov["dirty"] = None
    try:
        blob = subprocess.run(
            ["git", "show", f"HEAD:{HYPOTHESIS_FILE}"],
            capture_output=True, timeout=10)  # bytes (no text=): hash raw blob
        if blob.returncode == 0:
            prov["hypothesis_sha256"] = hashlib.sha256(blob.stdout).hexdigest()
    except Exception:
        prov["hypothesis_sha256"] = ""
    return prov


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
    _s = prices["SPY"]
    spy = _s if asof is None else _s.loc[:pd.Timestamp(asof)]
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
# TAX LOT TRACKING (FIFO, 25% capital gains)
# ============================================================
def fifo_realize(tax_lots, shares_sold, price):
    """
    Realize gain/loss by selling shares_sold at price using FIFO.
    Returns (realized_gain, remaining_lots).
    """
    remaining = shares_sold
    realized = 0.0
    new_lots = []
    for lot in tax_lots:
        if remaining <= 0:
            new_lots.append(lot)
            continue
        sell_from_lot = min(lot["shares"], remaining)
        gain = sell_from_lot * (price - lot["cost_basis"])
        realized += gain
        remaining -= sell_from_lot
        leftover = lot["shares"] - sell_from_lot
        if leftover > 1e-9:
            new_lots.append({"shares": leftover, "cost_basis": lot["cost_basis"]})
    return realized, new_lots


def apply_tax(realized_gain, loss_carryforward, tax_rate=None):
    """
    Apply tax to realized gain, using loss carryforward.
    Returns (tax_paid, remaining_carryforward).
    """
    if tax_rate is None:
        tax_rate = CONFIG["tax_rate"]
    net_gain = realized_gain - loss_carryforward
    if net_gain <= 0:
        return 0.0, -net_gain
    return net_gain * tax_rate, 0.0


def update_tax_lots(state, trades, date):
    """Update tax lots for all trades (buys add lots, sells realize gains)."""
    if "tax_lots" not in state:
        state["tax_lots"] = {}
    if "loss_carryforward" not in state:
        state["loss_carryforward"] = 0.0
    if "total_realized_tax" not in state:
        state["total_realized_tax"] = 0.0

    for tr in trades:
        t = tr["ticker"]
        if t not in state["tax_lots"]:
            state["tax_lots"][t] = []

        if tr["shares"] > 0:  # BUY
            state["tax_lots"][t].append({
                "shares": tr["shares"],
                "cost_basis": tr["price"],
                "date": str(date.date()),
            })
        else:  # SELL
            shares_to_sell = -tr["shares"]
            realized, remaining = fifo_realize(state["tax_lots"][t], shares_to_sell, tr["price"])
            state["tax_lots"][t] = remaining
            tax, state["loss_carryforward"] = apply_tax(realized, state["loss_carryforward"])
            state["total_realized_tax"] += tax


def compute_after_tax_nav(state, prices_row):
    """
    Compute after-tax liquidation NAV: what would remain if all positions
    were sold now and taxes paid on unrealized gains.
    """
    pre_tax_nav = state["cash"] + sum(
        state["shares"].get(t, 0.0) * prices_row[t] for t in prices_row.index)

    # Compute unrealized gains for each position
    total_unrealized = 0.0
    for t, shares in state["shares"].items():
        if t not in state.get("tax_lots", {}):
            continue
        for lot in state["tax_lots"][t]:
            unrealized = lot["shares"] * (prices_row[t] - lot["cost_basis"])
            total_unrealized += unrealized

    # Apply tax on net unrealized gain (after loss carryforward)
    net_gain = total_unrealized - state.get("loss_carryforward", 0.0)
    if net_gain > 0:
        tax_due = net_gain * CONFIG["tax_rate"]
    else:
        tax_due = 0.0

    return pre_tax_nav - tax_due


# ============================================================
# STATIC BENCHMARK (55/35/10, no MA200 filter)
# ============================================================
def init_benchmark_state(start_date, start_capital):
    """Initialize the static benchmark portfolio state."""
    return {
        "start_date": str(start_date),
        "start_capital": start_capital,
        "cash": start_capital,
        "shares": {},
        "total_costs": 0.0,
        "tax_lots": {},
        "loss_carryforward": 0.0,
        "total_realized_tax": 0.0,
        "last_trade_month": None,
        "nav": start_capital,
    }


def update_benchmark(state, prices, latest, regime):
    """
    Update the static benchmark portfolio.
    Uses the same rebalancing logic but with fixed weights (no regime switch).
    """
    if "benchmark" not in state:
        state["benchmark"] = init_benchmark_state(latest, CONFIG["start_capital"])

    bench = state["benchmark"]
    month_key = f"{latest.year}-{latest.month:02d}"

    # Rebalance on month-end (same timing as SmartPassive)
    if (is_last_trading_day_of_month(prices.index, latest)
            and bench.get("last_trade_month") != month_key):
        tw = dict(CONFIG["static_benchmark"])  # Fixed weights, no regime
        row0 = prices.loc[latest]
        trades, drift, nav = plan_trades(bench, row0, tw)
        if trades:
            apply_trades(bench, trades, latest, "BENCH_REBALANCE")
            update_tax_lots(bench, trades, latest)
            bench["last_trade_month"] = month_key

    # Mark-to-market
    row0 = prices.loc[latest]
    bench["nav"] = float(bench["cash"] + sum(
        bench["shares"].get(t, 0.0) * row0[t] for t in row0.index))
    bench["after_tax_nav"] = compute_after_tax_nav(bench, row0)


def compute_mar_ratio(nav_series, start_capital):
    """
    Compute MAR ratio = CAGR / |Max Drawdown|.
    nav_series: pd.Series of after-tax NAV values.
    """
    if len(nav_series) < 2:
        return 0.0, 0.0, 0.0

    # CAGR
    years = len(nav_series) / 252
    if years <= 0:
        return 0.0, 0.0, 0.0
    total_return = nav_series.iloc[-1] / start_capital
    cagr = (total_return ** (1 / years)) - 1 if total_return > 0 else -1.0

    # Max drawdown
    cummax = nav_series.cummax()
    dd = (nav_series - cummax) / cummax
    max_dd = abs(dd.min()) if len(dd) > 0 else 1.0

    mar = cagr / max_dd if max_dd > 0 else 0.0
    return mar, cagr, max_dd


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
    if not write_header:
        with open(path, "rb") as f:
            f.seek(0, 2)
            if f.tell() > 0:
                f.seek(-1, 2)
                if f.read(1) != b"\n":
                    with open(path, "a", encoding="utf-8") as fa:
                        fa.write("\n")
    tmp = path + ".tmp"
    if write_header:
        df.to_csv(tmp, mode="w", header=True, index=False)
    else:
        import shutil
        shutil.copy2(path, tmp)
        df.to_csv(tmp, mode="a", header=False, index=False)
    os.replace(tmp, path)


def read_nav_history():
    if os.path.exists(NAV_FILE):
        return pd.read_csv(NAV_FILE, parse_dates=["date"])
    return pd.DataFrame(columns=["date", "nav", "spy_bench_nav", "regime"])


# ============================================================
# REPORT
# ============================================================
def write_report(state, nav_df, last_signal, prov=None):
    if prov is None:
        prov = provenance()
    prov_commit = prov.get("commit") or "n/a"
    prov_dirty = prov.get("dirty")
    prov_hash = (prov.get("hypothesis_sha256") or "")[:16] or "n/a"
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
<p style="color:#666;font-size:12px">Provenance: commit <code>{prov_commit}</code> · dirty={prov_dirty} · hypothesis sha256 <code>{prov_hash}</code></p>
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
    # Capture provenance BEFORE any file writes. forward/ artifacts are tracked,
    # so stamping after save_state()/append_csv() would flip `dirty` on every run
    # by construction (the bot dirtying its own output). Captured here, `dirty`
    # reflects the tree at run start -- i.e. whether the code that is about to run
    # differs from HEAD -- which is what the docstring promises.
    prov = provenance()
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
            # Keep summary.json provenance current even on a no-op rerun, but
            # touch ONLY the provenance key: status/asof/numbers describe the
            # last real run and must not be clobbered with NO_ACTION here.
            if os.path.exists(SUMMARY_FILE):
                try:
                    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    existing["provenance"] = prov
                    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
                        json.dump(existing, f, indent=2)
                except Exception:
                    pass
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
            "tax_lots": {},
            "loss_carryforward": 0.0,
            "total_realized_tax": 0.0,
            "risk_off_cycles": [],
            "current_risk_off_entry": None,
        }
        tw = target_weights(regime)
        row0 = prices.loc[latest]
        trades, drift, nav = plan_trades(state, row0, tw)
        trade_rows = apply_trades(state, trades, latest, "INITIAL_ALLOCATION")
        update_tax_lots(state, trades, latest)
        state["last_trade_month"] = f"{latest.year}-{latest.month:02d}"
        status = "INITIAL_ALLOCATION"
        # Initialize benchmark at the same time
        state["benchmark"] = init_benchmark_state(latest, CONFIG["start_capital"])
        bench_trades, _, _ = plan_trades(state["benchmark"], row0, dict(CONFIG["static_benchmark"]))
        apply_trades(state["benchmark"], bench_trades, latest, "BENCH_INITIAL")
        update_tax_lots(state["benchmark"], bench_trades, latest)
        state["benchmark"]["last_trade_month"] = f"{latest.year}-{latest.month:02d}"
    else:
        month_key = f"{latest.year}-{latest.month:02d}"
        if (is_last_trading_day_of_month(prices.index, latest)
                and state.get("last_trade_month") != month_key):
            tw = target_weights(regime)
            row0 = prices.loc[latest]
            trades, drift, nav = plan_trades(state, row0, tw)
            if trades:
                trade_rows = apply_trades(state, trades, latest, "MONTH_END_REBALANCE")
                update_tax_lots(state, trades, latest)
                state["last_trade_month"] = month_key
                status = "REBALANCED"
            else:
                status = f"MONTH_END_NO_TRADE(drift={drift*100:.2f}%)"
        else:
            status = "HOLD(not month-end)"

        # Track RISK_OFF cycles
        if regime == "RISK_OFF" and state.get("current_risk_off_entry") is None:
            state["current_risk_off_entry"] = str(latest.date())
        elif regime == "RISK_ON" and state.get("current_risk_off_entry") is not None:
            state["risk_off_cycles"].append({
                "entry": state["current_risk_off_entry"],
                "exit": str(latest.date()),
                "completed": False,
            })
            state["current_risk_off_entry"] = None

        # Update benchmark
        update_benchmark(state, prices, latest, regime)

    # -------- mark-to-market --------
    row0 = prices.loc[latest]
    nav = state["cash"] + sum(state["shares"].get(t, 0.0) * row0[t]
                              for t in row0.index)
    state["nav"] = float(nav)
    state["after_tax_nav"] = compute_after_tax_nav(state, row0)
    state["spy_bench_nav"] = float(
        CONFIG["start_capital"] * spy_price / state["spy_start_price"])
    state["last_run_date"] = str(latest.date())

    # Benchmark NAV
    bench_nav = state.get("benchmark", {}).get("nav", CONFIG["start_capital"])
    bench_after_tax = state.get("benchmark", {}).get("after_tax_nav", CONFIG["start_capital"])

    nav_df = read_nav_history()
    if len(nav_df) == 0 or pd.to_datetime(nav_df["date"]).max() < pd.Timestamp(latest):
        append_csv(NAV_FILE, [{
            "date": str(latest.date()), "nav": round(nav, 2),
            "after_tax_nav": round(state["after_tax_nav"], 2),
            "bench_nav": round(bench_nav, 2),
            "bench_after_tax_nav": round(bench_after_tax, 2),
            "spy_bench_nav": round(state["spy_bench_nav"], 2),
            "regime": regime}],
            ["date", "nav", "after_tax_nav", "bench_nav", "bench_after_tax_nav", "spy_bench_nav", "regime"])
        nav_df = read_nav_history()

    append_csv(TRADES_FILE, trade_rows,
               ["date", "ticker", "reason", "price", "shares", "value", "cost"])
    append_csv(RUNS_FILE, [{
        "run_ts": datetime.now().isoformat(timespec="seconds"),
        "asof": str(latest.date()), "status": status,
        "trades": len(trade_rows), "nav": round(nav, 2)}],
        ["run_ts", "asof", "status", "trades", "nav"])

    save_state(state)
    write_report(state, nav_df, signal_row, prov)

    # Compute criteria status
    completed_cycles = sum(1 for c in state.get("risk_off_cycles", []) if c.get("completed"))
    days_elapsed = (latest - pd.Timestamp(state["start_date"])).days
    years_elapsed = days_elapsed / 365.25
    horizon_met = (years_elapsed >= CONFIG["min_years"] and
                   completed_cycles >= CONFIG["min_risk_off_cycles"])
    hard_stop_reached = latest >= pd.Timestamp(CONFIG["hard_stop_date"])

    if not horizon_met and not hard_stop_reached:
        verdict = "NOT_READY"
    elif hard_stop_reached and not horizon_met:
        verdict = "INCONCLUSIVE"
    else:
        # Compute SP-C1 and SP-C2
        c1_pass = state["after_tax_nav"] > bench_after_tax
        # MAR requires full history; for now mark as pending
        c2_pass = None  # Will be computed from nav_history at final evaluation
        # ---------------------------------------------------------------------
        # KNOWN GAP (C2 verdict) -- documented, deliberately NOT fixed here.
        # SP-C2 (after-tax MAR) is unimplemented, so c2_pass is None. But this
        # branch only runs once horizon_met (>= min_years AND min_risk_off
        # cycles) or the hard stop is hit -- i.e. from ~2029 onward. When it
        # does, `not c2_pass` evaluates `not None == True`, so a FAILING C1
        # (c1_pass=False) with c2 still unimplemented takes the second arm and
        # stamps verdict="REFUTED" on the strength of a criterion that was never
        # computed. Refuting the hypothesis on an uncomputed criterion is wrong.
        # Required guard BEFORE the first real verdict: treat c2_pass is None as
        # INCONCLUSIVE, not REFUTED (e.g. `if c2_pass is None: verdict =
        # "INCONCLUSIVE"`). This does NOT change the locked SP-C2 criterion --
        # it only stops an un-run criterion from producing a false REFUTED.
        # See preregistration SMARTPASSIVE_hypothesis.md sec. 5 (do not edit
        # that file: its sha256 is the provenance seal).
        # ---------------------------------------------------------------------
        if c1_pass and c2_pass:
            verdict = "NOT_REFUTED"
        elif not c1_pass and not c2_pass:
            verdict = "REFUTED"
        else:
            verdict = "PARTIAL"

    summary = {
        "experiment_id": CONFIG["experiment_id"],
        "asof": str(latest.date()),
        "status": status,
        "provenance": prov,
        "regime": regime,
        "spy_price": spy_price,
        "ma200": ma,
        "nav": round(nav, 2),
        "after_tax_nav": round(state["after_tax_nav"], 2),
        "bench_nav": round(bench_nav, 2),
        "bench_after_tax_nav": round(bench_after_tax, 2),
        "total_return_pct": round((nav / CONFIG["start_capital"] - 1) * 100, 3),
        "bench_return_pct": round((bench_nav / CONFIG["start_capital"] - 1) * 100, 3),
        "spy_bench_return_pct": round(
            (state["spy_bench_nav"] / CONFIG["start_capital"] - 1) * 100, 3),
        "n_trades": len(trade_rows),
        "holdings": state["shares"],
        "cash": round(state["cash"], 2),
        "risk_off_cycles_completed": completed_cycles,
        "years_elapsed": round(years_elapsed, 2),
        "horizon_met": horizon_met,
        "verdict": verdict,
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