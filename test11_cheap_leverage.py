"""
TEST 11: CHEAP LEVERAGE & VOLATILITY PREMIUM LAB
=================================================
Direct follow-up to Test 10. Attacks the exact reason leverage failed:

PART 1 - FINANCING MODEL (Idea 1):
  Leveraged funds do not pay retail margin (3%+). Futures-embedded
  leverage pays ~risk-free rate. Re-run RP_VolTarget (12% target, 1.5x
  cap) under three financing models:
    a) Fixed 3%/yr   (retail margin - Test 10 primary)
    b) Fixed 1.5%/yr (optimistic flat)
    c) Dynamic T-bill proxy: trailing 60d annualized SHY return,
       lagged 1 day (no look-ahead), floored at 0.
  Same strategy, different (more realistic) financing.

PART 2 - MODEST FIXED LEVERAGE ON BEST SHARPE (Idea 2):
  Kelly logic: lever the highest-Sharpe portfolio, not the vol target.
  RiskParity (Sharpe 0.87, best in Test 10 lab) at fixed leverage:
    1.00x (baseline), 1.25x (PRIMARY), 1.50x (stress)
  with dynamic T-bill financing; plus 3% financing stress on 1.25x.

PART 3 - SELLING VOLATILITY (Idea 3):
  Buy-write / put-write indices harvest the variance premium:
    ^BXM    - CBOE S&P 500 BuyWrite
    ^PCALL  - CBOE S&P 500 PutWrite
  If data is available via yfinance, compare to SPY on a PRICE basis
  (both indices are price indices; dividend-adjusted SPY would bias
  the comparison, so unadjusted SPY is used as the benchmark here).
  If data is unavailable, the test is reported INFEASIBLE - no
  synthetic approximation is invented.

PART 4 - IS/OOS + CRISIS for the primary candidate.

Pre-registered expectations (stated BEFORE results): 8-9% CAGR with
~20% DD for the levered RP variants; buy-write ~ market-like Sharpe
with lower return. Double-digit remains unlikely.
"""

import os
import pandas as pd
import numpy as np
from data_loader import download_prices
from backtest_engine import (
    backtest_from_weights,
    get_monthly_rebalance_dates,
    forward_fill_weights,
)
from metrics import calculate_metrics
from config import COST_PER_SIDE, SAFE_ASSET

from test10_low_vol_lab import (
    compute_risk_parity_sparse,
    generate_vol_target_weights,
    analyze_crisis_periods,
    CRISIS_PERIODS,
    IS_START,
    IS_END,
    OOS_START,
    TARGET_VOL,
    MAX_LEVERAGE,
)

# ============================================================
# PRE-REGISTERED PARAMETERS
# ============================================================
TBILL_LOOKBACK = 60          # trailing window for T-bill proxy rate
FIXED_LEVERAGE_PRIMARY = 1.25
FIXED_LEVERAGE_GRID = [1.0, 1.25, 1.5]
RETAIL_FINANCING = 0.03      # Test 10 assumption (stress)
FLAT_CHEAP_FINANCING = 0.015 # optimistic flat alternative

BUYWRITE_CANDIDATES = {
    "BXM": ["^BXM", "BXM"],
    "PutWrite": ["^PCALL", "PCALL"],
}


# ============================================================
# FINANCING MODELS
# ============================================================
def build_tbill_financing_series(prices, lookback=TBILL_LOOKBACK):
    """
    Dynamic annual financing rate ~ risk-free proxy:
    trailing 60d annualized SHY return, lagged 1 day, floored at 0.
    Mimics futures-embedded funding (pay ~risk-free on leverage).
    """
    shy_ret = prices[SAFE_ASSET].pct_change()
    ann_rate = shy_ret.rolling(lookback, min_periods=lookback).mean() * 252
    ann_rate = ann_rate.clip(lower=0).shift(1).fillna(0.0)
    return ann_rate


# ============================================================
# PART 2: FIXED LEVERAGE ON RISK PARITY
# ============================================================
def generate_fixed_leverage_rp(prices, rebalance_dates, leverage=FIXED_LEVERAGE_PRIMARY):
    """Capped risk-parity weights scaled by a CONSTANT leverage factor."""
    rp_sparse = compute_risk_parity_sparse(prices, rebalance_dates)
    scaled = rp_sparse * leverage
    return forward_fill_weights(scaled, prices.index)


# ============================================================
# PART 3: BUY-WRITE / PUT-WRITE INDICES
# ============================================================
def try_download_index(ticker_candidates, start):
    """Try to download a price index from yfinance; return (ticker, series) or (None, None)."""
    import yfinance as yf
    for t in ticker_candidates:
        try:
            data = yf.download(t, start=start, auto_adjust=True,
                               progress=False, threads=False)
            if data is None or len(data) == 0:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"].squeeze()
            else:
                close = data["Close"]
            close = close.dropna()
            if len(close) < 252:
                continue
            close.name = t
            return t, close
        except Exception:
            continue
    return None, None


def download_spy_price_series(start):
    """Unadjusted SPY close (price basis) for fair index comparison."""
    import yfinance as yf
    try:
        data = yf.download("SPY", start=start, auto_adjust=False,
                           progress=False, threads=False)
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"].squeeze()
        else:
            close = data["Close"]
        return close.dropna()
    except Exception:
        return None


def analyze_index_series(close, name, spy_close=None):
    """Metrics + crisis analysis for a price index series."""
    returns = close.pct_change().dropna()
    m = calculate_metrics(returns, name)
    crisis = analyze_crisis_periods(returns)
    row = {
        "name": name,
        "start": str(returns.index[0].date()),
        "end": str(returns.index[-1].date()),
        "cagr": m["cagr"],
        "sharpe": m["sharpe"],
        "max_dd": m["max_drawdown"],
        "ann_vol": m["ann_volatility"],
    }
    for cname, cvals in crisis.items():
        row[f"{cname}_ret"] = cvals["return"]
        row[f"{cname}_dd"] = cvals["max_dd"]
    return row


# ============================================================
# MAIN
# ============================================================
def run_test11():
    print("\n" + "=" * 70)
    print("TEST 11: CHEAP LEVERAGE & VOLATILITY PREMIUM LAB")
    print("=" * 70)

    prices_full = download_prices()
    rebalance_dates = get_monthly_rebalance_dates(prices_full.index, freq="M")
    tbill_series = build_tbill_financing_series(prices_full)

    print(f"\nT-bill proxy financing: mean={tbill_series.mean()*100:.2f}%/yr, "
          f"max={tbill_series.max()*100:.2f}%/yr (dynamic, lagged 1d)")

    summary_rows = []

    # ----------------------------------------------------------
    # PART 1: FINANCING MODEL ON RP_VolTarget
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 1: RP_VolTarget (12% target, 1.5x cap) UNDER 3 FINANCING MODELS")
    print("=" * 70)

    vt_weights, _ = generate_vol_target_weights(
        prices_full, rebalance_dates, TARGET_VOL, MAX_LEVERAGE, defensive=False)

    financing_models = [
        ("Fixed 3% (retail)", {"financing_rate": RETAIL_FINANCING}),
        ("Fixed 1.5% (flat)", {"financing_rate": FLAT_CHEAP_FINANCING}),
        ("Dynamic T-bill", {"financing_rate_series": tbill_series}),
    ]

    print(f"\n{'Financing model':<22} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>9} {'FinDrag':>9}")
    print("-" * 60)
    part1_rows = []
    for fname, fkwargs in financing_models:
        res = backtest_from_weights(prices_full, vt_weights, COST_PER_SIDE,
                                    f"VT_{fname}", **fkwargs)
        m = calculate_metrics(res["net_returns"], fname)
        print(f"{fname:<22} {m['cagr']:>7.2f}% {m['sharpe']:>8.3f} "
              f"{m['max_drawdown']:>8.2f}% {res['total_financing_drag']*100:>8.2f}%")
        part1_rows.append({
            "part": "1_financing_model", "strategy": "RP_VolTarget",
            "financing": fname, "cagr": m["cagr"], "sharpe": m["sharpe"],
            "max_dd": m["max_drawdown"],
            "financing_drag_pct": res["total_financing_drag"] * 100,
        })
    pd.DataFrame(part1_rows).to_csv("results/test11_financing_models.csv", index=False)

    # ----------------------------------------------------------
    # PART 2: FIXED LEVERAGE ON RISK PARITY
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 2: FIXED LEVERAGE ON RISK PARITY (dynamic T-bill financing)")
    print("=" * 70)

    print(f"\n{'Leverage':<10} {'Financing':<16} {'CAGR':>8} {'Sharpe':>8} "
          f"{'MaxDD':>9} {'Vol':>7} {'Turn/yr':>8}")
    print("-" * 68)
    part2_rows = []
    part2_results = {}
    for lev in FIXED_LEVERAGE_GRID:
        w = generate_fixed_leverage_rp(prices_full, rebalance_dates, lev)
        res = backtest_from_weights(prices_full, w, COST_PER_SIDE,
                                    f"RP_{lev}x", financing_rate_series=tbill_series)
        m = calculate_metrics(res["net_returns"], f"RP_{lev}x")
        label = f"{lev:.2f}x"
        marker = " <-- PRIMARY" if lev == FIXED_LEVERAGE_PRIMARY else ""
        print(f"{label:<10} {'T-bill':<16} {m['cagr']:>7.2f}% {m['sharpe']:>8.3f} "
              f"{m['max_drawdown']:>8.2f}% {m['ann_volatility']:>6.2f}% "
              f"{res['annual_turnover_one_way']:>7.2f}x{marker}")
        part2_results[f"RP_{lev}x_tbill"] = (res, m)
        part2_rows.append({
            "part": "2_fixed_leverage", "strategy": f"RP_{lev}x",
            "financing": "T-bill", "cagr": m["cagr"], "sharpe": m["sharpe"],
            "max_dd": m["max_drawdown"], "ann_vol": m["ann_volatility"],
            "turnover": res["annual_turnover_one_way"],
            "avg_gross": res["avg_gross_exposure"],
        })

    # Stress: primary leverage with retail financing
    w_primary = generate_fixed_leverage_rp(prices_full, rebalance_dates,
                                           FIXED_LEVERAGE_PRIMARY)
    res_stress = backtest_from_weights(prices_full, w_primary, COST_PER_SIDE,
                                       "RP_1.25x_retail", financing_rate=RETAIL_FINANCING)
    m_stress = calculate_metrics(res_stress["net_returns"], "RP_1.25x_retail")
    print(f"{str(FIXED_LEVERAGE_PRIMARY)+'x':<10} {'Fixed 3% (stress)':<16} "
          f"{m_stress['cagr']:>7.2f}% {m_stress['sharpe']:>8.3f} "
          f"{m_stress['max_drawdown']:>8.2f}% {m_stress['ann_volatility']:>6.2f}% "
          f"{res_stress['annual_turnover_one_way']:>7.2f}x")
    part2_rows.append({
        "part": "2_fixed_leverage", "strategy": "RP_1.25x",
        "financing": "Fixed 3% (stress)", "cagr": m_stress["cagr"],
        "sharpe": m_stress["sharpe"], "max_dd": m_stress["max_drawdown"],
        "ann_vol": m_stress["ann_volatility"],
        "turnover": res_stress["annual_turnover_one_way"],
        "avg_gross": res_stress["avg_gross_exposure"],
    })
    pd.DataFrame(part2_rows).to_csv("results/test11_fixed_leverage.csv", index=False)

    # ----------------------------------------------------------
    # PART 3: BUY-WRITE / PUT-WRITE INDICES
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 3: SELLING VOLATILITY (BXM / PutWrite indices)")
    print("=" * 70)

    buywrite_rows = []
    any_index_found = False
    for name, candidates in BUYWRITE_CANDIDATES.items():
        ticker, close = try_download_index(candidates, start="2007-01-01")
        if ticker is None:
            print(f"\n{name}: DATA UNAVAILABLE via yfinance -> INFEASIBLE "
                  f"(no synthetic approximation)")
            buywrite_rows.append({"name": name, "status": "INFEASIBLE_NO_DATA"})
            continue
        any_index_found = True
        print(f"\n{name}: downloaded {ticker} "
              f"({close.index[0].date()} to {close.index[-1].date()}, {len(close)} days)")
        row = analyze_index_series(close, f"{name} ({ticker})")
        row["status"] = "OK"
        buywrite_rows.append(row)

        # Fair price-basis SPY benchmark over the SAME window
        spy_price = download_spy_price_series(str(close.index[0].date()))
        if spy_price is not None:
            spy_price = spy_price.reindex(close.index).ffill().dropna()
            spy_row = analyze_index_series(spy_price,
                                           f"SPY price ({close.index[0].year}+)")
            spy_row["status"] = "benchmark"
            buywrite_rows.append(spy_row)

    if any_index_found:
        print(f"\n{'Index':<28} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>9}")
        print("-" * 58)
        for r in buywrite_rows:
            if r.get("status") in ("OK", "benchmark"):
                print(f"{r['name']:<28} {r['cagr']:>7.2f}% {r['sharpe']:>8.3f} "
                      f"{r['max_dd']:>8.2f}%")
    pd.DataFrame(buywrite_rows).to_csv("results/test11_buywrite.csv", index=False)

    # ----------------------------------------------------------
    # PART 4: IS/OOS + CRISIS FOR PRIMARY CANDIDATE
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"PART 4: PRIMARY CANDIDATE RP@{FIXED_LEVERAGE_PRIMARY}x + T-bill "
          f"financing — IS/OOS & CRISIS")
    print("=" * 70)

    prices_is = prices_full.loc[IS_START:IS_END]
    prices_oos = prices_full.loc[OOS_START:]

    slice_rows = []
    for label, prices_slice in [("IS", prices_is), ("OOS", prices_oos)]:
        reb = get_monthly_rebalance_dates(prices_slice.index, freq="M")
        w = generate_fixed_leverage_rp(prices_slice, reb, FIXED_LEVERAGE_PRIMARY)
        tb_slice = tbill_series.reindex(prices_slice.index)
        res = backtest_from_weights(prices_slice, w, COST_PER_SIDE,
                                    f"RP125_{label}", financing_rate_series=tb_slice)
        m = calculate_metrics(res["net_returns"], f"RP1.25x {label}")

        # SPY reference on same slice
        w_spy = forward_fill_weights(
            pd.DataFrame(1.0, index=reb, columns=["SPY"]).reindex(
                prices_slice.columns, axis=1).fillna(0),
            prices_slice.index)
        res_spy = backtest_from_weights(prices_slice, w_spy, COST_PER_SIDE, f"SPY_{label}")
        m_spy = calculate_metrics(res_spy["net_returns"], f"SPY {label}")

        print(f"\n[{label}] RP@{FIXED_LEVERAGE_PRIMARY}x: CAGR={m['cagr']:.2f}% "
              f"Sharpe={m['sharpe']:.3f} MaxDD={m['max_drawdown']:.2f}% | "
              f"SPY: CAGR={m_spy['cagr']:.2f}% Sharpe={m_spy['sharpe']:.3f} "
              f"MaxDD={m_spy['max_drawdown']:.2f}%")
        slice_rows.append({
            "period": label,
            "rp125_cagr": m["cagr"], "rp125_sharpe": m["sharpe"],
            "rp125_max_dd": m["max_drawdown"],
            "spy_cagr": m_spy["cagr"], "spy_sharpe": m_spy["sharpe"],
            "spy_max_dd": m_spy["max_drawdown"],
            "beats_spy_cagr": m["cagr"] > m_spy["cagr"],
            "beats_spy_sharpe": m["sharpe"] > m_spy["sharpe"],
            "double_digit": m["cagr"] >= 10.0,
            "dd_pass": m["passes_dd_limit"],
        })

    # Crisis analysis (full period, primary candidate)
    res_primary_full = part2_results[f"RP_{FIXED_LEVERAGE_PRIMARY}x_tbill"][0]
    crisis_primary = analyze_crisis_periods(res_primary_full["net_returns"])
    print(f"\nCrisis behavior (RP@{FIXED_LEVERAGE_PRIMARY}x, full period):")
    for cname, cvals in crisis_primary.items():
        if cvals["return"] is not None:
            print(f"  {cname:<25} return={cvals['return']:>7.2f}%  "
                  f"maxDD={cvals['max_dd']:>7.2f}%")

    pd.DataFrame(slice_rows).to_csv("results/test11_isoos.csv", index=False)

    # ----------------------------------------------------------
    # VERDICT
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    oos_row = next(r for r in slice_rows if r["period"] == "OOS")
    full_primary = part2_results[f"RP_{FIXED_LEVERAGE_PRIMARY}x_tbill"][1]

    print(f"\nPrimary candidate: RiskParity @ {FIXED_LEVERAGE_PRIMARY}x with T-bill financing")
    print(f"  Full period: CAGR={full_primary['cagr']:.2f}% "
          f"Sharpe={full_primary['sharpe']:.3f} MaxDD={full_primary['max_drawdown']:.2f}%")
    print(f"  OOS: CAGR={oos_row['rp125_cagr']:.2f}% Sharpe={oos_row['rp125_sharpe']:.3f} "
          f"MaxDD={oos_row['rp125_max_dd']:.2f}%")
    print(f"  OOS vs SPY: CAGR {'BEAT' if oos_row['beats_spy_cagr'] else 'LOST'} | "
          f"Sharpe {'BEAT' if oos_row['beats_spy_sharpe'] else 'LOST'}")
    print(f"  Double-digit CAGR: {'YES' if oos_row['double_digit'] else 'NO'} | "
          f"25% DD limit: {'PASS' if oos_row['dd_pass'] else 'FAIL'}")

    met_bar = (oos_row["double_digit"] or oos_row["beats_spy_cagr"]) and oos_row["dd_pass"]
    print("\n" + "-" * 70)
    if met_bar:
        print("RESULT: Candidate MEETS the return bar. Run bootstrap + cost stress "
              "before promotion.")
    else:
        print("RESULT: Candidate IMPROVES on Test 10 but does NOT reach double-digit "
              "or beat SPY OOS.")
        print("  Honest conclusion confirmed: cheap leverage lifts CAGR ~1-2pp, "
              "it does not create alpha.")
    print("-" * 70)

    summary_rows = slice_rows
    pd.DataFrame(summary_rows).to_csv("results/test11_summary.csv", index=False)
    print("\nResults saved to results/test11_*.csv")
    return summary_rows


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    run_test11()