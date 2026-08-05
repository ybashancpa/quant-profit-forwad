"""
TEST 8: THE ULTIMATE BASELINE
==============================
Final test: Does the MA200 filter actually add value, or is it
just adding whipsaw costs?

Comparison:
1. SmartPassive 55/35/10 (MA200 filter, switches to SHY)
2. Static 55/35/10 (no filter, monthly rebalance only)
3. Static 60/40 (classic benchmark)

Key Question: Does Static 55/35/10 survive 2008 within 25% DD?
- If YES: MA200 filter is unnecessary, static allocation wins
- If NO: MA200 filter is DD insurance, 2022 is the premium

Crisis Analysis Focus:
- 2008: Slow, persistent bear market (MA200 should shine)
- 2022: Choppy, volatile bear market (MA200 whipsaws)
"""

import pandas as pd
import numpy as np
from data_loader import download_prices
from backtest_engine import backtest_from_weights, get_monthly_rebalance_dates, forward_fill_weights
from metrics import calculate_metrics
from config import COST_PER_SIDE


# ============================================================
# PORTFOLIO DEFINITIONS
# ============================================================
STATIC_55_35_10 = {"SPY": 0.55, "IEF": 0.35, "GLD": 0.10}
STATIC_60_40 = {"SPY": 0.60, "IEF": 0.40}
RISK_ON_WEIGHTS = {"SPY": 0.55, "IEF": 0.35, "GLD": 0.10}
RISK_OFF_WEIGHTS = {"SHY": 0.55, "IEF": 0.35, "GLD": 0.10}
MA_LOOKBACK = 200

CRISIS_PERIODS = {
    "2008 Financial Crisis": ("2008-01-01", "2009-03-31"),
    "2020 Corona Crash": ("2020-02-01", "2020-04-30"),
    "2022 Inflation/Rates": ("2022-01-01", "2022-12-31"),
}


def generate_static_weights(prices, rebalance_dates, allocation):
    """Generate static portfolio weights (monthly rebalance only)."""
    sparse_weights = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    
    for date in rebalance_dates:
        for asset, weight in allocation.items():
            if asset in prices.columns:
                sparse_weights.loc[date, asset] = weight
    
    daily_weights = forward_fill_weights(sparse_weights, prices.index)
    return daily_weights


def generate_smart_passive_weights(prices, rebalance_dates):
    """Generate SmartPassive weights (MA200 filter)."""
    moving_avg = prices["SPY"].rolling(window=MA_LOOKBACK, min_periods=MA_LOOKBACK).mean()
    
    sparse_weights = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    
    for date in rebalance_dates:
        if date not in prices.index:
            continue
        
        spy_price = prices.loc[date, "SPY"]
        spy_ma = moving_avg.loc[date]
        
        if pd.isna(spy_ma) or spy_price <= spy_ma:
            # Risk-Off
            for asset, weight in RISK_OFF_WEIGHTS.items():
                if asset in prices.columns:
                    sparse_weights.loc[date, asset] = weight
        else:
            # Risk-On
            for asset, weight in RISK_ON_WEIGHTS.items():
                if asset in prices.columns:
                    sparse_weights.loc[date, asset] = weight
    
    daily_weights = forward_fill_weights(sparse_weights, prices.index)
    return daily_weights


def analyze_crisis_periods(returns):
    """Analyze performance during specific crisis periods."""
    results = {}
    for crisis_name, (start, end) in CRISIS_PERIODS.items():
        mask = (returns.index >= start) & (returns.index <= end)
        crisis_returns = returns[mask]
        
        if len(crisis_returns) == 0:
            results[crisis_name] = {"return": None, "max_dd": None}
            continue
        
        total_return = (1 + crisis_returns).prod() - 1
        cumulative = (1 + crisis_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        results[crisis_name] = {
            "return": total_return * 100,
            "max_dd": max_dd * 100,
        }
    return results


def find_max_dd_location(returns):
    """Find where the max drawdown occurred."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd_date = drawdown.idxmin()
    max_dd_value = drawdown.min()
    return max_dd_date, max_dd_value


def run_test8():
    """Run Test 8: Ultimate Baseline Comparison."""
    print("\n" + "="*70)
    print("TEST 8: THE ULTIMATE BASELINE")
    print("="*70)
    print("\nQuestion: Does the MA200 filter add value, or just whipsaw costs?")
    print("Key Test: Does Static 55/35/10 survive 2008 within 25% DD?")
    
    # Load data
    prices_full = download_prices()
    rebalance_dates = get_monthly_rebalance_dates(prices_full.index, freq="M")
    
    # ============================================================
    # FULL PERIOD PERFORMANCE
    # ============================================================
    print("\n" + "-"*70)
    print("FULL PERIOD PERFORMANCE (2007-2026)")
    print("-"*70)
    
    # SmartPassive (with MA200)
    weights_smart = generate_smart_passive_weights(prices_full, rebalance_dates)
    result_smart = backtest_from_weights(prices_full, weights_smart, COST_PER_SIDE, "SmartPassive")
    metrics_smart = calculate_metrics(result_smart["net_returns"], "SmartPassive")
    
    # Static 55/35/10
    weights_static = generate_static_weights(prices_full, rebalance_dates, STATIC_55_35_10)
    result_static = backtest_from_weights(prices_full, weights_static, COST_PER_SIDE, "Static55/35/10")
    metrics_static = calculate_metrics(result_static["net_returns"], "Static55/35/10")
    
    # Static 60/40
    weights_6040 = generate_static_weights(prices_full, rebalance_dates, STATIC_60_40)
    result_6040 = backtest_from_weights(prices_full, weights_6040, COST_PER_SIDE, "Static60/40")
    metrics_6040 = calculate_metrics(result_6040["net_returns"], "Static60/40")
    
    print(f"\n{'Strategy':<25} {'CAGR':>8} {'Sharpe':>8} {'Max DD':>10} {'Turnover/yr':>12}")
    print("-"*70)
    print(f"{'SmartPassive (MA200)':<25} {metrics_smart['cagr']:>7.2f}% {metrics_smart['sharpe']:>8.3f} {metrics_smart['max_drawdown']:>9.2f}% {result_smart['annual_turnover_one_way']:>11.2f}x")
    print(f"{'Static 55/35/10':<25} {metrics_static['cagr']:>7.2f}% {metrics_static['sharpe']:>8.3f} {metrics_static['max_drawdown']:>9.2f}% {result_static['annual_turnover_one_way']:>11.2f}x")
    print(f"{'Static 60/40':<25} {metrics_6040['cagr']:>7.2f}% {metrics_6040['sharpe']:>8.3f} {metrics_6040['max_drawdown']:>9.2f}% {result_6040['annual_turnover_one_way']:>11.2f}x")
    
    # ============================================================
    # CRISIS ANALYSIS (2008 vs 2022)
    # ============================================================
    print("\n" + "-"*70)
    print("CRISIS ANALYSIS: 2008 (Slow Bear) vs 2022 (Choppy Bear)")
    print("-"*70)
    
    crisis_smart = analyze_crisis_periods(result_smart["net_returns"])
    crisis_static = analyze_crisis_periods(result_static["net_returns"])
    crisis_6040 = analyze_crisis_periods(result_6040["net_returns"])
    
    print(f"\n{'Crisis':<25} {'SmartPassive':<15} {'Static55/35/10':<15} {'Static60/40':<15}")
    print("-"*70)
    for crisis_name in CRISIS_PERIODS.keys():
        smart_ret = crisis_smart[crisis_name]["return"]
        smart_dd = crisis_smart[crisis_name]["max_dd"]
        static_ret = crisis_static[crisis_name]["return"]
        static_dd = crisis_static[crisis_name]["max_dd"]
        sixty_ret = crisis_6040[crisis_name]["return"]
        sixty_dd = crisis_6040[crisis_name]["max_dd"]
        
        print(f"\n{crisis_name}:")
        print(f"  {'Return':<12} {smart_ret:>8.2f}% {static_ret:>12.2f}% {sixty_ret:>12.2f}%")
        print(f"  {'Max DD':<12} {smart_dd:>8.2f}% {static_dd:>12.2f}% {sixty_dd:>12.2f}%")
    
    # ============================================================
    # MAX DD LOCATION
    # ============================================================
    print("\n" + "-"*70)
    print("MAX DD LOCATION (Where did the worst drawdown occur?)")
    print("-"*70)
    
    smart_dd_date, smart_dd_val = find_max_dd_location(result_smart["net_returns"])
    static_dd_date, static_dd_val = find_max_dd_location(result_static["net_returns"])
    sixty_dd_date, sixty_dd_val = find_max_dd_location(result_6040["net_returns"])
    
    print(f"\nSmartPassive: {smart_dd_val*100:.2f}% on {smart_dd_date.date()}")
    print(f"Static 55/35/10: {static_dd_val*100:.2f}% on {static_dd_date.date()}")
    print(f"Static 60/40: {sixty_dd_val*100:.2f}% on {sixty_dd_date.date()}")
    
    # ============================================================
    # VERDICT
    # ============================================================
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    # Key question: Does Static 55/35/10 survive 2008 within 25% DD?
    static_2008_dd = crisis_static["2008 Financial Crisis"]["max_dd"]
    static_survives_2008 = abs(static_2008_dd) <= 25.0
    
    print(f"\nKEY QUESTION: Does Static 55/35/10 survive 2008 within 25% DD?")
    print(f"  Static 55/35/10 Max DD in 2008: {static_2008_dd:.2f}%")
    print(f"  Answer: {'YES - survives' if static_survives_2008 else 'NO - breaches limit'}")
    
    if static_survives_2008:
        print(f"\n  CONCLUSION: MA200 filter is UNNECESSARY.")
        print(f"  Static allocation provides adequate DD protection.")
        print(f"  The whipsaw costs in 2022 outweigh the 2008 benefits.")
        print(f"  RECOMMENDATION: Use Static 55/35/10 (simpler, lower turnover).")
    else:
        print(f"\n  CONCLUSION: MA200 filter IS VALUABLE as DD insurance.")
        print(f"  Static allocation breaches the 25% DD limit in 2008.")
        print(f"  The 2022 whipsaw is the premium paid for this insurance.")
        print(f"  RECOMMENDATION: Use SmartPassive if DD constraint is binding.")
    
    # Secondary comparison
    print(f"\nSecondary Comparison (Full Period):")
    print(f"  SmartPassive Sharpe: {metrics_smart['sharpe']:.3f}, Max DD: {metrics_smart['max_drawdown']:.2f}%")
    print(f"  Static 55/35/10 Sharpe: {metrics_static['sharpe']:.3f}, Max DD: {metrics_static['max_drawdown']:.2f}%")
    
    # Save results
    summary = {
        "smart_sharpe": metrics_smart["sharpe"],
        "smart_max_dd": metrics_smart["max_drawdown"],
        "static_sharpe": metrics_static["sharpe"],
        "static_max_dd": metrics_static["max_drawdown"],
        "static_2008_dd": static_2008_dd,
        "static_survives_2008": static_survives_2008,
        "6040_sharpe": metrics_6040["sharpe"],
        "6040_max_dd": metrics_6040["max_drawdown"],
    }
    
    pd.DataFrame([summary]).to_csv("results/test8_ultimate_baseline_summary.csv", index=False)
    print(f"\nResults saved to results/test8_ultimate_baseline_summary.csv")
    
    return summary


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run_test8()