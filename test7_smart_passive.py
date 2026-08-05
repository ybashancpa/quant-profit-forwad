"""
TEST 7: SMART PASSIVE PORTFOLIO VALIDATION
============================================
Final validation of the recommended portfolio.

Portfolio Design:
- Risk-On (SPY > MA200): 55% SPY, 35% IEF, 10% GLD
- Risk-Off (SPY < MA200): 55% SHY, 35% IEF, 10% GLD
- Rebalance: Monthly

Key Design Principle:
- Only the equity sleeve (SPY) is actively managed via MA200 filter
- IEF and GLD remain passive anchors (simplicity over complexity)
- The 10% GLD provides partial hedge for 2022-type scenarios
- Residual IEF exposure in Risk-Off is the accepted trade-off

This test validates the recommendation with the same rigor
applied to all other strategies.
"""

import pandas as pd
import numpy as np
from data_loader import download_prices
from backtest_engine import backtest_from_weights, get_monthly_rebalance_dates, forward_fill_weights
from metrics import calculate_metrics
from config import COST_PER_SIDE


# ============================================================
# PORTFOLIO DEFINITION
# ============================================================
RISK_ON_WEIGHTS = {"SPY": 0.55, "IEF": 0.35, "GLD": 0.10}
RISK_OFF_WEIGHTS = {"SHY": 0.55, "IEF": 0.35, "GLD": 0.10}
MA_LOOKBACK = 200

CRISIS_PERIODS = {
    "2008 Financial Crisis": ("2008-01-01", "2009-03-31"),
    "2020 Corona Crash": ("2020-02-01", "2020-04-30"),
    "2022 Inflation/Rates": ("2022-01-01", "2022-12-31"),
}


def generate_smart_passive_weights(prices, rebalance_dates):
    """
    Generate Smart Passive portfolio weights.
    
    Risk-On (SPY > MA200): 55% SPY, 35% IEF, 10% GLD
    Risk-Off (SPY < MA200): 55% SHY, 35% IEF, 10% GLD
    """
    moving_avg = prices["SPY"].rolling(window=MA_LOOKBACK, min_periods=MA_LOOKBACK).mean()
    
    sparse_weights = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    
    for date in rebalance_dates:
        if date not in prices.index:
            continue
        
        spy_price = prices.loc[date, "SPY"]
        spy_ma = moving_avg.loc[date]
        
        if pd.isna(spy_ma):
            # Not enough history, go defensive
            for asset, weight in RISK_OFF_WEIGHTS.items():
                if asset in prices.columns:
                    sparse_weights.loc[date, asset] = weight
        elif spy_price > spy_ma:
            # Risk-On
            for asset, weight in RISK_ON_WEIGHTS.items():
                if asset in prices.columns:
                    sparse_weights.loc[date, asset] = weight
        else:
            # Risk-Off
            for asset, weight in RISK_OFF_WEIGHTS.items():
                if asset in prices.columns:
                    sparse_weights.loc[date, asset] = weight
    
    daily_weights = forward_fill_weights(sparse_weights, prices.index)
    return daily_weights


def generate_benchmark_weights(prices, rebalance_dates, mode="60_40"):
    """Generate benchmark weights."""
    sparse_weights = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    
    for date in rebalance_dates:
        if mode == "spy_100":
            sparse_weights.loc[date, "SPY"] = 1.0
        elif mode == "60_40":
            sparse_weights.loc[date, "SPY"] = 0.6
            sparse_weights.loc[date, "IEF"] = 0.4
        elif mode == "smart_60_40":
            # 60/40 with MA200 filter (from Test 6)
            moving_avg = prices["SPY"].rolling(window=MA_LOOKBACK, min_periods=MA_LOOKBACK).mean()
            spy_price = prices.loc[date, "SPY"]
            spy_ma = moving_avg.loc[date]
            
            if pd.isna(spy_ma) or spy_price <= spy_ma:
                sparse_weights.loc[date, "IEF"] = 1.0
            else:
                sparse_weights.loc[date, "SPY"] = 0.6
                sparse_weights.loc[date, "IEF"] = 0.4
    
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


def run_test7():
    """Run Test 7: Smart Passive Portfolio Validation."""
    print("\n" + "="*70)
    print("TEST 7: SMART PASSIVE PORTFOLIO VALIDATION")
    print("="*70)
    print(f"\nPortfolio Design:")
    print(f"  Risk-On (SPY > MA200): 55% SPY, 35% IEF, 10% GLD")
    print(f"  Risk-Off (SPY < MA200): 55% SHY, 35% IEF, 10% GLD")
    print(f"  MA Lookback: {MA_LOOKBACK} days")
    print(f"  Cost: {COST_PER_SIDE*100:.2f}% per side")
    
    # Load data
    prices_full = download_prices()
    rebalance_dates = get_monthly_rebalance_dates(prices_full.index, freq="M")
    
    # ============================================================
    # FULL PERIOD PERFORMANCE
    # ============================================================
    print("\n" + "-"*70)
    print("FULL PERIOD PERFORMANCE (2007-2026)")
    print("-"*70)
    
    # Smart Passive
    weights_smart = generate_smart_passive_weights(prices_full, rebalance_dates)
    result_smart = backtest_from_weights(prices_full, weights_smart, COST_PER_SIDE, "SmartPassive")
    metrics_smart = calculate_metrics(result_smart["net_returns"], "SmartPassive")
    
    # Benchmarks
    weights_spy = generate_benchmark_weights(prices_full, rebalance_dates, "spy_100")
    result_spy = backtest_from_weights(prices_full, weights_spy, COST_PER_SIDE, "SPY")
    metrics_spy = calculate_metrics(result_spy["net_returns"], "SPY")
    
    weights_6040 = generate_benchmark_weights(prices_full, rebalance_dates, "60_40")
    result_6040 = backtest_from_weights(prices_full, weights_6040, COST_PER_SIDE, "60/40")
    metrics_6040 = calculate_metrics(result_6040["net_returns"], "60/40")
    
    weights_smart6040 = generate_benchmark_weights(prices_full, rebalance_dates, "smart_60_40")
    result_smart6040 = backtest_from_weights(prices_full, weights_smart6040, COST_PER_SIDE, "Smart60/40")
    metrics_smart6040 = calculate_metrics(result_smart6040["net_returns"], "Smart60/40")
    
    print(f"\n{'Strategy':<25} {'CAGR':>8} {'Sharpe':>8} {'Max DD':>10} {'Turnover/yr':>12}")
    print("-"*70)
    print(f"{'SmartPassive 55/35/10':<25} {metrics_smart['cagr']:>7.2f}% {metrics_smart['sharpe']:>8.3f} {metrics_smart['max_drawdown']:>9.2f}% {result_smart['annual_turnover_one_way']:>11.2f}x")
    print(f"{'Smart 60/40 (MA200)':<25} {metrics_smart6040['cagr']:>7.2f}% {metrics_smart6040['sharpe']:>8.3f} {metrics_smart6040['max_drawdown']:>9.2f}% {result_smart6040['annual_turnover_one_way']:>11.2f}x")
    print(f"{'Simple 60/40':<25} {metrics_6040['cagr']:>7.2f}% {metrics_6040['sharpe']:>8.3f} {metrics_6040['max_drawdown']:>9.2f}% {result_6040['annual_turnover_one_way']:>11.2f}x")
    print(f"{'SPY (B&H)':<25} {metrics_spy['cagr']:>7.2f}% {metrics_spy['sharpe']:>8.3f} {metrics_spy['max_drawdown']:>9.2f}% {result_spy['annual_turnover_one_way']:>11.2f}x")
    
    # ============================================================
    # CRISIS ANALYSIS
    # ============================================================
    print("\n" + "-"*70)
    print("CRISIS ANALYSIS")
    print("-"*70)
    
    crisis_smart = analyze_crisis_periods(result_smart["net_returns"])
    crisis_smart6040 = analyze_crisis_periods(result_smart6040["net_returns"])
    crisis_6040 = analyze_crisis_periods(result_6040["net_returns"])
    crisis_spy = analyze_crisis_periods(result_spy["net_returns"])
    
    print(f"\n{'Crisis':<25} {'SmartPassive':<15} {'Smart60/40':<15} {'Simple60/40':<15} {'SPY':<15}")
    print("-"*85)
    for crisis_name in CRISIS_PERIODS.keys():
        smart_ret = crisis_smart[crisis_name]["return"]
        smart_dd = crisis_smart[crisis_name]["max_dd"]
        smart6040_ret = crisis_smart6040[crisis_name]["return"]
        smart6040_dd = crisis_smart6040[crisis_name]["max_dd"]
        sixty_ret = crisis_6040[crisis_name]["return"]
        sixty_dd = crisis_6040[crisis_name]["max_dd"]
        spy_ret = crisis_spy[crisis_name]["return"]
        spy_dd = crisis_spy[crisis_name]["max_dd"]
        
        print(f"\n{crisis_name}:")
        print(f"  {'Return':<12} {smart_ret:>8.2f}% {smart6040_ret:>12.2f}% {sixty_ret:>12.2f}% {spy_ret:>12.2f}%")
        print(f"  {'Max DD':<12} {smart_dd:>8.2f}% {smart6040_dd:>12.2f}% {sixty_dd:>12.2f}% {spy_dd:>12.2f}%")
    
    # ============================================================
    # 2022 DEEP DIVE (IEF Residual Exposure)
    # ============================================================
    print("\n" + "-"*70)
    print("2022 DEEP DIVE: IEF Residual Exposure Cost")
    print("-"*70)
    
    # Calculate IEF performance in 2022
    ief_2022 = prices_full.loc["2022-01-01":"2022-12-31", "IEF"]
    ief_2022_return = (ief_2022.iloc[-1] / ief_2022.iloc[0] - 1) * 100
    
    print(f"\nIEF performance in 2022: {ief_2022_return:.2f}%")
    print(f"SmartPassive IEF allocation: 35% (constant)")
    print(f"Estimated IEF drag on SmartPassive in 2022: {ief_2022_return * 0.35:.2f}%")
    
    # ============================================================
    # VERDICT
    # ============================================================
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    print(f"\nSmartPassive 55/35/10 Performance:")
    print(f"  CAGR: {metrics_smart['cagr']:.2f}%")
    print(f"  Sharpe: {metrics_smart['sharpe']:.3f}")
    print(f"  Max DD: {metrics_smart['max_drawdown']:.2f}%")
    print(f"  Turnover: {result_smart['annual_turnover_one_way']:.2f}x/year")
    print(f"  Passes 25% DD limit: {'YES' if metrics_smart['passes_dd_limit'] else 'NO'}")
    
    # Compare to Smart 60/40
    beats_smart6040 = metrics_smart["sharpe"] > metrics_smart6040["sharpe"]
    lower_dd = metrics_smart["max_drawdown"] > metrics_smart6040["max_drawdown"]
    
    print(f"\nvs Smart 60/40 (MA200):")
    print(f"  Sharpe: {metrics_smart['sharpe']:.3f} vs {metrics_smart6040['sharpe']:.3f} -> {'BEAT' if beats_smart6040 else 'LOST'}")
    print(f"  Max DD: {metrics_smart['max_drawdown']:.2f}% vs {metrics_smart6040['max_drawdown']:.2f}% -> {'BETTER' if lower_dd else 'WORSE'}")
    
    # Save results
    summary = {
        "smart_passive_cagr": metrics_smart["cagr"],
        "smart_passive_sharpe": metrics_smart["sharpe"],
        "smart_passive_max_dd": metrics_smart["max_drawdown"],
        "smart_passive_turnover": result_smart["annual_turnover_one_way"],
        "smart_6040_sharpe": metrics_smart6040["sharpe"],
        "simple_6040_sharpe": metrics_6040["sharpe"],
        "spy_sharpe": metrics_spy["sharpe"],
    }
    
    pd.DataFrame([summary]).to_csv("results/test7_smart_passive_summary.csv", index=False)
    print(f"\nResults saved to results/test7_smart_passive_summary.csv")
    
    return summary


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run_test7()