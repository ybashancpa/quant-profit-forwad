"""
TEST 6: ROBUSTNESS & STRESS TESTING
=====================================
Final validation before verdict. Three components:

1. COST STRESS TEST:
   Run strategy under 3 cost regimes: 0.05%, 0.10%, 0.20% per side.
   If strategy collapses at 0.20%, it's too fragile for $10k portfolio.

2. ADAPTIVE WALK-FORWARD ANALYSIS (WFA):
   - 5-year training window, re-select best config from grid
   - Apply to next 1 year (pure OOS)
   - Roll forward until 2026
   - Track config selection stability

3. SMART BENCHMARK:
   60/40 with MA200 filter on SPY.
   If this simple rule matches our DD protection, the complex
   strategy is not justified.

This is the FINAL TEST. If strategy survives, we have something real.
"""

import pandas as pd
import numpy as np
from data_loader import download_prices
from backtest_engine import backtest_from_weights, get_monthly_rebalance_dates, forward_fill_weights
from metrics import calculate_metrics
from config import TICKERS, SAFE_ASSET


# ============================================================
# PARAMETERS
# ============================================================
MOMENTUM_LOOKBACKS = [6, 9]  # months
MA_LOOKBACKS = [100, 150, 200]  # days
TOP_K = 3
COST_LEVELS = [0.0005, 0.001, 0.002]  # 0.05%, 0.10%, 0.20% per side

# WFA parameters
WFA_TRAIN_YEARS = 5
WFA_TEST_YEARS = 1

CRISIS_PERIODS = {
    "2008 Financial Crisis": ("2008-01-01", "2009-03-31"),
    "2020 Corona Crash": ("2020-02-01", "2020-04-30"),
    "2022 Inflation/Rates": ("2022-01-01", "2022-12-31"),
}


def generate_dual_momentum_weights(prices, lookback_months, ma_lookback, rebalance_dates, safe_asset=SAFE_ASSET):
    """Generate Dual Momentum portfolio weights."""
    lookback_days = lookback_months * 21
    
    trailing_returns = prices.pct_change(lookback_days)
    moving_avg = prices.rolling(window=ma_lookback, min_periods=ma_lookback).mean()
    
    sparse_weights = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    
    for date in rebalance_dates:
        if date not in trailing_returns.index:
            continue
        
        rets = trailing_returns.loc[date].dropna()
        if len(rets) < TOP_K:
            continue
        
        top_assets = rets.nlargest(TOP_K).index.tolist()
        weight_per_asset = 1.0 / TOP_K
        safe_weight = 0.0
        
        for asset in top_assets:
            if asset not in prices.columns:
                continue
            
            price_today = prices.loc[date, asset]
            ma_today = moving_avg.loc[date, asset]
            
            if pd.isna(ma_today):
                safe_weight += weight_per_asset
            elif price_today > ma_today:
                sparse_weights.loc[date, asset] = weight_per_asset
            else:
                safe_weight += weight_per_asset
        
        if safe_weight > 0 and safe_asset in prices.columns:
            sparse_weights.loc[date, safe_asset] = safe_weight
    
    daily_weights = forward_fill_weights(sparse_weights, prices.index)
    return daily_weights


def generate_smart_benchmark_weights(prices, rebalance_dates, ma_lookback=200):
    """
    Generate Smart Benchmark weights: 60/40 with MA200 filter on SPY.
    
    If SPY > MA200: 60% SPY, 40% IEF
    If SPY < MA200: 100% IEF (defensive)
    """
    moving_avg = prices["SPY"].rolling(window=ma_lookback, min_periods=ma_lookback).mean()
    
    sparse_weights = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    
    for date in rebalance_dates:
        if date not in prices.index:
            continue
        
        spy_price = prices.loc[date, "SPY"]
        spy_ma = moving_avg.loc[date]
        
        if pd.isna(spy_ma):
            # Not enough history, go defensive
            sparse_weights.loc[date, "IEF"] = 1.0
        elif spy_price > spy_ma:
            # Uptrend: 60/40
            sparse_weights.loc[date, "SPY"] = 0.6
            sparse_weights.loc[date, "IEF"] = 0.4
        else:
            # Downtrend: defensive
            sparse_weights.loc[date, "IEF"] = 1.0
    
    daily_weights = forward_fill_weights(sparse_weights, prices.index)
    return daily_weights


def run_cost_stress_test(prices_full):
    """Run cost stress test under different cost regimes."""
    print("\n" + "="*70)
    print("PART 1: COST STRESS TEST")
    print("="*70)
    
    # Use full period for stress test
    rebalance_dates = get_monthly_rebalance_dates(prices_full.index, freq="M")
    
    # Generate weights once (6M/MA150 - our winner)
    weights = generate_dual_momentum_weights(prices_full, 6, 150, rebalance_dates)
    
    results = []
    for cost in COST_LEVELS:
        result = backtest_from_weights(prices_full, weights, cost, f"Cost {cost*100:.2f}%")
        metrics = calculate_metrics(result["net_returns"], f"Cost {cost*100:.2f}%")
        
        results.append({
            "cost_per_side": cost * 100,
            "net_cagr": metrics["cagr"],
            "net_sharpe": metrics["sharpe"],
            "max_dd": metrics["max_drawdown"],
            "turnover": result["annual_turnover_one_way"],
            "total_cost_drag": result["total_cost_drag"] * 100,
        })
    
    results_df = pd.DataFrame(results)
    print("\nCost Stress Test Results (Full Period 2007-2026):")
    print(results_df.to_string(index=False))
    
    # Check if strategy survives 0.20% cost
    stress_result = results_df[results_df["cost_per_side"] == 0.2].iloc[0]
    survives_stress = stress_result["net_sharpe"] > 0 and stress_result["net_cagr"] > 0
    
    print(f"\nStress test (0.20%/side): Sharpe {stress_result['net_sharpe']:.3f}, CAGR {stress_result['net_cagr']:.2f}%")
    print(f"Verdict: {'SURVIVES' if survives_stress else 'COLLAPSES'} under 2x cost stress")
    
    return results_df, survives_stress


def run_adaptive_wfa(prices_full):
    """Run Adaptive Walk-Forward Analysis."""
    print("\n" + "="*70)
    print("PART 2: ADAPTIVE WALK-FORWARD ANALYSIS")
    print("="*70)
    print(f"\nTraining window: {WFA_TRAIN_YEARS} years")
    print(f"Test window: {WFA_TEST_YEARS} year")
    
    # Get all years
    all_years = sorted(prices_full.index.year.unique())
    start_year = all_years[0]
    end_year = all_years[-1]
    
    # Need at least WFA_TRAIN_YEARS + WFA_TEST_YEARS to start
    first_test_year = start_year + WFA_TRAIN_YEARS
    
    wfa_results = []
    config_selections = []
    
    for test_year in range(first_test_year, end_year + 1):
        train_start = test_year - WFA_TRAIN_YEARS
        train_end = test_year - 1
        
        # Get training data
        train_prices = prices_full.loc[f"{train_start}-01-01":f"{train_end}-12-31"]
        if len(train_prices) < 252 * 2:  # Need at least 2 years of data
            continue
        
        train_rebalance_dates = get_monthly_rebalance_dates(train_prices.index, freq="M")
        
        # Select best config on training data
        best_config = None
        best_sharpe = -np.inf
        
        for mom_lb in MOMENTUM_LOOKBACKS:
            for ma_lb in MA_LOOKBACKS:
                weights = generate_dual_momentum_weights(train_prices, mom_lb, ma_lb, train_rebalance_dates)
                result = backtest_from_weights(train_prices, weights, 0.001, f"{mom_lb}M/MA{ma_lb}")
                metrics = calculate_metrics(result["net_returns"], "train")
                
                if metrics["sharpe"] > best_sharpe:
                    best_sharpe = metrics["sharpe"]
                    best_config = (mom_lb, ma_lb)
        
        if best_config is None:
            continue
        
        # Apply best config to test year
        test_prices = prices_full.loc[f"{test_year}-01-01":f"{test_year}-12-31"]
        if len(test_prices) < 20:
            continue
        
        test_rebalance_dates = get_monthly_rebalance_dates(test_prices.index, freq="M")
        
        mom_lb, ma_lb = best_config
        weights_test = generate_dual_momentum_weights(test_prices, mom_lb, ma_lb, test_rebalance_dates)
        result_test = backtest_from_weights(test_prices, weights_test, 0.001, f"{mom_lb}M/MA{ma_lb}")
        metrics_test = calculate_metrics(result_test["net_returns"], "test")
        
        wfa_results.append({
            "test_year": test_year,
            "train_period": f"{train_start}-{train_end}",
            "selected_config": f"{mom_lb}M/MA{ma_lb}",
            "train_sharpe": best_sharpe,
            "oos_return": metrics_test["total_return"],
            "oos_sharpe": metrics_test["sharpe"],
            "oos_max_dd": metrics_test["max_drawdown"],
        })
        
        config_selections.append(f"{mom_lb}M/MA{ma_lb}")
    
    wfa_df = pd.DataFrame(wfa_results)
    
    print("\nWalk-Forward Results:")
    print(wfa_df.to_string(index=False))
    
    # Config selection stability
    print("\nConfig Selection Frequency:")
    config_counts = pd.Series(config_selections).value_counts()
    print(config_counts.to_string())
    
    # Calculate aggregate OOS performance
    if len(wfa_df) > 0:
        avg_oos_return = wfa_df["oos_return"].mean()
        avg_oos_sharpe = wfa_df["oos_sharpe"].mean()
        worst_oos_return = wfa_df["oos_return"].min()
        best_oos_return = wfa_df["oos_return"].max()
        
        print(f"\nAggregate OOS Performance:")
        print(f"  Average annual return: {avg_oos_return:.2f}%")
        print(f"  Average annual Sharpe: {avg_oos_sharpe:.3f}")
        print(f"  Worst year: {worst_oos_return:.2f}%")
        print(f"  Best year: {best_oos_return:.2f}%")
        
        # Count positive years
        positive_years = (wfa_df["oos_return"] > 0).sum()
        print(f"  Positive years: {positive_years}/{len(wfa_df)}")
    
    return wfa_df, config_counts


def run_smart_benchmark_comparison(prices_full):
    """Compare strategy to Smart Benchmark (60/40 + MA200)."""
    print("\n" + "="*70)
    print("PART 3: SMART BENCHMARK COMPARISON")
    print("="*70)
    
    rebalance_dates = get_monthly_rebalance_dates(prices_full.index, freq="M")
    
    # Our strategy (6M/MA150)
    weights_strategy = generate_dual_momentum_weights(prices_full, 6, 150, rebalance_dates)
    result_strategy = backtest_from_weights(prices_full, weights_strategy, 0.001, "DualMom")
    metrics_strategy = calculate_metrics(result_strategy["net_returns"], "DualMom")
    
    # Smart benchmark (60/40 + MA200)
    weights_benchmark = generate_smart_benchmark_weights(prices_full, rebalance_dates, 200)
    result_benchmark = backtest_from_weights(prices_full, weights_benchmark, 0.001, "Smart60/40")
    metrics_benchmark = calculate_metrics(result_benchmark["net_returns"], "Smart60/40")
    
    # Simple 60/40 (no filter)
    weights_simple = pd.DataFrame(0.0, index=rebalance_dates, columns=prices_full.columns)
    for date in rebalance_dates:
        weights_simple.loc[date, "SPY"] = 0.6
        weights_simple.loc[date, "IEF"] = 0.4
    weights_simple = forward_fill_weights(weights_simple, prices_full.index)
    result_simple = backtest_from_weights(prices_full, weights_simple, 0.001, "Simple60/40")
    metrics_simple = calculate_metrics(result_simple["net_returns"], "Simple60/40")
    
    print(f"\n{'Strategy':<25} {'CAGR':>8} {'Sharpe':>8} {'Max DD':>10} {'Turnover/yr':>12}")
    print("-"*70)
    print(f"{'DualMom 6M/MA150':<25} {metrics_strategy['cagr']:>7.2f}% {metrics_strategy['sharpe']:>8.3f} {metrics_strategy['max_drawdown']:>9.2f}% {result_strategy['annual_turnover_one_way']:>11.2f}x")
    print(f"{'Smart 60/40 (MA200)':<25} {metrics_benchmark['cagr']:>7.2f}% {metrics_benchmark['sharpe']:>8.3f} {metrics_benchmark['max_drawdown']:>9.2f}% {result_benchmark['annual_turnover_one_way']:>11.2f}x")
    print(f"{'Simple 60/40':<25} {metrics_simple['cagr']:>7.2f}% {metrics_simple['sharpe']:>8.3f} {metrics_simple['max_drawdown']:>9.2f}% {result_simple['annual_turnover_one_way']:>11.2f}x")
    
    # Crisis analysis
    print("\nCrisis Analysis:")
    
    def analyze_crisis(returns):
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
            results[crisis_name] = {"return": total_return * 100, "max_dd": drawdown.min() * 100}
        return results
    
    crisis_strategy = analyze_crisis(result_strategy["net_returns"])
    crisis_benchmark = analyze_crisis(result_benchmark["net_returns"])
    crisis_simple = analyze_crisis(result_simple["net_returns"])
    
    print(f"\n{'Crisis':<25} {'DualMom':<15} {'Smart60/40':<15} {'Simple60/40':<15}")
    print("-"*70)
    for crisis_name in CRISIS_PERIODS.keys():
        strat_ret = crisis_strategy[crisis_name]["return"]
        strat_dd = crisis_strategy[crisis_name]["max_dd"]
        bench_ret = crisis_benchmark[crisis_name]["return"]
        bench_dd = crisis_benchmark[crisis_name]["max_dd"]
        simple_ret = crisis_simple[crisis_name]["return"]
        simple_dd = crisis_simple[crisis_name]["max_dd"]
        
        print(f"\n{crisis_name}:")
        print(f"  {'Return':<12} {strat_ret:>8.2f}% {bench_ret:>12.2f}% {simple_ret:>12.2f}%")
        print(f"  {'Max DD':<12} {strat_dd:>8.2f}% {bench_dd:>12.2f}% {simple_dd:>12.2f}%")
    
    return metrics_strategy, metrics_benchmark, metrics_simple


def run_test6():
    """Run full robustness test."""
    print("\n" + "="*70)
    print("TEST 6: ROBUSTNESS & STRESS TESTING")
    print("="*70)
    
    prices_full = download_prices()
    
    # Part 1: Cost stress test
    cost_results, survives_cost = run_cost_stress_test(prices_full)
    
    # Part 2: Adaptive WFA
    wfa_results, config_counts = run_adaptive_wfa(prices_full)
    
    # Part 3: Smart benchmark comparison
    metrics_strategy, metrics_benchmark, metrics_simple = run_smart_benchmark_comparison(prices_full)
    
    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    print(f"\n1. Cost Stress Test: {'PASS' if survives_cost else 'FAIL'}")
    print(f"   Strategy {'survives' if survives_cost else 'collapses under'} 2x cost stress (0.20%/side)")
    
    if len(wfa_results) > 0:
        avg_oos_sharpe = wfa_results["oos_sharpe"].mean()
        positive_years = (wfa_results["oos_return"] > 0).sum()
        print(f"\n2. Walk-Forward Analysis:")
        print(f"   Average OOS Sharpe: {avg_oos_sharpe:.3f}")
        print(f"   Positive years: {positive_years}/{len(wfa_results)}")
        print(f"   Config stability: {config_counts.to_dict()}")
    
    print(f"\n3. Smart Benchmark Comparison:")
    print(f"   DualMom Sharpe: {metrics_strategy['sharpe']:.3f}, Max DD: {metrics_strategy['max_drawdown']:.2f}%")
    print(f"   Smart60/40 Sharpe: {metrics_benchmark['sharpe']:.3f}, Max DD: {metrics_benchmark['max_drawdown']:.2f}%")
    
    beats_smart = metrics_strategy["sharpe"] > metrics_benchmark["sharpe"]
    lower_dd = metrics_strategy["max_drawdown"] > metrics_benchmark["max_drawdown"]  # Less negative = better
    
    print(f"\n   DualMom vs Smart60/40:")
    print(f"     Sharpe: {'BEAT' if beats_smart else 'LOST'}")
    print(f"     Max DD: {'BETTER' if lower_dd else 'WORSE'}")
    
    # Save results
    summary = {
        "survives_cost_stress": survives_cost,
        "wfa_avg_oos_sharpe": wfa_results["oos_sharpe"].mean() if len(wfa_results) > 0 else None,
        "strategy_sharpe": metrics_strategy["sharpe"],
        "strategy_max_dd": metrics_strategy["max_drawdown"],
        "smart_benchmark_sharpe": metrics_benchmark["sharpe"],
        "smart_benchmark_max_dd": metrics_benchmark["max_drawdown"],
    }
    
    pd.DataFrame([summary]).to_csv("results/test6_robustness_summary.csv", index=False)
    print(f"\nResults saved to results/test6_robustness_summary.csv")
    
    return summary


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run_test6()