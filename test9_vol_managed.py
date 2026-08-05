"""
TEST 9: VOLATILITY-MANAGED EQUITY SLEEVE
==========================================
Hypothesis: Realized volatility clusters. Reducing equity exposure
when vol is high improves Sharpe and drawdown.

Rule (PRE-REGISTERED, NO FITTING):
- Base portfolio: 55% equity sleeve, 35% IEF, 10% GLD
- Equity sleeve: SPY_fraction = min(1, TARGET_VOL / realized_vol)
- realized_vol = annualized std of SPY returns over 60 trading days
- TARGET_VOL = 15% (primary)
- Rebalance: Monthly
- Signal computed at rebalance date, position effective next day

Comparisons:
1. Static 55/35/10 (passive baseline)
2. SmartPassive MA200 (existing DD insurance)
3. Vol-managed 60d/15% (primary candidate)
4. Static matched-exposure (diagnostic: same avg SPY exposure)

Sensitivity (report only, no selection):
- Window: 20d vs 60d
- Target vol: 12% vs 15% vs 18%
"""

import pandas as pd
import numpy as np
from data_loader import download_prices
from backtest_engine import backtest_from_weights, get_monthly_rebalance_dates, forward_fill_weights
from metrics import calculate_metrics
from config import COST_PER_SIDE


# ============================================================
# PRE-REGISTERED PARAMETERS (NO FITTING)
# ============================================================
PRIMARY_WINDOW = 60      # Trading days for vol calculation
PRIMARY_TARGET_VOL = 0.15  # 15% annualized target
BASE_EQUITY_WEIGHT = 0.55
BASE_IEF_WEIGHT = 0.35
BASE_GLD_WEIGHT = 0.10

# Sensitivity parameters (report only)
SENSITIVITY_WINDOWS = [20, 60]
SENSITIVITY_TARGET_VOLS = [0.12, 0.15, 0.18]

CRISIS_PERIODS = {
    "2008 Financial Crisis": ("2008-01-01", "2009-03-31"),
    "2020 Corona Crash": ("2020-02-01", "2020-04-30"),
    "2022 Inflation/Rates": ("2022-01-01", "2022-12-31"),
}


def generate_vol_managed_weights(prices, rebalance_dates, window=PRIMARY_WINDOW, target_vol=PRIMARY_TARGET_VOL):
    """
    Generate volatility-managed portfolio weights.
    
    Rule: SPY_fraction = min(1, target_vol / realized_vol)
    - realized_vol = annualized std of SPY returns over `window` days
    - Signal computed at rebalance date
    - Position effective next day (handled by backtest engine)
    """
    # Calculate realized volatility (annualized)
    spy_returns = prices["SPY"].pct_change()
    realized_vol = spy_returns.rolling(window=window, min_periods=window).std() * np.sqrt(252)
    
    sparse_weights = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    
    for date in rebalance_dates:
        if date not in prices.index:
            continue
        
        vol = realized_vol.loc[date]
        
        if pd.isna(vol) or vol <= 0:
            # Not enough history, use full equity exposure
            spy_fraction = 1.0
        else:
            # Vol-managed fraction: reduce exposure when vol is high
            spy_fraction = min(1.0, target_vol / vol)
        
        # Equity sleeve allocation
        spy_weight = BASE_EQUITY_WEIGHT * spy_fraction
        shy_weight = BASE_EQUITY_WEIGHT - spy_weight  # Remainder to SHY
        
        sparse_weights.loc[date, "SPY"] = spy_weight
        sparse_weights.loc[date, "SHY"] = shy_weight
        sparse_weights.loc[date, "IEF"] = BASE_IEF_WEIGHT
        sparse_weights.loc[date, "GLD"] = BASE_GLD_WEIGHT
    
    daily_weights = forward_fill_weights(sparse_weights, prices.index)
    return daily_weights


def generate_static_weights(prices, rebalance_dates, allocation):
    """Generate static portfolio weights (monthly rebalance only)."""
    sparse_weights = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    
    for date in rebalance_dates:
        for asset, weight in allocation.items():
            if asset in prices.columns:
                sparse_weights.loc[date, asset] = weight
    
    daily_weights = forward_fill_weights(sparse_weights, prices.index)
    return daily_weights


def generate_smart_passive_weights(prices, rebalance_dates, ma_lookback=200):
    """Generate SmartPassive weights (MA200 filter)."""
    risk_on = {"SPY": 0.55, "IEF": 0.35, "GLD": 0.10}
    risk_off = {"SHY": 0.55, "IEF": 0.35, "GLD": 0.10}
    
    moving_avg = prices["SPY"].rolling(window=ma_lookback, min_periods=ma_lookback).mean()
    
    sparse_weights = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    
    for date in rebalance_dates:
        if date not in prices.index:
            continue
        
        spy_price = prices.loc[date, "SPY"]
        spy_ma = moving_avg.loc[date]
        
        if pd.isna(spy_ma) or spy_price <= spy_ma:
            weights = risk_off
        else:
            weights = risk_on
        
        for asset, weight in weights.items():
            if asset in prices.columns:
                sparse_weights.loc[date, asset] = weight
    
    daily_weights = forward_fill_weights(sparse_weights, prices.index)
    return daily_weights


def generate_matched_exposure_weights(prices, rebalance_dates, avg_spy_exposure):
    """
    Generate static portfolio with same average SPY exposure as vol-managed.
    This is a diagnostic to check if vol-managed adds value beyond
    simply having lower equity exposure.
    """
    # Scale the base allocation to match average SPY exposure
    scale = avg_spy_exposure / BASE_EQUITY_WEIGHT
    
    allocation = {
        "SPY": BASE_EQUITY_WEIGHT * scale,
        "SHY": BASE_EQUITY_WEIGHT * (1 - scale),
        "IEF": BASE_IEF_WEIGHT,
        "GLD": BASE_GLD_WEIGHT,
    }
    
    return generate_static_weights(prices, rebalance_dates, allocation)


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


def calculate_avg_spy_exposure(weights):
    """Calculate average SPY exposure over the period."""
    return weights["SPY"].mean()


def run_test9():
    """Run Test 9: Volatility-Managed Equity Sleeve."""
    print("\n" + "="*70)
    print("TEST 9: VOLATILITY-MANAGED EQUITY SLEEVE")
    print("="*70)
    print(f"\nPre-registered rule:")
    print(f"  Base: 55% equity sleeve, 35% IEF, 10% GLD")
    print(f"  SPY_fraction = min(1, {PRIMARY_TARGET_VOL*100:.0f}% / realized_vol)")
    print(f"  Vol window: {PRIMARY_WINDOW} trading days")
    print(f"  Rebalance: Monthly")
    print(f"  Cost: {COST_PER_SIDE*100:.2f}% per side")
    
    # Load data
    prices_full = download_prices()
    rebalance_dates = get_monthly_rebalance_dates(prices_full.index, freq="M")
    
    # ============================================================
    # PRIMARY COMPARISON
    # ============================================================
    print("\n" + "-"*70)
    print("PRIMARY COMPARISON (Full Period 2007-2026)")
    print("-"*70)
    
    # 1. Static 55/35/10
    static_alloc = {"SPY": 0.55, "IEF": 0.35, "GLD": 0.10}
    weights_static = generate_static_weights(prices_full, rebalance_dates, static_alloc)
    result_static = backtest_from_weights(prices_full, weights_static, COST_PER_SIDE, "Static55/35/10")
    metrics_static = calculate_metrics(result_static["net_returns"], "Static55/35/10")
    
    # 2. SmartPassive MA200
    weights_smart = generate_smart_passive_weights(prices_full, rebalance_dates)
    result_smart = backtest_from_weights(prices_full, weights_smart, COST_PER_SIDE, "SmartPassive")
    metrics_smart = calculate_metrics(result_smart["net_returns"], "SmartPassive")
    
    # 3. Vol-managed (primary)
    weights_vol = generate_vol_managed_weights(prices_full, rebalance_dates)
    result_vol = backtest_from_weights(prices_full, weights_vol, COST_PER_SIDE, "VolManaged")
    metrics_vol = calculate_metrics(result_vol["net_returns"], "VolManaged")
    
    # 4. Static matched-exposure (diagnostic)
    avg_spy_exposure = calculate_avg_spy_exposure(result_vol["weights"])
    weights_matched = generate_matched_exposure_weights(prices_full, rebalance_dates, avg_spy_exposure)
    result_matched = backtest_from_weights(prices_full, weights_matched, COST_PER_SIDE, "MatchedExposure")
    metrics_matched = calculate_metrics(result_matched["net_returns"], "MatchedExposure")
    
    print(f"\n{'Strategy':<25} {'CAGR':>8} {'Sharpe':>8} {'Max DD':>10} {'Turnover':>10} {'Avg SPY%':>10}")
    print("-"*75)
    print(f"{'Static 55/35/10':<25} {metrics_static['cagr']:>7.2f}% {metrics_static['sharpe']:>8.3f} {metrics_static['max_drawdown']:>9.2f}% {result_static['annual_turnover_one_way']:>9.2f}x {55.0:>9.1f}%")
    print(f"{'SmartPassive MA200':<25} {metrics_smart['cagr']:>7.2f}% {metrics_smart['sharpe']:>8.3f} {metrics_smart['max_drawdown']:>9.2f}% {result_smart['annual_turnover_one_way']:>9.2f}x {calculate_avg_spy_exposure(result_smart['weights'])*100:>9.1f}%")
    print(f"{'Vol-Managed 60d/15%':<25} {metrics_vol['cagr']:>7.2f}% {metrics_vol['sharpe']:>8.3f} {metrics_vol['max_drawdown']:>9.2f}% {result_vol['annual_turnover_one_way']:>9.2f}x {avg_spy_exposure*100:>9.1f}%")
    print(f"{'Matched-Exposure Static':<25} {metrics_matched['cagr']:>7.2f}% {metrics_matched['sharpe']:>8.3f} {metrics_matched['max_drawdown']:>9.2f}% {result_matched['annual_turnover_one_way']:>9.2f}x {avg_spy_exposure*100:>9.1f}%")
    
    # ============================================================
    # CRISIS ANALYSIS
    # ============================================================
    print("\n" + "-"*70)
    print("CRISIS ANALYSIS")
    print("-"*70)
    
    crisis_static = analyze_crisis_periods(result_static["net_returns"])
    crisis_smart = analyze_crisis_periods(result_smart["net_returns"])
    crisis_vol = analyze_crisis_periods(result_vol["net_returns"])
    crisis_matched = analyze_crisis_periods(result_matched["net_returns"])
    
    print(f"\n{'Crisis':<25} {'Static':<12} {'SmartPassive':<14} {'VolManaged':<12} {'Matched':<12}")
    print("-"*75)
    for crisis_name in CRISIS_PERIODS.keys():
        static_ret = crisis_static[crisis_name]["return"]
        smart_ret = crisis_smart[crisis_name]["return"]
        vol_ret = crisis_vol[crisis_name]["return"]
        matched_ret = crisis_matched[crisis_name]["return"]
        
        print(f"\n{crisis_name}:")
        print(f"  {'Return':<12} {static_ret:>8.2f}% {smart_ret:>10.2f}% {vol_ret:>8.2f}% {matched_ret:>8.2f}%")
    
    # ============================================================
    # SENSITIVITY ANALYSIS (Report only, no selection)
    # ============================================================
    print("\n" + "-"*70)
    print("SENSITIVITY ANALYSIS (Report only, no selection)")
    print("-"*70)
    
    print(f"\n{'Window':<10} {'Target Vol':<12} {'CAGR':>8} {'Sharpe':>8} {'Max DD':>10} {'Turnover':>10}")
    print("-"*60)
    
    sensitivity_results = []
    for window in SENSITIVITY_WINDOWS:
        for target_vol in SENSITIVITY_TARGET_VOLS:
            weights_sens = generate_vol_managed_weights(prices_full, rebalance_dates, window, target_vol)
            result_sens = backtest_from_weights(prices_full, weights_sens, COST_PER_SIDE, f"Vol_{window}d_{target_vol}")
            metrics_sens = calculate_metrics(result_sens["net_returns"], f"Vol_{window}d_{target_vol}")
            
            is_primary = (window == PRIMARY_WINDOW and target_vol == PRIMARY_TARGET_VOL)
            marker = " <-- PRIMARY" if is_primary else ""
            
            print(f"{window:<10} {target_vol*100:<11.0f}% {metrics_sens['cagr']:>7.2f}% {metrics_sens['sharpe']:>8.3f} {metrics_sens['max_drawdown']:>9.2f}% {result_sens['annual_turnover_one_way']:>9.2f}x{marker}")
            
            sensitivity_results.append({
                "window": window,
                "target_vol": target_vol,
                "cagr": metrics_sens["cagr"],
                "sharpe": metrics_sens["sharpe"],
                "max_dd": metrics_sens["max_drawdown"],
                "turnover": result_sens["annual_turnover_one_way"],
            })
    
    # ============================================================
    # VERDICT
    # ============================================================
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    # Key comparisons
    beats_static = metrics_vol["sharpe"] > metrics_static["sharpe"]
    beats_smart = metrics_vol["sharpe"] > metrics_smart["sharpe"]
    beats_matched = metrics_vol["sharpe"] > metrics_matched["sharpe"]
    passes_dd = metrics_vol["passes_dd_limit"]
    
    print(f"\nVol-Managed 60d/15% Performance:")
    print(f"  CAGR: {metrics_vol['cagr']:.2f}%")
    print(f"  Sharpe: {metrics_vol['sharpe']:.3f}")
    print(f"  Max DD: {metrics_vol['max_drawdown']:.2f}%")
    print(f"  Turnover: {result_vol['annual_turnover_one_way']:.2f}x/year")
    print(f"  Avg SPY Exposure: {avg_spy_exposure*100:.1f}%")
    
    print(f"\nKey Comparisons:")
    print(f"  vs Static 55/35/10: Sharpe {metrics_vol['sharpe']:.3f} vs {metrics_static['sharpe']:.3f} -> {'BEAT' if beats_static else 'LOST'}")
    print(f"  vs SmartPassive MA200: Sharpe {metrics_vol['sharpe']:.3f} vs {metrics_smart['sharpe']:.3f} -> {'BEAT' if beats_smart else 'LOST'}")
    print(f"  vs Matched-Exposure: Sharpe {metrics_vol['sharpe']:.3f} vs {metrics_matched['sharpe']:.3f} -> {'BEAT' if beats_matched else 'LOST'}")
    print(f"  Passes 25% DD limit: {'YES' if passes_dd else 'NO'}")
    
    # Final verdict
    print(f"\n" + "-"*70)
    if passes_dd and beats_static and beats_matched:
        if beats_smart:
            print("VERDICT: PROMOTE to paper trading")
            print("Vol-managed beats all benchmarks including SmartPassive.")
        else:
            print("VERDICT: CONTINUE evaluation")
            print("Vol-managed beats static but not SmartPassive.")
            print("May be complementary DD insurance, not replacement.")
    elif passes_dd and beats_static and not beats_matched:
        print("VERDICT: REJECT as edge")
        print("Vol-managed only beats static because of lower equity exposure.")
        print("No timing skill demonstrated.")
    else:
        print("VERDICT: REJECT")
        print("Vol-managed fails key criteria.")
    print("-"*70)
    
    # Save results
    summary = {
        "vol_managed_cagr": metrics_vol["cagr"],
        "vol_managed_sharpe": metrics_vol["sharpe"],
        "vol_managed_max_dd": metrics_vol["max_drawdown"],
        "vol_managed_turnover": result_vol["annual_turnover_one_way"],
        "vol_managed_avg_spy_exposure": avg_spy_exposure,
        "static_sharpe": metrics_static["sharpe"],
        "smart_sharpe": metrics_smart["sharpe"],
        "matched_sharpe": metrics_matched["sharpe"],
        "beats_static": beats_static,
        "beats_smart": beats_smart,
        "beats_matched": beats_matched,
        "passes_dd": passes_dd,
    }
    
    pd.DataFrame([summary]).to_csv("results/test9_vol_managed_summary.csv", index=False)
    pd.DataFrame(sensitivity_results).to_csv("results/test9_sensitivity.csv", index=False)
    print(f"\nResults saved to results/test9_vol_managed_summary.csv")
    
    return summary


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run_test9()