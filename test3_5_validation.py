"""
TEST 3.5: RIGOROUS VALIDATION
==============================
Protocol:
1. Run all Dual Momentum configs on IN-SAMPLE ONLY (2007-2017)
2. Select THE winner based on IS net Sharpe
3. Lock it in - NO further parameter changes
4. Run winner ONCE on OUT-OF-SAMPLE (2018-2026)
5. Compare against passive benchmarks (SPY, 60/40)
6. Crisis analysis: 2008, 2020, 2022

This is the TRUE test of whether we have an edge.
"""

import pandas as pd
import numpy as np
from data_loader import download_prices
from backtest_engine import backtest_from_weights, get_monthly_rebalance_dates, forward_fill_weights
from metrics import calculate_metrics
from config import COST_PER_SIDE, TICKERS, SAFE_ASSET, TS_MOM_MA_LOOKBACKS


# ============================================================
# PERIOD DEFINITIONS
# ============================================================
IS_START = "2007-01-01"
IS_END = "2017-12-31"
OOS_START = "2018-01-01"
OOS_END = None  # To present

# Crisis periods
CRISIS_PERIODS = {
    "2008 Financial Crisis": ("2008-01-01", "2009-03-31"),
    "2020 Corona Crash": ("2020-02-01", "2020-04-30"),
    "2022 Inflation/Rates": ("2022-01-01", "2022-12-31"),
}


def generate_dual_momentum_weights(prices, lookback_months, top_k, ma_lookback, rebalance_dates, safe_asset=SAFE_ASSET):
    """Generate Dual Momentum portfolio weights (same as Test 3)."""
    lookback_days = lookback_months * 21
    
    trailing_returns = prices.pct_change(lookback_days)
    moving_avg = prices.rolling(window=ma_lookback, min_periods=ma_lookback).mean()
    
    sparse_weights = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    
    for date in rebalance_dates:
        if date not in trailing_returns.index:
            continue
        
        rets = trailing_returns.loc[date].dropna()
        if len(rets) < top_k:
            continue
        
        top_assets = rets.nlargest(top_k).index.tolist()
        weight_per_asset = 1.0 / top_k
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


def generate_benchmark_weights(prices, rebalance_dates, mode="spy_100"):
    """
    Generate benchmark weights.
    
    Args:
        mode: "spy_100" for 100% SPY, "60_40" for 60% SPY / 40% IEF
    """
    sparse_weights = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    
    for date in rebalance_dates:
        if mode == "spy_100":
            sparse_weights.loc[date, "SPY"] = 1.0
        elif mode == "60_40":
            sparse_weights.loc[date, "SPY"] = 0.6
            sparse_weights.loc[date, "IEF"] = 0.4
    
    daily_weights = forward_fill_weights(sparse_weights, prices.index)
    return daily_weights


def analyze_crisis_periods(returns, name):
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


def run_validation():
    """Run the full validation protocol."""
    print("\n" + "="*70)
    print("TEST 3.5: RIGOROUS VALIDATION (IS/OOS + Benchmarks + Crisis)")
    print("="*70)
    
    # Load full data
    prices_full = download_prices()
    
    # ============================================================
    # STEP 1: IN-SAMPLE (2007-2017) - Select winner
    # ============================================================
    print("\n" + "-"*70)
    print("STEP 1: IN-SAMPLE (2007-2017) - Parameter Selection")
    print("-"*70)
    
    prices_is = prices_full.loc[IS_START:IS_END]
    rebalance_dates_is = get_monthly_rebalance_dates(prices_is.index, freq="M")
    
    # Test all configurations on IS only
    momentum_lookbacks = [6, 9]
    top_k = 3
    is_results = []
    
    for mom_lb in momentum_lookbacks:
        for ma_lb in TS_MOM_MA_LOOKBACKS:
            name = f"DualMom {mom_lb}M/MA{ma_lb}"
            
            weights = generate_dual_momentum_weights(
                prices_is, mom_lb, top_k, ma_lb, rebalance_dates_is
            )
            
            result = backtest_from_weights(prices_is, weights, COST_PER_SIDE, name)
            net_metrics = calculate_metrics(result["net_returns"], name)
            
            is_results.append({
                "name": name,
                "mom_lb": mom_lb,
                "ma_lb": ma_lb,
                "net_sharpe": net_metrics["sharpe"],
                "net_cagr": net_metrics["cagr"],
                "max_dd": net_metrics["max_drawdown"],
            })
    
    is_df = pd.DataFrame(is_results)
    print("\nIn-Sample Results (2007-2017):")
    print(is_df.to_string(index=False))
    
    # SELECT THE WINNER (based on IS net Sharpe)
    winner = is_df.loc[is_df["net_sharpe"].idxmax()]
    winner_name = winner["name"]
    winner_mom_lb = int(winner["mom_lb"])
    winner_ma_lb = int(winner["ma_lb"])
    
    print(f"\n{'='*70}")
    print(f"WINNER SELECTED (IS only): {winner_name}")
    print(f"IS Net Sharpe: {winner['net_sharpe']:.3f}, CAGR: {winner['net_cagr']:.2f}%, Max DD: {winner['max_dd']:.2f}%")
    print(f"{'='*70}")
    
    # ============================================================
    # STEP 2: OUT-OF-SAMPLE (2018-2026) - Single look
    # ============================================================
    print("\n" + "-"*70)
    print("STEP 2: OUT-OF-SAMPLE (2018-2026) - Single Look, No Changes")
    print("-"*70)
    
    prices_oos = prices_full.loc[OOS_START:]
    rebalance_dates_oos = get_monthly_rebalance_dates(prices_oos.index, freq="M")
    
    # Run winner on OOS
    weights_oos = generate_dual_momentum_weights(
        prices_oos, winner_mom_lb, top_k, winner_ma_lb, rebalance_dates_oos
    )
    result_oos = backtest_from_weights(prices_oos, weights_oos, COST_PER_SIDE, winner_name)
    oos_metrics = calculate_metrics(result_oos["net_returns"], f"{winner_name} OOS")
    
    print(f"\nOOS Results for {winner_name}:")
    print(f"  Net CAGR: {oos_metrics['cagr']:.2f}%")
    print(f"  Net Sharpe: {oos_metrics['sharpe']:.3f}")
    print(f"  Max DD: {oos_metrics['max_drawdown']:.2f}%")
    print(f"  Passes 25% DD: {'YES' if oos_metrics['passes_dd_limit'] else 'NO'}")
    
    # ============================================================
    # STEP 3: BENCHMARKS
    # ============================================================
    print("\n" + "-"*70)
    print("STEP 3: PASSIVE BENCHMARKS")
    print("-"*70)
    
    # IS benchmarks
    weights_spy_is = generate_benchmark_weights(prices_is, rebalance_dates_is, "spy_100")
    weights_6040_is = generate_benchmark_weights(prices_is, rebalance_dates_is, "60_40")
    
    result_spy_is = backtest_from_weights(prices_is, weights_spy_is, COST_PER_SIDE, "SPY IS")
    result_6040_is = backtest_from_weights(prices_is, weights_6040_is, COST_PER_SIDE, "60/40 IS")
    
    spy_is_metrics = calculate_metrics(result_spy_is["net_returns"], "SPY IS")
    sixty40_is_metrics = calculate_metrics(result_6040_is["net_returns"], "60/40 IS")
    
    # OOS benchmarks
    weights_spy_oos = generate_benchmark_weights(prices_oos, rebalance_dates_oos, "spy_100")
    weights_6040_oos = generate_benchmark_weights(prices_oos, rebalance_dates_oos, "60_40")
    
    result_spy_oos = backtest_from_weights(prices_oos, weights_spy_oos, COST_PER_SIDE, "SPY OOS")
    result_6040_oos = backtest_from_weights(prices_oos, weights_6040_oos, COST_PER_SIDE, "60/40 OOS")
    
    spy_oos_metrics = calculate_metrics(result_spy_oos["net_returns"], "SPY OOS")
    sixty40_oos_metrics = calculate_metrics(result_6040_oos["net_returns"], "60/40 OOS")
    
    print(f"\n{'Strategy':<25} {'Period':<8} {'CAGR':>8} {'Sharpe':>8} {'Max DD':>10}")
    print("-"*65)
    print(f"{'SPY (B&H)':<25} {'IS':<8} {spy_is_metrics['cagr']:>7.2f}% {spy_is_metrics['sharpe']:>8.3f} {spy_is_metrics['max_drawdown']:>9.2f}%")
    print(f"{'60/40':<25} {'IS':<8} {sixty40_is_metrics['cagr']:>7.2f}% {sixty40_is_metrics['sharpe']:>8.3f} {sixty40_is_metrics['max_drawdown']:>9.2f}%")
    print(f"{winner_name:<25} {'IS':<8} {winner['net_cagr']:>7.2f}% {winner['net_sharpe']:>8.3f} {winner['max_dd']:>9.2f}%")
    print("-"*65)
    print(f"{'SPY (B&H)':<25} {'OOS':<8} {spy_oos_metrics['cagr']:>7.2f}% {spy_oos_metrics['sharpe']:>8.3f} {spy_oos_metrics['max_drawdown']:>9.2f}%")
    print(f"{'60/40':<25} {'OOS':<8} {sixty40_oos_metrics['cagr']:>7.2f}% {sixty40_oos_metrics['sharpe']:>8.3f} {sixty40_oos_metrics['max_drawdown']:>9.2f}%")
    print(f"{winner_name:<25} {'OOS':<8} {oos_metrics['cagr']:>7.2f}% {oos_metrics['sharpe']:>8.3f} {oos_metrics['max_drawdown']:>9.2f}%")
    
    # ============================================================
    # STEP 4: CRISIS ANALYSIS
    # ============================================================
    print("\n" + "-"*70)
    print("STEP 4: CRISIS PERIOD ANALYSIS")
    print("-"*70)
    
    # Full period returns for crisis analysis
    weights_full = generate_dual_momentum_weights(
        prices_full, winner_mom_lb, top_k, winner_ma_lb, 
        get_monthly_rebalance_dates(prices_full.index, freq="M")
    )
    result_full = backtest_from_weights(prices_full, weights_full, COST_PER_SIDE, winner_name)
    
    weights_spy_full = generate_benchmark_weights(prices_full, get_monthly_rebalance_dates(prices_full.index, freq="M"), "spy_100")
    weights_6040_full = generate_benchmark_weights(prices_full, get_monthly_rebalance_dates(prices_full.index, freq="M"), "60_40")
    result_spy_full = backtest_from_weights(prices_full, weights_spy_full, COST_PER_SIDE, "SPY")
    result_6040_full = backtest_from_weights(prices_full, weights_6040_full, COST_PER_SIDE, "60/40")
    
    crisis_strategy = analyze_crisis_periods(result_full["net_returns"], winner_name)
    crisis_spy = analyze_crisis_periods(result_spy_full["net_returns"], "SPY")
    crisis_6040 = analyze_crisis_periods(result_6040_full["net_returns"], "60/40")
    
    print(f"\n{'Crisis':<25} {'Strategy':<15} {'SPY':<15} {'60/40':<15}")
    print("-"*70)
    for crisis_name in CRISIS_PERIODS.keys():
        strat_ret = crisis_strategy[crisis_name]["return"]
        strat_dd = crisis_strategy[crisis_name]["max_dd"]
        spy_ret = crisis_spy[crisis_name]["return"]
        spy_dd = crisis_spy[crisis_name]["max_dd"]
        sixty_ret = crisis_6040[crisis_name]["return"]
        sixty_dd = crisis_6040[crisis_name]["max_dd"]
        
        print(f"\n{crisis_name}:")
        print(f"  {'Return':<12} {strat_ret:>8.2f}% {spy_ret:>12.2f}% {sixty_ret:>12.2f}%")
        print(f"  {'Max DD':<12} {strat_dd:>8.2f}% {spy_dd:>12.2f}% {sixty_dd:>12.2f}%")
    
    # ============================================================
    # STEP 5: MAX DD LOCATION
    # ============================================================
    print("\n" + "-"*70)
    print("STEP 5: WHERE DID THE MAX DD OCCUR?")
    print("-"*70)
    
    full_returns = result_full["net_returns"]
    cumulative = (1 + full_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd_date = drawdown.idxmin()
    max_dd_value = drawdown.min()
    
    print(f"\nFull-period Max DD: {max_dd_value*100:.2f}% occurred on {max_dd_date.date()}")
    
    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    # Compare OOS Sharpe to benchmarks
    oos_beats_spy = oos_metrics["sharpe"] > spy_oos_metrics["sharpe"]
    oos_beats_6040 = oos_metrics["sharpe"] > sixty40_oos_metrics["sharpe"]
    oos_dd_ok = oos_metrics["passes_dd_limit"]
    
    print(f"\nOOS Sharpe: {oos_metrics['sharpe']:.3f}")
    print(f"  vs SPY OOS Sharpe: {spy_oos_metrics['sharpe']:.3f} -> {'BEAT' if oos_beats_spy else 'LOST'}")
    print(f"  vs 60/40 OOS Sharpe: {sixty40_oos_metrics['sharpe']:.3f} -> {'BEAT' if oos_beats_6040 else 'LOST'}")
    print(f"  OOS Max DD: {oos_metrics['max_drawdown']:.2f}% -> {'PASS' if oos_dd_ok else 'FAIL'} (limit 25%)")
    
    # Save results
    summary = {
        "winner_config": winner_name,
        "is_sharpe": winner["net_sharpe"],
        "is_cagr": winner["net_cagr"],
        "is_max_dd": winner["max_dd"],
        "oos_sharpe": oos_metrics["sharpe"],
        "oos_cagr": oos_metrics["cagr"],
        "oos_max_dd": oos_metrics["max_drawdown"],
        "spy_oos_sharpe": spy_oos_metrics["sharpe"],
        "6040_oos_sharpe": sixty40_oos_metrics["sharpe"],
    }
    
    pd.DataFrame([summary]).to_csv("results/test3_5_validation_summary.csv", index=False)
    print(f"\nResults saved to results/test3_5_validation_summary.csv")
    
    return summary


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run_validation()