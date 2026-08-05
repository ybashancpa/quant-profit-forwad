"""
TEST 3: Dual Momentum (H1 + H2 Combined)
=========================================
Hypothesis: Combining relative strength (cross-sectional) with 
absolute trend filter (time-series) provides both return AND 
drawdown protection.

Strategy:
1. At month-end, rank all assets by trailing N-month return
2. Select top K assets
3. For each selected asset, check absolute filter:
   - Price > MA(lookback) → hold the asset
   - Price < MA(lookback) → move to safe asset (SHY)
4. Rebalance monthly

This is the CORE STRATEGY. The absolute filter should protect
us during broad bear markets (2008, 2020, 2022).

Economic rationale: 
- Cross-sectional: Behavioral underreaction, herding
- Time-series: Trend persistence, slow information diffusion
- Combined: Avoids momentum crashes by going defensive
"""

import pandas as pd
import numpy as np
from data_loader import download_prices
from backtest_engine import backtest_from_weights, get_monthly_rebalance_dates, forward_fill_weights
from metrics import calculate_metrics, print_metrics
from config import COST_PER_SIDE, TICKERS, SAFE_ASSET, TS_MOM_MA_LOOKBACKS


def generate_dual_momentum_weights(prices, lookback_months, top_k, ma_lookback, rebalance_dates, safe_asset=SAFE_ASSET):
    """
    Generate Dual Momentum portfolio weights.
    
    Args:
        prices: pd.DataFrame of daily prices
        lookback_months: Momentum lookback in months (for ranking)
        top_k: Number of top assets to select
        ma_lookback: Moving average lookback for absolute filter
        rebalance_dates: Dates when rebalancing occurs
        safe_asset: Ticker for defensive allocation (default SHY)
    
    Returns:
        pd.DataFrame: Daily portfolio weights
    """
    lookback_days = lookback_months * 21  # Approx trading days per month
    
    # Calculate trailing returns (for ranking)
    trailing_returns = prices.pct_change(lookback_days)
    
    # Calculate moving average (for absolute filter)
    moving_avg = prices.rolling(window=ma_lookback, min_periods=ma_lookback).mean()
    
    # Create sparse weights at rebalance dates
    sparse_weights = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    
    for date in rebalance_dates:
        if date not in trailing_returns.index:
            continue
        
        # Get returns on rebalance date
        rets = trailing_returns.loc[date].dropna()
        
        if len(rets) < top_k:
            continue
        
        # Step 1: Rank and select top K (cross-sectional)
        top_assets = rets.nlargest(top_k).index.tolist()
        
        # Step 2: Apply absolute filter (time-series)
        weight_per_asset = 1.0 / top_k
        safe_weight = 0.0
        
        for asset in top_assets:
            if asset not in prices.columns:
                continue
            
            price_today = prices.loc[date, asset]
            ma_today = moving_avg.loc[date, asset]
            
            if pd.isna(ma_today):
                # Not enough history, go to safe
                safe_weight += weight_per_asset
            elif price_today > ma_today:
                # Uptrend: hold the asset
                sparse_weights.loc[date, asset] = weight_per_asset
            else:
                # Downtrend: move to safe asset
                safe_weight += weight_per_asset
        
        # Allocate safe weight
        if safe_weight > 0 and safe_asset in prices.columns:
            sparse_weights.loc[date, safe_asset] = safe_weight
    
    # Forward fill to daily
    daily_weights = forward_fill_weights(sparse_weights, prices.index)
    
    return daily_weights


def run_test3():
    """Run Test 3: Dual Momentum."""
    print("\n" + "="*70)
    print("TEST 3: DUAL MOMENTUM (H1 + H2 COMBINED)")
    print("="*70)
    print(f"\nCost model: {COST_PER_SIDE*100:.2f}% per side ({COST_PER_SIDE*200:.2f}% round trip)")
    print(f"Safe asset: {SAFE_ASSET}")
    print(f"Momentum lookbacks: [6, 9] months (best from Test 2)")
    print(f"Top K: [3] (best from Test 2)")
    print(f"MA lookbacks for absolute filter: {TS_MOM_MA_LOOKBACKS} days")
    
    # Load data
    prices = download_prices()
    
    # Get monthly rebalance dates
    rebalance_dates = get_monthly_rebalance_dates(prices.index, freq="M")
    print(f"Rebalance dates: {len(rebalance_dates)} months")
    
    # Store results
    all_results = []
    
    # Test configurations (based on Test 2 findings)
    momentum_lookbacks = [6, 9]
    top_k = 3
    
    for mom_lb in momentum_lookbacks:
        for ma_lb in TS_MOM_MA_LOOKBACKS:
            name = f"DualMom {mom_lb}M/MA{ma_lb}"
            
            # Generate weights
            weights = generate_dual_momentum_weights(
                prices, 
                lookback_months=mom_lb, 
                top_k=top_k, 
                ma_lookback=ma_lb, 
                rebalance_dates=rebalance_dates
            )
            
            # Run backtest
            result = backtest_from_weights(
                prices, 
                weights, 
                cost_per_side=COST_PER_SIDE,
                name=name
            )
            
            # Calculate metrics
            gross_metrics = calculate_metrics(result["gross_returns"], name=f"{name} GROSS")
            net_metrics = calculate_metrics(result["net_returns"], name=f"{name} NET")
            
            # Store summary
            all_results.append({
                "momentum_lookback": mom_lb,
                "ma_lookback": ma_lb,
                "top_k": top_k,
                "gross_cagr": gross_metrics["cagr"],
                "net_cagr": net_metrics["cagr"],
                "gross_sharpe": gross_metrics["sharpe"],
                "net_sharpe": net_metrics["sharpe"],
                "max_dd": net_metrics["max_drawdown"],
                "annual_turnover": result["annual_turnover_one_way"],
                "total_cost_drag": result["total_cost_drag"] * 100,
                "passes_dd": net_metrics["passes_dd_limit"],
            })
            
            print(f"\n{name}:")
            print(f"  Gross CAGR: {gross_metrics['cagr']:.2f}% | Net CAGR: {net_metrics['cagr']:.2f}%")
            print(f"  Gross Sharpe: {gross_metrics['sharpe']:.3f} | Net Sharpe: {net_metrics['sharpe']:.3f}")
            print(f"  Max DD: {net_metrics['max_drawdown']:.2f}% | Turnover/yr: {result['annual_turnover_one_way']:.2f}x")
            print(f"  Passes 25% DD limit: {'✓ YES' if net_metrics['passes_dd_limit'] else '✗ NO'}")
    
    # Summary table
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv("results/test3_dual_momentum_results.csv", index=False)
    print(f"\nResults saved to results/test3_dual_momentum_results.csv")
    
    # Verdict
    print("\n" + "="*70)
    print("TEST 3 VERDICT")
    print("="*70)
    
    net_sharpe_positive = results_df[results_df["net_sharpe"] > 0]
    passes_dd = results_df[results_df["passes_dd"] == True]
    
    print(f"\nConfigurations with positive net Sharpe: {len(net_sharpe_positive)}/{len(results_df)}")
    print(f"Configurations passing 25% Max DD limit: {len(passes_dd)}/{len(results_df)}")
    
    if len(results_df) > 0:
        best_net_sharpe = results_df.loc[results_df["net_sharpe"].idxmax()]
        print(f"\nBest net Sharpe: {best_net_sharpe['momentum_lookback']}M/MA{best_net_sharpe['ma_lookback']} = {best_net_sharpe['net_sharpe']:.3f}")
        
        # Check if any pass DD limit
        passing = results_df[results_df["passes_dd"] == True]
        if len(passing) > 0:
            best_passing = passing.loc[passing["net_sharpe"].idxmax()]
            print(f"Best PASSING config: {best_passing['momentum_lookback']}M/MA{best_passing['ma_lookback']} = Sharpe {best_passing['net_sharpe']:.3f}, DD {best_passing['max_dd']:.2f}%")
    
    return results_df


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run_test3()