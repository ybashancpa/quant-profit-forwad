"""
TEST 2: Cross-Sectional Momentum (H1) - Monthly Rebalance
==========================================================
Hypothesis: Assets with stronger recent performance continue to 
outperform weaker ones (relative strength / winner effect).

Strategy: 
- At month-end, rank all assets by trailing N-month return
- Hold top K assets with equal weight
- Rebalance monthly

Economic rationale: Behavioral underreaction, herding, and 
slow information diffusion create persistent relative trends.

Note: This is PURE cross-sectional - no absolute filter yet.
That comes in Test 3 (Dual Momentum).
"""

import pandas as pd
import numpy as np
from data_loader import download_prices
from backtest_engine import backtest_from_weights, get_monthly_rebalance_dates, forward_fill_weights
from metrics import calculate_metrics, print_metrics
from config import MOMENTUM_LOOKBACKS_MONTHS, COST_PER_SIDE, TICKERS, TRADING_DAYS_PER_YEAR


def generate_xsmom_weights(prices, lookback_months, top_k, rebalance_dates):
    """
    Generate cross-sectional momentum portfolio weights.
    
    Args:
        prices: pd.DataFrame of daily prices
        lookback_months: Momentum lookback in months (approx 21 trading days each)
        top_k: Number of top assets to hold
        rebalance_dates: Dates when rebalancing occurs
    
    Returns:
        pd.DataFrame: Daily portfolio weights
    """
    lookback_days = lookback_months * 21  # Approx trading days per month
    
    # Calculate trailing returns
    trailing_returns = prices.pct_change(lookback_days)
    
    # Create sparse weights at rebalance dates
    sparse_weights = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    
    for date in rebalance_dates:
        if date not in trailing_returns.index:
            continue
        
        # Get returns on rebalance date
        rets = trailing_returns.loc[date].dropna()
        
        if len(rets) < top_k:
            continue
        
        # Rank and select top K
        top_assets = rets.nlargest(top_k).index.tolist()
        
        # Equal weight
        weight = 1.0 / top_k
        for asset in top_assets:
            sparse_weights.loc[date, asset] = weight
    
    # Forward fill to daily
    daily_weights = forward_fill_weights(sparse_weights, prices.index)
    
    return daily_weights


def run_test2():
    """Run Test 2: Cross-Sectional Momentum."""
    print("\n" + "="*70)
    print("TEST 2: CROSS-SECTIONAL MOMENTUM (H1) - MONTHLY REBALANCE")
    print("="*70)
    print(f"\nCost model: {COST_PER_SIDE*100:.2f}% per side ({COST_PER_SIDE*200:.2f}% round trip)")
    print(f"Momentum lookbacks tested: {MOMENTUM_LOOKBACKS_MONTHS} months")
    print(f"Top K assets to hold: [1, 2, 3]")
    
    # Load data
    prices = download_prices()
    
    # Get monthly rebalance dates
    rebalance_dates = get_monthly_rebalance_dates(prices.index, freq="M")
    print(f"Rebalance dates: {len(rebalance_dates)} months")
    
    # Store results
    all_results = []
    
    # Test different lookbacks and top_k
    top_k_options = [1, 2, 3]
    
    for lookback in MOMENTUM_LOOKBACKS_MONTHS:
        for top_k in top_k_options:
            name = f"XS-Mom {lookback}M Top{top_k}"
            
            # Generate weights
            weights = generate_xsmom_weights(prices, lookback, top_k, rebalance_dates)
            
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
                "lookback_months": lookback,
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
    
    # Summary table
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv("results/test2_xsmom_results.csv", index=False)
    print(f"\nResults saved to results/test2_xsmom_results.csv")
    
    # Verdict
    print("\n" + "="*70)
    print("TEST 2 VERDICT")
    print("="*70)
    
    net_sharpe_positive = results_df[results_df["net_sharpe"] > 0]
    passes_dd = results_df[results_df["passes_dd"] == True]
    
    print(f"\nConfigurations with positive net Sharpe: {len(net_sharpe_positive)}/{len(results_df)}")
    print(f"Configurations passing 25% Max DD limit: {len(passes_dd)}/{len(results_df)}")
    
    if len(results_df) > 0:
        best_net_sharpe = results_df.loc[results_df["net_sharpe"].idxmax()]
        print(f"\nBest net Sharpe: {best_net_sharpe['lookback_months']}M Top{best_net_sharpe['top_k']} = {best_net_sharpe['net_sharpe']:.3f}")
    
    return results_df


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run_test2()