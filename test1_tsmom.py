"""
TEST 1: Time-Series Momentum (H2) - Individual Asset Level
===========================================================
Hypothesis: Assets in uptrends (price > MA) continue to outperform.
Strategy: Long when price > MA(lookback), cash (0 weight) otherwise.

This test evaluates each asset individually to understand:
1. Which assets exhibit trend-following characteristics
2. The gross vs net return impact (cost drag)
3. Turnover levels for different MA lookbacks

Economic rationale: Investor underreaction to information, 
herding behavior, and slow diffusion of news create momentum.
"""

import pandas as pd
import numpy as np
from data_loader import download_prices
from backtest_engine import backtest_from_weights
from metrics import calculate_metrics, print_metrics
from config import TS_MOM_MA_LOOKBACKS, COST_PER_SIDE, TICKERS


def generate_tsmom_signals(prices, ma_lookback):
    """
    Generate time-series momentum signals for a single asset.
    
    Signal: 1 (long) if price > MA, 0 (cash) otherwise.
    
    Args:
        prices: pd.Series of prices for one asset
        ma_lookback: Moving average lookback in days
    
    Returns:
        pd.Series: Position weights (0 or 1)
    """
    ma = prices.rolling(window=ma_lookback, min_periods=ma_lookback).mean()
    signal = (prices > ma).astype(float)
    return signal


def run_test1():
    """Run Test 1: Time-Series Momentum on each asset."""
    print("\n" + "="*70)
    print("TEST 1: TIME-SERIES MOMENTUM (H2) - INDIVIDUAL ASSET ANALYSIS")
    print("="*70)
    print(f"\nCost model: {COST_PER_SIDE*100:.2f}% per side ({COST_PER_SIDE*200:.2f}% round trip)")
    print(f"MA lookbacks tested: {TS_MOM_MA_LOOKBACKS} days")
    
    # Load data
    prices = download_prices()
    
    # Store results for summary
    all_results = []
    
    for ticker in TICKERS:
        print(f"\n{'─'*70}")
        print(f"ASSET: {ticker}")
        print(f"{'─'*70}")
        
        asset_prices = prices[ticker].dropna()
        
        for ma_lb in TS_MOM_MA_LOOKBACKS:
            # Generate signals
            signals = generate_tsmom_signals(asset_prices, ma_lb)
            
            # Create weights DataFrame (single asset)
            weights = pd.DataFrame({ticker: signals}, index=asset_prices.index)
            
            # Run backtest
            result = backtest_from_weights(
                prices[[ticker]], 
                weights, 
                cost_per_side=COST_PER_SIDE,
                name=f"{ticker}_MA{ma_lb}"
            )
            
            # Calculate metrics for gross and net
            gross_metrics = calculate_metrics(result["gross_returns"], name=f"{ticker} MA{ma_lb} GROSS")
            net_metrics = calculate_metrics(result["net_returns"], name=f"{ticker} MA{ma_lb} NET")
            
            # Store summary
            all_results.append({
                "ticker": ticker,
                "ma_lookback": ma_lb,
                "gross_cagr": gross_metrics["cagr"],
                "net_cagr": net_metrics["cagr"],
                "gross_sharpe": gross_metrics["sharpe"],
                "net_sharpe": net_metrics["sharpe"],
                "max_dd": net_metrics["max_drawdown"],
                "annual_turnover": result["annual_turnover_one_way"],
                "total_cost_drag": result["total_cost_drag"] * 100,
                "passes_dd": net_metrics["passes_dd_limit"],
            })
            
            # Print detailed for 200-day MA only (to reduce output)
            if ma_lb == 200:
                print(f"\n  [MA{ma_lb}] Gross CAGR: {gross_metrics['cagr']:.2f}% | Net CAGR: {net_metrics['cagr']:.2f}%")
                print(f"  [MA{ma_lb}] Gross Sharpe: {gross_metrics['sharpe']:.3f} | Net Sharpe: {net_metrics['sharpe']:.3f}")
                print(f"  [MA{ma_lb}] Max DD: {net_metrics['max_drawdown']:.2f}% | Turnover/yr: {result['annual_turnover_one_way']:.2f}x")
                print(f"  [MA{ma_lb}] Total cost drag: {result['total_cost_drag']*100:.2f}% over full period")
    
    # Summary table
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "="*70)
    print("SUMMARY TABLE (All assets, all MA lookbacks)")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv("results/test1_tsmom_results.csv", index=False)
    print(f"\nResults saved to results/test1_tsmom_results.csv")
    
    # Verdict
    print("\n" + "="*70)
    print("TEST 1 VERDICT")
    print("="*70)
    
    # Count how many pass
    net_sharpe_positive = results_df[results_df["net_sharpe"] > 0]
    passes_dd = results_df[results_df["passes_dd"] == True]
    
    print(f"\nConfigurations with positive net Sharpe: {len(net_sharpe_positive)}/{len(results_df)}")
    print(f"Configurations passing 25% Max DD limit: {len(passes_dd)}/{len(results_df)}")
    
    # Best configurations
    if len(results_df) > 0:
        best_net_sharpe = results_df.loc[results_df["net_sharpe"].idxmax()]
        print(f"\nBest net Sharpe: {best_net_sharpe['ticker']} MA{best_net_sharpe['ma_lookback']} = {best_net_sharpe['net_sharpe']:.3f}")
    
    return results_df


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run_test1()