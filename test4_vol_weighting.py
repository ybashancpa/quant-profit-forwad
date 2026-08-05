"""
TEST 4: Inverse Volatility Weighting (H4)
==========================================
Hypothesis: Equal-weighting ignores risk differences. Inverse-vol
weighting balances risk contribution across assets, potentially
improving Sharpe and reducing drawdown.

Strategy: Same as DualMom 6M/MA150 (winner from Test 3.5), but
instead of equal weight (33.3% each), weight by inverse volatility.

LOCKED PARAMETERS (no searching):
- Momentum lookback: 6 months
- MA filter: 150 days
- Volatility lookback: 60 days
- Top K: 3

Key concern: Inverse-vol rebalances weights even when holdings
don't change, increasing turnover and costs. The test is whether
Sharpe improvement (if any) survives the extra turnover.

Protocol: IS (2007-2017) then OOS (2018-2026), compare to
equal-weight baseline and 60/40 benchmark.
"""

import pandas as pd
import numpy as np
from data_loader import download_prices
from backtest_engine import backtest_from_weights, get_monthly_rebalance_dates, forward_fill_weights
from metrics import calculate_metrics
from config import COST_PER_SIDE, TICKERS, SAFE_ASSET


# ============================================================
# LOCKED PARAMETERS (from Test 3.5 winner)
# ============================================================
MOMENTUM_LOOKBACK_MONTHS = 6
MA_LOOKBACK = 150
TOP_K = 3
VOL_LOOKBACK_DAYS = 60  # Locked, no searching

# Period definitions (same as Test 3.5)
IS_START = "2007-01-01"
IS_END = "2017-12-31"
OOS_START = "2018-01-01"

CRISIS_PERIODS = {
    "2008 Financial Crisis": ("2008-01-01", "2009-03-31"),
    "2020 Corona Crash": ("2020-02-01", "2020-04-30"),
    "2022 Inflation/Rates": ("2022-01-01", "2022-12-31"),
}


def generate_vol_weighted_dual_momentum(prices, rebalance_dates, safe_asset=SAFE_ASSET):
    """
    Generate Dual Momentum weights with inverse-volatility weighting.
    
    Args:
        prices: pd.DataFrame of daily prices
        rebalance_dates: Dates when rebalancing occurs
        safe_asset: Ticker for defensive allocation
    
    Returns:
        pd.DataFrame: Daily portfolio weights
    """
    lookback_days = MOMENTUM_LOOKBACK_MONTHS * 21
    
    # Calculate trailing returns (for ranking)
    trailing_returns = prices.pct_change(lookback_days)
    
    # Calculate moving average (for absolute filter)
    moving_avg = prices.rolling(window=MA_LOOKBACK, min_periods=MA_LOOKBACK).mean()
    
    # Calculate daily returns and rolling volatility (for weighting)
    daily_returns = prices.pct_change()
    rolling_vol = daily_returns.rolling(window=VOL_LOOKBACK_DAYS, min_periods=VOL_LOOKBACK_DAYS).std()
    
    # Create sparse weights at rebalance dates
    sparse_weights = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    
    for date in rebalance_dates:
        if date not in trailing_returns.index:
            continue
        
        rets = trailing_returns.loc[date].dropna()
        if len(rets) < TOP_K:
            continue
        
        # Step 1: Rank and select top K (cross-sectional)
        top_assets = rets.nlargest(TOP_K).index.tolist()
        
        # Step 2: Apply absolute filter and collect risky assets
        risky_assets = []
        safe_weight = 0.0
        weight_per_slot = 1.0 / TOP_K
        
        for asset in top_assets:
            if asset not in prices.columns:
                safe_weight += weight_per_slot
                continue
            
            price_today = prices.loc[date, asset]
            ma_today = moving_avg.loc[date, asset]
            
            if pd.isna(ma_today):
                safe_weight += weight_per_slot
            elif price_today > ma_today:
                risky_assets.append(asset)
            else:
                safe_weight += weight_per_slot
        
        # Step 3: Inverse-vol weighting for risky assets
        if len(risky_assets) > 0:
            # Get volatilities at rebalance date (up to and including today)
            vols = rolling_vol.loc[date, risky_assets]
            
            # Handle NaN vols (not enough history)
            valid_assets = [a for a in risky_assets if not pd.isna(vols.get(a, np.nan))]
            
            if len(valid_assets) > 0:
                vols = vols[valid_assets]
                
                # Inverse volatility weights
                inv_vols = 1.0 / vols
                inv_vol_sum = inv_vols.sum()
                
                if inv_vol_sum > 0:
                    # Normalize to sum to 1 for risky portion
                    risky_weights = inv_vols / inv_vol_sum
                    
                    # Scale by fraction allocated to risky (1 - safe_weight)
                    risky_fraction = 1.0 - safe_weight
                    for asset in valid_assets:
                        sparse_weights.loc[date, asset] = risky_weights[asset] * risky_fraction
        
        # Step 4: Allocate safe weight
        if safe_weight > 0 and safe_asset in prices.columns:
            sparse_weights.loc[date, safe_asset] = safe_weight
    
    # Forward fill to daily
    daily_weights = forward_fill_weights(sparse_weights, prices.index)
    return daily_weights


def generate_equal_weight_dual_momentum(prices, rebalance_dates, safe_asset=SAFE_ASSET):
    """Generate equal-weight Dual Momentum weights (baseline from Test 3.5)."""
    lookback_days = MOMENTUM_LOOKBACK_MONTHS * 21
    
    trailing_returns = prices.pct_change(lookback_days)
    moving_avg = prices.rolling(window=MA_LOOKBACK, min_periods=MA_LOOKBACK).mean()
    
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


def generate_benchmark_weights(prices, rebalance_dates, mode="60_40"):
    """Generate benchmark weights."""
    sparse_weights = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    
    for date in rebalance_dates:
        if mode == "spy_100":
            sparse_weights.loc[date, "SPY"] = 1.0
        elif mode == "60_40":
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


def run_test4():
    """Run Test 4: Inverse Volatility Weighting."""
    print("\n" + "="*70)
    print("TEST 4: INVERSE VOLATILITY WEIGHTING (H4)")
    print("="*70)
    print(f"\nLocked parameters:")
    print(f"  Momentum lookback: {MOMENTUM_LOOKBACK_MONTHS} months")
    print(f"  MA filter: {MA_LOOKBACK} days")
    print(f"  Volatility lookback: {VOL_LOOKBACK_DAYS} days")
    print(f"  Top K: {TOP_K}")
    print(f"  Cost: {COST_PER_SIDE*100:.2f}% per side")
    
    # Load full data
    prices_full = download_prices()
    
    # ============================================================
    # IN-SAMPLE (2007-2017)
    # ============================================================
    print("\n" + "-"*70)
    print("IN-SAMPLE (2007-2017)")
    print("-"*70)
    
    prices_is = prices_full.loc[IS_START:IS_END]
    rebalance_dates_is = get_monthly_rebalance_dates(prices_is.index, freq="M")
    
    # Equal weight baseline
    weights_ew_is = generate_equal_weight_dual_momentum(prices_is, rebalance_dates_is)
    result_ew_is = backtest_from_weights(prices_is, weights_ew_is, COST_PER_SIDE, "EqualWeight")
    ew_is_metrics = calculate_metrics(result_ew_is["net_returns"], "EqualWeight IS")
    
    # Inverse vol
    weights_iv_is = generate_vol_weighted_dual_momentum(prices_is, rebalance_dates_is)
    result_iv_is = backtest_from_weights(prices_is, weights_iv_is, COST_PER_SIDE, "InverseVol")
    iv_is_metrics = calculate_metrics(result_iv_is["net_returns"], "InverseVol IS")
    
    # 60/40 benchmark
    weights_6040_is = generate_benchmark_weights(prices_is, rebalance_dates_is, "60_40")
    result_6040_is = backtest_from_weights(prices_is, weights_6040_is, COST_PER_SIDE, "60/40")
    sixty40_is_metrics = calculate_metrics(result_6040_is["net_returns"], "60/40 IS")
    
    print(f"\n{'Strategy':<20} {'CAGR':>8} {'Sharpe':>8} {'Max DD':>10} {'Turnover/yr':>12}")
    print("-"*65)
    print(f"{'EqualWeight':<20} {ew_is_metrics['cagr']:>7.2f}% {ew_is_metrics['sharpe']:>8.3f} {ew_is_metrics['max_drawdown']:>9.2f}% {result_ew_is['annual_turnover_one_way']:>11.2f}x")
    print(f"{'InverseVol':<20} {iv_is_metrics['cagr']:>7.2f}% {iv_is_metrics['sharpe']:>8.3f} {iv_is_metrics['max_drawdown']:>9.2f}% {result_iv_is['annual_turnover_one_way']:>11.2f}x")
    print(f"{'60/40':<20} {sixty40_is_metrics['cagr']:>7.2f}% {sixty40_is_metrics['sharpe']:>8.3f} {sixty40_is_metrics['max_drawdown']:>9.2f}% {result_6040_is['annual_turnover_one_way']:>11.2f}x")
    
    # ============================================================
    # OUT-OF-SAMPLE (2018-2026)
    # ============================================================
    print("\n" + "-"*70)
    print("OUT-OF-SAMPLE (2018-2026)")
    print("-"*70)
    
    prices_oos = prices_full.loc[OOS_START:]
    rebalance_dates_oos = get_monthly_rebalance_dates(prices_oos.index, freq="M")
    
    # Equal weight baseline
    weights_ew_oos = generate_equal_weight_dual_momentum(prices_oos, rebalance_dates_oos)
    result_ew_oos = backtest_from_weights(prices_oos, weights_ew_oos, COST_PER_SIDE, "EqualWeight")
    ew_oos_metrics = calculate_metrics(result_ew_oos["net_returns"], "EqualWeight OOS")
    
    # Inverse vol
    weights_iv_oos = generate_vol_weighted_dual_momentum(prices_oos, rebalance_dates_oos)
    result_iv_oos = backtest_from_weights(prices_oos, weights_iv_oos, COST_PER_SIDE, "InverseVol")
    iv_oos_metrics = calculate_metrics(result_iv_oos["net_returns"], "InverseVol OOS")
    
    # 60/40 benchmark
    weights_6040_oos = generate_benchmark_weights(prices_oos, rebalance_dates_oos, "60_40")
    result_6040_oos = backtest_from_weights(prices_oos, weights_6040_oos, COST_PER_SIDE, "60/40")
    sixty40_oos_metrics = calculate_metrics(result_6040_oos["net_returns"], "60/40 OOS")
    
    print(f"\n{'Strategy':<20} {'CAGR':>8} {'Sharpe':>8} {'Max DD':>10} {'Turnover/yr':>12}")
    print("-"*65)
    print(f"{'EqualWeight':<20} {ew_oos_metrics['cagr']:>7.2f}% {ew_oos_metrics['sharpe']:>8.3f} {ew_oos_metrics['max_drawdown']:>9.2f}% {result_ew_oos['annual_turnover_one_way']:>11.2f}x")
    print(f"{'InverseVol':<20} {iv_oos_metrics['cagr']:>7.2f}% {iv_oos_metrics['sharpe']:>8.3f} {iv_oos_metrics['max_drawdown']:>9.2f}% {result_iv_oos['annual_turnover_one_way']:>11.2f}x")
    print(f"{'60/40':<20} {sixty40_oos_metrics['cagr']:>7.2f}% {sixty40_oos_metrics['sharpe']:>8.3f} {sixty40_oos_metrics['max_drawdown']:>9.2f}% {result_6040_oos['annual_turnover_one_way']:>11.2f}x")
    
    # ============================================================
    # CRISIS ANALYSIS (Full period)
    # ============================================================
    print("\n" + "-"*70)
    print("CRISIS ANALYSIS (Full Period)")
    print("-"*70)
    
    rebalance_dates_full = get_monthly_rebalance_dates(prices_full.index, freq="M")
    
    weights_ew_full = generate_equal_weight_dual_momentum(prices_full, rebalance_dates_full)
    weights_iv_full = generate_vol_weighted_dual_momentum(prices_full, rebalance_dates_full)
    weights_6040_full = generate_benchmark_weights(prices_full, rebalance_dates_full, "60_40")
    
    result_ew_full = backtest_from_weights(prices_full, weights_ew_full, COST_PER_SIDE, "EqualWeight")
    result_iv_full = backtest_from_weights(prices_full, weights_iv_full, COST_PER_SIDE, "InverseVol")
    result_6040_full = backtest_from_weights(prices_full, weights_6040_full, COST_PER_SIDE, "60/40")
    
    crisis_ew = analyze_crisis_periods(result_ew_full["net_returns"])
    crisis_iv = analyze_crisis_periods(result_iv_full["net_returns"])
    crisis_6040 = analyze_crisis_periods(result_6040_full["net_returns"])
    
    print(f"\n{'Crisis':<25} {'EqualWeight':<15} {'InverseVol':<15} {'60/40':<15}")
    print("-"*70)
    for crisis_name in CRISIS_PERIODS.keys():
        ew_ret = crisis_ew[crisis_name]["return"]
        ew_dd = crisis_ew[crisis_name]["max_dd"]
        iv_ret = crisis_iv[crisis_name]["return"]
        iv_dd = crisis_iv[crisis_name]["max_dd"]
        sixty_ret = crisis_6040[crisis_name]["return"]
        sixty_dd = crisis_6040[crisis_name]["max_dd"]
        
        print(f"\n{crisis_name}:")
        print(f"  {'Return':<12} {ew_ret:>8.2f}% {iv_ret:>12.2f}% {sixty_ret:>12.2f}%")
        print(f"  {'Max DD':<12} {ew_dd:>8.2f}% {iv_dd:>12.2f}% {sixty_dd:>12.2f}%")
    
    # ============================================================
    # VERDICT
    # ============================================================
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    # Calculate Sharpe improvement
    is_sharpe_delta = iv_is_metrics["sharpe"] - ew_is_metrics["sharpe"]
    oos_sharpe_delta = iv_oos_metrics["sharpe"] - ew_oos_metrics["sharpe"]
    
    # Turnover increase
    is_turnover_delta = result_iv_is["annual_turnover_one_way"] - result_ew_is["annual_turnover_one_way"]
    oos_turnover_delta = result_iv_oos["annual_turnover_one_way"] - result_ew_oos["annual_turnover_one_way"]
    
    print(f"\nSharpe change (InverseVol - EqualWeight):")
    print(f"  IS:  {is_sharpe_delta:+.3f}")
    print(f"  OOS: {oos_sharpe_delta:+.3f}")
    
    print(f"\nTurnover increase:")
    print(f"  IS:  {is_turnover_delta:+.2f}x/year")
    print(f"  OOS: {oos_turnover_delta:+.2f}x/year")
    
    # Check if beats 60/40
    beats_6040_oos = iv_oos_metrics["sharpe"] > sixty40_oos_metrics["sharpe"]
    margin = iv_oos_metrics["sharpe"] - sixty40_oos_metrics["sharpe"]
    
    print(f"\nOOS vs 60/40:")
    print(f"  InverseVol Sharpe: {iv_oos_metrics['sharpe']:.3f}")
    print(f"  60/40 Sharpe: {sixty40_oos_metrics['sharpe']:.3f}")
    print(f"  Margin: {margin:+.3f} -> {'BEAT' if beats_6040_oos else 'LOST'}")
    
    # Save results
    summary = {
        "ew_is_sharpe": ew_is_metrics["sharpe"],
        "iv_is_sharpe": iv_is_metrics["sharpe"],
        "ew_oos_sharpe": ew_oos_metrics["sharpe"],
        "iv_oos_sharpe": iv_oos_metrics["sharpe"],
        "6040_oos_sharpe": sixty40_oos_metrics["sharpe"],
        "ew_oos_turnover": result_ew_oos["annual_turnover_one_way"],
        "iv_oos_turnover": result_iv_oos["annual_turnover_one_way"],
    }
    
    pd.DataFrame([summary]).to_csv("results/test4_vol_weighting_summary.csv", index=False)
    print(f"\nResults saved to results/test4_vol_weighting_summary.csv")
    
    return summary


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run_test4()