"""
TEST 9.5: VOL-MANAGED VALIDATION
=================================
Two missing tests that determine if the vol-managed edge is real:

1. BLOCK BOOTSTRAP: Is the Sharpe difference (0.971 vs 0.885) statistically significant?
   - Block bootstrap preserves autocorrelation in returns
   - 10,000 resamples
   - Report p-value for H0: Sharpe_vol <= Sharpe_matched

2. COST STRESS: Does vol-managed survive 20bps and 50bps per side?
   - Vol-managed makes small, frequent adjustments
   - More sensitive to costs than MA200
   - Must report net Sharpe at each cost level
"""

import pandas as pd
import numpy as np
from data_loader import download_prices
from backtest_engine import backtest_from_weights, get_monthly_rebalance_dates, forward_fill_weights
from metrics import calculate_metrics
from config import COST_PER_SIDE


# ============================================================
# PARAMETERS (same as Test 9)
# ============================================================
PRIMARY_WINDOW = 60
PRIMARY_TARGET_VOL = 0.15
BASE_EQUITY_WEIGHT = 0.55
BASE_IEF_WEIGHT = 0.35
BASE_GLD_WEIGHT = 0.10

BOOTSTRAP_N = 10000
BOOTSTRAP_BLOCK_SIZE = 21  # ~1 month of trading days
RANDOM_SEED = 42

COST_LEVELS = [0.0005, 0.0010, 0.0020, 0.0050]  # 5, 10, 20, 50 bps


def generate_vol_managed_weights(prices, rebalance_dates, window=PRIMARY_WINDOW, target_vol=PRIMARY_TARGET_VOL):
    """Generate volatility-managed portfolio weights."""
    spy_returns = prices["SPY"].pct_change()
    realized_vol = spy_returns.rolling(window=window, min_periods=window).std() * np.sqrt(252)
    
    sparse_weights = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    
    for date in rebalance_dates:
        if date not in prices.index:
            continue
        
        vol = realized_vol.loc[date]
        
        if pd.isna(vol) or vol <= 0:
            spy_fraction = 1.0
        else:
            spy_fraction = min(1.0, target_vol / vol)
        
        spy_weight = BASE_EQUITY_WEIGHT * spy_fraction
        shy_weight = BASE_EQUITY_WEIGHT - spy_weight
        
        sparse_weights.loc[date, "SPY"] = spy_weight
        sparse_weights.loc[date, "SHY"] = shy_weight
        sparse_weights.loc[date, "IEF"] = BASE_IEF_WEIGHT
        sparse_weights.loc[date, "GLD"] = BASE_GLD_WEIGHT
    
    daily_weights = forward_fill_weights(sparse_weights, prices.index)
    return daily_weights


def generate_static_weights(prices, rebalance_dates, allocation):
    """Generate static portfolio weights."""
    sparse_weights = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    
    for date in rebalance_dates:
        for asset, weight in allocation.items():
            if asset in prices.columns:
                sparse_weights.loc[date, asset] = weight
    
    daily_weights = forward_fill_weights(sparse_weights, prices.index)
    return daily_weights


def generate_matched_exposure_weights(prices, rebalance_dates, avg_spy_exposure):
    """Generate static portfolio with same average SPY exposure as vol-managed."""
    scale = avg_spy_exposure / BASE_EQUITY_WEIGHT
    
    allocation = {
        "SPY": BASE_EQUITY_WEIGHT * scale,
        "SHY": BASE_EQUITY_WEIGHT * (1 - scale),
        "IEF": BASE_IEF_WEIGHT,
        "GLD": BASE_GLD_WEIGHT,
    }
    
    return generate_static_weights(prices, rebalance_dates, allocation)


def calculate_sharpe(returns):
    """Calculate annualized Sharpe ratio."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(252)


def block_bootstrap_sharpe_diff(returns_a, returns_b, n_bootstrap=BOOTSTRAP_N, 
                                  block_size=BOOTSTRAP_BLOCK_SIZE, seed=RANDOM_SEED):
    """
    Block bootstrap test for Sharpe difference.
    
    H0: Sharpe_a <= Sharpe_b
    H1: Sharpe_a > Sharpe_b
    
    Returns:
        - observed_diff: Sharpe_a - Sharpe_b
        - p_value: probability of observing diff <= 0 under bootstrap
        - ci_lower, ci_upper: 95% confidence interval for the difference
    """
    np.random.seed(seed)
    
    n = len(returns_a)
    n_blocks = int(np.ceil(n / block_size))
    
    observed_sharpe_a = calculate_sharpe(returns_a)
    observed_sharpe_b = calculate_sharpe(returns_b)
    observed_diff = observed_sharpe_a - observed_sharpe_b
    
    bootstrap_diffs = []
    
    for _ in range(n_bootstrap):
        # Sample blocks with replacement
        block_starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
        
        # Build bootstrap samples
        boot_a = []
        boot_b = []
        for start in block_starts:
            boot_a.append(returns_a.values[start:start + block_size])
            boot_b.append(returns_b.values[start:start + block_size])
        
        boot_a = np.concatenate(boot_a)[:n]
        boot_b = np.concatenate(boot_b)[:n]
        
        # Calculate Sharpe difference
        sharpe_a = calculate_sharpe(pd.Series(boot_a))
        sharpe_b = calculate_sharpe(pd.Series(boot_b))
        bootstrap_diffs.append(sharpe_a - sharpe_b)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # p-value: proportion of bootstrap samples where diff <= 0
    p_value = np.mean(bootstrap_diffs <= 0)
    
    # 95% confidence interval
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    
    return {
        "observed_diff": observed_diff,
        "sharpe_a": observed_sharpe_a,
        "sharpe_b": observed_sharpe_b,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_bootstrap": n_bootstrap,
        "block_size": block_size,
    }


def run_test9_5():
    """Run Test 9.5: Vol-Managed Validation."""
    print("\n" + "="*70)
    print("TEST 9.5: VOL-MANAGED VALIDATION")
    print("="*70)
    print("\nTwo missing tests:")
    print("1. Block bootstrap: Is Sharpe diff statistically significant?")
    print("2. Cost stress: Does vol-managed survive 20/50 bps?")
    
    # Load data
    prices_full = download_prices()
    rebalance_dates = get_monthly_rebalance_dates(prices_full.index, freq="M")
    
    # Generate strategies
    weights_vol = generate_vol_managed_weights(prices_full, rebalance_dates)
    result_vol = backtest_from_weights(prices_full, weights_vol, COST_PER_SIDE, "VolManaged")
    
    avg_spy_exposure = result_vol["weights"]["SPY"].mean()
    weights_matched = generate_matched_exposure_weights(prices_full, rebalance_dates, avg_spy_exposure)
    result_matched = backtest_from_weights(prices_full, weights_matched, COST_PER_SIDE, "MatchedExposure")
    
    # ============================================================
    # TEST 1: BLOCK BOOTSTRAP
    # ============================================================
    print("\n" + "-"*70)
    print("TEST 1: BLOCK BOOTSTRAP (Sharpe Difference)")
    print("-"*70)
    print(f"\nH0: Sharpe_vol <= Sharpe_matched")
    print(f"H1: Sharpe_vol > Sharpe_matched")
    print(f"Bootstrap: {BOOTSTRAP_N} resamples, block size {BOOTSTRAP_BLOCK_SIZE} days")
    
    bootstrap_result = block_bootstrap_sharpe_diff(
        result_vol["net_returns"],
        result_matched["net_returns"]
    )
    
    print(f"\nResults:")
    print(f"  Sharpe Vol-Managed: {bootstrap_result['sharpe_a']:.3f}")
    print(f"  Sharpe Matched: {bootstrap_result['sharpe_b']:.3f}")
    print(f"  Observed Difference: {bootstrap_result['observed_diff']:.3f}")
    print(f"  95% CI: [{bootstrap_result['ci_lower']:.3f}, {bootstrap_result['ci_upper']:.3f}]")
    print(f"  p-value: {bootstrap_result['p_value']:.4f}")
    
    if bootstrap_result['p_value'] < 0.05:
        print(f"\n  VERDICT: SIGNIFICANT at 5% level")
        print(f"  The Sharpe difference is unlikely to be noise.")
    elif bootstrap_result['p_value'] < 0.10:
        print(f"\n  VERDICT: MARGINAL (p < 0.10)")
        print(f"  Suggestive but not conclusive.")
    else:
        print(f"\n  VERDICT: NOT SIGNIFICANT")
        print(f"  The Sharpe difference could be noise.")
    
    # ============================================================
    # TEST 2: COST STRESS
    # ============================================================
    print("\n" + "-"*70)
    print("TEST 2: COST STRESS (Net Sharpe at Different Cost Levels)")
    print("-"*70)
    
    print(f"\n{'Cost/Side':<12} {'Vol Sharpe':>12} {'Matched Sharpe':>15} {'Diff':>8} {'Vol Turnover':>14}")
    print("-"*65)
    
    cost_stress_results = []
    
    for cost in COST_LEVELS:
        # Vol-managed
        result_vol_cost = backtest_from_weights(prices_full, weights_vol, cost, "VolManaged")
        metrics_vol_cost = calculate_metrics(result_vol_cost["net_returns"], "VolManaged")
        
        # Matched
        result_matched_cost = backtest_from_weights(prices_full, weights_matched, cost, "Matched")
        metrics_matched_cost = calculate_metrics(result_matched_cost["net_returns"], "Matched")
        
        diff = metrics_vol_cost["sharpe"] - metrics_matched_cost["sharpe"]
        
        print(f"{cost*10000:<11.0f}bps {metrics_vol_cost['sharpe']:>12.3f} {metrics_matched_cost['sharpe']:>15.3f} {diff:>8.3f} {result_vol_cost['annual_turnover_one_way']:>13.2f}x")
        
        cost_stress_results.append({
            "cost_bps": cost * 10000,
            "vol_sharpe": metrics_vol_cost["sharpe"],
            "matched_sharpe": metrics_matched_cost["sharpe"],
            "diff": diff,
            "vol_cagr": metrics_vol_cost["cagr"],
            "vol_turnover": result_vol_cost["annual_turnover_one_way"],
        })
    
    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    bootstrap_pass = bootstrap_result['p_value'] < 0.05
    cost_50bps_pass = cost_stress_results[-1]["diff"] > 0  # At 50 bps
    
    print(f"\n1. Bootstrap test: {'PASS' if bootstrap_pass else 'FAIL'} (p={bootstrap_result['p_value']:.4f})")
    print(f"2. Cost stress (50bps): {'PASS' if cost_50bps_pass else 'FAIL'} (diff={cost_stress_results[-1]['diff']:.3f})")
    
    if bootstrap_pass and cost_50bps_pass:
        print(f"\nVERDICT: VALIDATED")
        print("Vol-managed demonstrates genuine timing skill.")
        print("The Sharpe difference is statistically significant and survives cost stress.")
    elif bootstrap_pass and not cost_50bps_pass:
        print(f"\nVERDICT: PARTIALLY VALIDATED")
        print("Sharpe difference is significant, but sensitive to high costs.")
        print("Viable only with low-cost execution.")
    elif not bootstrap_pass and cost_50bps_pass:
        print(f"\nVERDICT: NOT SIGNIFICANT")
        print("Survives costs, but Sharpe difference could be noise.")
    else:
        print(f"\nVERDICT: REJECT")
        print("Fails both significance and cost stress tests.")
    
    # Save results
    summary = {
        "bootstrap_p_value": bootstrap_result["p_value"],
        "bootstrap_ci_lower": bootstrap_result["ci_lower"],
        "bootstrap_ci_upper": bootstrap_result["ci_upper"],
        "observed_sharpe_diff": bootstrap_result["observed_diff"],
        "sharpe_at_50bps_vol": cost_stress_results[-1]["vol_sharpe"],
        "sharpe_at_50bps_matched": cost_stress_results[-1]["matched_sharpe"],
        "diff_at_50bps": cost_stress_results[-1]["diff"],
        "bootstrap_pass": bootstrap_pass,
        "cost_50bps_pass": cost_50bps_pass,
    }
    
    pd.DataFrame([summary]).to_csv("results/test9_5_validation_summary.csv", index=False)
    pd.DataFrame(cost_stress_results).to_csv("results/test9_5_cost_stress.csv", index=False)
    print(f"\nResults saved to results/test9_5_validation_summary.csv")
    
    return summary


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run_test9_5()