

"""
Performance Metrics Module
==========================
All metrics calculated on NET returns (after costs).
"""

import numpy as np
import pandas as pd
from config import TRADING_DAYS_PER_YEAR, MAX_DRAWDOWN_LIMIT


def calculate_metrics(returns, name="Strategy", rf_rate=0.0):
    """
    Calculate comprehensive performance metrics.
    
    Args:
        returns: pd.Series of daily returns (net of costs)
        name: Strategy name for reporting
        rf_rate: Annual risk-free rate (default 0 for simplicity)
    
    Returns:
        dict: Performance metrics
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return {"name": name, "error": "No returns data"}

    n_days = len(returns)
    n_years = n_days / TRADING_DAYS_PER_YEAR

    # Total return
    total_return = (1 + returns).prod() - 1

    # CAGR
    if n_years > 0 and (1 + total_return) > 0:
        cagr = (1 + total_return) ** (1 / n_years) - 1
    else:
        cagr = np.nan

    # Volatility (annualized)
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Sharpe Ratio (annualized)
    daily_rf = rf_rate / TRADING_DAYS_PER_YEAR
    excess_returns = returns - daily_rf
    if excess_returns.std() > 0:
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sharpe = np.nan

    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        downside_std = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        sortino = (cagr - rf_rate) / downside_std
    else:
        sortino = np.nan

    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    # Max drawdown date
    max_dd_date = drawdown.idxmin() if not drawdown.empty else None

    # Calmar Ratio
    if max_dd != 0:
        calmar = cagr / abs(max_dd)
    else:
        calmar = np.nan

    # Win rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else np.nan

    # Profit Factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Skewness and Kurtosis
    skew = returns.skew()
    kurt = returns.kurtosis()

    return {
        "name": name,
        "start_date": returns.index[0].date(),
        "end_date": returns.index[-1].date(),
        "n_years": round(n_years, 2),
        "total_return": round(total_return * 100, 2),
        "cagr": round(cagr * 100, 2),
        "ann_volatility": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "max_dd_date": str(max_dd_date.date()) if max_dd_date else None,
        "calmar": round(calmar, 3),
        "win_rate": round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 3),
        "skewness": round(skew, 3),
        "kurtosis": round(kurt, 3),
        "passes_dd_limit": abs(max_dd) <= MAX_DRAWDOWN_LIMIT,
    }


def print_metrics(metrics):
    """Pretty print metrics."""
    print(f"\n{'='*60}")
    print(f"  {metrics['name']}")
    print(f"{'='*60}")
    print(f"  Period:              {metrics['start_date']} to {metrics['end_date']} ({metrics['n_years']} years)")
    print(f"  Total Return:        {metrics['total_return']:.2f}%")
    print(f"  CAGR:                {metrics['cagr']:.2f}%")
    print(f"  Annualized Vol:      {metrics['ann_volatility']:.2f}%")
    print(f"  Sharpe Ratio:        {metrics['sharpe']:.3f}")
    print(f"  Sortino Ratio:       {metrics['sortino']:.3f}")
    print(f"  Max Drawdown:        {metrics['max_drawdown']:.2f}% (on {metrics['max_dd_date']})")
    print(f"  Calmar Ratio:        {metrics['calmar']:.3f}")
    print(f"  Win Rate:            {metrics['win_rate']:.2f}%")
    print(f"  Profit Factor:       {metrics['profit_factor']:.3f}")
    print(f"  Skewness:            {metrics['skewness']:.3f}")
    print(f"  Kurtosis:            {metrics['kurtosis']:.3f}")
    print(f"  {'─'*56}")
    dd_status = "✓ PASS" if metrics['passes_dd_limit'] else "✗ FAIL (>25%)"
    print(f"  Max DD Limit (25%):  {dd_status}")
    print(f"{'='*60}\n")


def calculate_turnover(weights_history):
    """
    Calculate portfolio turnover from weight history.
    
    Args:
        weights_history: pd.DataFrame of portfolio weights (dates x assets)
    
    Returns:
        float: Annualized turnover (one-way)
    """
    # Turnover = sum of absolute weight changes per rebalance
    weight_changes = weights_history.diff().abs().sum(axis=1)
    # Average one-way turnover per period
    avg_turnover_per_period = weight_changes.mean() / 2  # Divide by 2 for one-way
    return avg_turnover_per_period


def calculate_costs_from_turnover(weights_history, cost_per_side):
    """
    Calculate total cost drag from turnover.
    
    Args:
        weights_history: pd.DataFrame of portfolio weights
        cost_per_side: Cost per trade side (e.g., 0.001 for 10bps)
    
    Returns:
        pd.Series: Daily cost drag
    """
    weight_changes = weights_history.diff().abs().sum(axis=1)
    # Cost = total traded value * cost per side (both buy and sell)
    daily_costs = weight_changes * cost_per_side
    return daily_costs.fillna(0)