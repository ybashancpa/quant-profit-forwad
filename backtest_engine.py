"""
Backtest Engine
===============
Core backtesting infrastructure with proper cost modeling.
"""

import numpy as np
import pandas as pd
from config import COST_PER_SIDE, TRADING_DAYS_PER_YEAR


def backtest_from_weights(prices, weights, cost_per_side=COST_PER_SIDE, name="Strategy",
                          financing_rate=0.0, financing_rate_series=None):
    """
    Run backtest given a weights DataFrame.
    
    Args:
        prices: pd.DataFrame of daily prices (dates x assets)
        weights: pd.DataFrame of target weights (dates x assets)
                 Weights are applied with 1-day lag (signal on day t, position on day t+1)
                 Weights may sum to more than 1 (leverage) or less than 1 (cash in that
                 case is assumed to earn zero return).
        cost_per_side: Transaction cost per side
        name: Strategy name
        financing_rate: Annual financing rate charged on borrowed amount, i.e. on the
                 portion of gross exposure above 100%. Applied daily as
                 max(0, gross_exposure - 1) * financing_rate / 252.
                 Default 0.0 keeps backward compatibility with existing tests.
        financing_rate_series: Optional pd.Series of time-varying ANNUAL financing
                 rates indexed by date (e.g. futures-embedded funding ~ risk-free
                 rate). If provided, overrides financing_rate. The series should
                 already be lagged by the caller if look-ahead must be avoided.
    
    Returns:
        dict with:
            - gross_returns: pd.Series of gross daily returns
            - net_returns: pd.Series of net daily returns (after costs and financing)
            - costs: pd.Series of daily cost drag
            - financing_costs: pd.Series of daily financing cost on leverage
            - weights: pd.DataFrame of actual weights
            - turnover: pd.Series of daily turnover
            - avg_gross_exposure: average gross exposure (sum of abs weights)
    """
    # Align prices and weights
    common_dates = prices.index.intersection(weights.index)
    prices = prices.loc[common_dates]
    weights = weights.loc[common_dates]
    
    # Ensure all assets in weights exist in prices
    weights = weights[[c for c in weights.columns if c in prices.columns]]
    
    # Calculate daily asset returns
    asset_returns = prices.pct_change().fillna(0)
    
    # Shift weights by 1 day (signal today, position tomorrow)
    # This prevents look-ahead bias
    position_weights = weights.shift(1).fillna(0)
    
    # Gross portfolio returns
    gross_returns = (position_weights * asset_returns).sum(axis=1)
    
    # Calculate turnover (sum of absolute weight changes)
    weight_changes = position_weights.diff().abs().sum(axis=1)
    weight_changes.iloc[0] = position_weights.iloc[0].abs().sum()  # Initial position
    
    # Cost drag: turnover * cost per side
    # When we change weights, we trade the difference, paying cost on both sides
    costs = weight_changes * cost_per_side
    
    # Gross exposure and financing cost on leverage (exposure above 100%)
    gross_exposure = position_weights.abs().sum(axis=1)
    if financing_rate_series is not None:
        daily_rate = financing_rate_series.reindex(gross_exposure.index).fillna(0.0)
        financing_costs = (gross_exposure - 1.0).clip(lower=0) * daily_rate / TRADING_DAYS_PER_YEAR
    else:
        financing_costs = (gross_exposure - 1.0).clip(lower=0) * financing_rate / TRADING_DAYS_PER_YEAR
    
    # Net returns
    net_returns = gross_returns - costs - financing_costs
    
    # Calculate annualized turnover
    n_years = len(gross_returns) / TRADING_DAYS_PER_YEAR
    total_turnover = weight_changes.sum()
    annual_turnover = total_turnover / n_years / 2  # One-way turnover
    
    return {
        "name": name,
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "costs": costs,
        "financing_costs": financing_costs,
        "weights": position_weights,
        "turnover": weight_changes,
        "annual_turnover_one_way": annual_turnover,
        "total_cost_drag": costs.sum(),
        "total_financing_drag": financing_costs.sum(),
        "avg_gross_exposure": float(gross_exposure.mean()),
    }


def get_monthly_rebalance_dates(dates, freq="M"):
    """
    Get month-end rebalance dates from a daily date index.
    
    Args:
        dates: pd.DatetimeIndex of trading days
        freq: 'M' for month-end, 'W' for week-end
    
    Returns:
        pd.DatetimeIndex: Rebalance dates (actual trading days)
    """
    # Resample to get period-end dates
    if freq == "M":
        # Get last trading day of each month
        rebalance_dates = dates.to_series().groupby(
            [dates.year, dates.month]
        ).last()
    elif freq == "W":
        rebalance_dates = dates.to_series().groupby(
            [dates.year, dates.weekofyear]
        ).last()
    else:
        raise ValueError(f"Unsupported frequency: {freq}")
    
    return pd.DatetimeIndex(rebalance_dates.values)


def forward_fill_weights(signal_weights, all_dates):
    """
    Forward-fill sparse rebalance weights to daily frequency.
    
    Args:
        signal_weights: pd.DataFrame with weights only on rebalance dates
        all_dates: pd.DatetimeIndex of all trading days
    
    Returns:
        pd.DataFrame: Daily weights (forward-filled)
    """
    return signal_weights.reindex(all_dates).ffill().fillna(0)