"""
Configuration for Quant Research Backtesting Framework
=====================================================
Universe: 10 liquid ETFs across asset classes
Cost model: 0.1% per side (conservative flat baseline)
Risk constraint: Max drawdown 25% hard kill
"""

# ============================================================
# UNIVERSE DEFINITION
# ============================================================
UNIVERSE = {
    # US Equities
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    # International Equities
    "EFA": "Developed Markets (ex-US)",
    "EEM": "Emerging Markets",
    # Bonds
    "TLT": "US Treasury 20+ Year",
    "IEF": "US Treasury 7-10 Year",
    "SHY": "US Treasury 1-3 Year (cash proxy)",
    # Real Assets
    "GLD": "Gold",
    "DBC": "Commodities",
    "VNQ": "US Real Estate (REITs)",
}

TICKERS = list(UNIVERSE.keys())

# Safe haven / cash proxy for defensive allocation
SAFE_ASSET = "SHY"

# ============================================================
# DATA PARAMETERS
# ============================================================
DATA_START = "2007-01-01"  # Include 2008 crisis, Eurozone crisis, etc.
DATA_END = None  # None = today
DATA_INTERVAL = "1d"
DATA_DIR = "data"
PRICE_FILE = "etf_prices.csv"

# ============================================================
# COST MODEL (Conservative flat baseline)
# ============================================================
COST_PER_SIDE = 0.001  # 0.1% = 10 bps per side (commission + spread + slippage)
# Round trip cost = 0.2%

# ============================================================
# RISK CONSTRAINTS
# ============================================================
MAX_DRAWDOWN_LIMIT = 0.25  # 25% hard kill
INITIAL_CAPITAL = 10_000  # $10,000 portfolio

# ============================================================
# BACKTEST PARAMETERS
# ============================================================
TRADING_DAYS_PER_YEAR = 252
REBALANCE_FREQ = "M"  # Monthly rebalancing

# Momentum lookback candidates (in months)
MOMENTUM_LOOKBACKS_MONTHS = [3, 6, 9, 12]

# Time-series momentum: moving average lookbacks (days)
TS_MOM_MA_LOOKBACKS = [100, 150, 200]

# Volatility lookback for inverse-vol weighting (days)
VOL_LOOKBACK_DAYS = 60

# ============================================================
# WALK-FORWARD / VALIDATION SPLITS
# ============================================================
# In-sample: 2007-2019, Out-of-sample: 2020-present
OOS_START = "2020-01-01"