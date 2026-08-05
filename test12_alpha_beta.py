"""
TEST 12: GLOBAL MOMENTUM ALPHA/BETA REGRESSION (CLOSING MEASUREMENT)
=====================================================================
The one measurement never run explicitly: an alpha/beta regression of
the global (cross-asset) momentum strategy against the market, with
proper HAC (Newey-West) inference and IS/OOS split.

Strategy under test: DualMom 6M/MA150 (locked winner of Test 3.5),
NET of costs — momentum across the full 10-ETF global universe
(equities, bonds, gold, commodities, REITs) with absolute trend filter.

Model (daily, rf = 0, consistent with metrics.py):
    r_strategy,t = alpha + beta * r_SPY,t + eps,t

Inference: OLS + Newey-West HAC standard errors (Bartlett kernel,
21 lags pre-registered — daily returns with monthly-rebalance overlap
are autocorrelated; plain OLS t-stats would be overstated).

Pre-registered prediction (stated before results):
  - alpha positive and significant IN-SAMPLE
  - alpha NOT significant out-of-sample
  - most of the average return explained by beta (disguised beta)

Pass criterion for "alpha with beta <= 1.5":
  OOS alpha t-stat >= 1.96 AND beta <= 1.5 AND alpha > 0.
"""

import os
import numpy as np
import pandas as pd
from data_loader import download_prices
from backtest_engine import backtest_from_weights, get_monthly_rebalance_dates
from config import COST_PER_SIDE

from test3_dual_momentum import generate_dual_momentum_weights
from test3_5_validation import IS_START, IS_END, OOS_START

# ============================================================
# PRE-REGISTERED PARAMETERS
# ============================================================
MOM_LOOKBACK = 6       # months (locked Test 3.5 winner)
MA_LOOKBACK = 150      # days   (locked Test 3.5 winner)
TOP_K = 3
NW_LAGS = 21           # Newey-West lags (~1 month), pre-registered
T_CRIT = 1.96          # two-sided 5%
BETA_LIMIT = 1.5


# ============================================================
# OLS + NEWEY-WEST (manual, numpy-only; statsmodels not installed)
# ============================================================
def newey_west_ols(y, X, lags=NW_LAGS):
    """
    OLS with Newey-West HAC covariance.
    X must include an intercept column. Returns dict with
    coefficients, SEs, t-stats, R^2 and n.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    T, k = X.shape

    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta_hat

    ss_res = float(resid @ resid)
    y_mean = y.mean()
    ss_tot = float(((y - y_mean) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    XtX_inv = np.linalg.inv(X.T @ X)

    # HAC long-run covariance of x_t * e_t (Bartlett kernel)
    xe = X * resid[:, None]
    S = xe.T @ xe  # Gamma_0
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        gamma = xe[lag:].T @ xe[:-lag]
        S += w * (gamma + gamma.T)

    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    t_stats = np.where(se > 0, beta_hat / se, np.nan)

    return {
        "coef": beta_hat,
        "se": se,
        "t": t_stats,
        "r2": r2,
        "n": T,
    }


def run_alpha_beta_regression(strategy_returns, market_returns, label):
    """Run the CAPM-style regression for a given window; return row dict."""
    common = strategy_returns.index.intersection(market_returns.index)
    y = strategy_returns.loc[common].values
    x = market_returns.loc[common].values
    X = np.column_stack([np.ones(len(y)), x])

    fit = newey_west_ols(y, X)
    alpha_d, beta = fit["coef"]
    alpha_t, beta_t = fit["t"]

    # Decomposition of average annualized return
    ann_alpha = alpha_d * 252 * 100
    ann_beta_part = beta * x.mean() * 252 * 100
    ann_total = y.mean() * 252 * 100
    beta_share = (ann_beta_part / ann_total * 100) if ann_total != 0 else np.nan

    row = {
        "period": label,
        "n_days": fit["n"],
        "alpha_annualized_pct": round(ann_alpha, 2),
        "alpha_daily": alpha_d,
        "alpha_tstat": round(alpha_t, 3),
        "alpha_significant": bool(abs(alpha_t) >= T_CRIT and alpha_d > 0),
        "beta": round(beta, 3),
        "beta_tstat": round(beta_t, 3),
        "beta_le_1_5": bool(beta <= BETA_LIMIT),
        "r2": round(fit["r2"], 3),
        "ann_return_pct": round(ann_total, 2),
        "beta_explained_ann_pct": round(ann_beta_part, 2),
        "beta_share_of_return_pct": round(beta_share, 1),
    }
    return row


def print_row(row):
    print(f"\n[{row['period']}] n={row['n_days']} days, R2={row['r2']:.3f}")
    print(f"  Alpha (ann): {row['alpha_annualized_pct']:>6.2f}%  "
          f"t={row['alpha_tstat']:>6.2f}  -> "
          f"{'SIGNIFICANT' if row['alpha_significant'] else 'not significant'}")
    print(f"  Beta:        {row['beta']:>6.3f}  t={row['beta_tstat']:>6.2f}  "
          f"-> {'<= 1.5 OK' if row['beta_le_1_5'] else '> 1.5'}")
    print(f"  Return decomposition (ann): total={row['ann_return_pct']:.2f}% = "
          f"alpha {row['alpha_annualized_pct']:.2f}% + "
          f"beta*market {row['beta_explained_ann_pct']:.2f}% "
          f"({row['beta_share_of_return_pct']:.0f}% beta-explained)")


# ============================================================
# MAIN
# ============================================================
def run_test12():
    print("\n" + "=" * 70)
    print("TEST 12: GLOBAL MOMENTUM ALPHA/BETA REGRESSION")
    print("=" * 70)
    print(f"\nStrategy: DualMom {MOM_LOOKBACK}M/MA{MA_LOOKBACK} (locked), net of "
          f"{COST_PER_SIDE*10000:.0f}bps/side")
    print(f"Inference: OLS + Newey-West HAC, {NW_LAGS} lags | criterion: "
          f"OOS alpha t>= {T_CRIT}, beta <= {BETA_LIMIT}")
    print("\nPre-registered prediction: alpha significant IS, NOT significant OOS, "
          "mostly disguised beta.")

    prices_full = download_prices()
    rebalance_dates = get_monthly_rebalance_dates(prices_full.index, freq="M")

    # Strategy net returns (full period, then sliced — identical generation)
    weights = generate_dual_momentum_weights(
        prices_full, MOM_LOOKBACK, TOP_K, MA_LOOKBACK, rebalance_dates)
    result = backtest_from_weights(prices_full, weights, COST_PER_SIDE,
                                   "GlobalMomentum")
    strat_ret = result["net_returns"]

    # Market factor: SPY daily returns
    market_ret = prices_full["SPY"].pct_change().fillna(0)

    # ----------------------------------------------------------
    # REGRESSIONS: FULL / IS / OOS
    # ----------------------------------------------------------
    print("\n" + "-" * 70)
    print("REGRESSION RESULTS")
    print("-" * 70)

    rows = []
    for label, start, end in [("FULL", None, None),
                              ("IS (2007-2017)", IS_START, IS_END),
                              ("OOS (2018+)", OOS_START, None)]:
        s = strat_ret.loc[start:end] if (start or end) else strat_ret
        m = market_ret.loc[start:end] if (start or end) else market_ret
        row = run_alpha_beta_regression(s, m, label)
        rows.append(row)
        print_row(row)

    # ----------------------------------------------------------
    # DIAGNOSTIC: SmartPassive for context (same regression)
    # ----------------------------------------------------------
    from test9_vol_managed import generate_smart_passive_weights
    w_smart = generate_smart_passive_weights(prices_full, rebalance_dates)
    res_smart = backtest_from_weights(prices_full, w_smart, COST_PER_SIDE,
                                      "SmartPassive")
    print("\n" + "-" * 70)
    print("CONTEXT: SmartPassive MA200 (same regression, full period)")
    print("-" * 70)
    smart_row = run_alpha_beta_regression(res_smart["net_returns"], market_ret,
                                          "SmartPassive FULL")
    print_row(smart_row)

    # ----------------------------------------------------------
    # VERDICT
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    full_row = rows[0]
    is_row = rows[1]
    oos_row = rows[2]

    alpha_pass_oos = oos_row["alpha_significant"]
    beta_ok = oos_row["beta_le_1_5"]
    is_sig = is_row["alpha_significant"]

    print(f"\nIS  alpha: {is_row['alpha_annualized_pct']:+.2f}%/yr "
          f"(t={is_row['alpha_tstat']:.2f}) -> "
          f"{'SIGNIFICANT' if is_sig else 'not significant'}")
    print(f"OOS alpha: {oos_row['alpha_annualized_pct']:+.2f}%/yr "
          f"(t={oos_row['alpha_tstat']:.2f}) -> "
          f"{'SIGNIFICANT' if alpha_pass_oos else 'not significant'}")
    print(f"OOS beta:  {oos_row['beta']:.3f} -> "
          f"{'PASS' if beta_ok else 'FAIL'} (limit {BETA_LIMIT})")

    print("\n" + "-" * 70)
    if alpha_pass_oos and beta_ok:
        print("RESULT: GENUINE OOS ALPHA with beta <= 1.5. "
              "The prediction was WRONG — momentum has real skill.")
    elif is_sig and not alpha_pass_oos:
        print("RESULT: PREDICTION CONFIRMED. Alpha is significant in-sample "
              "but NOT out-of-sample — period-dependent, consistent with "
              "selection/overfitting. No tradable alpha.")
    elif not is_sig and not alpha_pass_oos:
        print("RESULT: NO ALPHA anywhere. Strategy performance is factor "
              "exposure (beta), not skill.")
    else:
        print("RESULT: MIXED — inspect rows above.")
    print("-" * 70)

    # Save
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows + [smart_row]).to_csv("results/test12_alpha_beta.csv",
                                            index=False)
    print("\nResults saved to results/test12_alpha_beta.csv")
    return rows


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    run_test12()