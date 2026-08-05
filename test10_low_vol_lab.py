"""
TEST 10: LOW-VOL STRATEGY LAB
==============================
Strategy lab comparing several low-volatility construction methods on the
same research infrastructure (backtest_engine + metrics + crisis analysis).

Candidates (all pre-registered, long-only, monthly rebalance):
  A. Static Low-Vol      - fixed diversified allocation (40/40/20)
  B. Risk Parity (IV)    - capped inverse-volatility weighting (60d vol)
  C. RP + Vol Targeting  - B scaled to TARGET_VOL, max leverage 150%
  D. Defensive VolTarget - C with SPY MA200 de-risking overlay

Benchmarks: SPY 100%, Static 60/40, SmartPassive MA200 (55/35/10).

Goal: test whether any candidate can beat the market (SPY) or achieve a
persistent double-digit CAGR after costs AND financing, while respecting
the 25% max-drawdown limit. No guarantee is assumed; results are reported
as-is, with OOS (2018+) kept separate from parameter design.

New cost realism: leverage above 100% gross exposure is charged an annual
financing rate (primary: 3%/yr on the borrowed portion), handled by the
extended backtest engine.

Sensitivity (report only, no selection):
  - Target vol: 10% / 12% / 15%
  - Financing rate: 2% / 3% / 5%
"""

import os
import pandas as pd
import numpy as np
from data_loader import download_prices
from backtest_engine import (
    backtest_from_weights,
    get_monthly_rebalance_dates,
    forward_fill_weights,
)
from metrics import calculate_metrics
from config import COST_PER_SIDE, SAFE_ASSET

# ============================================================
# PRE-REGISTERED PARAMETERS (NO FITTING)
# ============================================================
VOL_LOOKBACK_DAYS = 60        # realized vol window (trading days)
TARGET_VOL = 0.12             # primary annualized portfolio vol target
MAX_LEVERAGE = 1.5            # hard cap on gross exposure (150%)
PER_ASSET_CAP = 0.30          # per-asset cap in inverse-vol weights
DEFENSIVE_MA = 200            # MA lookback for defensive overlay
DEFENSIVE_SCALE = 0.5         # exposure multiplier when SPY < MA200
FINANCING_RATE = 0.03         # annual cost on borrowed amount (primary)

# Static low-vol allocation: 40% equity / 40% bonds / 20% real assets
STATIC_LOWVOL_ALLOC = {
    "SPY": 0.20, "QQQ": 0.05, "EFA": 0.10, "EEM": 0.05,
    "TLT": 0.10, "IEF": 0.20, "SHY": 0.10,
    "GLD": 0.10, "DBC": 0.05, "VNQ": 0.05,
}

STATIC_60_40 = {"SPY": 0.60, "IEF": 0.40}
SMART_RISK_ON = {"SPY": 0.55, "IEF": 0.35, "GLD": 0.10}
SMART_RISK_OFF = {"SHY": 0.55, "IEF": 0.35, "GLD": 0.10}

# Sensitivity grid (report only)
SENS_TARGET_VOLS = [0.10, 0.12, 0.15]
SENS_FINANCING_RATES = [0.02, 0.03, 0.05]

# Period split (consistent with Test 3.5 / Test 4)
IS_START = "2007-01-01"
IS_END = "2017-12-31"
OOS_START = "2018-01-01"

CRISIS_PERIODS = {
    "2008 Financial Crisis": ("2008-01-01", "2009-03-31"),
    "2020 Corona Crash": ("2020-02-01", "2020-04-30"),
    "2022 Inflation/Rates": ("2022-01-01", "2022-12-31"),
}


# ============================================================
# WEIGHT GENERATORS
# ============================================================
def cap_weights(weights, cap=PER_ASSET_CAP):
    """
    Iteratively cap per-asset weights and redistribute the excess
    proportionally to uncapped assets. Input/output: pd.Series.
    """
    w = weights.copy()
    for _ in range(20):  # safety bound on iterations
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        if w[under].sum() <= 0:
            break
        w[under] += excess * (w[under] / w[under].sum())
    return w


def generate_static_weights(prices, rebalance_dates, allocation):
    """Fixed allocation, monthly rebalance back to targets."""
    sparse = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    for date in rebalance_dates:
        for asset, weight in allocation.items():
            if asset in prices.columns:
                sparse.loc[date, asset] = weight
    return forward_fill_weights(sparse, prices.index)


def compute_risk_parity_sparse(prices, rebalance_dates, vol_lookback=VOL_LOOKBACK_DAYS):
    """
    Capped inverse-volatility weights at each rebalance date.
    Uses trailing realized vol only (no look-ahead).
    """
    daily_returns = prices.pct_change()
    rolling_vol = daily_returns.rolling(vol_lookback, min_periods=vol_lookback).std()

    sparse = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    for date in rebalance_dates:
        if date not in rolling_vol.index:
            continue
        vols = rolling_vol.loc[date].dropna()
        vols = vols[vols > 0]
        if len(vols) < 3:
            continue  # not enough history
        inv_vol = 1.0 / vols
        w = inv_vol / inv_vol.sum()
        w = cap_weights(w, PER_ASSET_CAP)
        sparse.loc[date, w.index] = w.values
    return sparse


def generate_risk_parity_weights(prices, rebalance_dates):
    """Daily forward-filled capped risk-parity weights (unlevered)."""
    sparse = compute_risk_parity_sparse(prices, rebalance_dates)
    return forward_fill_weights(sparse, prices.index)


def generate_vol_target_weights(prices, rebalance_dates, target_vol=TARGET_VOL,
                                max_leverage=MAX_LEVERAGE, defensive=False,
                                vol_lookback=VOL_LOOKBACK_DAYS):
    """
    Risk-parity weights scaled by target_vol / realized_portfolio_vol.

    - realized_portfolio_vol = annualized trailing `vol_lookback`-day std of
      the UNLEVERED risk-parity portfolio's own realized returns (positions
      lagged exactly as in the backtest engine -> no look-ahead).
    - scale = min(max_leverage, target_vol / realized_vol); NaN -> 1.0
    - Defensive overlay: if SPY <= MA(DEFENSIVE_MA), scale *= DEFENSIVE_SCALE
    - scale < 1: remainder goes to SHY; scale > 1: proportional leverage.

    Returns (daily_weights, scale_history).
    """
    rp_sparse = compute_risk_parity_sparse(prices, rebalance_dates, vol_lookback)
    rp_daily = forward_fill_weights(rp_sparse, prices.index)

    # Realized return of the unlevered RP portfolio (1-day lag, as in engine)
    asset_returns = prices.pct_change().fillna(0)
    rp_portfolio_returns = (rp_daily.shift(1).fillna(0) * asset_returns).sum(axis=1)
    realized_vol = (
        rp_portfolio_returns.rolling(vol_lookback, min_periods=vol_lookback).std()
        * np.sqrt(252)
    )

    ma = None
    if defensive:
        ma = prices["SPY"].rolling(DEFENSIVE_MA, min_periods=DEFENSIVE_MA).mean()

    sparse = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    scale_history = pd.Series(index=rebalance_dates, dtype=float)

    for date in rebalance_dates:
        if date not in prices.index:
            continue
        base = rp_sparse.loc[date] if date in rp_sparse.index else None
        if base is None or base.sum() <= 0:
            # Warm-up: hold safe asset
            if SAFE_ASSET in prices.columns:
                sparse.loc[date, SAFE_ASSET] = 1.0
            scale_history.loc[date] = 0.0
            continue

        vol = realized_vol.loc[date] if date in realized_vol.index else np.nan
        if pd.isna(vol) or vol <= 0:
            scale = 1.0
        else:
            scale = min(max_leverage, target_vol / vol)

        if defensive:
            spy_ma = ma.loc[date]
            if pd.isna(spy_ma) or prices.loc[date, "SPY"] <= spy_ma:
                scale *= DEFENSIVE_SCALE

        w = base * scale
        if scale < 1.0 and SAFE_ASSET in prices.columns:
            w[SAFE_ASSET] = w.get(SAFE_ASSET, 0.0) + (1.0 - scale)

        sparse.loc[date, w.index] = w.values
        scale_history.loc[date] = scale

    daily_weights = forward_fill_weights(sparse, prices.index)
    return daily_weights, scale_history


def generate_smart_passive_weights(prices, rebalance_dates):
    """SmartPassive MA200 benchmark (same rule as Tests 7-9)."""
    moving_avg = prices["SPY"].rolling(DEFENSIVE_MA, min_periods=DEFENSIVE_MA).mean()
    sparse = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    for date in rebalance_dates:
        if date not in prices.index:
            continue
        spy_ma = moving_avg.loc[date]
        if pd.isna(spy_ma) or prices.loc[date, "SPY"] <= spy_ma:
            alloc = SMART_RISK_OFF
        else:
            alloc = SMART_RISK_ON
        for asset, weight in alloc.items():
            if asset in prices.columns:
                sparse.loc[date, asset] = weight
    return forward_fill_weights(sparse, prices.index)


# ============================================================
# ANALYSIS HELPERS
# ============================================================
def analyze_crisis_periods(returns):
    """Return total return and max DD for each predefined crisis window."""
    results = {}
    for crisis_name, (start, end) in CRISIS_PERIODS.items():
        mask = (returns.index >= start) & (returns.index <= end)
        crisis_returns = returns[mask]
        if len(crisis_returns) == 0:
            results[crisis_name] = {"return": None, "max_dd": None}
            continue
        total_return = (1 + crisis_returns).prod() - 1
        cumulative = (1 + crisis_returns).cumprod()
        drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
        results[crisis_name] = {
            "return": total_return * 100,
            "max_dd": drawdown.min() * 100,
        }
    return results


def build_all_weights(prices, rebalance_dates):
    """Generate weights for every strategy in the lab."""
    weights = {}
    scales = {}

    weights["StaticLowVol"] = generate_static_weights(prices, rebalance_dates, STATIC_LOWVOL_ALLOC)
    weights["RiskParity"] = generate_risk_parity_weights(prices, rebalance_dates)

    vt_weights, vt_scale = generate_vol_target_weights(
        prices, rebalance_dates, TARGET_VOL, MAX_LEVERAGE, defensive=False)
    weights["RP_VolTarget"] = vt_weights
    scales["RP_VolTarget"] = vt_scale

    dvt_weights, dvt_scale = generate_vol_target_weights(
        prices, rebalance_dates, TARGET_VOL, MAX_LEVERAGE, defensive=True)
    weights["DefensiveVT"] = dvt_weights
    scales["DefensiveVT"] = dvt_scale

    weights["SPY"] = generate_static_weights(prices, rebalance_dates, {"SPY": 1.0})
    weights["60_40"] = generate_static_weights(prices, rebalance_dates, STATIC_60_40)
    weights["SmartPassive"] = generate_smart_passive_weights(prices, rebalance_dates)
    return weights, scales


def run_lab(prices, label="FULL", financing_rate=FINANCING_RATE, verbose=True):
    """Run all strategies on the given price window; return results dict."""
    rebalance_dates = get_monthly_rebalance_dates(prices.index, freq="M")
    weights, scales = build_all_weights(prices, rebalance_dates)

    results = {}
    for name, w in weights.items():
        res = backtest_from_weights(prices, w, COST_PER_SIDE, name,
                                    financing_rate=financing_rate)
        res["metrics"] = calculate_metrics(res["net_returns"], name)
        res["scale_history"] = scales.get(name)
        results[name] = res

    if verbose:
        print(f"\n[{label}] Period: {prices.index[0].date()} to {prices.index[-1].date()}"
              f" | financing={financing_rate*100:.0f}%/yr on leverage | cost={COST_PER_SIDE*10000:.0f}bps/side")
        header = (f"{'Strategy':<15} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>9} "
                  f"{'Vol':>7} {'Turn/yr':>8} {'AvgGross':>9} {'FinDrag':>8}")
        print("\n" + header)
        print("-" * len(header))
        for name, res in results.items():
            m = res["metrics"]
            print(f"{name:<15} {m['cagr']:>7.2f}% {m['sharpe']:>8.3f} "
                  f"{m['max_drawdown']:>8.2f}% {m['ann_volatility']:>6.2f}% "
                  f"{res['annual_turnover_one_way']:>7.2f}x "
                  f"{res['avg_gross_exposure']*100:>8.1f}% "
                  f"{res['total_financing_drag']*100:>7.2f}%")
    return results


def validate_weights(prices):
    """Sanity checks on the full-period weight paths."""
    print("\n" + "-" * 70)
    print("VALIDATION CHECKS")
    print("-" * 70)
    rebalance_dates = get_monthly_rebalance_dates(prices.index, freq="M")
    weights, scales = build_all_weights(prices, rebalance_dates)

    ok = True
    for name, w in weights.items():
        daily_sum = w.sum(axis=1)
        max_gross = w.abs().sum(axis=1).max()
        has_nan = bool(w.isna().any().any())
        sum_ok = bool(((daily_sum <= 1.001) | (name in ("RP_VolTarget", "DefensiveVT"))).all())
        lev_ok = max_gross <= MAX_LEVERAGE + 1e-6
        checks = []
        if has_nan:
            checks.append("NaN FOUND")
            ok = False
        if not sum_ok:
            checks.append("SUM>1 in unlevered strategy")
            ok = False
        if name in ("RP_VolTarget", "DefensiveVT") and not lev_ok:
            checks.append(f"LEVERAGE CAP BREACHED ({max_gross:.3f})")
            ok = False
        status = "OK" if not checks else "; ".join(checks)
        print(f"  {name:<15} max_gross={max_gross:6.3f}  min_sum={daily_sum.min():6.3f}  "
              f"max_sum={daily_sum.max():6.3f}  -> {status}")

    # Scale distribution for vol-targeted strategies
    for name in ("RP_VolTarget", "DefensiveVT"):
        s = scales[name].dropna()
        if len(s) > 0:
            levered_frac = (s > 1.0).mean() * 100
            print(f"  {name:<15} avg_scale={s.mean():.3f}  median={s.median():.3f}  "
                  f"max={s.max():.3f}  levered_months={levered_frac:.1f}%")
    print(f"\n  Overall validation: {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================
# MAIN
# ============================================================
def run_test10():
    print("\n" + "=" * 70)
    print("TEST 10: LOW-VOL STRATEGY LAB")
    print("=" * 70)
    print(f"\nPre-registered parameters:")
    print(f"  Vol lookback: {VOL_LOOKBACK_DAYS}d | Target vol: {TARGET_VOL*100:.0f}%")
    print(f"  Max leverage: {MAX_LEVERAGE*100:.0f}% | Per-asset cap: {PER_ASSET_CAP*100:.0f}%")
    print(f"  Defensive overlay: SPY<MA{DEFENSIVE_MA} -> scale x{DEFENSIVE_SCALE}")
    print(f"  Financing: {FINANCING_RATE*100:.0f}%/yr on borrowed amount")
    print(f"  Goal: beat SPY or reach double-digit CAGR, MaxDD <= 25%")

    prices_full = download_prices()

    # ----------------------------------------------------------
    # 1. VALIDATION
    # ----------------------------------------------------------
    validate_weights(prices_full)

    # ----------------------------------------------------------
    # 2. FULL PERIOD LAB
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("FULL PERIOD LAB (2007-2026)")
    print("=" * 70)
    full_results = run_lab(prices_full, label="FULL")

    # ----------------------------------------------------------
    # 3. CRISIS ANALYSIS
    # ----------------------------------------------------------
    print("\n" + "-" * 70)
    print("CRISIS ANALYSIS (return% / maxDD%)")
    print("-" * 70)
    crisis_rows = []
    key_strategies = ["StaticLowVol", "RiskParity", "RP_VolTarget",
                      "DefensiveVT", "SPY", "60_40", "SmartPassive"]
    for crisis_name in CRISIS_PERIODS.keys():
        print(f"\n{crisis_name}:")
        print(f"  {'Strategy':<15} {'Return':>9} {'MaxDD':>9}")
        for name in key_strategies:
            c = analyze_crisis_periods(full_results[name]["net_returns"])[crisis_name]
            if c["return"] is None:
                continue
            print(f"  {name:<15} {c['return']:>8.2f}% {c['max_dd']:>8.2f}%")
            crisis_rows.append({
                "crisis": crisis_name, "strategy": name,
                "return_pct": round(c["return"], 2), "max_dd_pct": round(c["max_dd"], 2),
            })

    # ----------------------------------------------------------
    # 4. IS / OOS SPLIT
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"IN-SAMPLE ({IS_START[:4]}-{IS_END[:4]}) vs OUT-OF-SAMPLE ({OOS_START[:4]}+)")
    print("=" * 70)
    prices_is = prices_full.loc[IS_START:IS_END]
    prices_oos = prices_full.loc[OOS_START:]
    is_results = run_lab(prices_is, label="IS")
    oos_results = run_lab(prices_oos, label="OOS")

    # ----------------------------------------------------------
    # 5. SENSITIVITY (report only)
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("SENSITIVITY: RP_VolTarget (full period, report only)")
    print("=" * 70)
    rebalance_dates = get_monthly_rebalance_dates(prices_full.index, freq="M")
    sens_rows = []
    print(f"\n{'TargetVol':<10} {'Financing':<10} {'CAGR':>8} {'Sharpe':>8} "
          f"{'MaxDD':>9} {'Turn/yr':>8}")
    print("-" * 58)
    for tv in SENS_TARGET_VOLS:
        for fr in SENS_FINANCING_RATES:
            w, _ = generate_vol_target_weights(prices_full, rebalance_dates,
                                               target_vol=tv, max_leverage=MAX_LEVERAGE)
            res = backtest_from_weights(prices_full, w, COST_PER_SIDE,
                                        f"VT_{tv}_{fr}", financing_rate=fr)
            m = calculate_metrics(res["net_returns"], f"VT_{tv}_{fr}")
            marker = " <-- PRIMARY" if (tv == TARGET_VOL and fr == FINANCING_RATE) else ""
            print(f"{tv*100:<9.0f}% {fr*100:<9.0f}% {m['cagr']:>7.2f}% {m['sharpe']:>8.3f} "
                  f"{m['max_drawdown']:>8.2f}% {res['annual_turnover_one_way']:>7.2f}x{marker}")
            sens_rows.append({
                "target_vol": tv, "financing_rate": fr, "cagr": m["cagr"],
                "sharpe": m["sharpe"], "max_dd": m["max_drawdown"],
                "turnover": res["annual_turnover_one_way"],
            })

    # ----------------------------------------------------------
    # 6. VERDICT
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    candidates = ["StaticLowVol", "RiskParity", "RP_VolTarget", "DefensiveVT"]
    spy_oos = oos_results["SPY"]["metrics"]
    smart_oos = oos_results["SmartPassive"]["metrics"]

    print(f"\nOOS reference: SPY CAGR={spy_oos['cagr']:.2f}% Sharpe={spy_oos['sharpe']:.3f} "
          f"MaxDD={spy_oos['max_drawdown']:.2f}% | SmartPassive CAGR={smart_oos['cagr']:.2f}% "
          f"Sharpe={smart_oos['sharpe']:.3f}")

    verdict_rows = []
    for name in candidates:
        m_full = full_results[name]["metrics"]
        m_oos = oos_results[name]["metrics"]
        beats_spy_cagr = m_oos["cagr"] > spy_oos["cagr"]
        beats_spy_sharpe = m_oos["sharpe"] > spy_oos["sharpe"]
        double_digit_full = m_full["cagr"] >= 10.0
        double_digit_oos = m_oos["cagr"] >= 10.0
        dd_pass_full = m_full["passes_dd_limit"]
        dd_pass_oos = m_oos["passes_dd_limit"]

        print(f"\n{name}:")
        print(f"  Full: CAGR={m_full['cagr']:.2f}% Sharpe={m_full['sharpe']:.3f} "
              f"MaxDD={m_full['max_drawdown']:.2f}%")
        print(f"  OOS:  CAGR={m_oos['cagr']:.2f}% Sharpe={m_oos['sharpe']:.3f} "
              f"MaxDD={m_oos['max_drawdown']:.2f}%")
        print(f"  OOS beats SPY CAGR: {'YES' if beats_spy_cagr else 'NO'} | "
              f"OOS beats SPY Sharpe: {'YES' if beats_spy_sharpe else 'NO'}")
        print(f"  Double-digit CAGR: full={'YES' if double_digit_full else 'NO'} "
              f"OOS={'YES' if double_digit_oos else 'NO'}")
        print(f"  25% DD limit: full={'PASS' if dd_pass_full else 'FAIL'} "
              f"OOS={'PASS' if dd_pass_oos else 'FAIL'}")

        verdict_rows.append({
            "strategy": name,
            "full_cagr": m_full["cagr"], "full_sharpe": m_full["sharpe"],
            "full_max_dd": m_full["max_drawdown"],
            "oos_cagr": m_oos["cagr"], "oos_sharpe": m_oos["sharpe"],
            "oos_max_dd": m_oos["max_drawdown"],
            "oos_beats_spy_cagr": beats_spy_cagr,
            "oos_beats_spy_sharpe": beats_spy_sharpe,
            "double_digit_full": double_digit_full,
            "double_digit_oos": double_digit_oos,
            "dd_pass_full": dd_pass_full, "dd_pass_oos": dd_pass_oos,
            "full_turnover": full_results[name]["annual_turnover_one_way"],
            "full_avg_gross_exposure": full_results[name]["avg_gross_exposure"],
            "full_financing_drag_pct": full_results[name]["total_financing_drag"] * 100,
        })

    # Best candidate selection (rule-based, reported not optimized)
    eligible = [r for r in verdict_rows
                if r["dd_pass_full"] and r["dd_pass_oos"]
                and (r["double_digit_oos"] or r["oos_beats_spy_cagr"])]
    print("\n" + "-" * 70)
    if eligible:
        best = max(eligible, key=lambda r: r["oos_sharpe"])
        print(f"LAB VERDICT: '{best['strategy']}' is the leading candidate.")
        print(f"  OOS CAGR={best['oos_cagr']:.2f}%, Sharpe={best['oos_sharpe']:.3f}, "
              f"MaxDD={best['oos_max_dd']:.2f}%")
        print("  Recommend cost stress + block bootstrap before paper trading.")
    else:
        print("LAB VERDICT: NO candidate met the bar "
              "(double-digit OOS CAGR or beating SPY, with DD <= 25%).")
        print("  Report honest negative result; do not force-fit parameters.")
    print("-" * 70)

    # ----------------------------------------------------------
    # 7. SAVE RESULTS
    # ----------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(verdict_rows).to_csv("results/test10_low_vol_lab_summary.csv", index=False)
    pd.DataFrame(sens_rows).to_csv("results/test10_sensitivity.csv", index=False)
    pd.DataFrame(crisis_rows).to_csv("results/test10_crisis.csv", index=False)
    print("\nResults saved to results/test10_low_vol_lab_summary.csv, "
          "test10_sensitivity.csv, test10_crisis.csv")

    return verdict_rows


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    run_test10()