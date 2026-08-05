"""
TEST 13: ADAPTIVE CRISIS ROTATION (the legitimate adaptive system)
===================================================================
Design principle (pre-registered): the ESCAPE UNIVERSE is fixed; the
CHOICE within it is dynamic and driven by live momentum. Never encode
"crisis -> bonds" from 2008 hindsight.

Layers
------
1. Core (calm): capped inverse-vol risk parity on the 10-ETF universe.
2. Regime detector (pre-defined, lagging by construction):
     trend stress: SPY <= MA200
     vol stress:   60d realized SPY vol > 25%/yr (pre-registered round level)
   PRIMARY hysteresis: enter stress if EITHER flag on; exit only when
   BOTH off (sticky exit). Symmetric exit tested as sensitivity only.
3. Escape universe (FIXED): TLT, IEF, GLD, DBC (+SHY fallback).
4. Rotation logic: in stress, hold the top-2 escape assets with
   POSITIVE 6m trailing momentum, equal weight. 1 positive -> 50/50
   with SHY. None positive -> 100% SHY. No pre-coded winner.
5. Guardrails: long-only, no leverage, rolling 252d beta vs SPY capped
   at 1.5 (scale down + SHY), 25% DD as reject criterion (not a
   path-dependent stop), monthly rebalance, 10bps/side.
6. Validation: IS/OOS, crisis log, Newey-West alpha, block bootstrap,
   multiple-testing correction, turnover split calm/stress.

Critical controls
-----------------
- SHY-only escape and IEF-fixed escape (the 2008 trap) benchmarks.
- Matched-beta SPY/SHY blend (disguised-beta control).
- STATIC-AVERAGE-ESCAPE (the decisive control): static portfolio with
  the time-average weights of the adaptive strategy.
    * implementable: average weights learned on IS, frozen for OOS
    * ex-post diagnostic: same-window average (non-tradable upper bound)
  If adaptive cannot beat the IS-trained static mix, "adaptivity" is a
  composition effect, not timing skill.

Multiple testing: project-wide Bonferroni alpha = 0.05/13 ~= 0.00385.
p<0.05 but >=0.00385 is NOMINAL ONLY. Holm within this test's family.

Modern addendum (n=1 case study): DBMF/KMLM/XLU/XLP added to the
escape universe from their common post-warmup start (~mid-2021). This
is a 2022 anecdote, NOT evidence, and never feeds the primary verdict.
"""

import os
import numpy as np
import pandas as pd
from data_loader import download_prices
from backtest_engine import (
    backtest_from_weights,
    get_monthly_rebalance_dates,
    forward_fill_weights,
)
from metrics import calculate_metrics
from config import COST_PER_SIDE, SAFE_ASSET

from test10_low_vol_lab import (
    compute_risk_parity_sparse,
    CRISIS_PERIODS,
    IS_START,
    IS_END,
    OOS_START,
)
from test12_alpha_beta import newey_west_ols
from test9_5_validation import block_bootstrap_sharpe_diff

# ============================================================
# PRE-REGISTERED PARAMETERS
# ============================================================
MA_TREND = 200
VOL_WINDOW = 60
VOL_STRESS_THRESHOLD = 0.25
MOM_LOOKBACK_MONTHS = 6
TOP_ESCAPE = 2
ESCAPE_RANKED = ["TLT", "IEF", "GLD", "DBC"]   # SHY = fallback only
BETA_LIMIT = 1.5
BETA_WINDOW = 252
N_PROJECT_TESTS = 13
ALPHA_PROJECT = 0.05 / N_PROJECT_TESTS          # ~0.00385
COST_STRESS = 0.005


# ============================================================
# REGIME DETECTION + ADAPTIVE WEIGHTS
# ============================================================
def stress_flags(prices, vol_thr=VOL_STRESS_THRESHOLD):
    """Daily trend/vol stress flags (point-in-time)."""
    ma = prices["SPY"].rolling(MA_TREND, min_periods=MA_TREND).mean()
    trend = prices["SPY"] <= ma
    rv = (prices["SPY"].pct_change()
          .rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std() * np.sqrt(252))
    vol = rv > vol_thr
    trend = trend.where(ma.notna(), False)
    vol = vol.where(rv.notna(), False)
    return trend.fillna(False), vol.fillna(False)


def escape_allocation(prices, date, mom_days, escape_ranked=ESCAPE_RANKED):
    """Top-2 positive-momentum escape assets; SHY fallback. Returns dict."""
    cand = [a for a in escape_ranked if a in prices.columns]
    moms = {}
    for a in cand:
        idx = prices.index.get_loc(date)
        if idx < mom_days:
            continue
        p0 = prices[a].iloc[idx - mom_days]
        p1 = prices[a].iloc[idx]
        if pd.notna(p0) and p0 > 0 and pd.notna(p1):
            moms[a] = p1 / p0 - 1.0
    pos = {a: m for a, m in moms.items() if m > 0}
    top = sorted(pos, key=pos.get, reverse=True)[:TOP_ESCAPE]
    w = {}
    if len(top) >= 2:
        for a in top:
            w[a] = 1.0 / TOP_ESCAPE
    elif len(top) == 1:
        w[top[0]] = 0.5
        w[SAFE_ASSET] = 0.5
    else:
        w[SAFE_ASSET] = 1.0
    return w, top


def generate_adaptive_weights(prices, rebalance_dates, symmetric_exit=False,
                              vol_thr=VOL_STRESS_THRESHOLD,
                              mom_months=MOM_LOOKBACK_MONTHS,
                              escape_ranked=ESCAPE_RANKED):
    """
    Adaptive crisis rotation weights + daily stress state.
    Returns (daily_weights, daily_state, selection_log).
    """
    mom_days = mom_months * 21
    trend, vol = stress_flags(prices, vol_thr)
    rp_sparse = compute_risk_parity_sparse(prices, rebalance_dates)

    sparse = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    state_at_reb = pd.Series(False, index=rebalance_dates)
    log = []
    in_stress = False

    for date in rebalance_dates:
        if date not in prices.index:
            continue
        t_flag = bool(trend.loc[date]) if date in trend.index else False
        v_flag = bool(vol.loc[date]) if date in vol.index else False

        if not in_stress:
            if t_flag or v_flag:
                in_stress = True
        else:
            if symmetric_exit:
                if not (t_flag and v_flag):
                    in_stress = False
            else:
                if not t_flag and not v_flag:
                    in_stress = False

        state_at_reb.loc[date] = in_stress
        if in_stress:
            esc_w, picked = escape_allocation(prices, date, mom_days, escape_ranked)
            for a, wt in esc_w.items():
                sparse.loc[date, a] = wt
            log.append({"date": date, "picked": "|".join(picked) if picked else "SHY",
                        "trend": t_flag, "vol": v_flag})
        else:
            base = rp_sparse.loc[date] if date in rp_sparse.index else None
            if base is not None and base.sum() > 0:
                sparse.loc[date, base.index] = base.values
            elif SAFE_ASSET in prices.columns:
                sparse.loc[date, SAFE_ASSET] = 1.0

    # Beta guardrail: rolling beta of the pre-guardrail strategy vs SPY
    daily_pre = forward_fill_weights(sparse, prices.index)
    asset_ret = prices["SPY"].pct_change().fillna(0)
    strat_ret = (daily_pre.shift(1).fillna(0) * prices.pct_change().fillna(0)).sum(axis=1)
    cov = strat_ret.rolling(BETA_WINDOW, min_periods=BETA_WINDOW).cov(asset_ret)
    var = asset_ret.rolling(BETA_WINDOW, min_periods=BETA_WINDOW).var()
    beta_roll = (cov / var).shift(1)  # lagged: no look-ahead

    for date in rebalance_dates:
        if date not in beta_roll.index:
            continue
        b = beta_roll.loc[date]
        if pd.notna(b) and b > BETA_LIMIT:
            row = sparse.loc[date]
            f = BETA_LIMIT / b
            scaled = row * f
            scaled[SAFE_ASSET] = scaled.get(SAFE_ASSET, 0.0) + (1.0 - f)
            sparse.loc[date] = scaled

    daily_weights = forward_fill_weights(sparse, prices.index)
    daily_state = state_at_reb.reindex(prices.index).ffill().fillna(False).astype(bool)
    return daily_weights, daily_state, pd.DataFrame(log)


def generate_fixed_escape_weights(prices, rebalance_dates, fixed_asset):
    """Calm = RP core; stress (same detector) -> 100% fixed_asset (control)."""
    trend, vol = stress_flags(prices)
    rp_sparse = compute_risk_parity_sparse(prices, rebalance_dates)
    sparse = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    in_stress = False
    for date in rebalance_dates:
        if date not in prices.index:
            continue
        t_flag = bool(trend.loc[date]); v_flag = bool(vol.loc[date])
        if not in_stress and (t_flag or v_flag):
            in_stress = True
        elif in_stress and not t_flag and not v_flag:
            in_stress = False
        if in_stress:
            if fixed_asset in prices.columns:
                sparse.loc[date, fixed_asset] = 1.0
        else:
            base = rp_sparse.loc[date] if date in rp_sparse.index else None
            if base is not None and base.sum() > 0:
                sparse.loc[date, base.index] = base.values
    return forward_fill_weights(sparse, prices.index)


def static_weights_from_series(prices, rebalance_dates, w_series):
    """Static portfolio holding fixed weights (monthly rebalance)."""
    sparse = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    for a in w_series.index:
        if a in sparse.columns and w_series[a] > 0:
            sparse[a] = w_series[a]
    return forward_fill_weights(sparse, prices.index)


# ============================================================
# ANALYSIS HELPERS
# ============================================================
def metrics_row(result, name):
    m = calculate_metrics(result["net_returns"], name)
    return m, result


def turnover_by_state(result, daily_state):
    """Split one-way turnover by calm/stress regime."""
    wc = result["turnover"]
    to = wc.abs().sum(axis=1) / 2 if wc.ndim > 1 else wc.abs() / 2
    st = daily_state.reindex(to.index).fillna(False).astype(bool)
    days_calm = int((~st).sum()); days_stress = int(st.sum())
    t_calm = float(to[~st].sum())
    t_stress = float(to[st].sum())
    n_years = len(to) / 252
    return {
        "days_calm": days_calm, "days_stress": days_stress,
        "pct_time_stress": round(days_stress / max(len(to), 1) * 100, 1),
        "turnover_calm_ann": round(t_calm / n_years, 2),
        "turnover_stress_ann": round(t_stress / n_years, 2),
    }


def nw_alpha_beta(strat_ret, market_ret, label):
    common = strat_ret.index.intersection(market_ret.index)
    y = strat_ret.loc[common].values
    x = market_ret.loc[common].values
    X = np.column_stack([np.ones(len(y)), x])
    fit = newey_west_ols(y, X)
    return {"label": label, "alpha_ann_pct": round(fit["coef"][0] * 252 * 100, 2),
            "alpha_t": round(fit["t"][0], 2), "beta": round(fit["coef"][1], 3),
            "r2": round(fit["r2"], 3), "n": fit["n"]}


def holm(pvals, alpha=0.05):
    """Holm step-down correction. Returns list of booleans (reject H0)."""
    order = np.argsort(pvals)
    m = len(pvals)
    reject = [False] * m
    for rank, idx in enumerate(order):
        thr = alpha / (m - rank)
        if pvals[idx] <= thr:
            reject[idx] = True
        else:
            break
    return reject


def print_table(rows, title):
    print(f"\n{title}")
    print(f"{'Strategy':<26} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'Vol':>6} {'Turn':>6}")
    print("-" * 64)
    for r in rows:
        print(f"{r['name']:<26} {r['cagr']:>6.2f}% {r['sharpe']:>7.3f} "
              f"{r['max_dd']:>7.2f}% {r['vol']:>5.2f}% {r['turn']:>5.2f}x")


# ============================================================
# MAIN
# ============================================================
def run_test13():
    print("\n" + "=" * 70)
    print("TEST 13: ADAPTIVE CRISIS ROTATION")
    print("=" * 70)
    print(f"Detector: SPY<=MA{MA_TREND} OR 60d vol>{VOL_STRESS_THRESHOLD*100:.0f}% | "
          f"hysteresis exit (both off)")
    print(f"Escape universe (FIXED): {ESCAPE_RANKED} + SHY fallback | "
          f"top-{TOP_ESCAPE} of {MOM_LOOKBACK_MONTHS}m momentum")
    print(f"Beta cap {BETA_LIMIT} (rolling {BETA_WINDOW}d) | no leverage | "
          f"project alpha={ALPHA_PROJECT:.4f} (Bonferroni N={N_PROJECT_TESTS})")

    prices = download_prices()
    reb = get_monthly_rebalance_dates(prices.index, freq="M")
    market_ret = prices["SPY"].pct_change().fillna(0)

    # ---------------- build all primary strategies ----------------
    w_adapt, state_adapt, sel_log = generate_adaptive_weights(prices, reb)
    w_rp = forward_fill_weights(compute_risk_parity_sparse(prices, reb), prices.index)
    w_shy_esc = generate_fixed_escape_weights(prices, reb, SAFE_ASSET)
    w_ief_esc = generate_fixed_escape_weights(prices, reb, "IEF")

    from test9_vol_managed import generate_smart_passive_weights
    w_smart = generate_smart_passive_weights(prices, reb)
    w_spy = forward_fill_weights(
        pd.DataFrame(1.0, index=reb, columns=prices.columns) * 0, prices.index)
    w_spy["SPY"] = 1.0

    strategies = {
        "AdaptiveRotation": w_adapt,
        "RiskParity": w_rp,
        "EscapeSHY": w_shy_esc,
        "EscapeIEF(2008trap)": w_ief_esc,
        "SmartPassive": w_smart,
        "SPY": w_spy,
    }
    results = {}
    for name, w in strategies.items():
        results[name] = backtest_from_weights(prices, w, COST_PER_SIDE, name)

    # ---------------- full-period table ----------------
    rows = []
    for name, res in results.items():
        m = calculate_metrics(res["net_returns"], name)
        rows.append({"name": name, "cagr": m["cagr"], "sharpe": m["sharpe"],
                     "max_dd": m["max_drawdown"], "vol": m["ann_volatility"],
                     "turn": res["annual_turnover_one_way"]})
    print_table(rows, "\nFULL PERIOD (2007-2026), net of 10bps/side")

    # ex-post static-average-escape (diagnostic only)
    avg_w_expost = w_adapt.mean()
    w_sae_expost = static_weights_from_series(prices, reb, avg_w_expost)
    res_sae_expost = backtest_from_weights(prices, w_sae_expost, COST_PER_SIDE, "SAE_expost")
    m_sae_expost = calculate_metrics(res_sae_expost["net_returns"], "SAE_expost")
    print(f"\nStatic-Average-Escape (EX-POST diagnostic): CAGR={m_sae_expost['cagr']:.2f}% "
          f"Sharpe={m_sae_expost['sharpe']:.3f} MaxDD={m_sae_expost['max_drawdown']:.2f}%")
    print(f"  avg composition: " +
          ", ".join(f"{k}:{v*100:.1f}%" for k, v in avg_w_expost[avg_w_expost > 0.005]
                    .sort_values(ascending=False).items()))

    # matched-beta control (diagnostic)
    strat_ret_pre = results["AdaptiveRotation"]["net_returns"]
    b_roll = (strat_ret_pre.rolling(BETA_WINDOW).cov(market_ret) /
              market_ret.rolling(BETA_WINDOW).var())
    avg_beta = float(b_roll.mean())
    w_mb = w_spy.copy() * 0
    w_mb["SPY"] = min(avg_beta, 1.0)
    w_mb[SAFE_ASSET] = 1.0 - min(avg_beta, 1.0)
    res_mb = backtest_from_weights(prices, w_mb, COST_PER_SIDE, "MatchedBeta")
    m_mb = calculate_metrics(res_mb["net_returns"], "MatchedBeta")
    print(f"Matched-Beta SPY/SHY (avg beta={avg_beta:.2f}, diagnostic): "
          f"CAGR={m_mb['cagr']:.2f}% Sharpe={m_mb['sharpe']:.3f}")

    # ---------------- turnover split + crisis log ----------------
    ts = turnover_by_state(results["AdaptiveRotation"], state_adapt)
    print(f"\nAdaptive regime stats: stress {ts['pct_time_stress']}% of time "
          f"({ts['days_stress']}d) | turnover/yr calm={ts['turnover_calm_ann']:.2f}x "
          f"stress={ts['turnover_stress_ann']:.2f}x")

    print("\nCrisis behavior (return% / maxDD%):")
    print(f"{'Crisis':<25} {'Adaptive':>10} {'EscSHY':>9} {'EscIEF':>9} {'SPY':>9}")
    crisis_rows = []
    for cname, (c0, c1) in CRISIS_PERIODS.items():
        line = {"crisis": cname}
        vals = []
        for name in ["AdaptiveRotation", "EscapeSHY", "EscapeIEF(2008trap)", "SPY"]:
            r = results[name]["net_returns"]
            seg = r[(r.index >= c0) & (r.index <= c1)]
            if len(seg) == 0:
                vals.append("   n/a   "); continue
            tr = float((1 + seg).prod() - 1) * 100
            cum = (1 + seg).cumprod()
            dd = float(((cum - cum.cummax()) / cum.cummax()).min()) * 100
            vals.append(f"{tr:5.1f}/{dd:5.1f}")
            line[name] = f"{tr:.2f}/{dd:.2f}"
        print(f"{cname:<25} {vals[0]:>10} {vals[1]:>9} {vals[2]:>9} {vals[3]:>9}")
        crisis_rows.append(line)

    # escape selections during crises
    for cname, (c0, c1) in CRISIS_PERIODS.items():
        seg = sel_log[(sel_log["date"] >= c0) & (sel_log["date"] <= c1)]
        if len(seg):
            picks = seg["picked"].tolist()
            print(f"  {cname}: escape picks by month: {picks}")

    # ---------------- IS / OOS with implementable SAE ----------------
    print("\n" + "=" * 70)
    print(f"IS ({IS_START[:4]}-{IS_END[:4]}) vs OOS ({OOS_START[:4]}+)")
    print("=" * 70)
    prices_is = prices.loc[IS_START:IS_END]
    prices_oos = prices.loc[OOS_START:]

    # IS-trained static-average-escape: average adaptive weights on IS only
    w_adapt_is, _, _ = generate_adaptive_weights(prices_is,
                                                 get_monthly_rebalance_dates(prices_is.index, "M"))
    avg_w_is = w_adapt_is.mean()

    slice_rows = []
    oos_results = {}
    for label, pslice in [("IS", prices_is), ("OOS", prices_oos)]:
        reb_s = get_monthly_rebalance_dates(pslice.index, "M")
        wa, sta, _ = generate_adaptive_weights(pslice, reb_s)
        wrp = forward_fill_weights(compute_risk_parity_sparse(pslice, reb_s), pslice.index)
        wse = generate_fixed_escape_weights(pslice, reb_s, SAFE_ASSET)
        wie = generate_fixed_escape_weights(pslice, reb_s, "IEF")
        # SAE: IS-trained weights frozen; on IS itself use ex-post diagnostic note
        w_sae_s = static_weights_from_series(pslice, reb_s, avg_w_is)

        block = {}
        for nm, ww in [("Adaptive", wa), ("RiskParity", wrp), ("EscapeSHY", wse),
                       ("EscapeIEF", wie), ("SAE_IStrained", w_sae_s)]:
            res = backtest_from_weights(pslice, ww, COST_PER_SIDE, nm)
            m = calculate_metrics(res["net_returns"], nm)
            block[nm] = (res, m)
        # SPY on slice
        wspy = pd.DataFrame(0.0, index=pslice.index, columns=pslice.columns)
        wspy["SPY"] = 1.0
        res_spy = backtest_from_weights(pslice, wspy, COST_PER_SIDE, "SPY")
        m_spy = calculate_metrics(res_spy["net_returns"], "SPY")
        block["SPY"] = (res_spy, m_spy)

        print(f"\n[{label}]")
        print(f"{'Strategy':<18} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8}")
        for nm, (res, m) in block.items():
            print(f"{nm:<18} {m['cagr']:>6.2f}% {m['sharpe']:>7.3f} {m['max_drawdown']:>7.2f}%")
            slice_rows.append({"period": label, "strategy": nm, "cagr": m["cagr"],
                               "sharpe": m["sharpe"], "max_dd": m["max_drawdown"]})
        if label == "OOS":
            oos_results = block

    # ---------------- bootstrap + multiple testing (OOS) ----------------
    print("\n" + "-" * 70)
    print("OOS BLOCK BOOTSTRAP: Adaptive vs controls (Sharpe diff)")
    print("-" * 70)
    adapt_oos = oos_results["Adaptive"][0]["net_returns"]
    comparisons = ["SAE_IStrained", "RiskParity", "EscapeSHY"]
    pvals, boot_rows = [], []
    for cmp_name in comparisons:
        b = block_bootstrap_sharpe_diff(adapt_oos, oos_results[cmp_name][0]["net_returns"])
        pvals.append(b["p_value"])
        boot_rows.append({"comparison": f"Adaptive-{cmp_name}",
                          "obs_diff": round(b["observed_diff"], 3),
                          "p_value": round(b["p_value"], 4),
                          "ci_low": round(b["ci_lower"], 3),
                          "ci_high": round(b["ci_upper"], 3)})
        print(f"  vs {cmp_name:<15} diff={b['observed_diff']:+.3f} "
              f"p={b['p_value']:.4f} CI[{b['ci_lower']:+.3f},{b['ci_upper']:+.3f}]")

    holm_reject = holm(pvals, 0.05)
    print(f"\n  Holm (family of {len(pvals)}, alpha=0.05): "
          f"{dict(zip(comparisons, holm_reject))}")
    print(f"  Project-wide Bonferroni threshold: p < {ALPHA_PROJECT:.4f}")
    for cmp_name, p in zip(comparisons, pvals):
        if p < ALPHA_PROJECT:
            grade = "PROJECT-SIGNIFICANT"
        elif p < 0.05:
            grade = "NOMINAL ONLY (multiple testing)"
        else:
            grade = "not significant"
        print(f"    Adaptive vs {cmp_name}: {grade}")

    # ---------------- Newey-West alpha (OOS + full) ----------------
    print("\n" + "-" * 70)
    print("NEWey-WEST ALPHA (21 lags)")
    print("-" * 70)
    nw_rows = []
    for label, sret, mret in [
        ("FULL", results["AdaptiveRotation"]["net_returns"], market_ret),
        ("OOS", adapt_oos, market_ret.loc[adapt_oos.index]),
    ]:
        r = nw_alpha_beta(sret, mret, label)
        nw_rows.append(r)
        print(f"  [{label}] alpha={r['alpha_ann_pct']:+.2f}%/yr t={r['alpha_t']:.2f} "
              f"beta={r['beta']:.3f} R2={r['r2']:.3f}")

    # ---------------- sensitivity (report only, full period) ----------------
    print("\n" + "-" * 70)
    print("SENSITIVITY (report only; primary = 6M/25%/hysteresis)")
    print("-" * 70)
    sens_rows = []
    variants = [
        ("PRIMARY hysteresis", dict()),
        ("symmetric exit", dict(symmetric_exit=True)),
        ("mom 3M", dict(mom_months=3)),
        ("mom 12M", dict(mom_months=12)),
        ("vol thr 20%", dict(vol_thr=0.20)),
        ("vol thr 30%", dict(vol_thr=0.30)),
    ]
    print(f"{'Variant':<22} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'%stress':>8}")
    for vname, kwargs in variants:
        wv, stv, _ = generate_adaptive_weights(prices, reb, **kwargs)
        resv = backtest_from_weights(prices, wv, COST_PER_SIDE, vname)
        mv = calculate_metrics(resv["net_returns"], vname)
        pct_stress = float(stv.mean() * 100)
        marker = " <--" if vname.startswith("PRIMARY") else ""
        print(f"{vname:<22} {mv['cagr']:>6.2f}% {mv['sharpe']:>7.3f} "
              f"{mv['max_drawdown']:>7.2f}% {pct_stress:>7.1f}%{marker}")
        sens_rows.append({"variant": vname, "cagr": mv["cagr"], "sharpe": mv["sharpe"],
                          "max_dd": mv["max_drawdown"], "pct_stress": pct_stress})

    # 2020 recovery cost of sticky exit
    r2020 = results["AdaptiveRotation"]["net_returns"]
    seg20 = r2020[(r2020.index >= "2020-02-01") & (r2020.index <= "2020-08-31")]
    cum20 = (1 + seg20).prod() - 1
    st20 = state_adapt[(state_adapt.index >= "2020-02-01") & (state_adapt.index <= "2020-08-31")]
    exit_date = st20[~st20].index.min() if (~st20).any() else None
    print(f"\n2020 sticky-exit cost check: Feb-Aug 2020 net return={cum20*100:.2f}%, "
          f"stress state until {exit_date.date() if exit_date is not None else 'n/a'}")

    # cost stress on adaptive (50bps)
    res_50 = backtest_from_weights(prices, w_adapt, COST_STRESS, "Adaptive50bps")
    m_50 = calculate_metrics(res_50["net_returns"], "Adaptive50bps")
    print(f"Cost stress 50bps/side: CAGR={m_50['cagr']:.2f}% Sharpe={m_50['sharpe']:.3f}")

    # ---------------- modern addendum (n=1 case study) ----------------
    print("\n" + "=" * 70)
    print("MODERN ADDENDUM (n=1 crisis anecdote: 2022) — NOT evidence")
    print("=" * 70)
    addendum_rows = []
    try:
        import yfinance as yf
        extra = ["DBMF", "KMLM", "XLU", "XLP"]
        data = yf.download(extra, start="2019-01-01", auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            px = data["Close"][extra]
        else:
            px = data[["Close"]]
        px = px.dropna(how="all")
        common_start = px.dropna().index[0]
        # need warmup for MA200+vol60+mom126 -> start later
        avail = prices.join(px, how="inner")
        warm_start = avail.loc[common_start:].index[min(260, len(avail.loc[common_start:]) - 1)]
        avail = avail.loc[warm_start:]
        ext_escape = ESCAPE_RANKED + [t for t in extra if t in avail.columns]
        reb_a = get_monthly_rebalance_dates(avail.index, "M")
        wa_ext, st_ext, log_ext = generate_adaptive_weights(
            avail, reb_a, escape_ranked=ext_escape)
        res_ext = backtest_from_weights(avail, wa_ext, COST_PER_SIDE, "AdaptiveExt")
        m_ext = calculate_metrics(res_ext["net_returns"], "AdaptiveExt")
        # 2022 slice
        seg22 = res_ext["net_returns"]["2022-01-01":"2022-12-31"]
        tr22 = float((1 + seg22).prod() - 1) * 100 if len(seg22) else np.nan
        picks22 = log_ext[(log_ext["date"] >= "2022-01-01") &
                          (log_ext["date"] <= "2022-12-31")]["picked"].tolist()
        print(f"Window: {avail.index[0].date()} to {avail.index[-1].date()} "
              f"(post DBMF/KMLM warmup)")
        print(f"AdaptiveExt: CAGR={m_ext['cagr']:.2f}% Sharpe={m_ext['sharpe']:.3f} "
              f"MaxDD={m_ext['max_drawdown']:.2f}% | 2022 return={tr22:.2f}%")
        print(f"2022 escape picks: {picks22}")
        print("Framing: single-crisis anecdote (n=1). No inference.")
        addendum_rows.append({"window_start": str(avail.index[0].date()),
                              "cagr": m_ext["cagr"], "sharpe": m_ext["sharpe"],
                              "max_dd": m_ext["max_drawdown"], "ret_2022": tr22,
                              "picks_2022": ";".join(picks22)})
    except Exception as e:
        print(f"Addendum INFEASIBLE: {e}")
        addendum_rows.append({"error": str(e)})

    # ---------------- verdict ----------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    oos_adapt = oos_results["Adaptive"][1]
    oos_sae = oos_results["SAE_IStrained"][1]
    oos_rp = oos_results["RiskParity"][1]
    p_sae = boot_rows[0]["p_value"]

    print(f"\nOOS Adaptive: CAGR={oos_adapt['cagr']:.2f}% Sharpe={oos_adapt['sharpe']:.3f} "
          f"MaxDD={oos_adapt['max_drawdown']:.2f}%")
    print(f"OOS SAE (IS-trained): Sharpe={oos_sae['sharpe']:.3f} | "
          f"OOS RiskParity: Sharpe={oos_rp['sharpe']:.3f}")
    beats_sae = oos_adapt["sharpe"] > oos_sae["sharpe"]
    dd_ok = oos_adapt["passes_dd_limit"]
    print(f"Beats static-average-escape (OOS): {'YES' if beats_sae else 'NO'} "
          f"(p={p_sae:.4f})")
    print(f"25% DD limit OOS: {'PASS' if dd_ok else 'FAIL'}")

    print("\n" + "-" * 70)
    if beats_sae and p_sae < ALPHA_PROJECT and dd_ok:
        print("RESULT: GENUINE TIMING VALUE — adaptive rotation beats its own "
              "static composition with project-level significance.")
    elif beats_sae and p_sae < 0.05 and dd_ok:
        print("RESULT: NOMINAL timing value only. After 13-test multiple-testing "
              "correction this is EXPLORATORY, not confirmed.")
    elif not beats_sae:
        print("RESULT: NO timing value — adaptive rotation does NOT beat the "
              "static-average-escape. The 'adaptivity' is a composition effect: "
              "holding the defensive mix statically does the same job.")
    else:
        print("RESULT: INCONCLUSIVE — inspect rows above.")
    print("-" * 70)

    # ---------------- save ----------------
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(slice_rows).to_csv("results/test13_isoos.csv", index=False)
    pd.DataFrame(boot_rows).to_csv("results/test13_bootstrap.csv", index=False)
    pd.DataFrame(sens_rows).to_csv("results/test13_sensitivity.csv", index=False)
    pd.DataFrame(crisis_rows).to_csv("results/test13_crisis.csv", index=False)
    pd.DataFrame(nw_rows).to_csv("results/test13_alpha.csv", index=False)
    pd.DataFrame(addendum_rows).to_csv("results/test13_addendum.csv", index=False)
    sel_log.to_csv("results/test13_escape_log.csv", index=False)
    print("\nResults saved to results/test13_*.csv")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    run_test13()