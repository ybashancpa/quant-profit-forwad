"""
TEST 14: OPTIONS INCOME / PREMIUM SELLING — EVIDENCE AUDIT
============================================================
Research question: do option-premium-selling strategies produce a
sustained positive expectation after costs for a private Israeli
investor with a $5k-$50k account?

This script is the quantitative core of OPTIONS_INCOME_RESEARCH.md.
It does NOT simulate option chains (none are available for free at
the required quality). It measures what CAN be measured from public
data:

PART 1 - Cboe strategy benchmark indices (BXM, PUT) vs SPX/SPY,
         full period and sub-periods (pre/post the ~2012 VRP break
         documented by Dew-Becker & Giglio 2025/2026).
PART 2 - Direct VRP measurement from our own data: VIX^2 vs
         subsequent realized variance, by sub-period.
PART 3 - Tail-event windows (2008, Volmageddon 2018, Feb-Mar 2020,
         2022, Aug 2024): returns, max DD, recovery time, years of
         returns erased.
PART 4 - Retail cost model (IBKR fixed pricing + regulatory/clearing
         fees + parameterized bid-ask), per strategy structure.
PART 5 - Account-size feasibility ($5k/$10k/$25k/$50k, 2% risk cap):
         granularity, break-even premium, cost drag.
PART 6 - Israeli tax wedge (25% capital gains) applied to index CAGRs.

Data: data/options_prices.csv, data/options_adj.csv
Outputs: results/test14_*.csv
"""

import os
import numpy as np
import pandas as pd

TRADING_DAYS = 252

os.makedirs("results", exist_ok=True)

PRICES = pd.read_csv("data/options_prices.csv", index_col=0, parse_dates=True)
ADJ = pd.read_csv("data/options_adj.csv", index_col=0, parse_dates=True)


# ============================================================
# helpers
# ============================================================
def metrics_from_level(level, name):
    """Performance metrics from a price/level series."""
    level = level.dropna()
    ret = level.pct_change().dropna()
    n_years = len(ret) / TRADING_DAYS
    total = level.iloc[-1] / level.iloc[0] - 1
    cagr = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = ret.mean() / ret.std() * np.sqrt(TRADING_DAYS) if ret.std() > 0 else np.nan
    downside = ret[ret < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (cagr / downside) if downside and downside > 0 else np.nan
    cummax = level.cummax()
    dd = level / cummax - 1
    max_dd = dd.min()
    worst_yr = ret.groupby(ret.index.year).apply(lambda x: (1 + x).prod() - 1).min()
    worst_mo = ret.groupby([ret.index.year, ret.index.month]).apply(
        lambda x: (1 + x).prod() - 1).min()
    return {
        "name": name,
        "start": str(ret.index[0].date()),
        "end": str(ret.index[-1].date()),
        "years": round(n_years, 1),
        "cagr_pct": round(cagr * 100, 2),
        "vol_pct": round(vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "worst_year_pct": round(worst_yr * 100, 2),
        "worst_month_pct": round(worst_mo * 100, 2),
        "calmar": round(cagr / abs(max_dd), 3) if max_dd < 0 else np.nan,
    }


def window_stats(level, start, end, name):
    """Return, max DD inside window, and recovery time AFTER the window ends."""
    level = level.dropna()
    pre_peak = level.loc[:start].max()
    seg = level.loc[start:end]
    if len(seg) < 2:
        return None
    ret = seg.iloc[-1] / seg.iloc[0] - 1
    dd_inside = (seg / seg.cummax() - 1).min()
    # recovery: first date AFTER the window where level >= pre-window peak
    after = level.loc[end:]
    recovered = after[after >= pre_peak]
    rec_days = (recovered.index[0] - seg.index[-1]).days if len(recovered) else None
    return {
        "name": name,
        "window": f"{start} -> {end}",
        "return_pct": round(ret * 100, 2),
        "max_dd_in_window_pct": round(dd_inside * 100, 2),
        "recovery_days_after_window_end": rec_days,
    }


# ============================================================
# PART 1: INDEX PERFORMANCE, FULL + SUB-PERIODS
# ============================================================
def part1_index_performance():
    print("\n" + "=" * 72)
    print("PART 1: CBOE STRATEGY INDICES vs BENCHMARKS")
    print("=" * 72)

    bxm = PRICES["^BXM"].dropna()
    put = PRICES["^PUT"].dropna()
    spx = PRICES["^SPX"].dropna()
    spy_tr = ADJ["SPY"].dropna()          # total return
    spy_px = PRICES["SPY"].dropna()        # price basis
    shy_tr = ADJ["SHY"].dropna()           # T-bill proxy (total return)

    series = {
        "PUT (Cboe PutWrite, price idx)": put,
        "BXM (Cboe BuyWrite, price idx)": bxm,
        "S&P 500 (price)": spx,
        "SPY (total return)": spy_tr,
        "SPY (price)": spy_px,
        "SHY (T-bill proxy, TR)": shy_tr,
    }

    rows = []
    for name, s in series.items():
        rows.append(metrics_from_level(s, name))
    df_full = pd.DataFrame(rows)
    print("\n--- Full period (2002+) ---")
    print(df_full.to_string(index=False))

    # Sub-periods aligned to the VRP structural-break literature
    windows = {
        "2002-2007 (pre-GFC)": ("2002-01-01", "2007-12-31"),
        "2008-2012 (GFC+EZ)": ("2008-01-01", "2012-12-31"),
        "2013-2019 (post-break)": ("2013-01-01", "2019-12-31"),
        "2020-2026 (recent)": ("2020-01-01", "2026-12-31"),
    }
    sub_rows = []
    for wname, (ws, we) in windows.items():
        for sname, s in [("PUT", put), ("BXM", bxm), ("SPX", spx),
                         ("SPY_TR", spy_tr), ("SHY_TR", shy_tr)]:
            seg = s.loc[ws:we]
            if len(seg) < 100:
                continue
            m = metrics_from_level(seg, f"{sname} {wname}")
            m["window"] = wname
            sub_rows.append(m)
    df_sub = pd.DataFrame(sub_rows)
    print("\n--- Sub-period CAGR% / Sharpe / MaxDD% ---")
    piv = df_sub.pivot_table(index="window", columns="name",
                             values=["cagr_pct", "sharpe", "max_dd_pct"])
    print(piv.to_string())

    # PUT decomposition: PUT return - SHY return = pure option-selling excess
    aligned = pd.concat([put.pct_change(), shy_tr.pct_change()], axis=1, join="inner").dropna()
    aligned.columns = ["put", "shy"]
    excess = aligned["put"] - aligned["shy"]
    n = len(excess)
    exc_ann = excess.mean() * TRADING_DAYS
    exc_vol = excess.std() * np.sqrt(TRADING_DAYS)
    print(f"\nPUT minus T-bill (pure premium-selling component): "
          f"{exc_ann*100:.2f}%/yr, vol {exc_vol*100:.2f}%, "
          f"IR {exc_ann/exc_vol:.3f}, % positive months "
          f"{(excess.resample('ME').sum() > 0).mean()*100:.1f}%")
    # by era
    for wname, (ws, we) in windows.items():
        seg = excess.loc[ws:we]
        if len(seg) > 252:
            print(f"  {wname}: excess {seg.mean()*TRADING_DAYS*100:+.2f}%/yr")

    df_full.to_csv("results/test14_index_performance.csv", index=False)
    df_sub.to_csv("results/test14_index_subperiods.csv", index=False)
    return df_full, df_sub


# ============================================================
# PART 2: VRP MEASUREMENT (VIX^2 vs realized variance)
# ============================================================
def part2_vrp():
    print("\n" + "=" * 72)
    print("PART 2: VARIANCE RISK PREMIUM — VIX^2 vs SUBSEQUENT REALIZED VARIANCE")
    print("=" * 72)

    spx = PRICES["^SPX"].dropna()
    vix = PRICES["^VIX"].dropna()
    logret = np.log(spx / spx.shift(1)).dropna()

    # monthly (21-trading-day) forward realized variance vs VIX^2 at start
    # annualization: sum of 21 daily squared log-returns * (252/21) = *12
    idx = logret.index
    rows = []
    step = 21
    for i in range(0, len(idx) - step, step):
        d0 = idx[i]
        rv21 = (logret.iloc[i:i + step] ** 2).sum() * (TRADING_DAYS / step)  # annualized
        v0 = vix.asof(d0)
        if v0 is None or np.isnan(v0) or np.isnan(rv21):
            continue
        rows.append({"date": d0, "iv2": (v0 / 100.0) ** 2,
                     "rv2": rv21, "vrp": (v0 / 100.0) ** 2 - rv21,
                     "vix": v0, "rv_vol": np.sqrt(rv21) * 100})
    df = pd.DataFrame(rows).set_index("date")

    eras = {
        "2002-2007": ("2002-01-01", "2007-12-31"),
        "2008-2012": ("2008-01-01", "2012-12-31"),
        "2013-2019": ("2013-01-01", "2019-12-31"),
        "2020-2026": ("2020-01-01", "2026-12-31"),
    }
    out = []
    print(f"\n{'era':<12}{'mean VRP':>10}{'med VRP':>10}{'>0 share':>10}"
          f"{'mean VIX':>10}{'mean RVvol':>12}")
    for ename, (ws, we) in eras.items():
        seg = df.loc[ws:we]
        if len(seg) < 12:
            continue
        row = {
            "era": ename,
            "mean_vrp_volpts": round(seg["vrp"].mean() * 100, 2),
            "median_vrp_volpts": round(seg["vrp"].median() * 100, 2),
            "pct_positive": round((seg["vrp"] > 0).mean() * 100, 1),
            "mean_vix": round(seg["vix"].mean(), 2),
            "mean_realized_vol": round(seg["rv_vol"].mean(), 2),
            "n_months": len(seg),
        }
        out.append(row)
        print(f"{ename:<12}{row['mean_vrp_volpts']:>10.2f}"
              f"{row['median_vrp_volpts']:>10.2f}{row['pct_positive']:>9.1f}%"
              f"{row['mean_vix']:>10.2f}{row['mean_realized_vol']:>12.2f}")

    full = {
        "era": "FULL 2002-2026",
        "mean_vrp_volpts": round(df["vrp"].mean() * 100, 2),
        "median_vrp_volpts": round(df["vrp"].median() * 100, 2),
        "pct_positive": round((df["vrp"] > 0).mean() * 100, 1),
        "mean_vix": round(df["vix"].mean(), 2),
        "mean_realized_vol": round(df["rv_vol"].mean(), 2),
        "n_months": len(df),
    }
    out.append(full)
    print(f"{'FULL':<12}{full['mean_vrp_volpts']:>10.2f}"
          f"{full['median_vrp_volpts']:>10.2f}{full['pct_positive']:>9.1f}%")

    res = pd.DataFrame(out)
    res.to_csv("results/test14_vrp_by_era.csv", index=False)
    df.to_csv("results/test14_vrp_monthly.csv")
    return res


# ============================================================
# PART 3: TAIL EVENTS
# ============================================================
CRISIS_WINDOWS = {
    "GFC 2008 (year)": ("2008-01-01", "2008-12-31"),
    "GFC peak-trough": ("2007-10-09", "2009-03-09"),
    "Volmageddon Feb-2018": ("2018-01-26", "2018-02-12"),
    "Corona Feb-Mar 2020": ("2020-02-19", "2020-03-23"),
    "2022 (year)": ("2022-01-01", "2022-12-31"),
    "Yen-carry Aug-2024": ("2024-07-16", "2024-08-05"),
}


def part3_tail_events():
    print("\n" + "=" * 72)
    print("PART 3: TAIL-EVENT WINDOWS")
    print("=" * 72)
    series = {
        "PUT": PRICES["^PUT"].dropna(),
        "BXM": PRICES["^BXM"].dropna(),
        "SPX": PRICES["^SPX"].dropna(),
        "SPY_TR": ADJ["SPY"].dropna(),
    }
    rows = []
    for cname, (ws, we) in CRISIS_WINDOWS.items():
        print(f"\n[{cname}]")
        for sname, s in series.items():
            w = window_stats(s, ws, we, f"{sname} | {cname}")
            if w is None:
                continue
            w["crisis"] = cname
            rows.append(w)
            rec = w["recovery_days_after_window_end"]
            print(f"  {sname:<8} ret={w['return_pct']:>8.2f}%  "
                  f"DD={w['max_dd_in_window_pct']:>8.2f}%  "
                  f"recovery(after window)={rec if rec is not None else 'NOT RECOVERED'} days")
    df = pd.DataFrame(rows)
    df.to_csv("results/test14_tail_events.csv", index=False)

    # years-of-return erased for PUT: compare crisis loss to prior 3yr cumulative
    put = PRICES["^PUT"].dropna()
    print("\nYears of PUT returns erased by each crisis:")
    for cname, (ws, we) in CRISIS_WINDOWS.items():
        seg = put.loc[ws:we]
        if len(seg) < 2:
            continue
        loss = seg.iloc[-1] / seg.iloc[0] - 1
        prior = put.loc[:ws]
        if len(prior) > 3 * TRADING_DAYS:
            p3 = prior.iloc[-1] / prior.iloc[-3 * TRADING_DAYS] - 1
            print(f"  {cname:<24} crisis {loss*100:+7.2f}%  vs prior-3yr {p3*100:+7.2f}%"
                  f"  -> {abs(loss)/p3:.1f}x wiped" if p3 > 0 else
                  f"  {cname:<24} crisis {loss*100:+7.2f}%")
    return df


# ============================================================
# PART 4: RETAIL COST MODEL (IBKR)
# ============================================================
IBKR_PER_CONTRACT = 0.65          # IBKR Pro fixed, <=10k contracts/mo (source: IBKR pricing page)
IBKR_MIN_PER_ORDER = 1.00
OCC_CLEARING = 0.025              # per contract
CAT_FEE = 0.0003                  # per contract
FINRA_TAF_SELL = 0.00329          # per contract sold
ORF_EST = 0.025                   # options regulatory fee estimate (varies by exchange)


def round_trip_cost(n_legs, contracts=1, premium_per_contract=None,
                    half_spread_ticks=1.0, tick=0.01):
    """
    Round-trip cost for one spread/structure of `contracts` spreads.
    n_legs: 1 (single), 2 (vertical), 4 (iron condor).
    Commissions charged per CONTRACT per ORDER side.
    """
    total_contracts = n_legs * contracts
    # open + close orders
    comm_open = max(IBKR_MIN_PER_ORDER, IBKR_PER_CONTRACT * total_contracts)
    comm_close = max(IBKR_MIN_PER_ORDER, IBKR_PER_CONTRACT * total_contracts)
    clearing = OCC_CLEARING * total_contracts * 2
    reg = (FINRA_TAF_SELL + CAT_FEE + ORF_EST) * total_contracts * 2
    spread_cost = half_spread_ticks * tick * 100 * total_contracts * 2  # cross twice
    total = comm_open + comm_close + clearing + reg + spread_cost
    return {
        "commissions": comm_open + comm_close,
        "clearing_reg": clearing + reg,
        "bid_ask_cost": spread_cost,
        "total_usd": total,
    }


def part4_costs():
    print("\n" + "=" * 72)
    print("PART 4: RETAIL COST MODEL (IBKR fixed pricing)")
    print("=" * 72)
    structures = [
        ("Cash-secured put / covered call (1 leg)", 1),
        ("Credit vertical spread (2 legs)", 2),
        ("Iron condor (4 legs)", 4),
    ]
    premiums = [0.20, 0.50, 1.00, 2.00, 5.00]  # per contract, in $ of underlying pts
    rows = []
    print(f"\nRound-trip cost per structure (1 contract/spread, 1-tick half-spread):")
    print(f"{'structure':<42}{'comm':>8}{'fees':>8}{'spread':>8}{'TOTAL':>8}")
    for sname, nlegs in structures:
        c = round_trip_cost(nlegs)
        print(f"{sname:<42}{c['commissions']:>8.2f}{c['clearing_reg']:>8.2f}"
              f"{c['bid_ask_cost']:>8.2f}{c['total_usd']:>8.2f}")

    print(f"\nCost as % of premium collected (per $100 of premium per contract):")
    print(f"{'premium/contract':<18}", end="")
    for sname, nlegs in structures:
        print(f"{str(nlegs)+'-leg':>10}", end="")
    print()
    for p in premiums:
        prem_usd = p * 100
        print(f"${p:<17.2f}", end="")
        for sname, nlegs in structures:
            c = round_trip_cost(nlegs)
            print(f"{c['total_usd']/prem_usd*100:>9.1f}%", end="")
        print()
        for sname, nlegs in structures:
            c = round_trip_cost(nlegs)
            rows.append({
                "structure": sname, "n_legs": nlegs,
                "premium_per_contract_usd": prem_usd,
                "round_trip_cost_usd": round(c["total_usd"], 2),
                "cost_pct_of_premium": round(c["total_usd"] / prem_usd * 100, 2),
            })
    # wider half-spread stress (illiquid single names)
    print(f"\nSpread-cost stress for 2-leg credit spread, $1.00 premium ($100):")
    for ticks in [1, 2, 5, 10]:
        c = round_trip_cost(2, half_spread_ticks=ticks)
        print(f"  half-spread {ticks} tick(s): total ${c['total_usd']:.2f} "
              f"= {c['total_usd']/100*100:.1f}% of $100 premium")
        rows.append({
            "structure": f"Credit spread half-spread {ticks}t", "n_legs": 2,
            "premium_per_contract_usd": 100.0,
            "round_trip_cost_usd": round(c["total_usd"], 2),
            "cost_pct_of_premium": round(c["total_usd"] / 100.0 * 100, 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv("results/test14_costs.csv", index=False)
    return df


# ============================================================
# PART 5: ACCOUNT-SIZE FEASIBILITY
# ============================================================
def part5_account_size():
    print("\n" + "=" * 72)
    print("PART 5: ACCOUNT-SIZE FEASIBILITY (2% risk cap per trade)")
    print("=" * 72)
    spy_last = float(PRICES["SPY"].dropna().iloc[-1])
    spx_last = float(PRICES["^SPX"].dropna().iloc[-1])
    print(f"\nReference prices: SPY=${spy_last:.2f}, SPX=${spx_last:.2f}")
    print(f"1 cash-secured SPY put contract requires ~${spy_last*100:,.0f} collateral")
    print(f"1 cash-secured SPX put contract requires ~${spx_last*100:,.0f} collateral")

    accounts = [5_000, 10_000, 25_000, 50_000]
    rows = []
    for acct in accounts:
        risk_budget = acct * 0.02
        # $1-wide credit put spread: max loss $100 per contract
        n_spreads_1w = int(risk_budget // 100)
        c1 = round_trip_cost(2, contracts=max(n_spreads_1w, 1))
        cost_per_spread_1w = round_trip_cost(2)["total_usd"]
        # break-even premium: cost / (1 - tax) ignored here; premium must exceed cost
        be_premium = cost_per_spread_1w  # $ per contract
        # $5-wide spread: max loss $500 per contract
        n_spreads_5w = int(risk_budget // 500)
        cost_per_spread_5w = round_trip_cost(2)["total_usd"]
        rows.append({
            "account_usd": acct,
            "risk_budget_2pct_usd": risk_budget,
            "csp_contracts_feasible": int(acct // (spy_last * 100)),
            "n_1wide_spreads_at_2pct": n_spreads_1w,
            "n_5wide_spreads_at_2pct": n_spreads_5w,
            "roundtrip_cost_per_spread_usd": round(cost_per_spread_1w, 2),
            "breakeven_premium_1wide_usd": round(be_premium, 2),
            "breakeven_premium_1wide_pct_of_width": round(be_premium / 100 * 100, 1),
            "cost_pct_of_20pct_credit_on_1wide": round(cost_per_spread_1w / 20 * 100, 1),
            "cost_pct_of_30pct_credit_on_1wide": round(cost_per_spread_1w / 30 * 100, 1),
            "cost_pct_of_50pct_credit_on_1wide": round(cost_per_spread_1w / 50 * 100, 1),
        })
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    df.to_csv("results/test14_account_size.csv", index=False)

    # annual cost drag for a monthly seller
    print("\nAnnualized cost drag, monthly rolling (12 trades/yr), 2-leg spreads:")
    for acct in accounts:
        n = int((acct * 0.02) // 100)
        if n == 0:
            print(f"  ${acct:>7,}: cannot fit even 1 $1-wide spread under 2% cap")
            continue
        cost_per_trade = round_trip_cost(2, contracts=n)["total_usd"]
        prem_20 = n * 20   # 20% of width credit
        prem_30 = n * 30
        print(f"  ${acct:>7,}: {n} spreads/trade; costs ${cost_per_trade:.2f}/trade = "
              f"{cost_per_trade/prem_20*100:.1f}% of 20%-credit (${prem_20}), "
              f"{cost_per_trade/prem_30*100:.1f}% of 30%-credit (${prem_30}); "
              f"${cost_per_trade*12:.0f}/yr = {cost_per_trade*12/acct*100:.2f}% of account")
    return df


# ============================================================
# PART 6: ISRAELI TAX WEDGE (25% capital gains, real)
# ============================================================
def part6_tax():
    print("\n" + "=" * 72)
    print("PART 6: ISRAELI 25% CAPITAL-GAINS TAX WEDGE ON INDEX CAGRs")
    print("=" * 72)
    tax = 0.25
    rows = []
    for name, s in [("PUT", PRICES["^PUT"].dropna()),
                    ("BXM", PRICES["^BXM"].dropna()),
                    ("SPX", PRICES["^SPX"].dropna()),
                    ("SPY_TR", ADJ["SPY"].dropna()),
                    ("SHY_TR", ADJ["SHY"].dropna())]:
        m = metrics_from_level(s, name)
        pre = m["cagr_pct"] / 100
        # simplified: tax on terminal gain only, paid at end (lower bound on drag)
        n = m["years"]
        terminal = (1 + pre) ** n
        after_tax_terminal = 1 + (terminal - 1) * (1 - tax)
        post = after_tax_terminal ** (1 / n) - 1
        rows.append({"name": name, "pre_tax_cagr_pct": round(pre * 100, 2),
                     "post_tax_cagr_pct_deferred_pct": round(post * 100, 2),
                     "drag_pp": round((pre - post) * 100, 2)})
        # annual-taxation scenario (realize gains yearly, e.g. short-term options trading)
        post_annual = pre * (1 - tax)
        rows[-1]["post_tax_cagr_pct_annual_pct"] = round(post_annual * 100, 2)
        rows[-1]["drag_annual_pp"] = round((pre - post_annual) * 100, 2)
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    print("\nNote: 'deferred' = tax once at end (best case, like index holding);")
    print("'annual' = gains realized & taxed every year (typical for active options trading).")
    df.to_csv("results/test14_tax_wedge.csv", index=False)
    return df


# ============================================================
# PART 7: ROLLING 10-YEAR CONSISTENCY (PUT vs SPY total return)
# ============================================================
def part7_rolling():
    print("\n" + "=" * 72)
    print("PART 7: ROLLING 10-YEAR CAGR — HOW OFTEN DOES PUT BEAT SPY TR?")
    print("=" * 72)
    put = PRICES["^PUT"].dropna()
    spy = ADJ["SPY"].dropna()
    bxm = PRICES["^BXM"].dropna()
    idx = put.index
    win = 10 * TRADING_DAYS
    rows = []
    for i in range(0, len(idx) - win, 21):  # monthly steps
        d0, d1 = idx[i], idx[i + win]
        seg_p = put.loc[d0:d1]
        seg_s = spy.loc[d0:d1]
        seg_b = bxm.loc[d0:d1]
        if len(seg_p) < win * 0.95 or len(seg_s) < win * 0.95:
            continue
        c_put = (seg_p.iloc[-1] / seg_p.iloc[0]) ** (1 / 10) - 1
        c_spy = (seg_s.iloc[-1] / seg_s.iloc[0]) ** (1 / 10) - 1
        c_bxm = ((seg_b.iloc[-1] / seg_b.iloc[0]) ** (1 / 10) - 1
                 if len(seg_b) > win * 0.95 else np.nan)
        rows.append({"end": d1, "put_cagr": c_put * 100, "spy_cagr": c_spy * 100,
                     "bxm_cagr": c_bxm * 100 if not np.isnan(c_bxm) else np.nan,
                     "put_beats_spy": c_put > c_spy})
    df = pd.DataFrame(rows).set_index("end")
    print(f"\n{len(df)} rolling 10y windows (monthly steps), 2002-2026:")
    print(f"  PUT beats SPY-TR in {df['put_beats_spy'].mean()*100:.1f}% of windows")
    print(f"  PUT 10y CAGR: mean {df['put_cagr'].mean():.2f}%, "
          f"min {df['put_cagr'].min():.2f}%, max {df['put_cagr'].max():.2f}%")
    print(f"  SPY 10y CAGR: mean {df['spy_cagr'].mean():.2f}%, "
          f"min {df['spy_cagr'].min():.2f}%, max {df['spy_cagr'].max():.2f}%")
    bxm_beats = (df['bxm_cagr'] > df['spy_cagr']).dropna()
    print(f"  BXM beats SPY-TR in {bxm_beats.mean()*100:.1f}% of windows (where available)")
    # windows ending post-2012 only (the post-VRP-break era)
    post = df.loc["2013-01-01":]
    if len(post):
        print(f"\n  Windows ending 2013+ (post-break era): n={len(post)}")
        print(f"    PUT beats SPY-TR in {post['put_beats_spy'].mean()*100:.1f}% of windows")
        print(f"    PUT 10y CAGR mean {post['put_cagr'].mean():.2f}% vs "
              f"SPY {post['spy_cagr'].mean():.2f}%")
    df.to_csv("results/test14_rolling_10y.csv")
    return df


# ============================================================
# MAIN
# ============================================================
def run_test14():
    print("\n" + "#" * 72)
    print("# TEST 14: OPTIONS INCOME — EVIDENCE AUDIT")
    print("#" * 72)
    part1_index_performance()
    part2_vrp()
    part3_tail_events()
    part4_costs()
    part5_account_size()
    part6_tax()
    part7_rolling()
    print("\nAll results saved to results/test14_*.csv")


if __name__ == "__main__":
    run_test14()