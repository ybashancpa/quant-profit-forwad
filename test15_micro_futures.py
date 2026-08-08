"""
TEST 15: INTRADAY EDGE IN CME MICRO FUTURES — EVIDENCE AUDIT
=============================================================
Research question: is there a sustained, exploitable intraday edge in
any of 15 CME micro futures contracts after commissions, spreads and
slippage, for a $5k-$50k private account?

The user ran a Momentum-Pullback strategy (VWAP direction + ADX>25 on
15m + EMA20/50 1h context + EMA20-touch trigger) on real contract data,
RTH only, Jun-Aug 2026 (41 trading days): ~56 configurations
(15 instruments x 4 stop multipliers), 6 profitable, 50 losing.
MYM: 8 trades, -$104. M2K: 14 trades, -$212.

This script quantifies the statistical and economic questions around
that result. It does NOT re-run the strategy (no intraday data here);
it measures what CAN be measured:

PART 1 - Multiple-testing verdict on the 56-trial experiment
         (sign test vs null, expected max Sharpe under null, DSR).
PART 2 - Statistical power: how many trades are needed to detect a
         given per-trade Sharpe; MinTRL; verdict on the 41-day sample.
PART 3 - Cost model per instrument (IBKR commissions + exchange fee
         estimates + bid-ask): round-trip cost, break-even edge.
PART 4 - Capital feasibility under 1-2% risk per trade.
PART 5 - Passive alternative over the same window (SPY TR Jun-Aug 2026).

Outputs: results/test15_*.csv
"""

import os
import math
import numpy as np
import pandas as pd

os.makedirs("results", exist_ok=True)

TRADING_DAYS = 252


# ---- minimal stats helpers (no scipy dependency) ----
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_ppf(p):
    # Acklam's inverse normal CDF approximation
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - plow:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def binom_cdf(k, n, p):
    total = 0.0
    for i in range(0, k + 1):
        total += math.comb(n, i) * p ** i * (1 - p) ** (n - i)
    return total


# ============================================================
# PART 1: MULTIPLE-TESTING VERDICT ON THE 56-TRIAL EXPERIMENT
# ============================================================
def part1_multiple_testing(n_trials=56, n_winners=6):
    print("\n" + "=" * 72)
    print("PART 1: MULTIPLE-TESTING VERDICT ON THE 56-TRIAL EXPERIMENT")
    print("=" * 72)

    # (a) sign test: under a zero-edge null, ~50% of configurations
    # should be profitable before costs. Observing 6/56 is far below.
    p_sign = binom_cdf(n_winners, n_trials, 0.5)
    print(f"\n(a) Sign test: {n_winners}/{n_trials} configurations profitable.")
    print(f"    Under zero-edge null, expected winners = {n_trials*0.5:.0f}.")
    print(f"    P(<= {n_winners} winners | null) = {p_sign:.3e}")
    print(f"    -> The family UNDERPERFORMS random chance (p < 1e-9).")

    # (b) expected max Sharpe under the null (Euler-Gauss approx,
    # Bailey & Lopez de Prado 2014): E[max SR] ~ (1-g)*Z(1-1/N)+g*Z(1-1/(N*e))
    gamma = 0.5772156649
    z1 = norm_ppf(1 - 1 / n_trials)
    z2 = norm_ppf(1 - 1 / (n_trials * math.e))
    e_max = (1 - gamma) * z1 + gamma * z2
    print(f"\n(b) Expected max t-stat/SR of {n_trials} pure-noise strategies:")
    print(f"    E[max] ~= {e_max:.2f}  (per-period units)")
    print(f"    -> A single 'best' configuration needs t > ~{e_max:.1f} JUST to")
    print(f"       be distinguishable from the best of 56 noise draws.")

    # (c) Deflated Sharpe: probability that observed SR beats the
    # expected max under null, for a range of observed SRs.
    # DSR = PSR(SR*) with SR* = E[max SR]; using normal approx:
    # DSR = Phi( (SR_obs - SR*) * sqrt(n-1) / sqrt(1 - skew*SR + (kurt-1)/4*SR^2) )
    print(f"\n(c) Deflated Sharpe Ratio (skew=-0.5, kurt=4, per-trade basis):")
    print(f"    {'n_trades':>9} {'SR_obs':>8} {'DSR prob':>10}")
    rows = []
    for n_trades in [8, 14, 30, 100, 300]:
        for sr_obs in [0.05, 0.10, 0.20, 0.30]:
            skew, kurt = -0.5, 4.0
            denom = math.sqrt(1 - skew * sr_obs + (kurt - 1) / 4 * sr_obs ** 2)
            dsr = norm_cdf((sr_obs - e_max) * math.sqrt(n_trades - 1) / denom)
            rows.append({"n_trades": n_trades, "sr_obs_per_trade": sr_obs,
                         "dsr_probability": round(dsr, 4)})
            print(f"    {n_trades:>9} {sr_obs:>8.2f} {dsr:>10.4f}")
    print(f"    -> With 8-14 trades, even a 0.30 per-trade SR cannot pass the")
    print(f"       deflated test; with 300 trades, 0.30 still fails (needs >~0.5).")

    df = pd.DataFrame(rows)
    df.to_csv("results/test15_dsr.csv", index=False)

    # (d) how many trials until a 'significant' hit is guaranteed noise
    print(f"\n(d) Trials vs expected-best-noise t-stat:")
    for N in [10, 56, 100, 500, 1000]:
        z1 = norm_ppf(1 - 1 / N)
        z2 = norm_ppf(1 - 1 / (N * math.e))
        print(f"    N={N:>5}: E[max t] ~= {(1-gamma)*z1 + gamma*z2:.2f}")
    return df


# ============================================================
# PART 2: STATISTICAL POWER / MINIMUM TRACK RECORD
# ============================================================
def part2_power():
    print("\n" + "=" * 72)
    print("PART 2: STATISTICAL POWER — HOW MANY TRADES DO YOU NEED?")
    print("=" * 72)

    # SE of Sharpe estimate (Lo 2002, normal approx):
    # SE(SR) ~ sqrt((1 + 0.5*SR^2)/n)
    print(f"\n(a) SE of per-trade Sharpe estimate by sample size:")
    print(f"    {'n_trades':>9} {'SE(SR=0)':>10} {'95% CI half-width':>18}")
    for n in [8, 14, 30, 50, 100, 300, 1000]:
        se = math.sqrt(1.0 / n)
        print(f"    {n:>9} {se:>10.3f} {1.96*se:>18.3f}")

    # trades needed to detect true per-trade SR with 80% power,
    # one-sided alpha=0.05 (and Bonferroni 0.05/56)
    print(f"\n(b) Trades needed to reject SR=0 (80% power):")
    print(f"    {'true SR':>8} {'n (a=0.05)':>12} {'n (a=0.05/56)':>15}")
    rows = []
    for sr in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        for alpha, label in [(0.05, "plain"), (0.05 / 56, "bonferroni")]:
            z_a = norm_ppf(1 - alpha)
            z_b = norm_ppf(0.80)
            # n ~ (z_a + z_b)^2 * (1 + 0.5*sr^2) / sr^2
            n_need = (z_a + z_b) ** 2 * (1 + 0.5 * sr ** 2) / sr ** 2
            rows.append({"true_per_trade_sr": sr, "alpha": label,
                         "trades_needed": int(math.ceil(n_need))})
        r1 = rows[-2]["trades_needed"]; r2 = rows[-1]["trades_needed"]
        print(f"    {sr:>8.2f} {r1:>12,} {r2:>15,}")

    # MinTRL (Bailey & Lopez de Prado): for annualized target SR,
    # expressed in trades: MinTRL = 1 + (1 - skew*SR + (kurt-1)/4 SR^2)
    # * (z_a / SR)^2  (z_a one-sided 95%)
    print(f"\n(c) Minimum Track Record Length (95% confidence, per-trade SR):")
    z_a = norm_ppf(0.95)
    for sr in [0.05, 0.10, 0.20, 0.30]:
        skew, kurt = -0.5, 4.0
        mintrl = 1 + (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) * (z_a / sr) ** 2
        print(f"    SR={sr:.2f}/trade -> MinTRL ~= {int(math.ceil(mintrl)):,} trades")

    # what does 41 days buy?
    print(f"\n(d) The 41-day sample:")
    print(f"    8 trades (MYM): SE(SR)=0.354 -> can only detect SR>~0.7/trade")
    print(f"    14 trades (M2K): SE(SR)=0.267 -> can only detect SR>~0.5/trade")
    print(f"    At ~0.2-0.5 trades/day/instrument, 100 trades = ~1 trading year.")
    print(f"    -> 41 days is UNINFORMATIVE about edge size per instrument,")
    print(f"       BUT the 6/56 sign test (Part 1) IS informative about the family.")

    df = pd.DataFrame(rows)
    df.to_csv("results/test15_power.csv", index=False)
    return df


# ============================================================
# PART 3: COST MODEL PER INSTRUMENT
# ============================================================
# IBKR fixed commissions (verified from IBKR pricing page, Aug 2026):
#   CME micro suite (MES,MNQ,M2K,MYM,MCL,MGC,SIL,...): $0.25/contract
#   E-micro FX (M6E,M6B,M6A,MJY): $0.15/contract
#   MBT $0.85, MET $0.20, MSL/MXP $2.25
# Exchange+clearing+NFA fees are pass-through; estimates below (per side).
INSTRUMENTS = {
    # sym: (name, ibkr_comm, exch_fee_est/side, tick_size, tick_value, typ_spread_ticks, rth_vol_note)
    "MES": ("Micro S&P 500",      0.25, 0.26, 0.25, 1.25, 1, "very high"),
    "MNQ": ("Micro Nasdaq-100",   0.25, 0.28, 0.25, 0.50, 1, "very high"),
    "MYM": ("Micro Dow",          0.25, 0.26, 1.00, 0.50, 1, "high"),
    "M2K": ("Micro Russell 2000", 0.25, 0.26, 0.10, 0.50, 1, "high"),
    "MCL": ("Micro WTI Crude",    0.25, 0.27, 0.01, 1.00, 1, "high"),
    "MGC": ("Micro Gold",         0.25, 0.27, 0.10, 1.00, 1, "high"),
    "SIL": ("Micro Silver",       0.25, 0.30, 0.005, 5.00, 1, "medium"),
    "MBT": ("Micro Bitcoin",      0.85, 0.40, 5.00, 0.50, 2, "medium"),
    "MET": ("Micro Ether",        0.20, 0.30, 0.05, 0.05, 2, "medium (tick value: verify)"),
    "MSL": ("Micro Solana",       2.25, 0.80, None, None, 3, "LOW (new, thin)"),
    "MXR": ("Micro XRP (MXP)",    2.25, 0.80, None, None, 3, "LOW (new, thin)"),
    "M6E": ("Micro EUR/USD",      0.15, 0.19, 0.0001, 1.25, 1, "medium-high"),
    "M6B": ("Micro GBP/USD",      0.15, 0.19, 0.0001, 0.625, 1, "medium"),
    "M6A": ("Micro AUD/USD",      0.15, 0.19, 0.0001, 1.00, 1, "medium"),
    "MJY": ("Micro JPY/USD",      0.15, 0.19, 0.0000005, 0.625, 1, "medium-low"),
}
NFA_FEE = 0.002  # per side


def part3_costs():
    print("\n" + "=" * 72)
    print("PART 3: COST MODEL PER INSTRUMENT (round trip, 1 contract)")
    print("=" * 72)
    rows = []
    print(f"\n{'sym':<6}{'IBKR':>6}{'exch+reg':>10}{'spread$':>9}{'TOTAL$':>8}"
          f"{'ticks':>7}  liquidity(RTH)")
    for sym, (name, comm, exch, tick, tv, spr, liq) in INSTRUMENTS.items():
        comm_rt = 2 * comm
        exch_rt = 2 * (exch + NFA_FEE)
        if tv is None:
            spread_usd = None
            total = comm_rt + exch_rt
            tot_ticks = None
        else:
            spread_usd = spr * tv
            total = comm_rt + exch_rt + spread_usd
            tot_ticks = total / tv
        rows.append({
            "symbol": sym, "name": name,
            "ibkr_comm_rt": round(comm_rt, 2),
            "exchange_reg_rt": round(exch_rt, 2),
            "spread_cost_usd": round(spread_usd, 2) if spread_usd else None,
            "total_round_trip_usd": round(total, 2),
            "total_in_ticks": round(tot_ticks, 1) if tot_ticks else None,
            "tick_value_usd": tv,
            "liquidity_rth": liq,
        })
        print(f"{sym:<6}{comm_rt:>6.2f}{exch_rt:>10.2f}"
              f"{spread_usd if spread_usd else 0:>9.2f}{total:>8.2f}"
              f"{tot_ticks if tot_ticks else 0:>7.1f}  {liq}")
    print(f"\nExchange/reg fees are ESTIMATES (CME pass-through, verify live).")
    print(f"Slippage on stops in fast moves adds 1-5+ ticks more (CFTC stop-loss")
    print(f"study; macro releases CPI/FOMC/NFP widen spreads several-fold).")
    df = pd.DataFrame(rows)
    df.to_csv("results/test15_costs.csv", index=False)
    return df


# ============================================================
# PART 4: CAPITAL FEASIBILITY (1-2% RISK PER TRADE)
# ============================================================
def part4_capital():
    print("\n" + "=" * 72)
    print("PART 4: CAPITAL FEASIBILITY UNDER 1-2% RISK PER TRADE")
    print("=" * 72)
    # stop distance scenarios in ticks (ATR-based stops on 15m for micros
    # typically land at 15-60 ticks depending on instrument/vol regime)
    stops_ticks = [20, 40, 60]
    rows = []
    print(f"\nRequired account ($) for 1 contract at 1% risk, by stop distance:")
    hdr = f"{'sym':<6}" + "".join(f"{'stop '+str(t)+'tk':>12}" for t in stops_ticks)
    print(hdr)
    for sym, (name, comm, exch, tick, tv, spr, liq) in INSTRUMENTS.items():
        if tv is None:
            print(f"{sym:<6}   tick value unverified — check CME spec")
            continue
        vals = []
        for t in stops_ticks:
            risk_usd = tv * t
            acct_1pct = risk_usd / 0.01
            vals.append(acct_1pct)
            rows.append({"symbol": sym, "stop_ticks": t,
                         "risk_usd_per_contract": round(risk_usd, 2),
                         "account_for_1pct_risk": round(acct_1pct, 0),
                         "account_for_2pct_risk": round(risk_usd / 0.02, 0)})
        print(f"{sym:<6}" + "".join(f"{v:>12,.0f}" for v in vals))
    df = pd.DataFrame(rows)

    # cost-vs-edge economics at $5k and $10k
    print(f"\nEconomics at $5,000 account, 1% risk ($50), 20-tick stop:")
    costs = {(r["symbol"]): r["total_round_trip_usd"] for _, r in df.iterrows()
             if False}  # placeholder
    cdf = pd.read_csv("results/test15_costs.csv")
    for sym in ["MES", "MNQ", "M2K", "MYM", "MCL"]:
        row = cdf[cdf["symbol"] == sym].iloc[0]
        tv = row["tick_value_usd"]
        risk = tv * 20
        if risk > 50:
            print(f"  {sym}: 20-tick stop risks ${risk:.0f} > $50 budget — infeasible at 1%")
            continue
        be_edge_ticks = row["total_round_trip_usd"] / tv
        print(f"  {sym}: risk ${risk:.2f}/trade; round-trip cost ${row['total_round_trip_usd']:.2f}"
              f" = {be_edge_ticks:.1f} ticks — every trade must beat {be_edge_ticks:.1f} ticks"
              f" of gross edge JUST to break even")
    df.to_csv("results/test15_capital.csv", index=False)
    return df


# ============================================================
# PART 5: PASSIVE ALTERNATIVE OVER THE SAME WINDOW
# ============================================================
def part5_passive():
    print("\n" + "=" * 72)
    print("PART 5: PASSIVE ALTERNATIVE, JUN-AUG 2026 (the test window)")
    print("=" * 72)
    try:
        adj = pd.read_csv("data/options_adj.csv", index_col=0, parse_dates=True)
        spy = adj["SPY"].dropna()
        seg = spy.loc["2026-06-01":"2026-08-31"]
        if len(seg) > 10:
            ret = seg.iloc[-1] / seg.iloc[0] - 1
            daily = seg.pct_change().dropna()
            sharpe_ann = daily.mean() / daily.std() * math.sqrt(TRADING_DAYS)
            print(f"\n  SPY total return {seg.index[0].date()} -> {seg.index[-1].date()}: "
                  f"{ret*100:+.2f}%  (ann. Sharpe {sharpe_ann:.2f}, DD "
                  f"{((seg/seg.cummax()-1).min())*100:.1f}%)")
            print(f"  Zero effort, zero skill, one decision. This is the bar.")
    except Exception as e:
        print(f"  (SPY window unavailable: {e})")


def run_test15():
    print("\n" + "#" * 72)
    print("# TEST 15: INTRADAY MICRO-FUTURES EDGE — EVIDENCE AUDIT")
    print("#" * 72)
    part1_multiple_testing()
    part2_power()
    part3_costs()
    part4_capital()
    part5_passive()
    print("\nAll results saved to results/test15_*.csv")


if __name__ == "__main__":
    run_test15()