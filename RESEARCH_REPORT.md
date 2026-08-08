# Quant Research Report: Systematic Trading Strategy for $10,000 Portfolio

**Date:** August 2026  
**Researcher:** Quant Research AI  
**Objective:** Determine whether a sustainable edge exists for active momentum trading after costs, for a $10,000 portfolio.

---

## Executive Summary

**VERDICT: NO SUSTAINABLE EDGE FOUND FOR COMPLEX ACTIVE STRATEGY**

After rigorous testing across 6 phases, including out-of-sample validation, walk-forward analysis, and benchmark comparison, we conclude that:

1. The complex Dual Momentum strategy does NOT provide risk-adjusted outperformance over a simple defensive rule.
2. The strategy's apparent edge is period-dependent and disappears under adaptive walk-forward testing.
3. A simple 60/40 + MA200 rule delivers better Sharpe with 70% lower turnover.
4. The original 8-9% annual return target is achievable through simple defensive rules, without complex automation.

---

## Research Methodology

### Principles Applied
1. **Economics before statistics** - Every signal required a causal hypothesis
2. **Net after costs is the only truth** - All results calculated after 0.1%/side costs
3. **Default assumption is overfitting** - All positive results assumed random until validated OOS
4. **Risk before return** - Max drawdown 25% hard limit enforced
5. **Small portfolio constraints** - Low turnover prioritized due to $10k size
6. **Full transparency** - All failures documented alongside successes

### Universe
10 liquid ETFs across asset classes:
- US Equities: SPY, QQQ
- International: EFA, EEM
- Bonds: TLT, IEF, SHY
- Real Assets: GLD, DBC, VNQ

### Data
- Source: Yahoo Finance (yfinance)
- Period: 2007-2026 (~19.5 years)
- Prices: Adjusted Close (dividend/split adjusted)
- No missing values

---

## Test Results Summary

### Test 1: Time-Series Momentum (Individual Assets)
**Result:** POSITIVE but insufficient alone
- 29/30 configurations showed positive net Sharpe
- SPY MA200: Net Sharpe 0.736, Max DD -21.6% (passes 25% limit)
- Most individual assets failed 25% DD constraint
- **Conclusion:** Valid building block, but needs diversification

### Test 2: Cross-Sectional Momentum (Monthly Rebalance)
**Result:** POSITIVE
- All 12 configurations showed positive net Sharpe
- Best: 6M Top3 - Net CAGR 10.49%, Sharpe 0.775, DD -22.5%
- **Conclusion:** Momentum effect is real and survives costs

### Test 3: Dual Momentum (Combined)
**Result:** STRONG PASS (initially)
- All 6 configurations passed 25% DD limit
- Best: 6M/MA150 - Net CAGR 10.63%, Sharpe 0.826, DD -17.9%
- **Conclusion:** Combining relative + absolute momentum reduces drawdowns

### Test 3.5: Rigorous Validation (IS/OOS Split)
**Result:** MIXED
- IS (2007-2017): Sharpe 0.823
- OOS (2018-2026): Sharpe 0.791 (only 4% decay - initially promising)
- **BUT:** OOS Sharpe (0.791) essentially tied with SPY (0.790), lost to 60/40 (0.837)
- **Crisis performance:** 2008: +5.25% vs SPY -43.90% (excellent protection)
- **Conclusion:** Edge is in DRAWDOWN PROTECTION, not Sharpe enhancement

### Test 4: Inverse Volatility Weighting
**Result:** MARGINAL IMPROVEMENT
- OOS Sharpe: 0.834 vs 0.791 (Equal Weight)
- Still lost to 60/40 (0.837) by 0.003
- Turnover increased by 0.5x/year
- **Conclusion:** Not worth the added complexity/costs

### Test 5: Mean-Reversion Entry Timing
**Result:** ABANDONED
- Internal contradiction: momentum + mean-reversion signals conflict
- High overfitting risk for marginal benefit
- **Conclusion:** Correctly rejected before implementation

### Test 6: Robustness & Stress Testing
**Result:** STRATEGY FAILS FINAL TEST

#### Part 1: Cost Stress Test - PASS
| Cost/Side | Net CAGR | Net Sharpe |
|-----------|----------|------------|
| 0.05% | 11.00% | 0.851 |
| 0.10% | 10.63% | 0.826 |
| 0.20% | 9.89% | 0.776 |

Strategy survives 2x cost stress.

#### Part 2: Adaptive Walk-Forward Analysis - FAIL
- **Average OOS Sharpe: 0.440** (vs 0.791 in single split)
- Positive years: 11/15
- Config selection unstable (5 different configs selected)
- **Conclusion:** Single IS/OOS split was flattering; true performance is much lower

#### Part 3: Smart Benchmark Comparison - LOST
| Strategy | CAGR | Sharpe | Max DD | Turnover |
|----------|------|--------|--------|----------|
| DualMom 6M/MA150 | 10.63% | 0.826 | -17.88% | 3.33x |
| **Smart 60/40 (MA200)** | **7.93%** | **0.960** | **-23.85%** | **0.98x** |
| Simple 60/40 | 8.49% | 0.780 | -31.39% | 0.03x |

**The simple 60/40 + MA200 rule BEATS the complex strategy on Sharpe with 70% lower turnover.**

---

## Key Findings

### 1. The Momentum Effect is Real (But Not Enough)
Cross-sectional and time-series momentum both show positive net Sharpe after costs. However, the edge is not large enough to justify active trading for a $10k portfolio.

### 2. Drawdown Protection is the Real Value
The strategy's primary benefit is reducing maximum drawdown (-17.9% vs -55% for SPY). However, this can be achieved more cheaply with simpler rules.

### 3. Period-Dependence is the Killer
The 0.79 OOS Sharpe from a single split was an artifact of the 2018 cutoff. Adaptive walk-forward revealed the true expected Sharpe is ~0.44.

### 4. Complexity Doesn't Justify Itself
A 2-line rule (60/40 + MA200) delivers:
- Higher Sharpe (0.960 vs 0.826)
- Lower turnover (0.98x vs 3.33x)
- Simpler implementation

### 5. The 2022 Niche
The only legitimate edge is in "nowhere to hide" scenarios (2022: stocks AND bonds falling). DualMom lost -10.6% vs SmartPassive -22.8%. However, this protection can be achieved more cheaply by holding gold/cash in a passive portfolio.

---

## Final Recommendation

### For a $10,000 Portfolio:

**DO NOT implement the complex Dual Momentum strategy.**

Instead, consider:
1. **Smart 60/40 (MA200):** 60% SPY / 40% IEF when SPY > MA200; 100% IEF otherwise
2. **Add diversification:** Small allocation to GLD (5-10%) for 2022-type scenarios
3. **Rebalance monthly** with awareness of tax implications

### Expected Realistic Returns:
- **CAGR:** 7-9% annually (not 20%)
- **Sharpe:** 0.7-0.9 (with some luck)
- **Max DD:** 20-25% in severe crises

### What This Research Proved:
- Active momentum trading does NOT beat simple defensive rules after costs
- The 8% original target IS achievable through passive/defensive approaches
- Complex automation adds cost and complexity without commensurate benefit
- 20% annual returns were never realistic for this approach

---

## Files Generated

| File | Description |
|------|-------------|
| `config.py` | Configuration parameters |
| `data_loader.py` | Data download and caching |
| `backtest_engine.py` | Core backtesting infrastructure |
| `metrics.py` | Performance metrics calculation |
| `test1_tsmom.py` | Time-series momentum test |
| `test2_xsmom.py` | Cross-sectional momentum test |
| `test3_dual_momentum.py` | Dual momentum test |
| `test3_5_validation.py` | IS/OOS validation |
| `test4_vol_weighting.py` | Inverse volatility weighting |
| `test6_robustness.py` | Final robustness testing |
| `results/*.csv` | All test results |

---

## Methodology Notes

### Cost Model
- Baseline: 0.1% per side (0.2% round trip)
- Stress test: 0.2% per side (0.4% round trip)
- Includes: commission, spread, slippage

### Validation Protocol
1. In-sample: 2007-2017 (parameter selection)
2. Out-of-sample: 2018-2026 (single look, no changes)
3. Adaptive walk-forward: 5-year train, 1-year test, rolling

### Multiple Testing Awareness
- Total configurations tested: ~60
- Deflated Sharpe considerations applied
- Walk-forward used to address selection bias

---

**Research Status: COMPLETE**  
**Verdict: NEGATIVE (no sustainable edge for complex strategy)**  
**Integrity: FULL (all results reported, including failures)**

---

## Addendum: Test 10 — Low-Vol Strategy Lab (August 2026)

**Question:** Can a low-vol construction (static allocation, capped inverse-vol risk parity, or a vol-targeting overlay with up to 150% leverage) beat SPY or deliver a persistent double-digit CAGR after costs **and financing**, within the 25% DD limit?

**Setup:** Same infrastructure (`backtest_engine`, `metrics`, crisis analysis). Monthly rebalance, 1-day lag, 10bps/side costs, and a new engine feature: annual financing charge on gross exposure above 100% (primary: 3%/yr). Pre-registered parameters: 60d vol window, 12% vol target, 1.5x leverage cap, 30% per-asset cap, MA200 defensive overlay (x0.5).

**Full-period results (2007-2026):**

| Strategy | CAGR | Sharpe | Max DD | Avg Gross | Financing Drag |
|----------|------|--------|--------|-----------|----------------|
| StaticLowVol 40/40/20 | 7.26% | 0.762 | -27.57% | 100% | — |
| RiskParity (capped IV) | 5.69% | 0.871 | -16.50% | 99% | — |
| RP + VolTarget 12%/1.5x | 6.46% | 0.721 | -24.37% | 144% | 26.2% cumulative |
| Defensive VolTarget | 5.73% | 0.769 | -17.45% | 136% | 21.1% cumulative |
| SPY (benchmark) | 10.98% | 0.630 | -55.19% | 100% | — |
| SmartPassive MA200 | 7.62% | 1.049 | -19.92% | 100% | — |

**OOS (2018+):** SPY CAGR 14.18%. No candidate beat SPY on CAGR; no candidate reached double-digit CAGR. RiskParity had the best candidate Sharpe (0.866) with -16.4% DD.

**Why leverage did not work:** the unlevered RP portfolio runs ~8% realized vol, so the 12% target pins the strategy at the 1.5x cap in 95% of months — it degenerates into static leverage, and the financing charge (~1.3%/yr) consumes the extra return. Sensitivity (10/12/15% targets × 2/3/5% financing) changed CAGR by less than 1.7pp; best cell (15% target, 2% financing) still only 7.12%.

**Crisis behavior (candidate strength):** DefensiveVT 2008: -2.67% (DD -12.1%); RiskParity 2008: -4.84%; all candidates protected far better than SPY (-43.9%), but 2022 hit every variant (-12% to -18%).

**VERDICT: NEGATIVE for the return objective.** Low-vol constructions in this universe are risk-reduction tools, not return engines. Double-digit CAGR is not achievable after costs and financing within the 25% DD constraint via these methods; claiming otherwise would require parameter fitting. RiskParity remains the best pure risk-adjusted low-vol option (Sharpe 0.87, DD -16.5%), complementary to — not a replacement for — SmartPassive.

Files: `test10_low_vol_lab.py`, `results/test10_low_vol_lab_summary.csv`, `results/test10_sensitivity.csv`, `results/test10_crisis.csv`.

---

## Addendum: Test 11 — Cheap Leverage & Volatility Premium (August 2026)

**Question:** Does Test 10's leverage failure come from an overly-pessimistic financing assumption, and can modest leverage on the best-Sharpe portfolio or selling volatility close the gap to double-digit returns?

**Part 1 — Financing model (RP_VolTarget 12%/1.5x):**

| Financing | CAGR | Sharpe | Max DD |
|-----------|------|--------|--------|
| Fixed 3% (retail) | 6.46% | 0.721 | -24.37% |
| Fixed 1.5% (optimistic) | 7.17% | 0.794 | -24.12% |
| Dynamic T-bill proxy (SHY-based, lagged) | 6.84% | 0.761 | -24.49% |

Cheaper financing is real but smaller than hoped: the dynamic T-bill proxy averaged **2.29%/yr** over 2007-2026 (not 1-1.5%), because 2023-2026 T-bill yields were 4-5%. Best case lifts CAGR by only ~0.7pp.

**Part 2 — Fixed leverage on RiskParity (T-bill financing):**

| Leverage | CAGR | Sharpe | Max DD |
|----------|------|--------|--------|
| 1.00x | 5.69% | 0.871 | -16.50% |
| 1.25x (primary) | 6.48% | 0.802 | -20.58% |
| 1.50x | 7.25% | 0.756 | -24.49% |

Leverage buys CAGR at a declining Sharpe price. OOS (2018+): RP@1.25x = 6.50% CAGR, Sharpe 0.787, DD -20.1% vs SPY 14.18% / 0.790. Lost on both.

**Part 3 — Selling volatility:** BXM (CBOE BuyWrite, 2007-2026): CAGR 5.94%, Sharpe 0.467, Max DD **-40.1%** vs SPY price-basis 8.88% / 0.531. Buy-write underperformed on both return and Sharpe and did NOT avoid 2008. PutWrite index unavailable via yfinance — reported INFEASIBLE, no synthetic approximation.

**VERDICT: NEGATIVE again.** All three directions behave exactly as pre-registered expectations suggested (or worse): cheap leverage adds ~1pp CAGR, modest leverage on the best Sharpe reaches ~6.5-7.3%, and selling vol is a worse risk profile than holding SPY. None reaches double-digit CAGR or beats SPY OOS within the 25% DD limit. The binding constraint is the low-vol universe's ~5.7% unlevered CAGR, not the financing model.

Files: `test11_cheap_leverage.py`, `results/test11_financing_models.csv`, `results/test11_fixed_leverage.csv`, `results/test11_buywrite.csv`, `results/test11_isoos.csv`, `results/test11_summary.csv`.

---

## Addendum: Test 12 — Global Momentum Alpha/Beta Regression (August 2026)

**The closing measurement.** Explicit alpha/beta regression of the global cross-asset momentum strategy (DualMom 6M/MA150, locked, net of costs) against SPY, with Newey-West HAC inference (21 lags, pre-registered; plain OLS t-stats would be overstated by monthly-rebalance overlap). Criterion: OOS alpha t >= 1.96 with beta <= 1.5.

**Pre-registered prediction:** alpha significant in-sample, NOT significant OOS, much of the return disguised beta.

| Period | Alpha (ann) | t-stat | Beta | Beta-explained share | R2 |
|--------|-------------|--------|------|----------------------|-----|
| Full (2007-2026) | +7.28% | 2.99 | 0.298 | 34% | 0.193 |
| IS (2007-2017) | +8.77% | **2.62 (sig)** | 0.240 | 21% | 0.126 |
| OOS (2018+) | +4.83% | 1.48 (not sig) | 0.379 | 55% | 0.312 |

Context: SmartPassive MA200 full-period alpha = +5.41%/yr, t = 4.01 — the simple defensive rule shows *stronger* statistical alpha than the complex momentum strategy on this metric.

**RESULT: PREDICTION CONFIRMED.** Alpha decays ~45% from IS to OOS (8.77% -> 4.83%) and loses statistical significance; the beta-explained share of return rises from 21% to 55%. Beta itself is low (<= 0.38), so the strategy is genuinely low-beta — but its skill component does not survive out-of-sample. No tradable alpha with beta <= 1.5.

This closes the research loop: every direction — active momentum, low-vol constructions, cheap leverage, selling volatility, and now explicit alpha inference — has been measured and none clears the bar. The robust, defensible result remains defensive risk management (SmartPassive-style rules), not return generation.

Files: `test12_alpha_beta.py`, `results/test12_alpha_beta.csv`.

---

## Addendum: Test 13 — Adaptive Crisis Rotation (August 2026)

**The legitimate adaptive design.** Fixed escape universe (TLT/IEF/GLD/DBC + SHY fallback); dynamic choice within it by live 6m momentum (top-2 positive, equal weight; none positive -> 100% SHY). Calm core = capped inverse-vol risk parity. Regime detector pre-registered: SPY<=MA200 OR 60d vol>25%/yr, hysteresis exit (both flags must clear). Beta cap 1.5 rolling, no leverage, 10bps/side.

**Decisive controls:** static-average-escape (SAE: static portfolio with the adaptive strategy's average weights; IS-trained version frozen for OOS + ex-post diagnostic), EscapeSHY, EscapeIEF (the 2008 trap), matched-beta SPY/SHY. Multiple testing: project-wide Bonferroni alpha = 0.05/13 ~= 0.0038; Holm within family.

**Full period:** Adaptive CAGR 5.69%, Sharpe **0.647**, DD -19.7%, turnover 1.85x — BELOW RiskParity (0.871), EscapeSHY (0.850), EscapeIEF (0.815), SAE ex-post (0.841), SmartPassive (1.049).

**The rotation DID adapt correctly in known crises** (the mechanism works as designed): 2008 +20.2% (rotated GLD/DBC -> TLT/IEF as the crisis evolved); 2020 +3.9% (TLT/IEF); 2022 -10.9% (DBC/GLD -> SHY, avoiding the IEF trap; EscapeIEF lost -15.6%). Stress = 24.7% of time; turnover calm 0.82x vs stress 1.03x per year.

**But OOS (2018+) kills it:** Adaptive Sharpe **0.583** vs SAE(IS-trained) **0.890**, RiskParity 0.866, EscapeSHY 0.683. Bootstrap Sharpe diffs all negative (p = 0.89 / 0.86 / 0.64). NW alpha: +5.40%/yr t=2.74 full period, decaying to +4.02% t=1.50 (not significant) OOS — same IS/OOS decay pattern as Test 12. Sensitivity: symmetric exit marginally better (0.694), 12M momentum collapses (Sharpe 0.344, DD -36.7%), vol threshold 20-30% nearly flat. Cost stress 50bps: Sharpe 0.485.

**Modern addendum (n=1, framed as anecdote):** with DBMF/KMLM/XLU/XLP added (window from Dec-2021 post-warmup), the system correctly picked KMLM/DBMF during 2022 stress — yet the window total is CAGR -0.30%, 2022 return -17.2%. Single-crisis observation; no inference.

**VERDICT: NEGATIVE.** The rotation mechanism is real and behaved adaptively in every known crisis — but the timing adds no OOS value: adaptive rotation does NOT beat its own static average composition (the SAE control), nor plain risk parity, nor escaping to SHY. The switching costs and detector lag consume the crisis gains in aggregate. Per the pre-registered decision rule, the "adaptivity" is a composition effect: holding the defensive mix statically does the same job, more cheaply.

Files: `test13_adaptive_crisis_rotation.py`, `results/test13_isoos.csv`, `results/test13_bootstrap.csv`, `results/test13_sensitivity.csv`, `results/test13_crisis.csv`, `results/test13_alpha.csv`, `results/test13_addendum.csv`, `results/test13_escape_log.csv`.

---

## Addendum: Test 14 — Options Income / Premium Selling Evidence Audit (August 2026)

**Question:** do option premium-selling strategies produce a sustained positive expectation after commissions, spreads, slippage and tax for a private Israeli account of $5k–$50k? Full critical report (Hebrew): `OPTIONS_INCOME_RESEARCH.md`.

**Evidence collected:**
- Full-history (2002–2026) measurement of the Cboe PUT and BXM benchmark indices vs SPX/SPY-TR/SHY; direct VRP measurement (VIX² vs forward 21-day realized variance, monthly); tail-event windows (2008, Volmageddon 2018, Mar 2020, 2022, Aug 2024); IBKR retail cost model; account-size feasibility under a 2% risk cap; Israeli 25% capital-gains wedge.
- Academic anchor: Dew-Becker & Giglio (Chicago Fed WP 2025-17, and June 2026 version) — S&P 500 option alphas became statistically indistinguishable from zero after ~2012; synthetic options never had negative alpha over ~100 years; VIX has converged to realized vol on average.

**Key measurements (our own, reproducible in `test14_options_income.py`):**

| Series (2002–2026) | CAGR | Sharpe | Max DD |
|---|---|---|---|
| PUT (Cboe PutWrite) | 7.43% | 0.532 | -37.1% |
| BXM (Cboe BuyWrite) | 6.16% | 0.493 | -40.1% |
| SPY total return | 10.00% | 0.598 | -55.2% |

- Sub-periods: PUT beat the index 2002–2012 (the VRP era), lost clearly 2013–2019 (6.99% vs 14.20% SPY-TR) and 2020–2026 (9.45% vs 15.62%).
- Rolling 10-year windows: PUT beats SPY-TR in only 13.1% of windows (6.7% for windows ending 2013+); BXM in 4.0%.
- Pure premium component (PUT − T-bill): +7.2%/yr, positive in 73% of months — real, but it is the compensation for the tail: PUT lost -28.9% in 23 trading days in Mar 2020 (1.6× the prior three years of cumulative gains) and -34.8% peak-to-trough in GFC.
- VRP itself: still positive on average (~1 variance point; IV>RV in 84% of months) but small, and per the academic evidence no longer convertible into significant hedged alpha.
- Retail costs (IBKR fixed): $4.11 / $6.81 / $13.63 round-trip per 1/2/4-leg structure at 1-tick half-spread = 23–34% of a typical $1-wide spread credit; ~1.6%/yr account drag at monthly frequency.
- Granularity: one cash-secured SPY put requires ~$77k collateral (Aug-2026 price) — infeasible for the entire $5k–$50k range; only defined-risk $1-wide spreads fit a 2% risk cap, and costs consume most of the credit.

**VERDICT: NEGATIVE for the stated account size.** The raw premium harvest exists but (a) the exploitable hedged alpha disappeared around 2012, (b) the benchmark selling indices underperform holding the index in 87–96% of rolling 10-year windows before costs and tax, and (c) costs + granularity + 25% tax eliminate the theoretical edge for accounts below ~$25k–$50k (and cash-secured index structures below ~$77k). Falsification criteria were pre-committed in the report and are already failed by existing data.

Files: `test14_options_income.py`, `OPTIONS_INCOME_RESEARCH.md`, `results/test14_*.csv`, primary sources in `research_sources/`.

---

## Addendum: Test 15 — Intraday Micro-Futures Edge Evidence Audit (August 2026)

**Question:** is there a sustained, exploitable intraday edge in any of 15 CME micro futures contracts (MES, MNQ, MYM, M2K, MCL, MGC, SIL, MBT, MET, MSL, MXR, M6E, M6B, M6A, MJY) after commissions, spreads and slippage, for a $5k–$50k private account? Full critical report (Hebrew): `MICRO_FUTURES_RESEARCH.md`.

**Evidence collected:**
- The Momentum-Pullback strategy (VWAP direction + ADX>25 on 15m + EMA20/50 1h context + EMA20-touch trigger) was run on real contract data, RTH only, Jun–Aug 2026 (41 trading days): ~56 configurations (15 instruments × 4 stop multipliers). Raw result: **6 profitable, 50 losing** (MYM: 8 trades, -$104; M2K: 14 trades, -$212).
- Multiple-testing audit, statistical-power analysis, per-instrument IBKR cost model, capital feasibility under a 1–2% risk cap, and the passive alternative over the same window.

**Key measurements (our own, reproducible in `test15_micro_futures.py`):**
- **Sign test: 6/56 profitable.** Under a zero-edge null ~28/56 should be profitable before costs; P(≤6 | null) = **5.1e-10**. The family underperforms random chance — a statistically significant negative signal, not noise.
- **Deflated Sharpe:** with 56 trials the expected best noise t-stat is ≈2.32; with only 8–14 trades per instrument even a 0.30 per-trade Sharpe cannot pass the deflated test.
- **Power:** detecting a true 0.10 per-trade Sharpe needs ~1,580 trades (Bonferroni, 80% power); 0.20 needs ~400. The 41-day sample (8–14 trades) can only detect an implausibly large edge (>0.5/trade).
- **Costs (round trip, 1 contract):** MES $2.27 (1.8 ticks), MNQ $1.56 (3.1), MYM/M2K $1.52 (3.0), MCL/MGC $2.04 (2.0), MBT $3.50 (7.0), SIL $6.10, MSL/MXR $6.10+ (thin). Every trade must clear 1.8–3.1 ticks of gross edge just to break even, before stop slippage.
- **Capital:** $5k suffices for the index micros at 1% risk (MES needs ~$2,500, MNQ/MYM/M2K ~$1,000 at a 20-tick stop); SIL needs ~$10k. Capital is NOT the binding constraint — the absence of edge is.
- **Passive bar:** SPY returned **+1.58%** over the same window (Jun 1 – Aug 6, 2026) with zero effort.

**VERDICT: NEGATIVE.** No exploitable intraday edge was found. The tested family loses systematically (worse than chance), costs consume 1.8–3.1 ticks before any edge, the sample is far too small to certify a small positive edge, and the passive alternative wins the window. Recommendation: do not deploy intraday micro-futures momentum-pullback trading; capital is better allocated to the passive/vol-managed strategies validated in Tests 1–13.

Files: `test15_micro_futures.py`, `MICRO_FUTURES_RESEARCH.md`, `results/test15_*.csv`.
