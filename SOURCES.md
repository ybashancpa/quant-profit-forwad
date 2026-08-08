# Research Sources Index

Bibliographic index for the PDFs and cost schedules in `research_sources/`.
The binary PDFs themselves are **not** committed (large, regenerable via the
`fetch_sources*.py` scripts); this file is the durable, reviewable record of
what was read, with enough metadata (title, authors, year, venue, DOI/URL) to
locate each source independently.

**Legend — type:** `primary` = read in full and used directly; `methodology`
= index/contract spec; `broker` = cost schedule; `secondary` = cited via
another source, not read in full.

**Reproduce:** `python fetch_sources.py` and `python fetch_sources_futures.py`
re-download the PDFs into `research_sources/` (see `_fetch_log*.txt` for the
recorded outcomes, including any 403 failures).

---

## 1. Options income / variance risk premium (Test 14)

| Local file | Citation | DOI / URL | Type | Read |
|---|---|---|---|---|
| `chicagofed_wp2025-17_vrp.pdf` | Dew-Becker & Giglio (2025), *The Decline of the Variance Risk Premium: Evidence from Traded and Synthetic Options*, Chicago Fed WP 2025-17 | `10.21033/wp-2025-17` | primary | full |
| `vrp_decline_dewbecker.pdf` | Dew-Becker & Giglio (2026), *The Decline of the S&P 500 Variance Risk Premium* (expanded June-2026 version) | dew-becker.org/documents/option_decline.pdf | primary | full |
| `cboe_benchmarks_factsheet.pdf` | Cboe (2020), *Benchmark Indexes* factsheet — **interested party** | cdn.cboe.com | methodology | full |
| `cboe_putwrite_methodology.pdf` | Cboe, *PutWrite Indices Methodology* | cdn.cboe.com | methodology | full |
| `cboe_buywrite_methodology.pdf` | Cboe, *BuyWrite Indices Methodology* | cdn.cboe.com | methodology | full |

Background cited through the above (not read in full): Coval & Shumway (2001);
Bakshi & Kapadia (2003); Broadie, Chernov & Johannes (2007); Bollerslev &
Todorov (2011); Bates (2022); Constantinides, Jackwerth & Savov (2013);
Carr & Wu (2009).

## 2. Intraday momentum / micro-futures (Test 15, H1, H2)

| Local file | Citation | DOI / URL | Type | Read |
|---|---|---|---|---|
| `gao_han_li_zhou_intraday_momentum.pdf` | ⚠️ **MISLABELED** — actually Limkriangkrai, Chai & Zheng (2023), *Market intraday momentum: APAC evidence*, Pacific-Basin Finance Journal 80, 102086 | `10.1016/j.pacfin.2023.102086` | primary | full |
| `intraday_tsm_international_lancaster.pdf` | Li (2020), international intraday time-series momentum, Lancaster FoFI WP 2020-092 | wp.lancs.ac.uk | primary | full |
| `intraday_tsm_centaur_reading.pdf` | University of Reading (Centaur) accepted version, intraday TSM | centaur.reading.ac.uk/95566 | primary | full |
| `park_irwin_reality_check_futures.pdf` | Park & Irwin, reality check on futures technical trading rules | farmdoc.illinois.edu | primary | full |

> **Note on the mislabeled file:** the fetch script intended the Gao, Han, Li
> & Zhou (2018, JFE) intraday-momentum paper under this filename, but the
> Monash URL returned the Limkriangkrai et al. (2023) APAC replication instead.
> The canonical Gao et al. (2018) result is cited in `H1_hypothesis.md` /
> `H2_hypothesis.md` from the literature, not from a locally archived copy.

## 3. Multiple-testing & backtest-overfitting methodology

| Local file | Citation | DOI / URL | Type | Read |
|---|---|---|---|---|
| `bailey_lopez_de_prado_deflated_sharpe.pdf` | Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*, J. Portfolio Management | ssrn.com/abstract=2460551 | primary | full |
| `white_reality_check_2000.pdf` | White (2000), *A Reality Check for Data Snooping*, Econometrica 68(5), 1097–1126 | — | primary | full |
| `mclean_pontiff_2016_anomaly_decay.pdf` | McLean & Pontiff (2016), *Does Academic Research Destroy Stock Return Predictability?*, J. Finance LXXI(1) | `10.1111/jofi.12365` | primary | full |

## 4. Day-trader performance literature

| Local file | Citation | DOI / URL | Type | Read |
|---|---|---|---|---|
| `barber_lee_liu_odean_taiwan_day_traders.pdf` | Barber, Lee, Liu & Odean, *Do Day Traders Profit?* (Taiwan) | faculty.haas.berkeley.edu/odean | primary | full |
| `odean_day_traders_learning.pdf` | Odean, day trading & learning | faculty.haas.berkeley.edu/odean | primary | full |
| `day_trading_for_a_living_brazil.pdf` | Brazilian day-trading-for-a-living study | ebicapital.nl | primary | full |

## 5. Broker cost schedules (IBKR)

| Local file | Content | Type |
|---|---|---|
| `ibkr_options_commissions.txt` | IBKR options commission schedule (fixed pricing) | broker |
| `ibkr_futures_commissions.html` / `.txt` | IBKR futures commissions page | broker |
| `ibkr_micro_futures_comparison.txt` | Micro-futures contract/commission comparison | broker |

> Broker schedules are point-in-time snapshots; re-verify against the live IBKR
> pricing pages before any live/Paper cost assumption is used in a decision.

---

## Known fetch failures
- `cftc_stoploss_futures_study.pdf` — **403 Forbidden** (not archived). Referenced
  in `fetch_sources_futures.py` but never downloaded.
- `data/cboe/*.csv` — CBOE CDN index-history fetch returned **403**; those files
  are error stubs and are git-ignored (see `data/cboe/_fetch_log.txt`).