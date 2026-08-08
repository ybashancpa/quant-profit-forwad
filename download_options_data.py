"""
Download and cache price history for the options-income research (Test 14).

Series:
  ^BXM  - Cboe S&P 500 BuyWrite (price index)
  ^PUT  - Cboe S&P 500 PutWrite (price index)
  ^SPX  - S&P 500 (price)
  ^VIX  - implied vol reference
  SPY   - total return proxy (Adj Close) and price basis
  SHY   - T-bill proxy for collateral yield
  IEF   - reference
  QQQ   - high-beta underlying reference
  TLT   - reference

Saved to data/options_prices.csv (Close, unadjusted) and data/options_adj.csv (Adj Close).
"""
import os
import yfinance as yf
import pandas as pd

TICKERS = ["^BXM", "^PUT", "^SPX", "^VIX", "SPY", "SHY", "IEF", "QQQ", "TLT"]
START = "2002-01-01"

os.makedirs("data", exist_ok=True)

close_frames, adj_frames = {}, {}
log = []
for t in TICKERS:
    try:
        d = yf.download(t, start=START, auto_adjust=False, progress=False, threads=False)
        if d is None or len(d) == 0:
            log.append(f"{t}: NO DATA")
            continue
        if isinstance(d.columns, pd.MultiIndex):
            c = d["Close"].iloc[:, 0]
            a = d["Adj Close"].iloc[:, 0]
        else:
            c = d["Close"]
            a = d["Adj Close"]
        c = pd.Series(c).dropna()
        a = pd.Series(a).dropna()
        close_frames[t] = c
        adj_frames[t] = a
        log.append(f"{t}: {len(c)} rows {c.index[0].date()} -> {c.index[-1].date()}")
    except Exception as e:
        log.append(f"{t}: ERROR {e}")

close = pd.DataFrame(close_frames)
adj = pd.DataFrame(adj_frames)
close.to_csv("data/options_prices.csv")
adj.to_csv("data/options_adj.csv")

with open("data/options_download_log.txt", "w") as f:
    f.write("\n".join(log))
print("\n".join(log))
print(f"\nSaved data/options_prices.csv ({close.shape}) and data/options_adj.csv ({adj.shape})")