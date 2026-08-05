"""Quick data verification script."""
import pandas as pd

df = pd.read_csv('data/etf_prices.csv', index_col=0, parse_dates=True)
print('='*60)
print('DATA VERIFICATION')
print('='*60)
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'Date range: {df.index[0].date()} to {df.index[-1].date()}')
print(f'Trading days: {len(df)}')
print(f'\nNaN counts per ticker:')
print(df.isna().sum())
print(f'\nFirst 3 rows:')
print(df.head(3))
print(f'\nLast 3 rows:')
print(df.tail(3))
print('='*60)