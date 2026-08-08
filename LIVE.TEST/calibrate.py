"""ממיר נתוני פרוקסי לרמות מדד אמיתיות ומריץ בקטסט"""
import pandas as pd
from config import INSTRUMENTS, RiskConfig, StrategyConfig
from strategy import MomentumPullbackStrategy
from backtest import Backtester

LEVELS = {'MNQ': 29421.0, 'MES': 7734.0, 'M6E': 1.1650}
FILES = {'MNQ':'MNQ_proxy_QQQ.csv','MES':'MES_proxy_SPY.csv','M6E':'M6E_proxy_FXE.csv'}

def load_scaled(sym):
    d = pd.read_csv(FILES[sym], index_col=0, parse_dates=True)
    k = LEVELS[sym] / d['close'].iloc[-1]
    for c in ['open','high','low','close']:
        d[c] *= k
    return d

if __name__ == '__main__':
    import sys
    syms = sys.argv[1:] or ['MNQ','MES','M6E']
    for s in syms:
        df = load_scaled(s)
        print(f"\n{'#'*66}\n  {s} | {len(df)} נרות | מחיר {df.close.iloc[-1]:,.4f}")
        for stop_m in [0.5, 0.75, 1.0, 1.5]:
            cfg = StrategyConfig(stop_atr_mult=stop_m)
            bt = Backtester(INSTRUMENTS[s], MomentumPullbackStrategy(cfg), RiskConfig(), 5000)
            bt.run(df)
            st = bt.stats()
            n = st.get('עסקאות',0)
            if n==0:
                rej = list(bt.rejections.keys())[:1]
                print(f"   stop {stop_m:<5} → 0 עסקאות   ({rej[0] if rej else '—'})")
            else:
                print(f"   stop {stop_m:<5} → {n:>3} עסקאות | הצלחה {st['אחוז הצלחה']:>6} | "
                      f"PF {st['Profit Factor']:>5} | {st['רווח נקי']:>10} | avgR {st['ממוצע R']:>6}")
