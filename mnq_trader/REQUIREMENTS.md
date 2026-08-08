# הרצת בדיקת H1 — הוראות

## ⚠️ התקנה בספרייה נפרדת

`config.py` כאן מתנגש עם `config.py` של מחקר ה-ETF.
שים את כל הקבצים בתיקייה משלהם:

```
mnq_trader/          ← כל הקבצים כאן
research_etf/        ← הפרויקט הקודם, נפרד
```

## התקנה

```bash
pip install pandas numpy yfinance pyarrow
```

## הרצה

```bash
cd mnq_trader
python test_h1.py --symbols MYM M2K
```

## סדר התלויות

```
config.py       ← אין תלויות (מפרטים, RiskConfig, StrategyConfig)
   ↓
indicators.py   ← config           (VWAP, ADX, ATR, EMA, יישור טווחי זמן)
   ↓
strategy.py     ← config, indicators   (MomentumPullbackStrategy)
risk.py         ← config               (size_position, CircuitBreaker)
   ↓
backtest.py     ← config, risk, strategy   (Backtester)
   ↓
screener.py     ← הכל                 (MICROS ×15, download, rth_only, to_instrument)
   ↓
test_h1.py      ← הכל                 (LullFilteredStrategy — הבדיקה)
```

**עזר (לא נדרש ל-H1):** `main.py`, `data.py`, `calibrate.py`, `fetch_and_export.py`

## מה נעול ואסור לשנות

`test_h1.py` שורות 20-24 — הקריטריונים מ-`H1_hypothesis.md`:

```python
LULL_START, LULL_END = time(11, 0), time(14, 0)
MIN_R_GAP = 0.15
MIN_TRADES_PER_DAY = 0.5
```

`StrategyConfig` ב-`config.py` — פרמטרי האסטרטגיה הנבדקת.

שינוי כלשהו בהם אחרי ראיית התוצאה פוסל את הבדיקה.

## אימות שהחבילה שלמה

```bash
python -c "import config, indicators, strategy, risk, backtest, screener, test_h1; print('OK')"
```

---

# הרצה בדמו

## התקנה נוספת

```bash
pip install ib_async
```

## IB Gateway (לא TWS)

TWS סוגר את עצמו פעם ביום ודורש התחברות מחדש. Gateway יציב יותר.

Configure → Settings → API:
- ✅ Enable ActiveX and Socket Clients
- ✅ Read-Only API — **כבוי**
- Socket port: **4002** (דמו) / 4001 (לייב)
- Trusted IP: 127.0.0.1

## הרצה

```bash
# יום ראשון — בלי לשלוח הזמנות בכלל
python live_trader.py --symbols MYM M2K --dry-run

# אחרי שה-dry-run נקי
python live_trader.py --symbols MYM M2K
```

## אחרי הסגירה, כל יום

```bash
python reconcile.py --latest
```

זו הבדיקה היומית. **לא P&L.**

## מה נחשב תקין

| בדיקה | תקין |
|---|---|
| שגיאות / ניתוקים | 0 |
| אי-התאמות סיגנל | 0 |
| סיגנלים שהוחמצו | 0 |
| החלקה | ≤ 1.0 טיקים בממוצע |
| פוזיציות פתוחות בסוף | 0 |

כל סטייה — לתקן לפני שמסתכלים על רווח.

## אם ההחלקה גבוהה מהמודל

`reconcile.py` יתריע. אם הממוצע בפועל הוא 2 טיקים במקום 1,
כל הבקטסטים אופטימיים. יש לעדכן `slippage_ticks` ב-`config.py`
ולהריץ מחדש — **ולתעד את השינוי**, כי הוא משנה תוצאות קודמות.
