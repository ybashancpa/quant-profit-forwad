# SmartPassive Forward Paper Test — Operations Guide

ניסוי forward של חצי שנה+ לשיטת SmartPassive 55/35/10 + MA200, שננעלה אחרי 13 מבחני מחקר (ראה `RESEARCH_REPORT.md`). **אין לשנות פרמטרים במהלך הניסוי.** כל שינוי = ניסוי חדש (`experiment_id` חדש).

## כללי המסחר (נעולים, config hash בתוך state.json)
| משטר | תנאי | הקצאה |
|---|---|---|
| RISK_ON | SPY > MA200 | 55% SPY, 35% IEF, 10% GLD |
| RISK_OFF | SPY ≤ MA200 | 55% SHY, 35% IEF, 10% GLD |

- מסחר רק ביום המסחר האחרון של החודש + הקצאה ראשונית חד-פעמית.
- tolerance band 3%; עלות paper של 10bps/side; מניות חלקיות; ללא מינוף/שורט.
- הון התחלתי $10,000. תאריך התחלה: 2026-08-04 (RISK_ON).

## הרצה מקומית
```bash
pip install -r requirements.txt
python test_forward_engine.py   # unit tests (ללא רשת)
python forward_test.py          # הרצה חיה מול סגירה אחרונה
python forward_test.py --date 2026-07-31   # סימולציה as-of (לבדיקות בלבד)
```

## קבצים
| קובץ | תפקיד |
|---|---|
| `forward/state.json` | מצב התיק (מקור אמת) |
| `forward/nav_history.csv` | NAV יומי + benchmark SPY (append-only) |
| `forward/trades.csv` | יומן עסקאות |
| `forward/signals.csv` | signal יומי (SPY, MA200, regime) |
| `forward/runs.csv` | יומן הרצות (סטטוס, שגיאות) |
| `forward/report.html` | דוח אנושי עם עקומת NAV |
| `forward/summary.json` | סיכום מכונה להרצה האחרונה |

## אוטומציה (GitHub Actions)
`.github/workflows/forward-test.yml` רץ Mon–Fri ב-22:30 UTC (אחרי סגירת ארה"ב):
1. בדיקות unit; 2. הרצת forward; 3. commit אוטומטי של `forward/`; 4. artifact של הדוח.

### חיבור ראשוני (חד-פעמי)
```bash
git init
git add -A
git commit -m "forward paper test: engine, tests, workflow, initial state"
# צור repo פרטי חדש בגיטהאב, ואז:
git remote add origin https://github.com/<OWNER>/<REPO>.git
git branch -M main
git push -u origin main
```
ודא ב-GitHub: Settings → Actions → General → **Allow all actions**; הרשאות write של ה-workflow כבר מוגדרות בקובץ. אם ה-repo פרטי, בדוק שיש דקות Actions פנויות (הרצה יומית ~2-3 דקות).

### התראות (מומלץ)
- GitHub כבר שולח מייל על כשלון workflow (Settings → Notifications → Actions).
- אופציונלי: הוסף step עם Slack/Telegram webhook ששולח את `forward/summary.json` רק כאשר `status` הוא `REBALANCED`/`INITIAL_ALLOCATION`/`ERROR`.

## ניטור ותחזוקה
- **אין לגעת** ב-`forward/state.json` ידנית. אם צריך לתקן — גבה קודם ותעד ב-commit.
- אם workflow נכשל: בדוק `forward/runs.csv` ו-artifacts. שגיאת נתונים לא מבצעת מסחר — זה בטיחות, לא באג.
- GitHub משבית scheduled workflows אחרי ~60 יום ללא פעילות ב-repo; commit אוטומטי יומי שומר על פעילות, אבל כדאי לבדוק מדי פעם את טאב Actions.
- בדיקה חודשית: פתח `forward/report.html` — זה כל מה שצריך.

## סיום הניסוי (אחרי 6+ חודשים)
השווה: NAV מול SPY benchmark ו-metrics (CAGR, Sharpe, MaxDD מתוך nav_history.csv). החלטה: להמשיך כ-money-neutral paper, לעבור לכסף אמיתי רק אם הניסוי עקבי עם ה-backtest, או לסגור. כל החלטה מתועדת ב-commit.