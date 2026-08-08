# SmartPassive Forward Paper Test — Operations Guide

ניסוי forward לשיטת SmartPassive 55/35/10 + MA200, שננעלה אחרי 13 מבחני מחקר (ראה `RESEARCH_REPORT.md`). **אין לשנות פרמטרים במהלך הניסוי.** כל שינוי = ניסוי חדש (`experiment_id` חדש).

**Preregistration:** `SMARTPASSIVE_hypothesis.md` (נכתב 2026-08-08, לפני החלפת משטר ראשונה).

**Benchmark רשמי:** תיק סטטי 55/35/10 (SPY/IEF/GLD) ללא פילטר MA200, עם אותן עלויות, tolerance ותזמון. SPY benchmark נשמר כתיאור בלבד.

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

## מודל מס (preregistered)
- 25% על רווח הון ממומש לפי FIFO, בדולר.
- הפסדים ממומשים נצברים ומקוזזים מול רווחים עתידיים (loss carry-forward).
- מתעלמים ממס דיבידנד, מט"ח ומס יסף 3%.
- בסוף הניסוי: liquidation רעיוני + מס על רווח לא ממומש.
- ההשוואה: `after_tax_terminal_wealth`, לא NAV לפני מס.

## קריטריוני הפרכה (preregistered)
| # | קריטריון | הגדרה |
|---|---|---|
| SP-C1 | after-tax terminal wealth | SmartPassive > Benchmark |
| SP-C2 | MAR after-tax | SmartPassive > Benchmark |

**MAR = after-tax CAGR ÷ |Max Drawdown|** (drawdown על after-tax liquidation NAV יומי).

### פסק דין
| תוצאה | תנאי |
|---|---|
| **מופרכת** | C1 ו-C2 נכשלים |
| **לא הופרכה, חלקית** | רק אחד עובר |
| **לא הופרכה** | שניהם עוברים |
| **לא ניתן להכריע** | תנאי האופק/המחזורים לא התקיימו |

## אופק הניסוי (preregistered)
פסק דין רק לאחר **המאוחר** מבין:
1. **3 שנים מלאות** מ-4.8.2026; ו-
2. **שני מחזורי RISK_OFF מלאים**.

**Hard stop:** אם עד 4.8.2034 לא הושלמו שני מחזורים → **לא ניתן להכריע**.

## סיום הניסוי
השווה: after-tax terminal wealth ו-MAR מול benchmark סטטי 55/35/10. החלטה: להמשיך כ-money-neutral paper, לעבור לכסף אמיתי רק אם הניסוי עקבי עם ה-backtest, או לסגור. כל החלטה מתועדת ב-commit.
