"""
test_h1.py — בדיקת השערת חלון הנזילות

מריץ את האסטרטגיה פעמיים על אותם נתונים:
  A. ללא סינון שעות (כמו קודם)
  B. עם סינון שעות השפל 11:00-14:00

ומשווה מול הקריטריונים שנקבעו מראש ב-H1_hypothesis.md.

⚠️ אל תשנה את הקריטריונים אחרי שראית את התוצאה.

שינויים ביחס לגרסה המקורית (הנדסה/מדידה בלבד — לא ספים, לא חלונות, לא מקור נתונים):
  1. סיווג עסקאות לפי זמן יצירת הסיגנל (signal_time) ולא זמן המילוי —
     המסמך מגדיר "סיגנלים שנוצרים" בשעות השפל.
  2. קידוד מפורש של חלונות הנזילות מהמסמך (09:45–11:00, 14:00–15:15).
  3. אכיפת כל ארבעת קריטריוני ההפרכה מהמסמך (כולל תוחלת מסוננת ≥ 0
     ו-≥ 0.5 עסקאות/יום למכשיר, שלא נאכפו בגרסה המקורית).
  4. snapshot נתונים: כל סימול יורד פעם אחת, נשמר לדיסק, ושתי הריצות
     (A/B) מקבלות את אותו DataFrame בדיוק.
  5. דיווח שגיאת תקן ורווח סמך 95% של פער ה-R (דיווח בלבד, לא קריטריון).

שימוש:
    python test_h1.py --symbols MYM M2K
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import Backtester
from config import RiskConfig, StrategyConfig
from screener import MICROS, download, rth_only, to_instrument
from strategy import MomentumPullbackStrategy

# ── הקריטריונים מ-H1_hypothesis.md — נעולים ──
LULL_START, LULL_END = time(11, 0), time(14, 0)
MIN_R_GAP = 0.15            # פער קטן מזה → H1 מופרכת
EXPECTED_R_GAP = 0.30       # התחזית
MIN_TRADES_PER_DAY = 0.5    # מתחת לזה אין מה להריץ

# ── קידוד מפורש של חלונות הנזילות כפי שנכתבו במסמך ──
# (אינו קריטריון חדש; רק מסיר עמימות מ-"~lull")
LIQUIDITY_WINDOWS = [
    (time(9, 45), time(11, 0)),
    (time(14, 0), time(15, 15)),
]

OUT_DIR = Path(__file__).parent / "results_h1"
DATA_DIR = OUT_DIR / "data"


def in_lull(ts) -> bool:
    return LULL_START <= ts.time() < LULL_END


def in_liquidity(ts) -> bool:
    t = ts.time()
    return any(a <= t < b for a, b in LIQUIDITY_WINDOWS)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gap_stats(act_r: pd.Series, lull_r: pd.Series) -> dict:
    """פער תוחלת R + שגיאת תקן (Welch) + רווח סמך 95%"""
    gap = act_r.mean() - lull_r.mean()
    se = np.sqrt(act_r.var(ddof=1) / len(act_r) + lull_r.var(ddof=1) / len(lull_r))
    return {
        "gap_r": round(float(gap), 4),
        "se_r": round(float(se), 4),
        "ci95_lo": round(float(gap - 1.96 * se), 4),
        "ci95_hi": round(float(gap + 1.96 * se), 4),
        "n_liquidity": int(len(act_r)),
        "n_lull": int(len(lull_r)),
    }


class LullFilteredStrategy(MomentumPullbackStrategy):
    """זהה לאסטרטגיה המקורית, אבל לא נכנסת בשעות השפל"""

    def generate_signal(self, row, ts):
        if in_lull(ts):
            return None
        return super().generate_signal(row, ts)


def run(symbol: str, df: pd.DataFrame, filtered: bool, risk_cfg: RiskConfig):
    inst = to_instrument(next(m for m in MICROS if m.symbol == symbol))
    cls = LullFilteredStrategy if filtered else MomentumPullbackStrategy
    bt = Backtester(inst, cls(StrategyConfig()), risk_cfg, risk_cfg.account_size)
    bt.run(df)
    return bt


def sanity_check(sym: str, df: pd.DataFrame) -> list[str]:
    issues = []
    if str(df.index.tz) != "America/New_York":
        issues.append(f"{sym}: timezone = {df.index.tz} (צריך America/New_York)")
    if not df.index.is_monotonic_increasing:
        issues.append(f"{sym}: אינדקס לא ממוין")
    if df.index.duplicated().any():
        issues.append(f"{sym}: timestamps כפולים")
    if df[["open", "high", "low", "close"]].isna().any().any():
        issues.append(f"{sym}: NaN במחירים")
    return issues


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["MYM", "M2K"])
    p.add_argument("--capital", type=float, default=5000.0)
    p.add_argument("--days", type=int, default=59)
    p.add_argument("--refresh", action="store_true",
                   help="כפה הורדה חדשה במקום snapshot שמור")
    args = p.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    # ── hash של המסמכים הנעולים (audit) ──
    locked_hashes = {}
    for f in ["H1_hypothesis.md", "config.py", "strategy.py"]:
        fp = Path(__file__).parent / f
        if fp.exists():
            locked_hashes[f] = sha256_file(fp)

    risk_cfg = RiskConfig(account_size=args.capital)

    print("╔" + "═" * 66 + "╗")
    print("║" + "  בדיקת H1 — חלון הנזילות התוך-יומי".ljust(66) + "║")
    print("╚" + "═" * 66 + "╝")
    print(f"  שעות שפל שנחסמות: {LULL_START:%H:%M}–{LULL_END:%H:%M} ET (לפי זמן הסיגנל)")
    print(f"  חלונות נזילות: 09:45–11:00, 14:00–15:15 ET")
    print(f"  קריטריון: פער ≥ {MIN_R_GAP}R, יציב בין מכשירים")
    print(f"  + תוחלת מסוננת ≥ 0, ו-≥ {MIN_TRADES_PER_DAY} עסקאות/יום למכשיר\n")

    run_meta = {
        "executed_at_utc": datetime.utcnow().isoformat(),
        "symbols": args.symbols,
        "capital": args.capital,
        "days_requested": args.days,
        "locked_criteria": {
            "LULL_START": "11:00", "LULL_END": "14:00",
            "MIN_R_GAP": MIN_R_GAP, "EXPECTED_R_GAP": EXPECTED_R_GAP,
            "MIN_TRADES_PER_DAY": MIN_TRADES_PER_DAY,
        },
        "locked_file_hashes": locked_hashes,
        "note": ("נתוני Yahoo continuous futures (YM=F/RTY=F), הורדה חד-פעמית "
                 "שנשמרה ל-snapshot. זה אינו ה-snapshot המקורי של 41 הימים."),
    }

    all_trades = []
    per_symbol = {}
    all_issues = []

    for sym in args.symbols:
        spec = next(m for m in MICROS if m.symbol == sym)

        # ── snapshot: אם קיים — משתמשים בו; אחרת מורידים פעם אחת ושומרים ──
        snap = DATA_DIR / f"{sym}_5m_snapshot.parquet"
        if snap.exists() and not args.refresh:
            df = pd.read_parquet(snap)
            print(f"  [snapshot] {sym}: נטען מקובץ שמור ({snap.name})")
        else:
            df = download(spec, "5m", args.days)
            if df is None:
                print(f"  ✗ {sym}: אין נתונים")
                per_symbol[sym] = {"error": "no_data"}
                continue
            df = rth_only(df)
            df.to_parquet(snap)
            df = pd.read_parquet(snap)  # שתי הריצות על הקובץ השמור

        issues = sanity_check(sym, df)
        all_issues += issues
        if issues:
            print(f"  ⚠ {sym}: בעיות sanity: {issues}")

        days = df.index.normalize().nunique()
        print("═" * 68)
        print(f"  {sym} | {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d} "
              f"| {days} ימי מסחר | {len(df):,} נרות 5m (RTH)")

        base = run(sym, df, False, risk_cfg)
        d = base.results()
        if d.empty:
            print(f"  ✗ {sym}: 0 עסקאות")
            per_symbol[sym] = {"error": "zero_trades"}
            continue

        # סיווג לפי זמן יצירת הסיגנל (fallback: זמן מילוי, עם סימון)
        if d["signal_time"].isna().any():
            all_issues.append(f"{sym}: חלק מהעסקאות ללא signal_time")
            d["sig_ts"] = d["signal_time"].fillna(d["entry_time"])
        else:
            d["sig_ts"] = d["signal_time"]

        d["lull"] = d.sig_ts.apply(in_lull)
        d["liquidity"] = d.sig_ts.apply(in_liquidity)
        d["symbol"] = sym
        all_trades.append(d)
        d.to_csv(OUT_DIR / f"trades_base_{sym}.csv", index=False)

        lull = d[d.lull]
        act = d[d.liquidity]
        other = d[~d.lull & ~d.liquidity]
        if len(other):
            all_issues.append(f"{sym}: {len(other)} עסקאות מחוץ לשני הסלים")

        print(f"  {'':16}{'עסקאות':>9}{'תוחלת R':>11}{'הצלחה':>9}{'נטו':>11}")
        print("  " + "-" * 60)
        for lbl, s in [("חלון נזילות", act), ("שעות שפל", lull)]:
            if len(s) == 0:
                print(f"  {lbl:16}{0:>9}")
                continue
            print(f"  {lbl:16}{len(s):>9}{s.r_multiple.mean():>11.3f}"
                  f"{(s.net_pnl>0).mean()*100:>8.1f}%"
                  f"{'$'+format(s.net_pnl.sum(),',.0f'):>11}")

        sym_rec = {"days": int(days), "n_base": int(len(d)),
                   "n_lull": int(len(lull)), "n_liquidity": int(len(act))}

        if len(lull) and len(act):
            g = gap_stats(act.r_multiple, lull.r_multiple)
            sym_rec["gap"] = g
            print(f"\n  פער: {g['gap_r']:+.3f}R | SE {g['se_r']:.3f}R "
                  f"| CI95 [{g['ci95_lo']:+.3f}, {g['ci95_hi']:+.3f}]", end="  ")
            print("✓ עובר סף" if g["gap_r"] >= MIN_R_GAP else "✗ מתחת לסף")
            print(f"  (התחזית הנעולה: {EXPECTED_R_GAP}R; הסף הנעול: {MIN_R_GAP}R)")
        else:
            sym_rec["gap"] = None
            print("\n  ⚠ אין עסקאות באחד הסלים — הפער לא ניתן לחישוב")

        # יציאות VWAP — תחזית 4 (תיאורית בלבד)
        vw = d[d.exit_reason.str.contains("VWAP", na=False)]
        if len(vw):
            sym_rec["vwap_exits_total"] = int(len(vw))
            sym_rec["vwap_exits_in_lull_pct"] = round(float(vw.lull.mean() * 100), 1)
            print(f"  יציאות VWAP: {len(vw)} סה\"כ, "
                  f"{int(vw.lull.sum())} מהן בשעות השפל ({vw.lull.mean()*100:.0f}%)")

        # ── ריצה מסוננת (B) על אותו snapshot ──
        filt = run(sym, df, True, risk_cfg)
        fd = filt.results()
        fd.to_csv(OUT_DIR / f"trades_filtered_{sym}.csv", index=False)
        n_filt = len(fd)
        per_day = n_filt / days if days else 0.0
        filt_mean_r = float(fd.r_multiple.mean()) if n_filt else float("nan")
        sym_rec["n_filtered"] = n_filt
        sym_rec["filtered_mean_r"] = round(filt_mean_r, 4) if n_filt else None
        sym_rec["filtered_trades_per_day"] = round(per_day, 3)

        print(f"\n  אחרי סינון: {n_filt} עסקאות "
              f"(היה {len(d)}, ירידה של {(1-n_filt/len(d))*100:.0f}%)")
        if n_filt:
            print(f"    תוחלת R: {filt_mean_r:+.3f} | "
                  f"נטו: ${fd.net_pnl.sum():,.0f} | "
                  f"{per_day:.2f} עסקאות ליום")
        per_symbol[sym] = sym_rec
        print()

    # ── פסק דין מול כל ארבעת הקריטריונים שנקבעו מראש ──
    print("═" * 68)
    print("  הכרעה מול הקריטריונים שנקבעו מראש (H1_hypothesis.md)")
    print("═" * 68)

    verdict = {"criteria": {}, "refuted": None, "reason": ""}

    if not all_trades:
        print("  ✗ אין נתונים להכרעה.")
        verdict["refuted"] = "INCONCLUSIVE"
        verdict["reason"] = "no_trades"
    else:
        gaps = {}
        for sym, g in pd.concat(all_trades).groupby("symbol"):
            l, ac = g[g.lull], g[g.liquidity]
            if len(l) and len(ac):
                gaps[sym] = ac.r_multiple.mean() - l.r_multiple.mean()

        if not gaps:
            print("  ✗ אין מספיק עסקאות בשתי הקטגוריות — לא ניתן להכריע.")
            verdict["refuted"] = "INCONCLUSIVE"
            verdict["reason"] = "insufficient_trades_in_buckets"
        else:
            for s, v in gaps.items():
                print(f"    {s}: {v:+.3f}R")

            # קריטריון 1: פער ≥ 0.15R בכל המכשירים
            big = bool(all(v >= MIN_R_GAP for v in gaps.values()))
            # קריטריון 2: יציבות סימן בין מכשירים
            stable = bool(len(gaps) > 1 and len(set(np.sign(list(gaps.values())))) == 1)
            # קריטריון 3: תוחלת כוללת אחרי סינון לא נשארת שלילית
            pooled_filt = [rec.get("filtered_mean_r") for rec in per_symbol.values()
                           if isinstance(rec, dict) and rec.get("filtered_mean_r") is not None]
            c3_ok = bool(pooled_filt) and bool(np.mean(pooled_filt) >= 0)
            # קריטריון 4: ≥ 0.5 עסקאות/יום למכשיר אחרי סינון
            tpd = {s: rec.get("filtered_trades_per_day", 0.0)
                   for s, rec in per_symbol.items() if isinstance(rec, dict) and "n_filtered" in rec}
            c4_ok = bool(tpd) and bool(all(v >= MIN_TRADES_PER_DAY for v in tpd.values()))

            verdict["criteria"] = {
                "1_gap_ge_0.15R_all_instruments": bool(big),
                "2_sign_stable_across_instruments": bool(stable),
                "3_filtered_expectancy_not_negative": c3_ok,
                "4_filtered_trades_per_day_ge_0.5": c4_ok,
                "gaps": {k: round(float(v), 4) for k, v in gaps.items()},
                "filtered_trades_per_day": tpd,
                "pooled_filtered_mean_r": round(float(np.mean(pooled_filt)), 4) if pooled_filt else None,
            }

            print()
            print(f"  1. פער ≥ {MIN_R_GAP}R בכל המכשירים:        {'✓' if big else '✗'}")
            print(f"  2. יציב בסימן בין מכשירים:               {'✓' if stable else '✗'}")
            print(f"  3. תוחלת מסוננת לא שלילית:               {'✓' if c3_ok else '✗'}")
            print(f"  4. ≥ {MIN_TRADES_PER_DAY} עסקאות/יום למכשיר:       {'✓' if c4_ok else '✗'}")
            print()

            if big and stable and c3_ok and c4_ok:
                verdict["refuted"] = False
                print("  → H1 לא הופרכה. מצדיק המשך בדיקה על נתונים ארוכים יותר.")
                print("     זה *אינו* אישור ליתרון — המדגם חסר הספק לכך.")
            else:
                verdict["refuted"] = True
                failed = [k for k, v in verdict["criteria"].items()
                          if k[0].isdigit() and v is False]
                verdict["reason"] = ",".join(failed)
                print("  → H1 מופרכת. אין להמשיך ל-H2 על אותם נתונים.")

    if all_issues:
        print("\n  ⚠ הערות sanity שנתגלו במהלך ההרצה:")
        for i in all_issues:
            print(f"    • {i}")

    print("═" * 68)

    # ── שמירת סיכום ──
    summary = {**run_meta, "per_symbol": per_symbol, "verdict": verdict,
               "sanity_issues": all_issues}
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"\n  פלט נשמר: {OUT_DIR}/ (summary.json, trades_*.csv, data/)")


if __name__ == "__main__":
    main()