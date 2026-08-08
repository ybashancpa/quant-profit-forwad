"""
test_h2.py — בדיקת השערה H2 (חלון הסגירה / Market Intraday Momentum)

מבצעת את המבחנים לפי H2_hypothesis.md + H2_appendix_A.md:
  - מבחן A: חישוב נטו שנתי מול תשואה חסרת סיכון (FC1)
  - מבחן B: מתאם בין r_ROD ל-r_LH (FC2)
  - FC3: PENDING (החלקה — נמדד ב-Paper)
  - FC4: PENDING (מרג'ין — נמדד מול TWS)
  - בקטסט תיאורי (אינו קריטריון)

⚠️ אל תשנה את הקריטריונים אחרי שראית את התוצאה.

שימוש:
    python test_h2.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

from h2_config import (
    BASE_SLIPPAGE_TICKS, CRITERIA, DATA_DIR, ET, LOCKED_ACCOUNT,
    LOCKED_COMMISSION_RT, LOCKED_CONTRACTS, LOCKED_EFFECT_SIZE,
    LOCKED_ENTRY_TIME, LOCKED_EXIT_TIME, LOCKED_FILES,
    LOCKED_MARGIN_ESTIMATE, LOCKED_MARGIN_THRESHOLD,
    LOCKED_MULTIPLIER, LOCKED_RISK_FREE_ANNUAL, LOCKED_STOP_PCT,
    LOCKED_SYMBOL, LOCKED_TICK_VALUE, OUT_DIR, SLIPPAGE_SCENARIOS,
    SNAPSHOT_PATH, VERDICT_INCONCLUSIVE, VERDICT_NOT_REFUTED_PARTIAL,
    VERDICT_REFUTED, assert_locked,
)

# ══════════════════════════════════════════════════════════════
# עזרים
# ══════════════════════════════════════════════════════════════

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pearson_with_pvalue(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Pearson correlation + p-value דו-צדדי.
    מימוש numpy בלבד (ללא scipy) לפי הנחיית המשתמש:
      t = r * sqrt((n-2)/(1-r^2))
      p = 2 * (1 - CDF_t(|t|, n-2))
    כאן n≈41, df=39 — קירוב נורמלי מספיק (t_39 ≈ z).
    """
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    if abs(r) >= 1.0:
        return r, 0.0
    t_stat = r * math.sqrt((n - 2) / (1 - r * r))
    # קירוב נורמלי ל-p (df=39, שגיאה זניחה)
    from math import erfc, sqrt
    p = erfc(abs(t_stat) / sqrt(2))  # = 2*(1 - Phi(|t|))
    return r, float(p)


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """רווח סמך 95% למתאם Pearson via Fisher-z."""
    if n < 4 or abs(r) >= 1.0:
        return (float("nan"), float("nan"))
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    z_crit = 1.96  # 95%
    lo = math.tanh(z - z_crit * se)
    hi = math.tanh(z + z_crit * se)
    return (lo, hi)


# ══════════════════════════════════════════════════════════════
# שלב 2 — בניית טבלת ימים
# ══════════════════════════════════════════════════════════════

def build_daily_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    בונה טבלת ימים: שורה אחת לכל יום מסחר.
    מחזיר (טבלה, dict של ספירות הסרה).
    """
    removals = {
        "half_day_sessions": 0,
        "day_after_half_day": 0,
        "missing_1530": 0,
        "missing_1555": 0,
        "missing_prev_close": 0,
        "zero_r_ROD": 0,
    }

    days = sorted(df.index.normalize().unique())
    rows = []
    prev_close = None
    prev_was_half_day = False

    for day in days:
        day_df = df[df.index.normalize() == day]

        # ── זיהוי חצי-יום: סגירה מוקדמת (אין נר 15:30) ──
        has_1530 = any(day_df.index.time == LOCKED_ENTRY_TIME)
        # נר 15:55 = הנר שמתחיל ב-15:55 (מסתיים ב-16:00)
        has_1555 = any(day_df.index.time == LOCKED_EXIT_TIME)

        if not has_1530 or not has_1555:
            # חצי-יום או יום חסר
            if prev_close is not None:
                removals["half_day_sessions"] += 1
            prev_was_half_day = True
            prev_close = None  # מאפסים — אין סגירה תקינה
            continue

        # ── יום אחרי חצי-יום: מוסר (נספח א' סעיף ה') ──
        if prev_was_half_day:
            removals["day_after_half_day"] += 1
            prev_was_half_day = False
            # מעדכנים prev_close מהיום הנוכחי להמשך
            bar_1555 = day_df[day_df.index.time == LOCKED_EXIT_TIME]
            prev_close = float(bar_1555["close"].iloc[0])
            continue

        prev_was_half_day = False

        # ── בדיקת prev_close ──
        if prev_close is None:
            removals["missing_prev_close"] += 1
            bar_1555 = day_df[day_df.index.time == LOCKED_EXIT_TIME]
            prev_close = float(bar_1555["close"].iloc[0])
            continue

        # ── חילוץ מחירים ──
        bar_1530 = day_df[day_df.index.time == LOCKED_ENTRY_TIME]
        bar_1555 = day_df[day_df.index.time == LOCKED_EXIT_TIME]

        if bar_1530.empty:
            removals["missing_1530"] += 1
            prev_close = float(bar_1555["close"].iloc[0]) if not bar_1555.empty else prev_close
            continue
        if bar_1555.empty:
            removals["missing_1555"] += 1
            prev_close = prev_close  # לא מעדכנים
            continue

        px_1530 = float(bar_1530["open"].iloc[0])   # פתיחת נר 15:30
        px_close = float(bar_1555["close"].iloc[0])  # סגירת נר 15:55

        r_ROD = px_1530 / prev_close - 1.0
        r_LH = px_close / px_1530 - 1.0

        # ── כיוון ──
        if r_ROD == 0.0:
            direction = 0
            removals["zero_r_ROD"] += 1
        else:
            direction = 1 if r_ROD > 0 else -1

        # ── תנועה נגדית מקסימלית (adverse) ──
        # חלון 15:30–16:00: כל הנרים מ-15:30 עד 15:55 (כולל)
        window = day_df[(day_df.index.time >= LOCKED_ENTRY_TIME) &
                        (day_df.index.time <= LOCKED_EXIT_TIME)]
        if direction == 1:  # Long — adverse = ירידה
            adverse = (px_1530 - window["low"].min()) / px_1530
        elif direction == -1:  # Short — adverse = עלייה
            adverse = (window["high"].max() - px_1530) / px_1530
        else:
            adverse = 0.0

        rows.append({
            "date": day.date(),
            "prev_close": prev_close,
            "px_1530": px_1530,
            "px_close": px_close,
            "r_ROD": r_ROD,
            "r_LH": r_LH,
            "direction": direction,
            "adverse": adverse,
        })

        # עדכון prev_close ליום הבא
        prev_close = px_close

    table = pd.DataFrame(rows)
    return table, removals


# ══════════════════════════════════════════════════════════════
# שלב 3 — מבחן A
# ══════════════════════════════════════════════════════════════

def test_a(table: pd.DataFrame) -> dict:
    """
    מבחן A: חישוב נטו שנתי.
    נספח א' סעיף א': sigma_dollars = std(r_LH × px_1530 × multiplier)
    """
    r_LH = table["r_LH"].values
    px_1530 = table["px_1530"].values

    # גישה ראשית: sigma_dollars ישירות
    pnl_per_contract = r_LH * px_1530 * LOCKED_MULTIPLIER
    sigma_dollars = float(np.std(pnl_per_contract, ddof=1))

    # גישה חלופית (לשקיפות): σ_LH × mean(px_1530) × multiplier
    sigma_r_LH = float(np.std(r_LH, ddof=1))
    mean_px = float(np.mean(px_1530))
    sigma_dollars_alt = sigma_r_LH * mean_px * LOCKED_MULTIPLIER

    expected = LOCKED_EFFECT_SIZE * sigma_dollars

    results = {
        "n_days": len(table),
        "sigma_dollars": round(sigma_dollars, 4),
        "sigma_dollars_alt": round(sigma_dollars_alt, 4),
        "sigma_r_LH": round(sigma_r_LH, 6),
        "mean_px_1530": round(mean_px, 2),
        "expected_gross_per_trade": round(expected, 4),
        "scenarios": {},
    }

    for slip_ticks in SLIPPAGE_SCENARIOS:
        slip_cost = slip_ticks * LOCKED_TICK_VALUE  # הלוך-חזור
        net = expected - LOCKED_COMMISSION_RT - slip_cost
        annual = net * 250
        pct_account = annual / LOCKED_ACCOUNT * 100
        results["scenarios"][f"slip_{slip_ticks}"] = {
            "slippage_ticks_rt": slip_ticks,
            "slippage_cost": round(slip_cost, 2),
            "net_per_trade": round(net, 4),
            "annual_net": round(annual, 2),
            "pct_account": round(pct_account, 2),
        }

    # break_even_ticks (נספח א' סעיף ג')
    rf_per_trade = LOCKED_RISK_FREE_ANNUAL * LOCKED_ACCOUNT / 250  # $0.70
    if expected - LOCKED_COMMISSION_RT - rf_per_trade > 0:
        be_ticks = (expected - LOCKED_COMMISSION_RT - rf_per_trade) / LOCKED_TICK_VALUE
    else:
        be_ticks = 0.0  # כבר מתחת לסף
    results["break_even_ticks"] = round(be_ticks, 2)
    results["rf_per_trade"] = round(rf_per_trade, 4)

    return results


# ══════════════════════════════════════════════════════════════
# שלב 4 — מבחן B
# ══════════════════════════════════════════════════════════════

def test_b(table: pd.DataFrame) -> dict:
    """מתאם Pearson בין r_ROD ל-r_LH."""
    valid = table[table["direction"] != 0]
    x = valid["r_ROD"].values
    y = valid["r_LH"].values
    n = len(x)

    r, p = pearson_with_pvalue(x, y)
    ci_lo, ci_hi = fisher_ci(r, n)

    return {
        "n": n,
        "corr": round(r, 6),
        "p_value": round(p, 6),
        "ci95_lo": round(ci_lo, 6),
        "ci95_hi": round(ci_hi, 6),
        "negative_significant": bool(r < 0 and p < 0.10),
    }


# ══════════════════════════════════════════════════════════════
# שלב 5 — בקטסט תיאורי
# ══════════════════════════════════════════════════════════════

def descriptive_backtest(table: pd.DataFrame) -> dict:
    """
    הרצת הכלל: כניסה ב-15:30, יציאה ב-16:00, חוזה אחד.
    ⚠️ תיאורי בלבד — אינו קריטריון.
    """
    trades = table[table["direction"] != 0].copy()
    if trades.empty:
        return {"n_trades": 0}

    # P&L גולמי לחוזה
    trades["gross_pnl"] = (trades["r_LH"] * trades["px_1530"]
                           * LOCKED_MULTIPLIER * trades["direction"])
    # עלויות: עמלה + החלקה בסיס (2 טיקים)
    slip_cost = BASE_SLIPPAGE_TICKS * LOCKED_TICK_VALUE
    trades["net_pnl"] = trades["gross_pnl"] - LOCKED_COMMISSION_RT - slip_cost

    n = len(trades)
    wins = int((trades["net_pnl"] > 0).sum())
    net_total = float(trades["net_pnl"].sum())
    mean_net = float(trades["net_pnl"].mean())

    # סטופ קטסטרופה: בדיקה כמה ימים חרגו
    stop_threshold = LOCKED_STOP_PCT
    stop_hits = int((trades["adverse"] >= stop_threshold).sum())

    return {
        "n_trades": n,
        "win_rate": round(wins / n * 100, 1),
        "net_total": round(net_total, 2),
        "mean_net_per_trade": round(mean_net, 4),
        "stop_hits_1pct": stop_hits,
        "note": "תיאורי בלבד — n קטן, אינו קריטריון",
    }


# ══════════════════════════════════════════════════════════════
# שלב 6 — פסק דין
# ══════════════════════════════════════════════════════════════

def evaluate_criteria(ta: dict, tb: dict) -> dict:
    """
    מעריך את ארבעת הקריטריונים.
    מחזיר dict עם סטטוס לכל FC + פסק דין סופי.
    """
    results = {}

    # ── FC1: נטו שנתי >= $175 (3.5% × $5,000) בתרחיש בסיס (2 טיקים) ──
    base_scenario = ta["scenarios"][f"slip_{BASE_SLIPPAGE_TICKS}"]
    fc1_annual = base_scenario["annual_net"]
    fc1_threshold = CRITERIA["FC1"]["threshold"]
    fc1_pass = fc1_annual >= fc1_threshold
    results["FC1"] = {
        "status": "PASS" if fc1_pass else "FAIL",
        "value": fc1_annual,
        "threshold": fc1_threshold,
        "detail": f"נטו שנתי ${fc1_annual:.0f} מול סף ${fc1_threshold:.0f}",
    }

    # ── FC2: מתאם שלילי מובהק → FAIL ──
    fc2_fail = tb["negative_significant"]
    results["FC2"] = {
        "status": "FAIL" if fc2_fail else "PASS",
        "value": tb["corr"],
        "p_value": tb["p_value"],
        "detail": (f"corr={tb['corr']:.4f}, p={tb['p_value']:.4f}"
                   + (" → שלילי מובהק" if fc2_fail else " → לא שלילי מובהק")),
    }

    # ── FC3: PENDING ──
    results["FC3"] = {
        "status": "PENDING",
        "value": None,
        "break_even_ticks": ta["break_even_ticks"],
        "detail": (f"נדרש Paper. break-even = {ta['break_even_ticks']:.1f} טיקים "
                   f"הלוך-חזור (מעל 2 טיקים → FC1 בסכנה)"),
    }

    # ── FC4: PENDING ──
    results["FC4"] = {
        "status": "PENDING",
        "value": None,
        "estimate": LOCKED_MARGIN_ESTIMATE,
        "threshold": LOCKED_MARGIN_THRESHOLD,
        "detail": (f"אומדן ${LOCKED_MARGIN_ESTIMATE:.0f} מול סף "
                   f"${LOCKED_MARGIN_THRESHOLD:.0f} — לאמת מול TWS"),
    }

    # ── פסק דין ──
    if results["FC1"]["status"] == "FAIL" or results["FC2"]["status"] == "FAIL":
        verdict = VERDICT_REFUTED
    elif results["FC1"]["status"] == "PASS" and results["FC2"]["status"] == "PASS":
        verdict = VERDICT_NOT_REFUTED_PARTIAL
    else:
        verdict = VERDICT_INCONCLUSIVE

    return {"criteria": results, "verdict": verdict}


# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════

def main():
    # ── שלב 0: assertions + audit ──
    assert_locked()

    # hash-ים של קבצים נעולים
    locked_hashes = {}
    for f in LOCKED_FILES:
        fp = Path(__file__).parent / f
        if fp.exists():
            locked_hashes[f] = sha256_file(fp)

    # hash של snapshot (אם קיים)
    snapshot_hash = None
    if SNAPSHOT_PATH.exists():
        snapshot_hash = sha256_file(SNAPSHOT_PATH)

    print("╔" + "═" * 66 + "╗")
    print("║" + "  בדיקת H2 — חלון הסגירה (Market Intraday Momentum)".ljust(66) + "║")
    print("╚" + "═" + "═" * 66 + "╝")
    print(f"  מכשיר: {LOCKED_SYMBOL} | כניסה: {LOCKED_ENTRY_TIME:%H:%M} | "
          f"יציאה: {LOCKED_EXIT_TIME:%H:%M} | חוזים: {LOCKED_CONTRACTS}")
    print(f"  effect size: {LOCKED_EFFECT_SIZE} | עמלה: ${LOCKED_COMMISSION_RT}")
    print(f"  קריטריונים: FC1 (נטו שנתי), FC2 (מתאם), FC3 (PENDING), FC4 (PENDING)")
    print()

    # ── שלב 1: טעינת snapshot ──
    if not SNAPSHOT_PATH.exists():
        print("  ✗ אין snapshot. יש להריץ קודם: python fetch_h2.py")
        sys.exit(1)

    df = pd.read_parquet(SNAPSHOT_PATH)
    print(f"  [snapshot] {len(df):,} נרות | "
          f"{df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}")

    # sanity: timezone
    if str(df.index.tz) != "America/New_York":
        print(f"  ⚠ timezone = {df.index.tz} (צריך America/New_York)")

    # ── שלב 2: טבלת ימים ──
    table, removals = build_daily_table(df)
    n_valid = len(table)
    print(f"\n  טבלת ימים: {n_valid} ימים תקפים")
    for k, v in removals.items():
        if v > 0:
            print(f"    הוסרו — {k}: {v}")

    if n_valid < 5:
        print("  ✗ פחות מ-5 ימים תקפים — לא ניתן להכריע")
        verdict_out = {"verdict": VERDICT_INCONCLUSIVE, "reason": "insufficient_days"}
        save_summary(locked_hashes, snapshot_hash, table, removals,
                     None, None, None, verdict_out)
        sys.exit(1)

    # ── שלב 3: מבחן A ──
    ta = test_a(table)
    print(f"\n  מבחן A:")
    print(f"    σ_dollars = ${ta['sigma_dollars']:.2f} "
          f"(חלופי: ${ta['sigma_dollars_alt']:.2f})")
    print(f"    תוחלת גולמית = ${ta['expected_gross_per_trade']:.2f}/עסקה")
    for key, sc in ta["scenarios"].items():
        marker = " ← בסיס" if sc["slippage_ticks_rt"] == BASE_SLIPPAGE_TICKS else ""
        print(f"    {key}: נטו ${sc['net_per_trade']:.2f}/עסקה | "
              f"שנתי ${sc['annual_net']:.0f} = {sc['pct_account']:.1f}%{marker}")
    print(f"    break-even: {ta['break_even_ticks']:.1f} טיקים הלוך-חזור")

    # ── שלב 4: מבחן B ──
    tb = test_b(table)
    print(f"\n  מבחן B:")
    print(f"    corr(r_ROD, r_LH) = {tb['corr']:.4f} | p = {tb['p_value']:.4f}")
    print(f"    CI95: [{tb['ci95_lo']:.4f}, {tb['ci95_hi']:.4f}] | n = {tb['n']}")
    if tb["negative_significant"]:
        print("    ⚠ מתאם שלילי מובהק — ראיה נגד H2")
    else:
        print("    מתאם לא שלילי מובהק — היעדר סתירה (אינו אישור)")

    # ── שלב 5: בקטסט תיאורי ──
    bt = descriptive_backtest(table)
    print(f"\n  בקטסט תיאורי (אינו קריטריון):")
    if bt.get("n_trades", 0) > 0:
        print(f"    {bt['n_trades']} עסקאות | הצלחה {bt['win_rate']:.0f}% | "
              f"נטו ${bt['net_total']:.0f} | תוחלת ${bt['mean_net_per_trade']:.2f}")
        print(f"    סטופ 1% הופעל: {bt['stop_hits_1pct']} ימים")

    # ── שלב 6: פסק דין ──
    verdict_out = evaluate_criteria(ta, tb)

    print(f"\n{'═' * 68}")
    print("  פסק דין מול קריטריוני ההפרכה")
    print("═" * 68)
    for fc_id in ["FC1", "FC2", "FC3", "FC4"]:
        fc = verdict_out["criteria"][fc_id]
        icon = {"PASS": "✓", "FAIL": "✗", "PENDING": "⏳"}[fc["status"]]
        print(f"  {icon} {fc_id}: {fc['detail']}")
    print()
    print(f"  → {verdict_out['verdict']}")
    if verdict_out["verdict"] == VERDICT_NOT_REFUTED_PARTIAL:
        print("    הצעד הבא: מדידת החלקה ב-Paper (FC3) + אימות מרג'ין (FC4)")
        print("    ⚠ אין לרכוש נתוני IBKR עד השלמת FC3/FC4")
    elif verdict_out["verdict"] == VERDICT_REFUTED:
        print("    H2 מופרכת. אין להמשיך.")
    print("═" * 68)

    # ── שמירת פלטים ──
    save_summary(locked_hashes, snapshot_hash, table, removals,
                 ta, tb, bt, verdict_out)
    save_outputs(table, ta)

    print(f"\n  פלט נשמר: {OUT_DIR}/")


def save_summary(hashes, snap_hash, table, removals, ta, tb, bt, verdict_out):
    """שומר summary.json."""
    OUT_DIR.mkdir(exist_ok=True)
    summary = {
        "executed_at_utc": datetime.now(tz=__import__('datetime').timezone.utc).isoformat(),
        "hypothesis": "H2",
        "symbol": LOCKED_SYMBOL,
        "locked_file_hashes": hashes,
        "snapshot_hash": snap_hash,
        "n_valid_days": len(table),
        "removals": removals,
        "test_a": ta,
        "test_b": tb,
        "descriptive_backtest": bt,
        "verdict": verdict_out,
        "protocol_notes": [
            "שני הנספחים (H2_appendix_A.md, H2_protocol_overlap_ruling.md) "
            "נשמרו לפני טעינת הנתונים — hash-ים כלולים לעיל.",
            "FC3 ו-FC4 הם PENDING בהגדרה — פסק הדין חלקי.",
            "הבדיקה אינה יכולה לאשר, רק לשלול (n קטן, הספק אפסי).",
        ],
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")


def save_outputs(table: pd.DataFrame, ta: dict):
    """שומר daily_table.csv ו-test_a.csv."""
    OUT_DIR.mkdir(exist_ok=True)
    table.to_csv(OUT_DIR / "daily_table.csv", index=False)

    # test_a.csv: שורה לכל תרחיש
    rows = []
    for key, sc in ta["scenarios"].items():
        rows.append({"scenario": key, **sc})
    pd.DataFrame(rows).to_csv(OUT_DIR / "test_a.csv", index=False)


if __name__ == "__main__":
    main()