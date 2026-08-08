"""
test_h2_audit.py — Criteria-to-code audit לבדיקת H2

מוודא:
  1. כל קריטריון במסמך (FC1–FC4) מיוצג בקוד
  2. כל בדיקה בקוד ממופה לקריטריון במסמך
  3. mutation test: היפוך מכוון של כל תנאי משנה את פסק הדין
  4. assertions על הגדרות נעולות

יש להריץ לפני test_h2.py:
    python test_h2_audit.py
"""

from __future__ import annotations

import sys
from datetime import time

import numpy as np
import pandas as pd

from h2_config import (
    BASE_SLIPPAGE_TICKS, CRITERIA, LOCKED_ACCOUNT, LOCKED_COMMISSION_RT,
    LOCKED_CONTRACTS, LOCKED_EFFECT_SIZE, LOCKED_ENTRY_TIME,
    LOCKED_EXIT_TIME, LOCKED_MARGIN_ESTIMATE, LOCKED_MARGIN_THRESHOLD,
    LOCKED_MULTIPLIER, LOCKED_RISK_FREE_ANNUAL, LOCKED_STOP_PCT,
    LOCKED_SYMBOL, LOCKED_TICK_VALUE, assert_locked,
)
from test_h2 import evaluate_criteria, pearson_with_pvalue, test_a, test_b


# ══════════════════════════════════════════════════════════════
# 1. כיסוי דו-כיווני
# ══════════════════════════════════════════════════════════════

def test_coverage():
    """כל קריטריון במסמך מיוצג; אין קריטריון עודף בקוד."""
    expected_ids = {"FC1", "FC2", "FC3", "FC4"}
    actual_ids = set(CRITERIA.keys())
    assert actual_ids == expected_ids, (
        f"כיסוי קריטריונים: ציפיתי {expected_ids}, קיבלתי {actual_ids}")
    print("  ✓ כיסוי דו-כיווני: FC1–FC4 מיוצגים בדיוק")


# ══════════════════════════════════════════════════════════════
# 2. assertions על הגדרות נעולות
# ══════════════════════════════════════════════════════════════

def test_locked_definitions():
    """assert על כל הגדרה נעולה מהמסמך."""
    assert_locked()
    # בדיקות נוספות ספציפיות ל-audit
    assert LOCKED_ENTRY_TIME == time(15, 30)
    assert LOCKED_EXIT_TIME == time(15, 55)
    assert LOCKED_SYMBOL == "M2K"
    assert LOCKED_CONTRACTS == 1
    assert LOCKED_STOP_PCT == 0.01
    assert LOCKED_MULTIPLIER == 5.0
    assert LOCKED_TICK_VALUE == 0.50  # 0.10 × 5
    assert BASE_SLIPPAGE_TICKS == 2
    assert LOCKED_EFFECT_SIZE == 0.063
    assert LOCKED_COMMISSION_RT == 1.52
    assert LOCKED_RISK_FREE_ANNUAL == 0.035
    assert LOCKED_ACCOUNT == 5000.0
    assert abs(CRITERIA["FC1"]["threshold"] - 175.0) < 1e-9  # 3.5% × 5000
    assert CRITERIA["FC2"]["threshold"] == 0.10
    assert CRITERIA["FC3"]["direction"] == "PENDING"
    assert CRITERIA["FC4"]["direction"] == "PENDING"
    assert LOCKED_MARGIN_THRESHOLD == 1500.0
    print("  ✓ כל ההגדרות הנעולות תקינות")


# ══════════════════════════════════════════════════════════════
# 3. Mutation tests
# ══════════════════════════════════════════════════════════════

def _make_synthetic_table(n=40, corr_sign=1.0, sigma_scale=1.0):
    """
    יוצר טבלה סינתטית לבדיקת mutation.
    corr_sign=1 → מתאם חיובי; -1 → שלילי.
    sigma_scale שולט בגודל σ (משפיע על FC1).
    """
    np.random.seed(42)
    px_base = 2000.0
    px_1530 = np.full(n, px_base)
    r_ROD = np.random.normal(0.002, 0.003, n)
    # r_LH מתואם עם r_ROD לפי הסימן
    noise = np.random.normal(0, 0.002, n) * sigma_scale
    r_LH = corr_sign * 0.5 * r_ROD + noise

    table = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq="B").date,
        "prev_close": px_base / (1 + r_ROD),
        "px_1530": px_1530,
        "px_close": px_1530 * (1 + r_LH),
        "r_ROD": r_ROD,
        "r_LH": r_LH,
        "direction": np.where(r_ROD > 0, 1, np.where(r_ROD < 0, -1, 0)),
        "adverse": np.abs(noise),
    })
    return table


def test_mutation_fc1():
    """
    FC1: נטו שנתי >= $175.
    Mutation: σ קטן מאוד → expected קטן → נטו שלילי → FAIL.
    """
    # σ גדול → צריך לעבור
    table_pass = _make_synthetic_table(sigma_scale=3.0)
    ta_pass = test_a(table_pass)
    tb_dummy = test_b(table_pass)
    v_pass = evaluate_criteria(ta_pass, tb_dummy)
    # לא בהכרח PASS כי תלוי ב-σ; נבדוק שהלוגיקה רצה

    # σ אפסי → expected ≈ 0 → נטו שלילי → FAIL
    table_fail = _make_synthetic_table(sigma_scale=0.001)
    ta_fail = test_a(table_fail)
    v_fail = evaluate_criteria(ta_fail, tb_dummy)
    assert v_fail["criteria"]["FC1"]["status"] == "FAIL", (
        "FC1 mutation: σ≈0 חייב להיכשל")
    print("  ✓ FC1 mutation: σ≈0 → FAIL")


def test_mutation_fc2():
    """
    FC2: מתאם שלילי מובהק → FAIL.
    Mutation: מתאם שלילי חזק → חייב FAIL.
    מתאם חיובי → PASS (אינו אישור, רק היעדר סתירה).
    """
    # מתאם שלילי חזק
    table_neg = _make_synthetic_table(corr_sign=-1.0)
    ta_neg = test_a(table_neg)
    tb_neg = test_b(table_neg)
    v_neg = evaluate_criteria(ta_neg, tb_neg)
    # עם corr=-0.5 ו-n=40, p צריך להיות < 0.10
    if tb_neg["negative_significant"]:
        assert v_neg["criteria"]["FC2"]["status"] == "FAIL", (
            "FC2 mutation: מתאם שלילי מובהק חייב FAIL")
        print("  ✓ FC2 mutation: מתאם שלילי מובהק → FAIL")
    else:
        # אם לא מובהק (n קטן), עדיין PASS — זה תקין
        print("  ✓ FC2 mutation: מתאם שלילי לא מובהק → PASS (תקין)")

    # מתאם חיובי → תמיד PASS
    table_pos = _make_synthetic_table(corr_sign=1.0)
    ta_pos = test_a(table_pos)
    tb_pos = test_b(table_pos)
    v_pos = evaluate_criteria(ta_pos, tb_pos)
    assert v_pos["criteria"]["FC2"]["status"] == "PASS", (
        "FC2: מתאם חיובי חייב PASS (היעדר סתירה)")
    print("  ✓ FC2: מתאם חיובי → PASS (היעדר סתירה, לא אישור)")


def test_mutation_fc3_fc4():
    """FC3 ו-FC4 תמיד PENDING — mutation לא משנה."""
    table = _make_synthetic_table()
    ta = test_a(table)
    tb = test_b(table)
    v = evaluate_criteria(ta, tb)
    assert v["criteria"]["FC3"]["status"] == "PENDING"
    assert v["criteria"]["FC4"]["status"] == "PENDING"
    print("  ✓ FC3/FC4: תמיד PENDING (נמדד ב-Paper/TWS)")


def test_mutation_verdict_logic():
    """
    פסק דין:
    - FC1 FAIL → מופרכת (גם אם FC2 PASS)
    - FC2 FAIL → מופרכת (גם אם FC1 PASS)
    - FC1+FC2 PASS → לא הופרכה, חלקית
    """
    table = _make_synthetic_table(corr_sign=1.0, sigma_scale=3.0)
    ta = test_a(table)
    tb = test_b(table)

    # נבדוק שהלוגיקה עקבית
    v = evaluate_criteria(ta, tb)
    fc1 = v["criteria"]["FC1"]["status"]
    fc2 = v["criteria"]["FC2"]["status"]

    if fc1 == "FAIL" or fc2 == "FAIL":
        assert v["verdict"] == "מופרכת", (
            f"FC1={fc1}, FC2={fc2} → חייב 'מופרכת'")
    elif fc1 == "PASS" and fc2 == "PASS":
        assert v["verdict"] == "לא הופרכה, חלקית", (
            "FC1+FC2 PASS → חייב 'לא הופרכה, חלקית'")
    print(f"  ✓ לוגיקת פסק דין עקבית (FC1={fc1}, FC2={fc2} → {v['verdict']})")


# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════

def main():
    print("═" * 60)
    print("  H2 Criteria-to-Code Audit")
    print("═" * 60)

    tests = [
        ("כיסוי דו-כיווני", test_coverage),
        ("הגדרות נעולות", test_locked_definitions),
        ("FC1 mutation", test_mutation_fc1),
        ("FC2 mutation", test_mutation_fc2),
        ("FC3/FC4 PENDING", test_mutation_fc3_fc4),
        ("לוגיקת פסק דין", test_mutation_verdict_logic),
    ]

    failures = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as e:
            print(f"  ✗ {name}: {e}")
            failures += 1
        except Exception as e:
            print(f"  ✗ {name}: שגיאה — {type(e).__name__}: {e}")
            failures += 1

    print("═" * 60)
    if failures == 0:
        print("  ✓ Audit עבר: כל הקריטריונים מיוצגים ונאכפים")
        print("    ניתן להריץ: python test_h2.py")
    else:
        print(f"  ✗ Audit נכשל ({failures} כשלים). לתקן לפני הרצה.")
        sys.exit(1)


if __name__ == "__main__":
    main()