"""
h2_config.py — קונפיגורציה נעולה לבדיקת H2 (חלון הסגירה)

כל הערכים כאן נגזרים מ-H2_hypothesis.md + H2_appendix_A.md.
אסור לשנות ערך כלשהו אחרי ראיית תוצאות.

⚠️ נכתב לפני ראיית נתונים כלשהם.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from zoneinfo import ZoneInfo

# ══════════════════════════════════════════════════════════════
# אזור זמן
# ══════════════════════════════════════════════════════════════
ET = ZoneInfo("America/New_York")

# ══════════════════════════════════════════════════════════════
# הגדרות נעולות מהמסמך (assert יאכוף)
# ══════════════════════════════════════════════════════════════
LOCKED_SYMBOL = "M2K"              # מכשיר יחיד — אין להחליף
LOCKED_YAHOO = "RTY=F"             # סימבול Yahoo לחוזה רציף
LOCKED_MULTIPLIER = 5.0            # M2K multiplier
LOCKED_TICK_SIZE = 0.10            # נקודות לטיק
LOCKED_TICK_VALUE = LOCKED_TICK_SIZE * LOCKED_MULTIPLIER  # $0.50
LOCKED_CONTRACTS = 1               # חוזה אחד. קבוע.
LOCKED_ENTRY_TIME = time(15, 30)   # פתיחת נר 15:30 ET
LOCKED_EXIT_TIME = time(15, 55)    # סגירת נר 15:55 ET (= 16:00)
LOCKED_STOP_PCT = 0.01             # סטופ קטסטרופה 1.0%
LOCKED_COMMISSION_RT = 1.52        # עמלה round-trip לחוזה (מהמסמך)
LOCKED_EFFECT_SIZE = 0.063         # Sharpe לעסקה מהספרות (Baltussen)
LOCKED_RISK_FREE_ANNUAL = 0.035    # 3.5% תשואה חסרת סיכון
LOCKED_ACCOUNT = 5000.0            # גודל חשבון
LOCKED_MARGIN_ESTIMATE = 800.0     # אומדן מרג'ין (config.py) — לא מדידה
LOCKED_MARGIN_THRESHOLD = 1500.0   # סף FC4

# ══════════════════════════════════════════════════════════════
# תרחישי החלקה (טיקים הלוך־חזור)
# ══════════════════════════════════════════════════════════════
SLIPPAGE_SCENARIOS = [1, 2, 4]     # 2 = בסיס FC1; 1,4 = רגישות
BASE_SLIPPAGE_TICKS = 2            # נספח א' סעיף ב'

# ══════════════════════════════════════════════════════════════
# קריטריוני הפרכה (FC1–FC4)
# ══════════════════════════════════════════════════════════════
CRITERIA = {
    "FC1": {
        "desc": "מבחן A: נטו שנתי < 3.5% על $5,000",
        "threshold": LOCKED_RISK_FREE_ANNUAL * LOCKED_ACCOUNT,  # $175
        "direction": "net_annual >= threshold",
        "source": "H2_hypothesis.md §5.1",
    },
    "FC2": {
        "desc": "מתאם שלילי מובהק בין r_ROD ל-r_LH",
        "threshold": 0.10,
        "direction": "NOT (corr < 0 AND p < 0.10)",
        "source": "H2_hypothesis.md §5.2",
    },
    "FC3": {
        "desc": "החלקה > 2 טיקים הופכת מבחן A לשלילי",
        "threshold": None,  # PENDING — נמדד ב-Paper
        "direction": "PENDING",
        "source": "H2_hypothesis.md §5.3 + נספח א' סעיף ג'",
    },
    "FC4": {
        "desc": "מרג'ין M2K > $1,500",
        "threshold": LOCKED_MARGIN_THRESHOLD,
        "direction": "PENDING",
        "source": "H2_hypothesis.md §5.4 + נספח א' סעיף ד'",
    },
}

# ══════════════════════════════════════════════════════════════
# פסק דין אפשריים (נספח א')
# ══════════════════════════════════════════════════════════════
VERDICT_REFUTED = "מופרכת"
VERDICT_NOT_REFUTED_PARTIAL = "לא הופרכה, חלקית"
VERDICT_INCONCLUSIVE = "לא ניתן להכריע"

# ══════════════════════════════════════════════════════════════
# נתיבי פלט
# ══════════════════════════════════════════════════════════════
from pathlib import Path

OUT_DIR = Path(__file__).parent / "results_h2"
DATA_DIR = OUT_DIR / "data"
SNAPSHOT_PATH = DATA_DIR / "M2K_5m.parquet"

# ══════════════════════════════════════════════════════════════
# קבצים נעולים שיש לשמור hash
# ══════════════════════════════════════════════════════════════
LOCKED_FILES = [
    "H2_hypothesis.md",
    "H2_appendix_A.md",
    "H2_protocol_overlap_ruling.md",
    "h2_config.py",
]


def assert_locked():
    """Assertions על ההגדרות הנעולות — נקרא לפני כל הרצה."""
    assert LOCKED_SYMBOL == "M2K", "מכשיר חייב להיות M2K"
    assert LOCKED_ENTRY_TIME == time(15, 30), "כניסה חייבת להיות 15:30"
    assert LOCKED_EXIT_TIME == time(15, 55), "יציאה חייבת להיות סגירת 15:55"
    assert LOCKED_CONTRACTS == 1, "חוזה אחד קבוע"
    assert LOCKED_STOP_PCT == 0.01, "סטופ קטסטרופה 1.0%"
    assert LOCKED_MULTIPLIER == 5.0, "M2K multiplier = 5"
    assert LOCKED_TICK_SIZE == 0.10, "M2K tick = 0.10"
    assert BASE_SLIPPAGE_TICKS == 2, "בסיס FC1 = 2 טיקים"
    assert LOCKED_EFFECT_SIZE == 0.063, "effect size מהספרות"
    assert len(CRITERIA) == 4, "חייבים להיות בדיוק 4 קריטריונים"
    assert set(CRITERIA.keys()) == {"FC1", "FC2", "FC3", "FC4"}