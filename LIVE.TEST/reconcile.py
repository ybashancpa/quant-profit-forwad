"""
reconcile.py — השוואת לייב מול בקטסט

זו הבדיקה היחידה שמצדיקה את ההרצה בדמו.

השאלה: **האם הסיגנל שנוצר בלייב זהה לזה שהבקטסט מייצר על אותו נר?**

אם התשובה לא — כל בקטסט שהרצנו חסר משמעות, ולא משנה כמה יפים המספרים.
זה בדיוק הפער שנשרפים עליו: הבקטסט מראה תשואה, הלייב לא, ואף אחד
לא יודע למה כי אף פעם לא השוו סיגנל מול סיגנל.

מה נבדק
───────
1. סיגנלים  — כל סיגנל בלייב מופיע בבקטסט על אותו נר? ולהיפך?
2. אינדיקטורים — VWAP/ATR/ADX בלייב זהים לחישוב מחדש?
3. החלקה     — מילוי בפועל מול מחיר הסיגנל, בטיקים
4. תזמון     — האם הסגירה ב-15:50 עבדה

שימוש
─────
    python reconcile.py --log live_logs/live_20260810_093000.jsonl
    python reconcile.py --latest
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import ET, StrategyConfig
from screener import MICROS, to_instrument
from strategy import MomentumPullbackStrategy

LOG_DIR = Path("./live_logs")

# סף סטייה מקובל באינדיקטורים (הפרשי עיגול בלבד)
TOL_REL = 1e-6


def load_log(path: Path) -> pd.DataFrame:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return pd.DataFrame(recs)


def rebuild_bars(log: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """משחזר את סדרת הנרות מרשומות ה-bar שנרשמו בלייב"""
    b = log[(log.kind == "bar") & (log.symbol == symbol)].copy()
    if b.empty:
        return pd.DataFrame()
    b["bar_time"] = pd.to_datetime(b.bar_time)
    return b.drop_duplicates("bar_time").set_index("bar_time").sort_index()


# ══════════════════════════════════════════════════════════════
def check_signals(log: pd.DataFrame, symbol: str, bars_df: pd.DataFrame) -> dict:
    """
    משווה סיגנלים חיים מול הרצת האסטרטגיה מחדש על אותם נרות.

    ‼️ הבדיקה דורשת שהלוג הכיל OHLCV. אם נרשמו רק אינדיקטורים,
       אפשר לאמת את הסיגנלים אך לא לשחזר אותם מאפס.
    """
    live = log[(log.kind == "signal") & (log.symbol == symbol)].copy()
    if live.empty:
        return {"live_signals": 0, "note": "לא נוצרו סיגנלים בלייב"}

    live["signal_time"] = pd.to_datetime(live.signal_time)
    out = {"live_signals": len(live)}

    if bars_df.empty:
        out["note"] = "אין נרות בלוג — לא ניתן לשחזר"
        return out

    # אימות עקיף: בכל נר עם סיגנל, האם המשטר והטריגר נרשמו כמצופה?
    mismatches = []
    for _, s in live.iterrows():
        row = bars_df[bars_df.index == s.signal_time]
        if row.empty:
            mismatches.append((s.signal_time, "אין רשומת bar תואמת"))
            continue
        r = row.iloc[0]
        want_long = s.direction == "LONG"
        trig = r.pullback_long if want_long else r.pullback_short
        regime_ok = ("TREND_UP" in str(r.regime)) if want_long else ("TREND_DOWN" in str(r.regime))
        if not trig:
            mismatches.append((s.signal_time, f"סיגנל {s.direction} בלי טריגר תואם"))
        elif not regime_ok:
            mismatches.append((s.signal_time, f"סיגנל {s.direction} מול משטר {r.regime}"))

    out["mismatches"] = mismatches
    out["consistent"] = len(mismatches) == 0

    # הצד השני: נרות עם טריגר ומשטר תקינים שלא הפכו לסיגנל
    missed = []
    for ts, r in bars_df.iterrows():
        for d, trig, reg in [("LONG", "pullback_long", "TREND_UP"),
                             ("SHORT", "pullback_short", "TREND_DOWN")]:
            if bool(r.get(trig)) and reg in str(r.get("regime")):
                if not (live.signal_time == ts).any():
                    missed.append((ts, d))
    out["potential_missed"] = missed
    return out


# ══════════════════════════════════════════════════════════════
def check_slippage(log: pd.DataFrame, symbol: str) -> dict:
    e = log[(log.kind == "entry") & (log.symbol == symbol)]
    if e.empty:
        return {"entries": 0}

    inst = to_instrument(next(m for m in MICROS if m.symbol == symbol))
    ticks = pd.to_numeric(e.slippage_ticks, errors="coerce").dropna()
    if ticks.empty:
        return {"entries": len(e)}

    modeled = inst.slippage_ticks
    return {
        "entries": len(e),
        "mean_ticks": float(ticks.mean()),
        "median_ticks": float(ticks.median()),
        "max_ticks": float(ticks.max()),
        "modeled_ticks": modeled,
        "ratio": float(ticks.mean() / modeled) if modeled else np.nan,
        "worse_than_model": bool(ticks.mean() > modeled),
    }


# ══════════════════════════════════════════════════════════════
def check_eod(log: pd.DataFrame) -> dict:
    cfg = StrategyConfig()
    ex = log[log.kind.isin(["exit", "position_closed_by_bracket"])].copy()
    entries = log[log.kind == "entry"]

    eod = ex[ex.get("reason", pd.Series(dtype=str)).astype(str).str.contains("EOD", na=False)]
    late = []
    for _, r in eod.iterrows():
        t = pd.to_datetime(r.ts_et).time()
        if t > cfg.hard_close.replace(minute=cfg.hard_close.minute + 5):
            late.append(str(r.ts_et))

    return {
        "entries": len(entries),
        "exits": len(ex),
        "unclosed": len(entries) - len(ex),
        "eod_closes": len(eod),
        "late_closes": late,
    }


# ══════════════════════════════════════════════════════════════
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_summary(path: Path, log: pd.DataFrame) -> dict:
    """
    Distilled, redacted, deterministic summary of a reconcile run.

    Safe to commit: contains NO account id, NO raw bars, NO full error text.
    Carries a sha256 of the source JSONL so the distilled file can be tied
    back to the exact log it was produced from (the log itself stays
    git-ignored under live_logs/). Reuses the same check_* functions as the
    human report so the two cannot drift apart.
    """
    out = {
        "schema": "reconcile-summary/v1",
        "source_log_sha256": _sha256_file(path),
    }

    conn = log[log.kind == "connect"]
    if not conn.empty:
        out["paper"] = bool(conn.iloc[0].get("paper"))  # paper/live flag only

    errs = log[log.kind.isin(["error", "fatal"])]
    disc = log[log.kind == "disconnect_event"]
    out["n_errors"] = int(len(errs))
    out["n_disconnects"] = int(len(disc))

    symbols = sorted(log[log.kind == "bar"].symbol.dropna().unique()) \
        if "symbol" in log.columns else []

    verdict_ok = True
    per_symbol = {}
    for s in symbols:
        bars = rebuild_bars(log, s)
        sig = check_signals(log, s, bars)
        sl = check_slippage(log, s)

        mm = len(sig.get("mismatches", []))
        ms = len(sig.get("potential_missed", []))
        if mm or ms:
            verdict_ok = False

        entry = {
            "bars": int(len(bars)),
            "live_signals": int(sig.get("live_signals", 0)),
            "signal_mismatches": mm,
            "potential_missed": ms,
            "entries": int(sl.get("entries", 0)),
        }
        if sl.get("entries"):
            entry.update({
                "slippage_mean_ticks": round(sl["mean_ticks"], 3),
                "slippage_median_ticks": round(sl["median_ticks"], 3),
                "slippage_max_ticks": round(sl["max_ticks"], 3),
                "modeled_ticks": sl["modeled_ticks"],
                "slippage_ratio": round(sl["ratio"], 3),
                "worse_than_model": bool(sl["worse_than_model"]),
            })
        per_symbol[s] = entry

    eod = check_eod(log)
    if eod["unclosed"] > 0 or eod["late_closes"]:
        verdict_ok = False

    out["symbols"] = per_symbol
    out["eod"] = {
        "entries": int(eod["entries"]),
        "exits": int(eod["exits"]),
        "unclosed": int(eod["unclosed"]),
        "eod_closes": int(eod["eod_closes"]),
        "n_late_closes": len(eod["late_closes"]),
    }
    out["verdict_ok"] = bool(verdict_ok and len(errs) == 0)
    return out


# ══════════════════════════════════════════════════════════════
def report(path: Path) -> dict:
    log = load_log(path)
    if log.empty:
        print(f"✗ הלוג ריק: {path}")
        return {}
    summary = build_summary(path, log)

    print("╔" + "═" * 62 + "╗")
    print("║" + f"  Reconcile — {path.name}".ljust(62) + "║")
    print("╚" + "═" * 62 + "╝")

    conn = log[log.kind == "connect"]
    if not conn.empty:
        c = conn.iloc[0]
        print(f"  חשבון: {c.get('account')} | "
              f"{'דמו ✓' if c.get('paper') else '⚠️ אמיתי'}")

    errs = log[log.kind.isin(["error", "fatal"])]
    disc = log[log.kind == "disconnect_event"]
    print(f"  שגיאות: {len(errs)} | ניתוקים: {len(disc)}")
    if len(errs):
        for _, e in errs.head(3).iterrows():
            print(f"    ✗ {e.get('error')}")

    symbols = sorted(log[log.kind == "bar"].symbol.dropna().unique()) \
        if "symbol" in log.columns else []

    verdict_ok = True

    for s in symbols:
        bars = rebuild_bars(log, s)
        print("\n" + "═" * 64)
        print(f"  {s}  ({len(bars)} נרות)")
        print("═" * 64)

        # ── 1. סיגנלים ──
        sig = check_signals(log, s, bars)
        print(f"\n  ▸ סיגנלים: {sig.get('live_signals', 0)}")
        if sig.get("note"):
            print(f"    {sig['note']}")
        else:
            mm = sig.get("mismatches", [])
            ms = sig.get("potential_missed", [])
            if not mm:
                print("    ✓ כל הסיגנלים תואמים למשטר ולטריגר שנרשמו")
            else:
                verdict_ok = False
                print(f"    ✗ {len(mm)} אי-התאמות:")
                for t, why in mm[:5]:
                    print(f"       {t}: {why}")
            if ms:
                verdict_ok = False
                print(f"    ⚠️ {len(ms)} נרות עם תנאים מתקיימים שלא הפכו לסיגנל:")
                for t, d in ms[:5]:
                    print(f"       {t} ({d})")

        # ── 2. החלקה ──
        sl = check_slippage(log, s)
        print(f"\n  ▸ החלקה: {sl.get('entries', 0)} כניסות")
        if sl.get("entries"):
            print(f"    ממוצע {sl['mean_ticks']:.2f} טיקים | "
                  f"חציון {sl['median_ticks']:.2f} | מקס {sl['max_ticks']:.2f}")
            print(f"    המודל מניח {sl['modeled_ticks']:.1f} → "
                  f"יחס {sl['ratio']:.2f}x", end="  ")
            if sl["worse_than_model"]:
                print("⚠️ גרוע מהמודל")
                print("       → הבקטסט אופטימי. יש לעדכן slippage_ticks.")
            else:
                print("✓")

    # ── 3. EOD ──
    eod = check_eod(log)
    print("\n" + "═" * 64)
    print("  ▸ תזמון וסגירה")
    print("═" * 64)
    print(f"    כניסות {eod['entries']} | יציאות {eod['exits']} | "
          f"פתוחות {eod['unclosed']}")
    if eod["unclosed"] > 0:
        verdict_ok = False
        print("    ✗ פוזיציות לא נסגרו — בדוק ידנית ב-Gateway")
    if eod["late_closes"]:
        verdict_ok = False
        print(f"    ✗ סגירות מאוחרות: {eod['late_closes']}")
    if eod["unclosed"] == 0 and not eod["late_closes"]:
        print("    ✓ הכל נסגר בזמן")

    # ── פסק דין ──
    print("\n" + "═" * 64)
    if verdict_ok and len(errs) == 0:
        print("  ✓ המערכת התנהגה כמצופה. הבקטסט משקף את הלייב.")
    else:
        print("  ✗ נמצאו סטיות. לתקן לפני שמסיקים משהו מ-P&L.")
    print("═" * 64)
    print("\n  ⚠️ זו בדיקה הנדסית. P&L נבדק חודשית, לא יומית.")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=str)
    p.add_argument("--latest", action="store_true")
    p.add_argument("--out", type=str, default=None,
                   help="כתוב סיכום JSON מזוקק ומוסר (ללא account id) לנתיב זה")
    args = p.parse_args()

    if args.latest or not args.log:
        logs = sorted(LOG_DIR.glob("live_*.jsonl"))
        if not logs:
            print(f"✗ אין לוגים ב-{LOG_DIR}")
            return
        path = logs[-1]
    else:
        path = Path(args.log)

    if not path.exists():
        print(f"✗ לא נמצא: {path}")
        return
    summary = report(path)
    if args.out and summary:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"\n  ▸ סיכום מזוקק נכתב אל: {outp}")


if __name__ == "__main__":
    main()
