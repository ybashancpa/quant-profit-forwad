"""
Weekly email report for the SmartPassive forward paper test.
Reads forward/ logs, builds an HTML summary, sends via Gmail SMTP.

Env vars (set as GitHub Secrets):
  GMAIL_USER          sender gmail address
  GMAIL_APP_PASSWORD  16-char app password

Usage:
  python send_weekly_email.py            # send
  python send_weekly_email.py --dry-run  # build only, save email_preview.html
"""

import argparse
import json
import os
import smtplib
import ssl
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate

import pandas as pd

FORWARD_DIR = os.environ.get("FORWARD_DIR", "forward")
# Use `or` so an empty/unset secret still falls back to the default address.
GMAIL_USER = (os.environ.get("GMAIL_USER") or "ybashan.cpa@gmail.com").strip()
TO_ADDR = (os.environ.get("REPORT_TO") or GMAIL_USER).strip()
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pct(value, start):
    if value is None or start is None:
        return None
    return (value / start - 1) * 100


def _fmt_pct(value):
    if value is None:
        return "—"
    return f"{value:+.2f}%"


def _fmt_gap(a, b):
    if a is None or b is None:
        return "—"
    return f"{a - b:+.2f}%"


def build_html():
    state = load_json(os.path.join(FORWARD_DIR, "state.json"))
    summary = load_json(os.path.join(FORWARD_DIR, "summary.json"))
    nav = pd.read_csv(os.path.join(FORWARD_DIR, "nav_history.csv"),
                      parse_dates=["date"])

    nav_now = state["nav"]
    start = state["start_capital"]
    total_ret = _pct(nav_now, start)

    bench_nav = summary.get("bench_nav")
    bench_ret = _pct(bench_nav, start)
    nav_for_gap = summary.get("nav", nav_now)
    gap_ret = _pct(nav_for_gap, start)
    after_tax_nav = summary.get("after_tax_nav")
    bench_after_tax_nav = summary.get("bench_after_tax_nav")
    after_tax_ret = _pct(after_tax_nav, start)
    bench_after_tax_ret = _pct(bench_after_tax_nav, start)
    spy_bench_ret = _pct(state.get("spy_bench_nav"), start)

    verdict = summary.get("verdict")
    risk_off_cycles = summary.get("risk_off_cycles_completed")
    years_elapsed = summary.get("years_elapsed")

    cummax = nav["nav"].cummax()
    dd = ((nav["nav"] - cummax) / cummax).min() * 100 if len(nav) else 0.0
    days = (pd.Timestamp(state["last_run_date"]) -
            pd.Timestamp(state["start_date"])).days

    cutoff = nav["date"].max() - pd.Timedelta(days=7)
    week_nav = nav[nav["date"] >= cutoff]
    trades = pd.read_csv(os.path.join(FORWARD_DIR, "trades.csv"))
    trades["date"] = pd.to_datetime(trades["date"])
    week_trades = trades[trades["date"] >= cutoff]
    signals = pd.read_csv(os.path.join(FORWARD_DIR, "signals.csv"))
    signals["date"] = pd.to_datetime(signals["date"])

    holdings = "".join(
        f"<tr><td>{t}</td><td>{v:,.4f}</td></tr>"
        for t, v in sorted(state["shares"].items()))

    if len(week_trades):
        trade_rows = "".join(
            f"<tr><td>{r['date'].date()}</td><td>{r['ticker']}</td>"
            f"<td>{'BUY' if r['value']>0 else 'SELL'}</td>"
            f"<td>${abs(r['value']):,.2f}</td><td>{r['reason']}</td></tr>"
            for _, r in week_trades.iterrows())
        trades_html = (f"<table><tr><th>תאריך</th><th>נייר</th><th>כיוון</th>"
                       f"<th>סכום</th><th>סיבה</th></tr>{trade_rows}</table>")
    else:
        trades_html = "<p>לא בוצעו עסקאות השבוע.</p>"

    if len(week_nav) >= 2 and "bench_nav" in week_nav.columns:
        week_ret = (week_nav["nav"].iloc[-1] / week_nav["nav"].iloc[0] - 1) * 100
        week_bench = (week_nav["bench_nav"].iloc[-1] /
                      week_nav["bench_nav"].iloc[0] - 1) * 100
    elif len(week_nav) >= 2:
        week_ret = (week_nav["nav"].iloc[-1] / week_nav["nav"].iloc[0] - 1) * 100
        week_bench = None
    else:
        week_ret = 0.0
        week_bench = None

    regime_he = {"RISK_ON": "סיכון-פעיל (SPY מעל MA200)",
                 "RISK_OFF": "סיכון-כבוי (SPY מתחת/על MA200)"}

    verdict_display = verdict or "—"
    verdict_color = "#c00" if verdict == "REFUTED" else "#666" if verdict == "NOT_READY" else "#080"

    cycles_display = f"{risk_off_cycles}/2" if risk_off_cycles is not None else "—"
    years_display = f"{years_elapsed:.2f}/3.00" if years_elapsed is not None else "—"

    week_line = f"{week_ret:+.2f}% (בנצ'מרק: {_fmt_pct(week_bench)})"

    html = f"""<html dir="rtl"><body style="font-family:Arial">
<h2>דוח שבועי — SmartPassive Forward Test</h2>
<p>ניסוי: <b>{state['experiment_id']}</b> | התחלה: {state['start_date']}
({days} ימים) | עודכן: {state['last_run_date']}</p>

<table cellpadding="8" cellspacing="0" style="border:2px solid {verdict_color};border-collapse:collapse;margin-bottom:12px">
<tr><td style="font-size:18px"><b>Verdict: <span style="color:{verdict_color}">{verdict_display}</span></b></td></tr>
<tr><td>מחזורי RISK_OFF שהושלמו: <b>{cycles_display}</b> | זמן שחלף: <b>{years_display} שנים</b></td></tr>
</table>

<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse">
<tr><td><b>NAV</b></td><td>${nav_now:,.2f}</td></tr>
<tr><td><b>NAV לאחר מס</b></td><td>{f'${after_tax_nav:,.2f}' if after_tax_nav is not None else '—'}</td></tr>
<tr><td><b>תשואה מצטברת</b></td><td>{_fmt_pct(total_ret)}</td></tr>
<tr><td><b>בנצ'מרק סטטי (55/35/10)</b></td><td>{_fmt_pct(bench_ret)}</td></tr>
<tr><td><b>פער מול בנצ'מרק</b></td><td>{_fmt_gap(gap_ret, bench_ret)}</td></tr>
<tr><td><b>פער לאחר מס</b></td><td>{_fmt_gap(after_tax_ret, bench_after_tax_ret)}</td></tr>
<tr><td style="color:#999"><b>SPY (להקשר בלבד)</b></td><td style="color:#999">{_fmt_pct(spy_bench_ret)}</td></tr>
<tr><td><b>תשואת השבוע</b></td><td>{week_line}</td></tr>
<tr><td><b>Max Drawdown</b></td><td>{dd:.2f}%</td></tr>
<tr><td><b>משטר</b></td><td>{regime_he.get(summary['regime'], summary['regime'])}</td></tr>
<tr><td><b>SPY / MA200</b></td><td>{summary['spy_price']:.2f} / {summary['ma200']:.2f}</td></tr>
<tr><td><b>מזומן</b></td><td>${state['cash']:,.2f}</td></tr>
<tr><td><b>עלויות מצטברות</b></td><td>${state.get('total_costs',0):,.2f}</td></tr>
</table>
<h3>אחזקות</h3>
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse">
<tr><th>נייר</th><th>מניות</th></tr>{holdings}</table>
<h3>עסקאות השבוע</h3>{trades_html}
<p style="color:#666">מסחר מתבצע רק ביום המסחר האחרון של החודש, בכפוף ל-drift
מעל 3% (סטייה מקסימלית לנייר יחיד מול משקל היעד).
דוח מלא: forward/report.html ב-repository.</p>
</body></html>"""
    return html, total_ret


def _build_subject(total_ret):
    state = load_json(os.path.join(FORWARD_DIR, "state.json"))
    summary = load_json(os.path.join(FORWARD_DIR, "summary.json"))
    verdict = summary.get("verdict", "—")
    bench_nav = summary.get("bench_nav")
    start = state["start_capital"]
    gap = _fmt_gap(_pct(state["nav"], start), _pct(bench_nav, start))
    return (f"SmartPassive [{verdict}] NAV ${state['nav']:,.0f} "
            f"(מול בנצ'מרק: {gap})")


def send(html, total_ret):
    app_pw = os.environ.get("GMAIL_APP_PASSWORD")
    if not app_pw:
        raise RuntimeError("GMAIL_APP_PASSWORD env var not set")
    if not GMAIL_USER:
        raise RuntimeError("GMAIL_USER is empty; set the GMAIL_USER secret")
    msg = MIMEMultipart("alternative")
    msg["Subject"] = _build_subject(total_ret)
    msg["From"] = GMAIL_USER
    msg["To"] = TO_ADDR
    msg["Date"] = formatdate(localtime=True)
    msg.attach(MIMEText(html, "html", "utf-8"))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
        server.login(GMAIL_USER, app_pw.replace(" ", ""))
        server.sendmail(GMAIL_USER, [a.strip() for a in TO_ADDR.split(",") if a.strip()],
                        msg.as_string())
    print(f"[EMAIL SENT] to {TO_ADDR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="build email HTML only, do not send")
    args = ap.parse_args()

    html, total_ret = build_html()
    preview = os.path.join(FORWARD_DIR, "email_preview.html")
    with open(preview, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[EMAIL BUILT] preview: {preview}")

    if args.dry_run:
        print("[DRY-RUN] not sending")
        return
    send(html, total_ret)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[EMAIL ERROR] {e}", file=sys.stderr)
        sys.exit(1)
