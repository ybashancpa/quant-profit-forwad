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
GMAIL_USER = os.environ.get("GMAIL_USER", "ybashan.cpa@gmail.com")
TO_ADDR = os.environ.get("REPORT_TO", GMAIL_USER)
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_html():
    state = load_json(os.path.join(FORWARD_DIR, "state.json"))
    summary = load_json(os.path.join(FORWARD_DIR, "summary.json"))
    nav = pd.read_csv(os.path.join(FORWARD_DIR, "nav_history.csv"),
                      parse_dates=["date"])

    nav_now = state["nav"]
    start = state["start_capital"]
    total_ret = (nav_now / start - 1) * 100
    bench_ret = (state["spy_bench_nav"] / start - 1) * 100
    cummax = nav["nav"].cummax()
    dd = ((nav["nav"] - cummax) / cummax).min() * 100 if len(nav) else 0.0
    days = (pd.Timestamp(state["last_run_date"]) -
            pd.Timestamp(state["start_date"])).days

    # last 7 days of NAV rows + trades + signals
    cutoff = nav["date"].max() - pd.Timedelta(days=7)
    week_nav = nav[nav["date"] >= cutoff]
    trades = pd.read_csv(os.path.join(FORWARD_DIR, "trades.csv"))
    trades["date"] = pd.to_datetime(trades["date"])
    week_trades = trades[trades["date"] >= cutoff]
    signals = pd.read_csv(os.path.join(FORWARD_DIR, "signals.csv"))
    signals["date"] = pd.to_datetime(signals["date"])
    week_signals = signals[signals["date"] >= cutoff]

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

    if len(week_nav) >= 2:
        week_ret = (week_nav["nav"].iloc[-1] / week_nav["nav"].iloc[0] - 1) * 100
        week_bench = (week_nav["spy_bench_nav"].iloc[-1] /
                      week_nav["spy_bench_nav"].iloc[0] - 1) * 100
    else:
        week_ret = week_bench = 0.0

    regime_he = {"RISK_ON": "סיכון-פעיל (SPY מעל MA200)",
                 "RISK_OFF": "סיכון-כבוי (SPY מתחת/על MA200)"}

    html = f"""<html dir="rtl"><body style="font-family:Arial">
<h2>דוח שבועי — SmartPassive Forward Test</h2>
<p>ניסוי: <b>{state['experiment_id']}</b> | התחלה: {state['start_date']}
({days} ימים) | עודכן: {state['last_run_date']}</p>
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse">
<tr><td><b>NAV</b></td><td>${nav_now:,.2f}</td></tr>
<tr><td><b>תשואה מצטברת</b></td><td>{total_ret:+.2f}%</td></tr>
<tr><td><b>תשואת SPY (benchmark)</b></td><td>{bench_ret:+.2f}%</td></tr>
<tr><td><b>פער מול SPY</b></td><td>{total_ret-bench_ret:+.2f}%</td></tr>
<tr><td><b>תשואת השבוע</b></td><td>{week_ret:+.2f}% (SPY: {week_bench:+.2f}%)</td></tr>
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
מעל 3%. דוח מלא: forward/report.html ב-repository.</p>
</body></html>"""
    return html, total_ret


def send(html, total_ret):
    app_pw = os.environ.get("GMAIL_APP_PASSWORD")
    if not app_pw:
        raise RuntimeError("GMAIL_APP_PASSWORD env var not set")
    msg = MIMEMultipart("alternative")
    msg["Subject"] = (f"SmartPassive Weekly: NAV ${json.load(open(os.path.join(FORWARD_DIR,'state.json')))['nav']:,.0f} "
                      f"({total_ret:+.2f}%)")
    msg["From"] = GMAIL_USER
    msg["To"] = TO_ADDR
    msg["Date"] = formatdate(localtime=True)
    msg.attach(MIMEText(html, "html", "utf-8"))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
        server.login(GMAIL_USER, app_pw.replace(" ", ""))
        server.sendmail(GMAIL_USER, [a.strip() for a in TO_ADDR.split(",")],
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