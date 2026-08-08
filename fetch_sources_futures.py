"""Download research source PDFs/pages for the micro-futures intraday study."""
import os
import ssl
import urllib.request

os.makedirs("research_sources", exist_ok=True)

SOURCES = {
    # intraday momentum literature
    "gao_han_li_zhou_intraday_momentum.pdf":
        "https://researchmgt.monash.edu/ws/files/519509174/494419119_oa.pdf",
    "intraday_tsm_international_lancaster.pdf":
        "http://wp.lancs.ac.uk/fofi2020/files/2020/04/FoFI-2020-092-Zeming-Li.pdf",
    "intraday_tsm_centaur_reading.pdf":
        "https://centaur.reading.ac.uk/95566/1/Accepted-Version.pdf",
    # day-trader performance literature
    "day_trading_for_a_living_brazil.pdf":
        "https://ebicapital.nl/wp-content/uploads/2022/05/day-trading.pdf",
    "barber_lee_liu_odean_taiwan_day_traders.pdf":
        "https://faculty.haas.berkeley.edu/odean/papers/Day%20Traders/Day%20Trade%20040330.pdf",
    "odean_day_traders_learning.pdf":
        "https://faculty.haas.berkeley.edu/odean/papers/Day%20Traders/Day%20Trading%20and%20Learning%20110217.pdf",
    # multiple testing / Sharpe statistics
    "bailey_lopez_de_prado_deflated_sharpe.pdf":
        "https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf",
    "white_reality_check_2000.pdf":
        "https://www.ssc.wisc.edu/~bhansen/718/White2000.pdf",
    "mclean_pontiff_2016_anomaly_decay.pdf":
        "https://gwern.net/doc/economics/2016-mclean.pdf",
    # stop-loss / execution
    "cftc_stoploss_futures_study.pdf":
        "https://www.cftc.gov/sites/default/files/Stoploss_final_ada.pdf",
    # futures technical rules reality check
    "park_irwin_reality_check_futures.pdf":
        "https://farmdoc.illinois.edu/assets/meetings/nccc134/conf_2005/pdf/confp09-05.pdf",
}

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

log = []
for name, url in SOURCES.items():
    dest = os.path.join("research_sources", name)
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/pdf,*/*",
        })
        with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
            raw = r.read()
        with open(dest, "wb") as f:
            f.write(raw)
        log.append(f"{name}: OK {len(raw)} bytes (pdf={raw[:5] == b'%PDF-'})")
    except Exception as e:
        log.append(f"{name}: FAILED {type(e).__name__}: {e}")

with open(os.path.join("research_sources", "_fetch_log_futures.txt"), "w") as f:
    f.write("\n".join(log))
print("\n".join(log))