"""Download research source PDFs for the options-income study."""
import os
import ssl
import urllib.request

os.makedirs("research_sources", exist_ok=True)

SOURCES = {
    "vrp_decline_dewbecker.pdf": "https://www.dew-becker.org/documents/option_decline.pdf",
    "chicagofed_wp2025-17_vrp.pdf": "https://www.chicagofed.org/-/media/publications/working-papers/2025/wp2025-17.pdf",
    "cboe_benchmarks_factsheet.pdf": "https://cdn.cboe.com/resources/education/research_publications/benchmarks-fact-sheet.pdf",
    "cboe_putwrite_methodology.pdf": "https://cdn.cboe.com/api/global/us_indices/governance/Cboe_PutWrite_Indices_Methodology.pdf",
    "cboe_buywrite_methodology.pdf": "https://cdn.cboe.com/api/global/us_indices/governance/Cboe_BuyWrite_Indices_Methodology.pdf",
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
        with urllib.request.urlopen(req, timeout=40, context=ctx) as r:
            raw = r.read()
        with open(dest, "wb") as f:
            f.write(raw)
        head = raw[:5]
        log.append(f"{name}: OK {len(raw)} bytes (pdf={head == b'%PDF-'})")
    except Exception as e:
        log.append(f"{name}: FAILED {type(e).__name__}: {e}")

with open(os.path.join("research_sources", "_fetch_log.txt"), "w") as f:
    f.write("\n".join(log))
print("\n".join(log))