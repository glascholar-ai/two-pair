#!/usr/bin/env python3
"""Download Binance funding rate history for all TradFi (stock) perpetuals, past 2 months."""
import json, time, csv, os
import urllib.request, urllib.parse

BASE = "https://fapi.binance.com"
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

ex = json.load(open(os.path.join(os.path.dirname(__file__), "exchangeInfo.json")))
stocks = [
    {"symbol": s["symbol"], "type": s["underlyingType"], "onboard": s["onboardDate"]}
    for s in ex["symbols"]
    if s["underlyingType"] in ("EQUITY", "KR_EQUITY", "HK_EQUITY")
    and s["status"] == "TRADING"
]
print(f"{len(stocks)} stock perps")

now_ms = int(time.time() * 1000)
start_ms = now_ms - 61 * 24 * 3600 * 1000  # ~2 months

def fetch(symbol, start, end):
    rows = []
    cur = start
    while True:
        q = urllib.parse.urlencode({"symbol": symbol, "startTime": cur, "endTime": end, "limit": 1000})
        with urllib.request.urlopen(f"{BASE}/fapi/v1/fundingRate?{q}", timeout=30) as r:
            batch = json.load(r)
        rows.extend(batch)
        if len(batch) < 1000:
            break
        cur = batch[-1]["fundingTime"] + 1
    return rows

all_rows = []
for i, s in enumerate(stocks):
    sym = s["symbol"]
    rows = fetch(sym, max(start_ms, s["onboard"]), now_ms)
    for r in rows:
        all_rows.append({
            "symbol": sym,
            "underlyingType": s["type"],
            "fundingTime": r["fundingTime"],
            "fundingRate": r["fundingRate"],
            "markPrice": r.get("markPrice", ""),
        })
    print(f"[{i+1}/{len(stocks)}] {sym}: {len(rows)} records")
    time.sleep(0.15)

out = os.path.join(OUT_DIR, "funding_history.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["symbol", "underlyingType", "fundingTime", "fundingRate", "markPrice"])
    w.writeheader()
    w.writerows(all_rows)
print(f"saved {len(all_rows)} rows -> {out}")
