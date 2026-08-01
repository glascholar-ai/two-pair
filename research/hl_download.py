#!/usr/bin/env python3
"""Download Hyperliquid HIP-3 stock-perp funding history (hourly), past 2 months."""
import json, time, csv, os
import urllib.request, urllib.error

API = "https://api.hyperliquid.xyz/info"
OUT = os.path.join(os.path.dirname(__file__), "data", "hl_funding_history.csv")
PAUSE = 1.0

NON_EQUITY = {  # indices / FX / commodities on the xyz & mkts dexes
    "XYZ100", "GOLD", "JPY", "EUR", "GBP", "SILVER", "CL", "COPPER", "NATGAS",
    "PLATINUM", "PALLADIUM", "KR200", "JP225", "BRENTOIL", "SP500", "DRAM",
    "PURRDAT", "US500", "USTECH",
}
KR = {"SKHX", "SMSN", "HYUNDAI", "SKHY"}
JP = {"SOFTBANK", "KIOXIA"}
HK = {"MINIMAX", "ZHIPU"}
CN = {"GIGADEV", "CXMT", "SHAZ"}

def market_of(base):
    for mkt, s in [("KR", KR), ("JP", JP), ("HK", HK), ("CN", CN)]:
        if base in s:
            return mkt
    return "US"

def post(payload):
    for attempt in range(8):
        try:
            req = urllib.request.Request(API, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 10 * (attempt + 1)
                print(f"  429, backing off {wait}s")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("too many 429s")

coins = []
for dex in ("xyz", "para"):
    meta = json.load(open(f"hl_meta_{dex}.json"))
    for a in meta["universe"]:
        if a.get("isDelisted"):
            continue
        base = a["name"].split(":", 1)[1]
        if base not in NON_EQUITY:
            coins.append(a["name"])
print(f"{len(coins)} equity perps")

done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        done = {r["coin"] for r in csv.DictReader(f)}
    print(f"resume: {len(done)} coins already fetched")
else:
    with open(OUT, "w", newline="") as f:
        csv.writer(f).writerow(["coin", "market", "time", "fundingRate", "premium"])

now_ms = int(time.time() * 1000)
start_ms = now_ms - 61 * 24 * 3600 * 1000

for i, coin in enumerate(coins):
    if coin in done:
        continue
    rows, cur = [], start_ms
    while True:
        batch = post({"type": "fundingHistory", "coin": coin, "startTime": cur, "endTime": now_ms})
        if not batch:
            break
        mkt = market_of(coin.split(":", 1)[1])
        rows.extend({"coin": coin, "market": mkt, "time": r["time"],
                     "fundingRate": r["fundingRate"], "premium": r["premium"]} for r in batch)
        if len(batch) < 500:
            break
        cur = batch[-1]["time"] + 1
        time.sleep(PAUSE)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["coin", "market", "time", "fundingRate", "premium"])
        w.writerows(rows)
    print(f"[{i+1}/{len(coins)}] {coin}: {len(rows)}")
    time.sleep(PAUSE)
print("done")
