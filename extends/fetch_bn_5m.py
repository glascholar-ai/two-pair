#!/usr/bin/env python3
"""Download full 5m kline history + funding history for all Binance equity perps."""
import json, time, urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "data" / "bn5m"
OUT.mkdir(parents=True, exist_ok=True)

info = json.load(urllib.request.urlopen("https://fapi.binance.com/fapi/v1/exchangeInfo"))
symbols = sorted(
    (s["symbol"], int(s["onboardDate"])) for s in info["symbols"]
    if s.get("contractType") == "TRADIFI_PERPETUAL" and s.get("status") == "TRADING"
    and s.get("underlyingType") in ("EQUITY", "PREMARKET", "KR_EQUITY", "HK_EQUITY", "JP_EQUITY"))
print(f"{len(symbols)} symbols", flush=True)

COLS = ["ts", "o", "h", "l", "c", "vol", "close_ts", "quote_vol", "trades", "tb_base", "tb_quote", "ig"]


def get(url, tries=6):
    for a in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                used = int(r.headers.get("X-MBX-USED-WEIGHT-1M", 0))
                d = json.load(r)
            if used > 2000:          # stay under the 2400/min weight cap
                time.sleep(30)
            return d
        except Exception as ex:
            print(f"retry {a}: {ex!r}", flush=True)
            time.sleep(5 * (a + 1))
    return None


fund_rows = []
for i, (sym, onboard) in enumerate(symbols):
    f = OUT / f"{sym}.parquet"
    if f.exists():
        continue
    rows, start = [], onboard
    while True:
        d = get(f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=5m"
                f"&startTime={start}&limit=1500")
        if not d:
            break
        rows.extend(d)
        if len(d) < 1500:
            break
        start = int(d[-1][0]) + 1
        time.sleep(0.3)
    if rows:
        df = pd.DataFrame(rows, columns=COLS)
        df = df[["ts", "o", "h", "l", "c", "vol", "quote_vol", "trades"]].astype(
            {"ts": "int64", "o": float, "h": float, "l": float, "c": float,
             "vol": float, "quote_vol": float, "trades": "int64"})
        df.to_parquet(f, index=False)
    fr = get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}&limit=1000")
    if fr:
        fund_rows.extend(fr)
    print(f"[{i+1}/{len(symbols)}] {sym}: {len(rows)} bars, {len(fr or [])} funding", flush=True)

pd.DataFrame(fund_rows).to_parquet(OUT / "_funding.parquet", index=False)
print("ALL_DONE", flush=True)
