#!/usr/bin/env python3
"""Self-contained funding cache for the dynamic-carry study — fetched straight
from Binance fapi and the Hyperliquid API, independent of perpfund.db (which
may be mid-migration in another session, and which keys one dex per ticker
while e.g. AVGO funding differs materially between xyz and para).

Output:
    data/dyn/funding_bn.parquet   cols ticker, ts, rate      (31d, all TradFi)
    data/dyn/funding_hl.parquet   cols ticker, dex, coin, ts, rate (31d, all
                                  stock-dex coins on xyz+para)
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).parent
DYN = ROOT / "data" / "dyn"
BN_FAPI = "https://fapi.binance.com"
HL_API = "https://api.hyperliquid.xyz/info"
HL_DEXES = ("xyz", "para")
CANONICAL = {"STXX": "STX", "SKHX": "SKHYNIX", "SMSN": "SAMSUNG"}
DAYS = 31
HL_EXCLUDE = {
    "GOLD", "SILVER", "CL", "COPPER", "NATGAS", "PLATINUM", "PALLADIUM",
    "BRENTOIL", "ALUMINIUM", "CORN", "10Y", "JPY", "EUR", "GBP",
    "XYZ100", "KR200", "JP225", "SP500", "TOTAL2", "OTHERS", "BTCD", "PURRDAT",
}


def get(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def post(payload: Dict[str, Any], tries: int = 4) -> Any:
    body = json.dumps(payload).encode()
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                HL_API, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as ex:  # noqa: BLE001
            print(f"  retry {attempt}: {ex!r}", flush=True)
            time.sleep(2.0 * (attempt + 1))
    return None


def fetch_bn(now: int) -> pd.DataFrame:
    info = get(f"{BN_FAPI}/fapi/v1/exchangeInfo")
    syms: List[Tuple[str, str]] = []
    for s in info["symbols"]:
        if (s.get("contractType") == "TRADIFI_PERPETUAL"
                and s.get("status") == "TRADING"):
            base = str(s["baseAsset"])
            syms.append((str(s["symbol"]), CANONICAL.get(base, base)))
    rows: List[Dict[str, Any]] = []
    start = now - DAYS * 86_400_000
    for sym, ticker in syms:
        try:
            data = get(f"{BN_FAPI}/fapi/v1/fundingRate?symbol={sym}"
                       f"&startTime={start}&limit=1000")
        except Exception as ex:  # noqa: BLE001
            print(f"  bn {sym}: {ex!r}")
            continue
        rows.extend({"ticker": ticker, "ts": int(r["fundingTime"]),
                     "rate": float(r["fundingRate"])} for r in data)
        time.sleep(0.12)
    df = pd.DataFrame(rows).drop_duplicates(["ticker", "ts"])
    return df.sort_values(["ticker", "ts"]).reset_index(drop=True)


def fetch_hl(now: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    start = now - DAYS * 86_400_000
    for dex in HL_DEXES:
        meta = post({"type": "meta", "dex": dex})
        for u in meta["universe"]:
            if u.get("isDelisted"):
                continue
            coin = str(u["name"])
            base = coin.split(":", 1)[1]
            if base in HL_EXCLUDE:
                continue
            ticker = CANONICAL.get(base, base)
            cursor = start
            while True:
                out = post({"type": "fundingHistory", "coin": coin,
                            "startTime": cursor})
                if not isinstance(out, list) or not out:
                    break
                rows.extend({"ticker": ticker, "dex": dex, "coin": coin,
                             "ts": int(r["time"]),
                             "rate": float(r["fundingRate"])} for r in out)
                last = int(out[-1]["time"])
                if len(out) < 500 or last <= cursor:
                    break
                cursor = last + 1
            time.sleep(0.15)
        print(f"  {dex} done", flush=True)
    df = pd.DataFrame(rows).drop_duplicates(["coin", "ts"])
    return df.sort_values(["coin", "ts"]).reset_index(drop=True)


def main() -> None:
    DYN.mkdir(parents=True, exist_ok=True)
    now = int(time.time() * 1000)
    bn = fetch_bn(now)
    bn.to_parquet(DYN / "funding_bn.parquet", index=False)
    print(f"bn: {len(bn)} rows, {bn['ticker'].nunique()} tickers")
    hl = fetch_hl(now)
    hl.to_parquet(DYN / "funding_hl.parquet", index=False)
    print(f"hl: {len(hl)} rows, {hl['coin'].nunique()} coins")


if __name__ == "__main__":
    main()
