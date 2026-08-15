#!/usr/bin/env python3
"""Fetch Hyperliquid HIP-3 (dex "xyz") stock-perp 5m candles and cache as parquet.

Usage: python scan_offhours_hl.py [NAME ...]   (default: representative list)
Cache: data/hl/xyz_<NAME>_5m.parquet with cols ts, o, h, l, c, vol, trades.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List

import pandas as pd

CACHE = Path(__file__).parent / "data" / "hl"
API = "https://api.hyperliquid.xyz/info"
DEFAULT = ["MU", "NVDA", "TSLA", "AAPL", "AMD", "PLTR", "COIN", "MSTR", "HOOD", "CRWV",
           "SNDK", "AVGO", "META", "INTC", "ORCL", "GOOGL", "AMZN", "MSFT", "NFLX", "LLY",
           "COST", "RIVN", "GME", "HIMS", "MRVL", "RKLB", "LITE", "CRCL", "BX", "DKNG"]
CHUNK_MS = 15 * 86_400_000  # 15d of 5m bars = 4320 < API cap (~5000)
EARLIEST_MS = 1_772_323_200_000  # 2026-03-01


def post(payload: Dict[str, object]) -> object:
    req = urllib.request.Request(API, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def fetch_candles(coin: str, start_ms: int, end_ms: int) -> List[Dict[str, str]]:
    out = post({"type": "candleSnapshot",
                "req": {"coin": coin, "interval": "5m", "startTime": start_ms, "endTime": end_ms}})
    return out if isinstance(out, list) else []


def fetch_history(name: str) -> pd.DataFrame:
    coin = f"xyz:{name}"
    end = int(time.time() * 1000)
    frames: List[pd.DataFrame] = []
    empty_streak = 0
    while end > EARLIEST_MS and empty_streak < 2:
        start = end - CHUNK_MS
        rows = fetch_candles(coin, start, end)
        if not rows:
            empty_streak += 1
        else:
            empty_streak = 0
            frames.append(pd.DataFrame(rows))
        end = start - 1
        time.sleep(1.0)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    df = pd.DataFrame({"ts": df["t"].astype("int64"), "o": df["o"].astype(float),
                       "h": df["h"].astype(float), "l": df["l"].astype(float),
                       "c": df["c"].astype(float), "vol": df["v"].astype(float),
                       "trades": df["n"].astype("int64")})
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def main() -> None:
    names = sys.argv[1:] or DEFAULT
    CACHE.mkdir(parents=True, exist_ok=True)
    for n in names:
        p = CACHE / f"xyz_{n}_5m.parquet"
        if p.exists():
            print(f"{n}: cached ({len(pd.read_parquet(p))} bars)")
            continue
        df = pd.DataFrame()
        for attempt in range(3):
            try:
                df = fetch_history(n)
                break
            except Exception as ex:  # noqa: BLE001
                print(f"{n}: attempt {attempt} failed {ex}")
                time.sleep(20)
        if df.empty:
            print(f"{n}: no data")
            continue
        df.to_parquet(p, index=False)
        t0 = pd.to_datetime(df["ts"].iloc[0], unit="ms", utc=True)
        print(f"{n}: {len(df)} bars from {t0:%Y-%m-%d}")


if __name__ == "__main__":
    main()
