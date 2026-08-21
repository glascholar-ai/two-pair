#!/usr/bin/env python3
"""Fetch Hyperliquid candles for dynamic-carry candidates (HL legs).

candleSnapshot keeps ~5000 bars per interval: 15m covers ~52d (full backtest
window), 5m ~17d (recent refinement). Cache: data/dyn/hl/<dex>_<name>_{5m,15m}
.parquet, cols ts,o,h,l,c,vol,trades.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

ROOT = Path(__file__).parent
DYN = ROOT / "data" / "dyn"
OUT = DYN / "hl"
API = "https://api.hyperliquid.xyz/info"


def post(payload: Dict[str, Any], tries: int = 5) -> Any:
    body = json.dumps(payload).encode()
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                API, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except Exception as ex:  # noqa: BLE001
            print(f"  retry {attempt}: {ex!r}", flush=True)
            time.sleep(2.0 * (attempt + 1))
    return None


def needed_coins() -> List[str]:
    cand = json.loads((DYN / "candidates.json").read_text())
    coins: Set[str] = set()
    for r in cand["type_a"]:
        if r["venue"] == "HL" and r.get("hl_coin"):
            coins.add(str(r["hl_coin"]))
    for r in cand["type_b"]:
        if r.get("hl_coin"):
            coins.add(str(r["hl_coin"]))
    return sorted(coins)


def fetch_interval(coin: str, interval: str, span_ms: int) -> pd.DataFrame:
    now = int(time.time() * 1000)
    rows: List[Dict[str, Any]] = []
    end = now
    step = span_ms // 3
    while end > now - span_ms:
        out = post({"type": "candleSnapshot",
                    "req": {"coin": coin, "interval": interval,
                            "startTime": end - step, "endTime": end}})
        if isinstance(out, list) and out:
            rows.extend(out)
        end -= step + 1
        time.sleep(0.25)
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    df = pd.DataFrame({
        "ts": raw["t"].astype("int64"),
        "o": raw["o"].astype(float), "h": raw["h"].astype(float),
        "l": raw["l"].astype(float), "c": raw["c"].astype(float),
        "vol": raw["v"].astype(float), "trades": raw["n"].astype("int64")})
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    coins = needed_coins()
    print(f"{len(coins)} HL coins")
    for coin in coins:
        stem = coin.replace(":", "_")
        for interval, span in (("15m", 53 * 86_400_000), ("5m", 18 * 86_400_000)):
            df = fetch_interval(coin, interval, span)
            if len(df):
                df.to_parquet(OUT / f"{stem}_{interval}.parquet", index=False)
            print(f"{coin:16s} {interval}: {len(df)} bars", flush=True)


if __name__ == "__main__":
    main()
