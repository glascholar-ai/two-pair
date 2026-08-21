#!/usr/bin/env python3
"""Bulk-download Binance TradFi perp aggTrades (data.binance.vision daily zips)
for dynamic-carry candidates and cache 1-second last-price bars.

Input:  data/dyn/candidates.json (scan_dyn_select.py)
Output: data/dyn/bn1s/<SYMBOL>.parquet  cols: ts (s, UTC), px (last), qty (sum)
Window: last N_DAYS full UTC days (daily zips lag ~1 day).
"""
from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Set, cast

import pandas as pd

ROOT = Path(__file__).parent
DYN = ROOT / "data" / "dyn"
OUT = DYN / "bn1s"
BULK = "https://data.binance.vision/data/futures/um/daily/aggTrades"
N_DAYS = 32


def needed_symbols() -> List[str]:
    cand = json.loads((DYN / "candidates.json").read_text())
    syms: Set[str] = set()
    for r in cand["type_a"]:
        if r["venue"] == "BN" and r.get("bn_symbol"):
            syms.add(str(r["bn_symbol"]))
    for r in cand["type_b"]:
        if r.get("bn_symbol"):
            syms.add(str(r["bn_symbol"]))
    return sorted(syms)


def fetch_day(symbol: str, day: str) -> Optional[pd.DataFrame]:
    """One daily aggTrades zip -> raw frame; None if file absent (404)."""
    url = f"{BULK}/{symbol}/{symbol}-aggTrades-{day}.zip"
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 dyncarry"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                blob = resp.read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    else:
        return None
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = zf.namelist()[0]
        df = pd.read_csv(io.BytesIO(zf.read(name)))
    if "transact_time" not in df.columns:      # headerless fallback
        df.columns = ["agg_trade_id", "price", "quantity", "first_trade_id",
                      "last_trade_id", "transact_time", "is_buyer_maker"][
                          : len(df.columns)]
    return df


def to_1s(df: pd.DataFrame) -> pd.DataFrame:
    """AggTrades -> per-second last price + summed qty."""
    d = pd.DataFrame({
        "sec": df["transact_time"].astype("int64") // 1000,
        "px": df["price"].astype(float),
        "qty": df["quantity"].astype(float)})
    g = d.groupby("sec").agg(px=("px", "last"), qty=("qty", "sum"))
    out = cast(pd.DataFrame, g.reset_index()).rename(columns={"sec": "ts"})
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date()
    days = [(today - timedelta(days=i)).isoformat() for i in range(N_DAYS, 0, -1)]
    syms = needed_symbols()
    print(f"{len(syms)} symbols x {len(days)} days")
    for sym in syms:
        path = OUT / f"{sym}.parquet"
        have: Set[str] = set()
        frames: List[pd.DataFrame] = []
        if path.exists():
            old = pd.read_parquet(path)
            have = set(pd.to_datetime(old["ts"], unit="s", utc=True)
                       .dt.strftime("%Y-%m-%d"))
            frames.append(old)
        t0 = time.time()
        n_new = 0
        for day in days:
            if day in have:
                continue
            raw = fetch_day(sym, day)
            if raw is not None and len(raw):
                frames.append(to_1s(raw))
                n_new += 1
            time.sleep(0.1)
        if frames:
            allf = pd.concat(frames).drop_duplicates("ts").sort_values("ts")
            allf.reset_index(drop=True).to_parquet(path, index=False)
            print(f"{sym:16s} days+{n_new:2d} rows={len(allf):8d} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        else:
            print(f"{sym:16s} no data", flush=True)


if __name__ == "__main__":
    main()
