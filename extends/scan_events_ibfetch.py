#!/usr/bin/env python3
"""Fetch ES / NQ continuous-future 5m bars (24h, TRADES) from IBKR TWS and cache to parquet.

Read-only. Steps back in 30-day chunks from now to START. Sleeps >=2s between requests.
Output: data/ib/<SYM>_5m.parquet with columns ts (UTC ms), o, h, l, c, vol.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, cast

import pandas as pd
from ib_insync import IB, Future, util

OUT = Path(__file__).parent / "data" / "ib"
OUT.mkdir(parents=True, exist_ok=True)


def _utc(d: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(d, tz="UTC"))


START = _utc("2026-03-20")


def fetch(ib: IB, sym: str, expiry: str, start: pd.Timestamp,
          end: pd.Timestamp) -> pd.DataFrame:
    """Download 5m bars for one specific futures contract between start and end."""
    c = Future(sym, expiry, "CME", includeExpired=True)
    ib.qualifyContracts(c)
    frames: List[pd.DataFrame] = []
    while end > start:
        bars = ib.reqHistoricalData(
            c, endDateTime=end.strftime("%Y%m%d-%H:%M:%S"), durationStr="30 D",
            barSizeSetting="5 mins", whatToShow="TRADES", useRTH=False, formatDate=2)
        time.sleep(2.5)
        if not bars:
            print(f"{sym}: empty chunk ending {end}")
            break
        df = util.df(bars)
        if df is None or df.empty:
            break
        df = df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c",
                                "volume": "vol"})
        df["ts"] = pd.to_datetime(df["date"], utc=True)
        frames.append(cast(pd.DataFrame, df[["ts", "o", "h", "l", "c", "vol"]]))
        first = cast(pd.Timestamp, cast(pd.Series, df["ts"]).min())
        print(f"{sym}: {len(df)} bars {first} .. {df['ts'].max()}", flush=True)
        if first <= start or first >= end:
            break
        end = first
    out = pd.concat(frames).drop_duplicates("ts").sort_values("ts")
    out["ts"] = (out["ts"].astype("int64") // 10**6).astype("int64")
    return out.reset_index(drop=True)


def main() -> None:
    ib = IB()
    ib.connect("127.0.0.1", 7496, clientId=133, readonly=True, timeout=20)
    try:
        for sym in ["ES", "NQ"]:
            f = OUT / f"{sym}_5m.parquet"
            # front contract per period (roll ~1 week before 3rd-Friday expiry)
            a = fetch(ib, sym, "202606", START, _utc("2026-06-13"))
            b = fetch(ib, sym, "202609", _utc("2026-06-05"),
                      cast(pd.Timestamp, pd.Timestamp.utcnow().ceil("1D")))
            a["contract"], b["contract"] = "M6", "U6"
            df = pd.concat([a, b]).sort_values(["ts", "contract"]).reset_index(drop=True)
            df.to_parquet(f, index=False)
            print(f"saved {f} rows={len(df)}")
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
