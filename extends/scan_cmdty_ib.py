#!/usr/bin/env python3
"""Fetch IBKR reference prices for commodity/FX perps (READ-ONLY, port 7496).

Contracts: COMEX GC/SI/HG, NYMEX CL/NG (front continuous future) plus spot
XAUUSD/XAGUSD/EURUSD/GBPUSD/USDJPY midpoint. 5-min bars, useRTH=False.
Cache: data/ib/<KEY>_5m.parquet with cols ts (ms, bar start UTC), o,h,l,c,vol
and data/ib/<KEY>_meta.json (resolved contract, expiry).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from ib_insync import IB, Contract, Forex, Future, util

CACHE = Path(__file__).parent / "data" / "ib"
DURATION = "30 D"
CLIENT_ID = 123

FUTS: List[Tuple[str, str, str]] = [("GC", "GC", "COMEX"), ("SI", "SI", "COMEX"),
                                    ("HG", "HG", "COMEX"), ("CL", "CL", "NYMEX"),
                                    ("NG", "NG", "NYMEX")]
FX: List[str] = ["XAUUSD", "XAGUSD", "EURUSD", "GBPUSD", "USDJPY"]


def bars_to_df(bars: List[Any]) -> pd.DataFrame:
    df = util.df(bars)
    if df is None or df.empty:
        return pd.DataFrame()
    ts = pd.to_datetime(df["date"], utc=True)
    return pd.DataFrame({"ts": (ts.astype("int64") // 1_000_000).astype("int64"),
                         "o": df["open"].astype(float), "h": df["high"].astype(float),
                         "l": df["low"].astype(float), "c": df["close"].astype(float),
                         "vol": df["volume"].astype(float)})


def front_future(ib: IB, sym: str, exch: str) -> Optional[Contract]:
    cds = ib.reqContractDetails(Future(sym, exchange=exch))
    time.sleep(1)
    cds = [c for c in cds if c.contract is not None and c.contract.lastTradeDateOrContractMonth]
    if not cds:
        return None
    today = time.strftime("%Y%m%d")
    # front = earliest expiry with >= 8 days left, to avoid delivery-month illiquidity
    def days_left(cd: Any) -> int:
        d = cd.contract.lastTradeDateOrContractMonth[:8]
        return int((pd.Timestamp(d) - pd.Timestamp(today)).days)
    live = sorted((c for c in cds if days_left(c) >= 8), key=days_left)
    return live[0].contract if live else None


def fetch_hist(ib: IB, contract: Contract, what: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    end = ""
    for _ in range(3):  # up to 3 x 30 D
        bars = ib.reqHistoricalData(contract, endDateTime=end, durationStr=DURATION,
                                    barSizeSetting="5 mins", whatToShow=what, useRTH=False,
                                    formatDate=2)
        time.sleep(2.5)
        df = bars_to_df(bars)
        if df.empty:
            break
        frames.append(df)
        first = pd.to_datetime(int(df["ts"].min()), unit="ms", utc=True)
        end = f"{first:%Y%m%d-%H:%M:%S}"
        if len(df) < 200:
            break
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def save(key: str, df: pd.DataFrame, meta: Dict[str, Any]) -> None:
    df.to_parquet(CACHE / f"{key}_5m.parquet", index=False)
    (CACHE / f"{key}_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"{key}: {len(df)} bars  {meta}", flush=True)


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else CLIENT_ID
    ib = IB()
    ib.connect("127.0.0.1", 7496, clientId=cid, readonly=True, timeout=20)
    try:
        for key, sym, exch in FUTS:
            if (CACHE / f"{key}_5m.parquet").exists():
                continue
            c = front_future(ib, sym, exch)
            if c is None:
                print(f"{key}: no contract", flush=True)
                continue
            df = fetch_hist(ib, c, "TRADES")
            save(key, df, {"symbol": sym, "exchange": exch, "conId": c.conId,
                           "expiry": c.lastTradeDateOrContractMonth,
                           "localSymbol": c.localSymbol, "multiplier": c.multiplier})
        for pair in FX:
            key = f"FX_{pair}"
            if (CACHE / f"{key}_5m.parquet").exists():
                continue
            fx = Forex(pair)
            ib.qualifyContracts(fx)
            time.sleep(1)
            df = fetch_hist(ib, fx, "MIDPOINT")
            if df.empty:
                print(f"{key}: no data", flush=True)
                continue
            save(key, df, {"pair": pair, "conId": fx.conId})
    finally:
        ib.disconnect()
    print("IB_DONE", flush=True)


if __name__ == "__main__":
    main()
