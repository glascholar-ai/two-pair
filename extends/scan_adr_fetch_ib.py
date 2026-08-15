#!/usr/bin/env python3
"""Fetch 5m bars from IBKR TWS (read-only) for home-market lines & FX, with cache.

Usage:  python3 scan_adr_fetch_ib.py [name ...]   (default: all in CONTRACTS)
Cache:  data/ib/<name>_5m.parquet   columns: ts (UTC ms), o,h,l,c,vol
Pacing: ~2s sleep between requests; each name pulled in chunks of DUR days.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from ib_insync import IB, Contract, Forex, Index, Stock, util

IB_DIR = Path(__file__).parent / "data" / "ib"
IB_DIR.mkdir(parents=True, exist_ok=True)

# name -> (contract, whatToShow)
CONTRACTS: Dict[str, Tuple[Contract, str]] = {
    "9988_HK": (Stock("9988", "SEHK", "HKD"), "TRADES"),
    "700_HK": (Stock("700", "SEHK", "HKD"), "TRADES"),
    "1810_HK": (Stock("1810", "SEHK", "HKD"), "TRADES"),
    "2330_TW": (Stock("2330", "TWSE", "TWD"), "TRADES"),
    "0050_TW": (Stock("0050", "TWSE", "TWD"), "TRADES"),
    "6758_T": (Stock("6758", "TSEJ", "JPY"), "TRADES"),
    "NOVOB_CPH": (Stock("NOVO.B", "CPH", "DKK"), "TRADES"),
    "ASML_AEB": (Stock("ASML", "AEB", "EUR"), "TRADES"),
    "005930_KRX": (Stock("005930", "KRX", "KRW"), "TRADES"),
    "000660_KRX": (Stock("000660", "KRX", "KRW"), "TRADES"),
    "005380_KRX": (Stock("005380", "KRX", "KRW"), "TRADES"),
    "K200_IDX": (Index("K200", "KSE"), "TRADES"),
    "N225_IDX": (Index("N225", "OSE.JPN"), "TRADES"),
    "HSI_IDX": (Index("HSI", "HKFE"), "TRADES"),
    "USDKRW": (Forex("USDKRW"), "MIDPOINT"),
    "USDHKD": (Forex("USDHKD"), "MIDPOINT"),
    "USDJPY": (Forex("USDJPY"), "MIDPOINT"),
    "EURUSD": (Forex("EURUSD"), "MIDPOINT"),
    "USDDKK": (Forex("USDDKK"), "MIDPOINT"),
    "USDCNH": (Forex("USDCNH"), "MIDPOINT"),
}
DUR = "10 D"          # per-request chunk for 5m bars
CHUNKS = 7            # 7 x 10 D ~ 70 calendar days


def load_cache(name: str) -> Optional[pd.DataFrame]:
    p = IB_DIR / f"{name}_5m.parquet"
    return pd.read_parquet(p) if p.exists() else None


def bars_to_df(bars: List) -> pd.DataFrame:
    df = util.df(bars)
    if df is None or df.empty:
        return pd.DataFrame({k: [] for k in ["ts", "o", "h", "l", "c", "vol"]})
    ts = pd.to_datetime(df["date"], utc=True)
    out = pd.DataFrame({
        "ts": (ts.astype("int64") // 10**6).values,
        "o": df["open"].values, "h": df["high"].values, "l": df["low"].values,
        "c": df["close"].values, "vol": df["volume"].values,
    })
    return out


def fetch_one(ib: IB, name: str, contract: Contract, what: str,
              chunks: int = CHUNKS) -> pd.DataFrame:
    """Pull `chunks` x DUR of 5m bars ending now, walking backwards; merge with cache."""
    ib.qualifyContracts(contract)
    frames: List[pd.DataFrame] = []
    cached = load_cache(name)
    if cached is not None and len(cached):
        frames.append(cached)
    end = ""
    for i in range(chunks):
        try:
            bars = ib.reqHistoricalData(
                contract, endDateTime=end, durationStr=DUR, barSizeSetting="5 mins",
                whatToShow=what, useRTH=False, formatDate=2, timeout=60)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name} chunk {i}: error {exc}")
            break
        df = bars_to_df(bars)
        print(f"  {name} chunk {i}: {len(df)} bars"
              + (f" {pd.to_datetime(df.ts.min(), unit='ms')} .. {pd.to_datetime(df.ts.max(), unit='ms')}"
                 if len(df) else ""))
        if not len(df):
            break
        frames.append(df)
        first = pd.to_datetime(int(df["ts"].min()), unit="ms", utc=True)
        # if we already have cache covering before `first`, stop
        if cached is not None and len(cached) and int(cached["ts"].max()) >= int(df["ts"].min()):
            # only need to fill forward gap; but keep going if cache is short
            if int(cached["ts"].min()) <= int(df["ts"].min()):
                break
        end = first.strftime("%Y%m%d-%H:%M:%S")
        time.sleep(6.0)
    if not frames:
        return pd.DataFrame({k: [] for k in ["ts", "o", "h", "l", "c", "vol"]})
    allb = pd.concat(frames).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    allb.to_parquet(IB_DIR / f"{name}_5m.parquet", index=False)
    return allb


def main(argv: List[str]) -> None:
    names = argv[1:] or list(CONTRACTS)
    ib = IB()
    ib.connect("127.0.0.1", 7496, clientId=113, timeout=20, readonly=True)
    for n in names:
        c, what = CONTRACTS[n]
        print(f"== {n}")
        df = fetch_one(ib, n, c, what)
        if len(df):
            print(f"   total {len(df)} bars, "
                  f"{pd.to_datetime(df.ts.min(), unit='ms')} .. {pd.to_datetime(df.ts.max(), unit='ms')}")
        time.sleep(6.0)
    ib.disconnect()


if __name__ == "__main__":
    main(sys.argv)
