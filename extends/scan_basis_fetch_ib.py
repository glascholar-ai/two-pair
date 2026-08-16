#!/usr/bin/env python3
"""Download IBKR 5-min extended-hours bars (TRADES / BID_ASK, SMART + OVERNIGHT).

Cached to data/ib/<SYM>_5m_ext.parquet with columns:
  ts (UTC bar start), src ('SMART'|'OVERNIGHT'), what ('TRADES'|'BID_ASK'),
  o, h, l, c, vol, bar_count.
For BID_ASK bars IBKR semantics: o = time-avg bid, c = time-avg ask,
h = max ask, l = min bid.
Read-only API; pacing ~1 request / 8 s.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
from ib_insync import IB, Stock  # type: ignore[import-untyped]  # ib_insync ships no stubs

OUT = Path(__file__).parent / "data" / "ib"
OUT.mkdir(parents=True, exist_ok=True)

SYMS = ["MU", "NVDA", "TSLA", "AMD", "HOOD", "COIN", "PLTR", "MSTR", "SNDK", "WDC",
        "META", "AAPL", "SPY", "QQQ", "ORCL", "SOFI", "RDDT", "MSFT", "JPM"]
PACE_S = 3.0
OVN_DUR = "60 D"      # OVERNIGHT farm is slow; 120 D took >1h under contention


def _req(ib: IB, con: Stock, what: str, dur: str, end: str = "") -> pd.DataFrame:
    """One reqHistoricalData call -> DataFrame (may be empty)."""
    t0 = time.time()
    bars = ib.reqHistoricalData(con, endDateTime=end, durationStr=dur,
                                barSizeSetting="5 mins", whatToShow=what,
                                useRTH=False, formatDate=2, timeout=240)
    print(f"  {con.symbol} {con.exchange} {what} {dur} end={end!r}: {len(bars)} bars "
          f"{time.time() - t0:.0f}s", flush=True)
    time.sleep(PACE_S)
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame([{"ts": b.date, "o": b.open, "h": b.high, "l": b.low, "c": b.close,
                        "vol": b.volume, "bar_count": b.barCount} for b in bars])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["src"] = con.exchange
    df["what"] = what
    return df


def _chunks(ib: IB, con: Stock, what: str, n_chunks: int) -> pd.DataFrame:
    """Walk back in 30 D chunks (BID_ASK cannot do 120 D in one call)."""
    frames: List[pd.DataFrame] = []
    end = ""
    for _ in range(n_chunks):
        df = _req(ib, con, what, "30 D", end)
        if df.empty:
            break
        frames.append(df)
        first = df["ts"].min()
        end = (first - pd.Timedelta(minutes=5)).strftime("%Y%m%d-%H:%M:%S")
    return pd.concat(frames) if frames else pd.DataFrame()


def fetch_symbol(ib: IB, sym: str, ba_chunks: int = 1, ovn_ba_chunks: int = 0) -> pd.DataFrame:
    smart = Stock(sym, "SMART", "USD")
    ovn = Stock(sym, "OVERNIGHT", "USD")
    ib.qualifyContracts(smart)
    ovn_ok = bool(ib.qualifyContracts(ovn))
    parts: List[pd.DataFrame] = []
    parts.append(_req(ib, smart, "TRADES", "120 D"))
    parts.append(_chunks(ib, smart, "BID_ASK", ba_chunks))
    if ovn_ok:
        parts.append(_req(ib, ovn, "TRADES", OVN_DUR))
        if ovn_ba_chunks:
            parts.append(_chunks(ib, ovn, "BID_ASK", ovn_ba_chunks))
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    df = df.drop_duplicates(["ts", "src", "what"]).sort_values(["what", "src", "ts"])
    return df.reset_index(drop=True)


def load_ib(sym: str) -> Optional[pd.DataFrame]:
    f = OUT / f"{sym}_5m_ext.parquet"
    return pd.read_parquet(f) if f.exists() else None


def main(argv: List[str]) -> None:
    syms = argv[1:] or SYMS
    ib = IB()
    for attempt in range(60):
        try:
            ib.connect("127.0.0.1", 7496, clientId=107 + attempt % 3, timeout=60, readonly=True)
            break
        except Exception as ex:  # pylint: disable=broad-except
            print(f"connect attempt {attempt} failed: {ex!r}", flush=True)
            time.sleep(60)
    if not ib.isConnected():
        raise SystemExit("cannot connect to TWS")
    for sym in syms:
        f = OUT / f"{sym}_5m_ext.parquet"
        if f.exists():
            print(f"{sym}: cached", flush=True)
            continue
        t0 = time.time()
        try:
            df = fetch_symbol(ib, sym)
        except Exception as ex:  # pylint: disable=broad-except
            print(f"{sym}: FAILED {ex!r}", flush=True)
            continue
        df.to_parquet(f, index=False)
        summ = df.groupby(["what", "src"])["ts"].agg(["count", "min", "max"])
        print(f"{sym}: {len(df)} rows in {time.time() - t0:.0f}s\n{summ}", flush=True)
    ib.disconnect()


if __name__ == "__main__":
    main(sys.argv)
