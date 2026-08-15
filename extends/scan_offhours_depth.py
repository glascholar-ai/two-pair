#!/usr/bin/env python3
"""Capacity check for off-hours liquidity provision on Binance stock perps.

1. Snapshot orderbook depth (fapi /depth) for representative names NOW and
   compute $ depth within 25/50/100 bps of mid on each side.
2. Compare 5m quote_vol per bar in DEAD (00-08 UTC) vs RTH (13:30-20:00) from
   the local parquet history.
Snapshots are cached under data/hl/depth_<UTC-stamp>.json.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, cast

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
D = HERE / "data" / "bn5m"
CACHE = HERE / "data" / "hl"
NAMES = ["MU", "NVDA", "TSLA", "AAPL", "AMD", "PLTR", "COIN", "MSTR", "HOOD", "CRWV",
         "SNDK", "AVGO", "META", "INTC", "ORCL"]
BANDS = (25, 50, 100)


def fetch_depth(sym: str) -> Dict[str, object]:
    url = f"https://fapi.binance.com/fapi/v1/depth?symbol={sym}USDT&limit=500"
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.load(r)


def depth_row(sym: str, book: Dict[str, object]) -> Dict[str, float]:
    bids = np.array(book["bids"], dtype=float)
    asks = np.array(book["asks"], dtype=float)
    mid = (bids[0, 0] + asks[0, 0]) / 2
    row: Dict[str, float] = {"mid": mid, "spread_bps": (asks[0, 0] - bids[0, 0]) / mid * 1e4}
    for b in BANDS:
        lo, hi = mid * (1 - b / 1e4), mid * (1 + b / 1e4)
        row[f"bid_{b}"] = float((bids[bids[:, 0] >= lo, 0] * bids[bids[:, 0] >= lo, 1]).sum())
        row[f"ask_{b}"] = float((asks[asks[:, 0] <= hi, 0] * asks[asks[:, 0] <= hi, 1]).sum())
    row["bid_full"] = float((bids[:, 0] * bids[:, 1]).sum())
    row["ask_full"] = float((asks[:, 0] * asks[:, 1]).sum())
    row["bid_worst_bps"] = float((mid - bids[-1, 0]) / mid * 1e4)
    row["ask_worst_bps"] = float((asks[-1, 0] - mid) / mid * 1e4)
    return row


def snapshot(names: List[str]) -> pd.DataFrame:
    stamp = datetime.now(timezone.utc)
    rows: Dict[str, Dict[str, float]] = {}
    raw: Dict[str, object] = {"utc": stamp.isoformat()}
    for s in names:
        try:
            book = fetch_depth(s)
        except Exception as ex:  # noqa: BLE001
            print(f"  {s}: fetch failed {ex}", file=sys.stderr)
            continue
        raw[s] = book
        rows[s] = depth_row(s, book)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"depth_{stamp.strftime('%Y%m%dT%H%M%SZ')}.json").write_text(json.dumps(raw))
    df = pd.DataFrame(rows).T
    print(f"orderbook snapshot at {stamp:%Y-%m-%d %H:%M:%S} UTC ({stamp:%A})")
    return df


def volume_by_segment(names: List[str]) -> pd.DataFrame:
    """Median quote_vol ($) per 5m bar and per-hour totals, DEAD vs RTH vs AH, weekdays."""
    rows = []
    for s in names:
        p = D / f"{s}USDT.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        idx = pd.DatetimeIndex(pd.to_datetime(cast(pd.Series, df["ts"]), unit="ms", utc=True))
        qv = pd.Series(cast(pd.Series, df["quote_vol"]).to_numpy(dtype=float), index=idx)
        dt = idx.to_series().dt
        qv = cast(pd.Series, qv[dt.dayofweek.to_numpy() < 5])
        dt = pd.DatetimeIndex(qv.index).to_series().dt
        minute = dt.hour.to_numpy() * 60 + dt.minute.to_numpy()
        segs = {"DEAD": (minute >= 0) & (minute < 480),
                "PRE": (minute >= 480) & (minute < 810),
                "RTH": (minute >= 810) & (minute < 1200),
                "AH": minute >= 1200}
        row: Dict[str, float] = {"days": float(len(qv) / 288)}
        for k, m in segs.items():
            row[f"{k}_med_bar"] = float(cast(pd.Series, qv[m]).median())
            row[f"{k}_mean_bar"] = float(cast(pd.Series, qv[m]).mean())
        rows.append(pd.Series(row, name=s))
    out = pd.DataFrame(rows)
    out["DEAD/RTH_med"] = out["DEAD_med_bar"] / out["RTH_med_bar"]
    return out


def main() -> None:
    pd.set_option("display.width", 200)
    dep = snapshot(NAMES)
    cols = ["mid", "spread_bps"] + [f"{s}_{b}" for b in BANDS for s in ("bid", "ask")] + \
        ["bid_full", "ask_full", "bid_worst_bps", "ask_worst_bps"]
    print(cast(pd.DataFrame, dep[cols]).round(0).to_string())
    print("\nmedian across names ($):")
    med = cast(pd.DataFrame, dep[[f"{s}_{b}" for b in BANDS for s in ("bid", "ask")]]).median()
    print(cast(pd.Series, med).round(0))
    print("\n== quote_vol per 5m bar by segment (weekdays, $) ==")
    vol = volume_by_segment(NAMES)
    print(vol.round(3).to_string())
    ratio = float(cast(pd.Series, vol["DEAD/RTH_med"]).median())
    dead_bar = float(cast(pd.Series, vol["DEAD_med_bar"]).median())
    print(f"\nmedian DEAD/RTH ratio: {ratio:.3f}")
    print(f"median DEAD $/5m bar: {dead_bar:.0f} -> $/hour: {dead_bar * 12:.0f}")


if __name__ == "__main__":
    main()
