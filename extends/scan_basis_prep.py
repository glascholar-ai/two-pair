#!/usr/bin/env python3
"""Shared loaders for the cross-venue (Binance perp vs IBKR stock) basis scans.

Merged 5-min frame per symbol (index = UTC bar start):
  bn        Binance perp close
  ib_last   IBKR last trade close (SMART 08:00-24:00 UTC, OVERNIGHT 00:00-08:00);
            NaN when the bar had no IBKR trade (stale print) or no IB session
  ib_last_ff  forward-filled ib_last (stale allowed)
  ib_vol    IBKR bar volume (shares)
  ib_bid / ib_ask / ib_mid / spread_bps   from BID_ASK bars (time-averaged)
  basis     ln(bn / ib_last)          [bps]
  basis_mid ln(bn / ib_mid)           [bps]  (mid = time-avg over the bar)
  basis_midc ln(bn / centred mid)      [bps]  centred mid = avg(mid_t, mid_t+1) ~ quote at bar close
  seg       AH | DEAD | PRE | OPEN | REST
  tday      US trading day the bar belongs to (AH/DEAD/PRE roll to next day)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
BN = ROOT / "data" / "bn5m"
IBD = ROOT / "data" / "ib"
SEG = [("AH", 20 * 60, 24 * 60), ("DEAD", 0, 8 * 60), ("PRE", 8 * 60, 13 * 60 + 30),
       ("OPEN", 13 * 60 + 30, 14 * 60 + 30), ("REST", 14 * 60 + 30, 20 * 60)]
SEG_ORDER = ["AH", "DEAD", "PRE", "OPEN", "REST"]
DST_START = pd.Timestamp("2026-03-09", tz="UTC")   # keep EDT-only sample (segments in UTC)


def load_bn(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(BN / f"{sym}USDT.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["ts"], unit="ms", utc=True))
    df = df[~df.index.duplicated()].sort_index()
    return cast(pd.DataFrame, df[["o", "h", "l", "c", "vol", "quote_vol", "trades"]])


def load_ib(sym: str) -> Optional[pd.DataFrame]:
    f = IBD / f"{sym}_5m_ext.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def minute_of_day(idx: pd.DatetimeIndex) -> np.ndarray:
    """UTC minute-of-day (0..1439) as int array (typed alternative to idx.hour/minute)."""
    return (np.asarray(idx.asi8, dtype=np.int64) // 60_000_000_000) % 1440


def segment_of(idx: pd.DatetimeIndex) -> pd.Series:
    minute = minute_of_day(idx)
    out = pd.Series("REST", index=idx, dtype=object)
    for name, a, b in SEG:
        out[(minute >= a) & (minute < b)] = name
    return out


def trading_day(idx: pd.DatetimeIndex) -> pd.Series:
    """US trading day a bar belongs to (bars from 20:00 UTC roll into the next day)."""
    minute = minute_of_day(idx)
    day0 = idx - pd.to_timedelta(minute, unit="m")
    day = pd.Series(day0, index=idx)
    day = day.where(minute < 20 * 60, day + pd.Timedelta(days=1))
    dow = (np.asarray(pd.DatetimeIndex(day).asi8, dtype=np.int64) // 86_400_000_000_000 + 3) % 7
    # Friday AH -> Saturday -> roll to Monday; Sunday bars -> Monday
    day = day + pd.to_timedelta(np.where(dow == 5, 2, np.where(dow == 6, 1, 0)), unit="D")
    return day


def merge_symbol(sym: str) -> Optional[pd.DataFrame]:
    """Return merged 5m frame or None if IBKR cache missing."""
    ib = load_ib(sym)
    if ib is None:
        return None
    bn = load_bn(sym)
    tr = cast(pd.DataFrame, ib[ib["what"] == "TRADES"])
    ba = cast(pd.DataFrame, ib[ib["what"] == "BID_ASK"])
    # SMART covers 08:00-24:00, OVERNIGHT 00:00-08:00; prefer SMART where both exist
    tr = tr.sort_values(by=["ts", "src"], ascending=[True, False])  # SMART > OVERNIGHT
    tr = tr.drop_duplicates("ts", keep="first").set_index("ts")
    ba = ba.sort_values(by=["ts", "src"], ascending=[True, False])
    ba = ba.drop_duplicates("ts", keep="first").set_index("ts")
    out = pd.DataFrame({"bn": bn["c"], "bn_vol": bn["quote_vol"]})
    out["ib_last"] = tr["c"]
    out["ib_vol"] = tr["vol"]
    out["ib_src"] = tr["src"]
    out["ib_bid"] = ba["o"]
    out["ib_ask"] = ba["c"]
    out["ib_mid"] = (ba["o"] + ba["c"]) / 2
    out["spread_bps"] = (ba["c"] - ba["o"]) / out["ib_mid"] * 1e4
    out = cast(pd.DataFrame, out[out.index >= DST_START])
    out = out.dropna(subset=["bn"])
    out["ib_last_ff"] = out["ib_last"].ffill()
    out.loc[out["ib_vol"].fillna(0) <= 0, "ib_last"] = np.nan   # no trade in bar -> stale
    out["basis"] = np.log(out["bn"] / out["ib_last"]) * 1e4
    out["basis_ff"] = np.log(out["bn"] / out["ib_last_ff"]) * 1e4
    out["basis_mid"] = np.log(out["bn"] / out["ib_mid"]) * 1e4
    mid_c = (out["ib_mid"] + out["ib_mid"].shift(-1)) / 2
    out["ib_midc"] = mid_c
    out["basis_midc"] = np.log(out["bn"] / mid_c) * 1e4
    idx = pd.DatetimeIndex(out.index)
    out["seg"] = segment_of(idx)
    out["tday"] = trading_day(idx)
    return out


def load_all(syms: List[str]) -> Dict[str, pd.DataFrame]:
    res: Dict[str, pd.DataFrame] = {}
    for s in syms:
        m = merge_symbol(s)
        if m is not None and len(m) > 0:
            res[s] = m
    return res


def load_special_funding() -> pd.DataFrame:
    """Binance 'Special' funding events = dividend settlements (ex-date 00:00 UTC)."""
    f = pd.read_parquet(BN / "_funding.parquet")
    f = f[f["rateType"] == "Special"].copy()
    f["t"] = pd.to_datetime(f["fundingTime"].astype("int64"), unit="ms", utc=True)
    f["rate"] = f["fundingRate"].astype(float)
    f["mark"] = f["markPrice"].astype(float)
    f["sym"] = [str(x).replace("USDT", "") for x in f["symbol"]]
    return cast(pd.DataFrame, f[["sym", "t", "rate", "mark"]]).sort_values(by="t").reset_index(drop=True)
