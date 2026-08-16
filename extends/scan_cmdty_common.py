#!/usr/bin/env python3
"""Shared loaders / session labelling for commodity-perp scans."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent / "data"
BN_DIR = ROOT / "bn5m_cmdty"
HL_DIR = ROOT / "hl"
IB_DIR = ROOT / "ib"
NY = ZoneInfo("America/New_York")

# Binance perp -> (IB future key, HL name, carry-adjust flag)
MAP: Dict[str, Dict[str, Optional[str]]] = {
    "XAUUSDT": {"ib": "GC", "hl": "GOLD", "carry": "y"},
    "XAGUSDT": {"ib": "SI", "hl": "SILVER", "carry": "y"},
    "CLUSDT": {"ib": "CL", "hl": "CL", "carry": None},
    "BZUSDT": {"ib": None, "hl": "BRENTOIL", "carry": None},
    "NATGASUSDT": {"ib": "NG", "hl": "NATGAS", "carry": None},
    "COPPERUSDT": {"ib": "HG", "hl": "COPPER", "carry": "y"},
    "XPTUSDT": {"ib": None, "hl": "PLATINUM", "carry": None},
    "XPDUSDT": {"ib": None, "hl": "PALLADIUM", "carry": None},
}
FX_HL: Dict[str, str] = {"EUR": "FX_EURUSD", "GBP": "FX_GBPUSD", "JPY": "FX_USDJPY"}


def col(df: pd.DataFrame, name: str) -> pd.Series:
    return cast(pd.Series, df[name])


def rows(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    return cast(pd.DataFrame, df.loc[np.asarray(mask, dtype=bool)])


def load_bn(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(BN_DIR / f"{sym}.parquet")
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("dt").sort_index()


def load_hl(name: str) -> pd.DataFrame:
    df = pd.read_parquet(HL_DIR / f"xyz_{name}_5m.parquet")
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("dt").sort_index()


def load_ib(key: str) -> pd.DataFrame:
    df = pd.read_parquet(IB_DIR / f"{key}_5m.parquet")
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("dt").sort_index()


def ib_meta(key: str) -> Dict[str, object]:
    p = IB_DIR / f"{key}_meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def bn_funding(sym: str) -> pd.Series:
    f = pd.read_parquet(BN_DIR / "_funding.parquet")
    f = f[f["symbol"] == sym]
    idx = pd.to_datetime(f["fundingTime"].astype("int64"), unit="ms", utc=True)
    return pd.Series(np.asarray(f["fundingRate"].astype(float)), index=idx).sort_index()


def hl_funding(name: str) -> pd.Series:
    p = HL_DIR / f"xyz_{name}_funding.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    f = pd.read_parquet(p)
    idx = pd.to_datetime(f["time"].astype("int64"), unit="ms", utc=True)
    return pd.Series(np.asarray(f["fundingRate"].astype(float)), index=idx).sort_index()


def cme_session(idx: pd.DatetimeIndex) -> pd.Series:
    """Label each bar start: 'open' (Globex trading), 'break' (17:00-18:00 ET
    daily maintenance), 'weekend' (Fri 17:00 ET -> Sun 18:00 ET)."""
    ny = pd.Series(idx.tz_convert(NY)).dt
    hm = np.asarray(ny.hour) * 60 + np.asarray(ny.minute)
    dow = np.asarray(ny.dayofweek)
    is_break = (hm >= 17 * 60) & (hm < 18 * 60)
    weekend = (dow == 5) | ((dow == 4) & (hm >= 17 * 60)) | ((dow == 6) & (hm < 18 * 60))
    lab = np.where(weekend, "weekend", np.where(is_break, "break", "open"))
    return pd.Series(lab, index=idx)


def fx_session(idx: pd.DatetimeIndex) -> pd.Series:
    """Spot FX: closed Fri 17:00 ET -> Sun 17:00 ET (no daily break)."""
    ny = pd.Series(idx.tz_convert(NY)).dt
    hm = np.asarray(ny.hour) * 60 + np.asarray(ny.minute)
    dow = np.asarray(ny.dayofweek)
    weekend = (dow == 5) | ((dow == 4) & (hm >= 17 * 60)) | ((dow == 6) & (hm < 17 * 60))
    return pd.Series(np.where(weekend, "weekend", "open"), index=idx)


def half_life(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 50:
        return float("nan")
    lag = np.asarray(x.shift(1).dropna(), dtype=float)
    y = np.asarray(x.iloc[1:], dtype=float)
    b = float(np.polyfit(lag - lag.mean(), y - y.mean(), 1)[0])
    if b <= 0 or b >= 1:
        return float("inf")
    return float(-np.log(2) / np.log(b))


def fmt(x: float, d: int = 1) -> str:
    return "nan" if x != x else f"{x:.{d}f}"


def dollar_depth(book: Dict[str, List[List[str]]], bps: float) -> Dict[str, float]:
    bids = [(float(p), float(q)) for p, q in book["bids"]]
    asks = [(float(p), float(q)) for p, q in book["asks"]]
    if not bids or not asks:
        return {"bid": 0.0, "ask": 0.0, "spread_bps": float("nan")}
    mid = (bids[0][0] + asks[0][0]) / 2
    lo, hi = mid * (1 - bps / 1e4), mid * (1 + bps / 1e4)
    b = sum(p * q for p, q in bids if p >= lo)
    a = sum(p * q for p, q in asks if p <= hi)
    return {"bid": b, "ask": a, "spread_bps": (asks[0][0] - bids[0][0]) / mid * 1e4}
