"""Market data access: Binance klines/funding, Yahoo FX, dataset assembly.

All fetchers return pandas objects indexed by tz-aware UTC timestamps.
The merged pair dataset has columns kr, us, fx and is what both the backtest
loader and the live engine's warmup use.
"""
from __future__ import annotations

import datetime as dt
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, List, Optional

import pandas as pd

BINANCE_FAPI = "https://fapi.binance.com"
YAHOO_FX_URL = ("https://query1.finance.yahoo.com/v8/finance/chart/KRW%3DX"
                "?interval=5m&range=1mo")
_RETRIES = 4


def _http_json(url: str, timeout: float = 30.0) -> Any:
    """GETs a URL and parses JSON, with exponential backoff on failure."""
    last_err: Optional[Exception] = None
    for attempt in range(_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
            last_err = err
            time.sleep(2.0 ** attempt)
    raise ConnectionError(f"GET {url} failed after {_RETRIES} tries: {last_err}")


def fetch_klines(symbol: str, interval: str, start_ms: int,
                 base: str = BINANCE_FAPI) -> pd.Series:
    """Fetches close prices for a Binance USDT-M futures symbol.

    Args:
        symbol: e.g. "SKHYNIXUSDT".
        interval: kline interval, e.g. "5m".
        start_ms: inclusive start time in epoch milliseconds.
        base: API base URL.

    Returns:
        Series of closes indexed by bar OPEN time (UTC). The caller decides
        how to label bar close times.
    """
    rows: List[Any] = []
    cur = start_ms
    while True:
        query = urllib.parse.urlencode({"symbol": symbol, "interval": interval,
                                        "startTime": cur, "limit": 1000})
        batch = _http_json(f"{base}/fapi/v1/klines?{query}")
        rows += batch
        if len(batch) < 1000:
            break
        cur = batch[-1][0] + 1
        time.sleep(0.2)
    idx = pd.to_datetime([int(r[0]) for r in rows], unit="ms", utc=True)
    return pd.Series([float(r[4]) for r in rows], index=idx, name=symbol)


def fetch_funding(symbol: str, start_ms: int,
                  base: str = BINANCE_FAPI) -> pd.Series:
    """Fetches historical funding settlements for a symbol.

    Returns:
        Series of funding rates (fraction, not %) indexed by settlement time
        rounded to the minute, duplicates dropped, sorted.
    """
    rows: List[Any] = []
    cur = start_ms
    while True:
        query = urllib.parse.urlencode({"symbol": symbol, "startTime": cur,
                                        "limit": 1000})
        batch = _http_json(f"{base}/fapi/v1/fundingRate?{query}")
        rows += batch
        if len(batch) < 1000:
            break
        cur = batch[-1]["fundingTime"] + 1
        time.sleep(0.2)
    stamps = pd.Series(pd.to_datetime(
        [int(r["fundingTime"]) for r in rows], unit="ms", utc=True))
    idx = pd.DatetimeIndex(stamps.dt.round("min"))
    ser = pd.Series([float(r["fundingRate"]) for r in rows], index=idx,
                    name=symbol)
    ser = ser.loc[~idx.duplicated(keep="first")]
    return ser.sort_index()


def fetch_fx_yahoo() -> pd.Series:
    """Fetches ~1 month of 5m USDKRW from Yahoo Finance."""
    payload = _http_json(YAHOO_FX_URL)
    result = payload["chart"]["result"][0]
    idx = pd.to_datetime(result["timestamp"], unit="s", utc=True)
    ser = pd.Series(result["indicators"]["quote"][0]["close"], index=idx,
                    name="usdkrw")
    return ser.dropna()


def build_pair_dataset(kr: pd.Series, us: pd.Series,
                       fx: pd.Series) -> pd.DataFrame:
    """Aligns the two legs and FX into the canonical pair dataset.

    Bars are joined on identical open times (inner join of the legs); FX is
    forward-filled onto the bar grid, then back-filled only for any leading
    gap.

    Returns:
        DataFrame with columns kr, us, fx indexed by bar open time (UTC).
    """
    df = pd.DataFrame({"kr": kr, "us": us}).dropna()
    df["fx"] = fx.reindex(df.index, method="ffill").ffill().bfill()
    return df


def load_pair_csv(path: str) -> pd.DataFrame:
    """Loads a saved pair dataset (columns ts, kr, us, fx)."""
    df = pd.read_csv(path, parse_dates=["ts"]).set_index("ts")
    if not {"kr", "us", "fx"}.issubset(df.columns):
        raise ValueError(f"{path} lacks kr/us/fx columns")
    return df


def load_funding_csv(path: str) -> pd.Series:
    """Loads a saved funding series (columns ts, rate)."""
    df = pd.read_csv(path, parse_dates=["ts"])
    ser = pd.Series(df["rate"].to_numpy(dtype=float),
                    index=pd.DatetimeIndex(df["ts"]))
    return ser.sort_index()


def save_funding_csv(ser: pd.Series, path: str) -> None:
    """Saves a funding series as columns ts, rate."""
    pd.DataFrame({"ts": ser.index, "rate": ser.to_numpy()}).to_csv(
        path, index=False)


def funding_between(ser: pd.Series, start: dt.datetime,
                    end: dt.datetime) -> float:
    """Sums funding settlements in the half-open interval (start, end]."""
    return float(ser[(ser.index > start) & (ser.index <= end)].sum())
