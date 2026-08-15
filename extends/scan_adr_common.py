#!/usr/bin/env python3
"""Shared loaders / pair definitions for the ADR-vs-home-line basis scans."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
BN_DIR = HERE / "data" / "bn5m"
IB_DIR = HERE / "data" / "ib"

# session windows in UTC minutes-of-day (start, end) — summer 2026 (Europe on DST)
SESS_HK = [(90, 240), (300, 480)]          # 01:30-04:00, 05:00-08:00
SESS_TW = [(60, 330)]                        # 01:00-05:30
SESS_JP = [(0, 150), (210, 390)]             # 00:00-02:30, 03:30-06:30
SESS_KR = [(0, 390)]                         # 00:00-06:30
SESS_CPH = [(420, 900)]                      # 07:00-15:00
SESS_AEB = [(420, 930)]                      # 07:00-15:30


@dataclass
class Pair:
    name: str
    perp: str                 # Binance perp symbol (without USDT)
    home: str                 # IB cache name
    fx: Optional[str]         # IB cache name for USD/XXX (None => none)
    fx_invert: bool           # True if fx is XXX/USD (EURUSD) -> implied = home*fx
    ratio: float              # implied USD price = home * ratio / fx (or * fx if invert)
    session: List[Tuple[int, int]]
    price_based: bool = True  # False for ETF-vs-index (level arbitrary; use demeaned basis)
    home_desc: str = ""
    notes: str = field(default="")


PAIRS_ADR: List[Pair] = [
    Pair("BABA/9988.HK", "BABA", "9988_HK", "USDHKD", False, 8.0, SESS_HK, True, "9988.HK x8 / USDHKD"),
    Pair("TSM/2330.TW", "TSM", "2330_TW", "USDTWD", False, 5.0, SESS_TW, True, "2330.TW x5 / USDTWD(1h Yahoo)"),
    Pair("SONY/6758.T", "SONY", "6758_T_yahoo", "USDJPY", False, 1.0, SESS_JP, True, "6758.T / USDJPY"),
    Pair("NVO/NOVO.B", "NVO", "NOVOB_CPH", "USDDKK", False, 1.0, SESS_CPH, True, "NOVO B / USDDKK"),
    Pair("ASML/ASML.AS", "ASML", "ASML_AEB", "EURUSD", True, 1.0, SESS_AEB, True, "ASML.AS x EURUSD"),
    Pair("EWY/K200", "EWY", "K200_IDX", "USDKRW", False, 1.0, SESS_KR, False, "KOSPI200 idx / USDKRW (scaled)"),
    Pair("EWY/005930", "EWY", "005930_KRX", "USDKRW", False, 1.0, SESS_KR, False, "Samsung 005930 / USDKRW (scaled)"),
    Pair("EWJ/N225", "EWJ", "N225_IDX", "USDJPY", False, 1.0, SESS_JP, False, "Nikkei225 idx / USDJPY (scaled)"),
    Pair("EWT/0050.TW", "EWT", "0050_TW", "USDTWD", False, 1.0, SESS_TW, False, "0050.TW / USDTWD (scaled)"),
    Pair("EWT/2330.TW", "EWT", "2330_TW", "USDTWD", False, 1.0, SESS_TW, False, "2330.TW / USDTWD (scaled)"),
    Pair("HK0700/700.HK", "HK0700", "700_HK", None, False, 1.0, SESS_HK, True, "700.HK (perp quoted in HKD)"),
    Pair("TENCENT/700.HK", "TENCENT", "700_HK", "USDHKD", False, 1.0, SESS_HK, True, "700.HK / USDHKD"),
    Pair("HK1810/1810.HK", "HK1810", "1810_HK", None, False, 1.0, SESS_HK, True, "1810.HK (perp quoted in HKD)"),
]

# reverse case: home-listed perps
PAIRS_HOME: List[Pair] = [
    Pair("SAMSUNG/005930", "SAMSUNG", "005930_KRX", "USDKRW", False, 1.0, SESS_KR, True),
    Pair("SKHYNIX/000660", "SKHYNIX", "000660_KRX", "USDKRW", False, 1.0, SESS_KR, True),
    Pair("HYUNDAI/005380", "HYUNDAI", "005380_KRX", "USDKRW", False, 1.0, SESS_KR, True),
    Pair("HK0700/700.HK", "HK0700", "700_HK", None, False, 1.0, SESS_HK, True),
    Pair("TENCENT/700.HK", "TENCENT", "700_HK", "USDHKD", False, 1.0, SESS_HK, True),
    Pair("HK1810/1810.HK", "HK1810", "1810_HK", None, False, 1.0, SESS_HK, True),
]


SeriesLike = Union[pd.Series, pd.DataFrame]


def col(df: pd.DataFrame, name: str) -> pd.Series:
    """Typed single-column accessor (pandas stubs return Series|DataFrame)."""
    return cast(pd.Series, df[name])


def lg(s: pd.Series) -> pd.Series:
    """Natural log as a Series (numpy ufunc stubs lose the Series type)."""
    return pd.Series(np.log(s.to_numpy(dtype=float)), index=s.index)


def dtidx(obj: Any) -> pd.DatetimeIndex:
    return cast(pd.DatetimeIndex, obj.index)


def days(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Midnight-normalised copy of a DatetimeIndex (stubs lack .normalize)."""
    return cast(pd.DatetimeIndex, cast(Any, idx).normalize())


def minute_of_day(idx: pd.DatetimeIndex) -> np.ndarray:
    i = cast(Any, idx)
    return np.asarray(i.hour * 60 + i.minute)


def dow(idx: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(cast(Any, idx).dayofweek)


def load_bn(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(BN_DIR / f"{sym}USDT.parquet")
    df = df.assign(ts=pd.to_datetime(df["ts"], unit="ms", utc=True)).set_index("ts")
    df = df[~df.index.duplicated()].sort_index()
    return cast(pd.DataFrame, df[["o", "h", "l", "c", "vol", "quote_vol"]])


def load_ib(name: str) -> pd.DataFrame:
    p = IB_DIR / f"{name}_5m.parquet"
    if name == "EURUSD":            # longer series cached by the FX scan
        p = IB_DIR / "FX_EURUSD_5m.parquet"
    if not p.exists():
        if name == "USDTWD":
            p = IB_DIR / "USDTWD_1h_yahoo.parquet"
        else:
            raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    df = df.assign(ts=pd.to_datetime(df["ts"], unit="ms", utc=True)).set_index("ts")
    df = df[~df.index.duplicated()].sort_index()
    return cast(pd.DataFrame, df)


def in_session(idx: pd.DatetimeIndex, sess: List[Tuple[int, int]]) -> np.ndarray:
    m = minute_of_day(idx)
    mask = np.zeros(len(idx), dtype=bool)
    for a, b in sess:
        mask |= (m >= a) & (m < b)
    mask &= dow(idx) < 5
    return mask


def fx_series(name: Optional[str], invert: bool, index: pd.DatetimeIndex) -> pd.Series:
    """USD-per-home-currency multiplier aligned to `index` (ffill, <=2h stale)."""
    if name is None:
        return pd.Series(1.0, index=index)
    fx = load_ib(name)["c"]
    fx = fx.reindex(fx.index.union(index)).ffill(limit=30).reindex(index)
    return fx if invert else 1.0 / fx


def implied_usd(p: Pair, index: pd.DatetimeIndex) -> pd.Series:
    home = load_ib(p.home)["c"].reindex(index)
    mult = fx_series(p.fx, p.fx_invert, index)
    return home * p.ratio * mult


def build_frame(p: Pair) -> pd.DataFrame:
    """Aligned 5m frame: perp close, implied home USD close, basis, session mask."""
    perp = load_bn(p.perp)
    home_raw = load_ib(p.home)
    pi, hi = dtidx(perp), dtidx(home_raw)
    lo = max(pd.Timestamp(cast(Any, pi[0])), pd.Timestamp(cast(Any, hi[0])))
    hi_t = min(pd.Timestamp(cast(Any, pi[-1])), pd.Timestamp(cast(Any, hi[-1])))
    idx = cast(pd.DatetimeIndex, pi.union(hi))
    idx = cast(pd.DatetimeIndex, idx[(idx >= lo) & (idx <= hi_t)])
    df = pd.DataFrame(index=idx)
    df["perp"] = perp["c"].reindex(idx)
    df["perp_qv"] = perp["quote_vol"].reindex(idx)
    df["home_usd"] = implied_usd(p, idx)
    df["home_raw"] = home_raw["c"].reindex(idx)
    df["sess"] = in_session(idx, p.session)
    both = df["perp"].notna() & df["home_usd"].notna()
    df["basis"] = np.where(both, np.log(df["perp"] / df["home_usd"]), np.nan)
    if not p.price_based:
        # ETF vs index: remove a slow (5-day rolling median) level so basis is deviation
        b = df["basis"].where(df["sess"])
        lvl = b.rolling(288 * 5, min_periods=50).median()
        df["basis"] = df["basis"] - lvl.ffill()
    return df


def half_life(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 30:
        return float("nan")
    arr = x.to_numpy(dtype=float)
    rho = float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
    if rho <= 0 or rho >= 1:
        return float("inf") if rho >= 1 else 0.0
    return float(-np.log(2) / np.log(rho))


def ols(y: SeriesLike, x: SeriesLike) -> Tuple[float, float, float, int]:
    """Return slope, t-stat, r2, n for y = a + b x."""
    d = pd.concat([y, x], axis=1).dropna()
    if len(d) < 10:
        return float("nan"), float("nan"), float("nan"), len(d)
    yy, xx = d.iloc[:, 0].values, d.iloc[:, 1].values
    xm, ym = xx.mean(), yy.mean()
    sxx = ((xx - xm) ** 2).sum()
    b = ((xx - xm) * (yy - ym)).sum() / sxx
    a = ym - b * xm
    res = yy - a - b * xx
    n = len(d)
    se = float(np.sqrt((res ** 2).sum() / (n - 2) / sxx))
    r2 = 1 - (res ** 2).sum() / ((yy - ym) ** 2).sum()
    return float(b), float(b / se) if se > 0 else float("nan"), float(r2), n
