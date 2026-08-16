#!/usr/bin/env python3
"""Shared helpers for the calendar/event-window scans (scan_events_*.py).

Data: data/bn5m/<SYM>USDT.parquet (Binance 5m klines, ts = bar open ms UTC),
      data/ib/ES_5m.parquet / NQ_5m.parquet (IBKR 5m bars, ts = bar open ms UTC).
Both use bar-open timestamps, so a bar with ts=T covers [T, T+5m) and its close is
the price *at* T+5m.  `px_at(s, t)` returns the close of the last bar that ended
at or before t, i.e. the price known at time t.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence,
                    Tuple, cast)

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
BN = ROOT / "data" / "bn5m"
IB = ROOT / "data" / "ib"


# --- typed accessors (pandas stubs are too loose for basic-mode pyright) ------------
def col(df: pd.DataFrame, name: str) -> pd.Series:
    """Single column of a DataFrame as a Series."""
    return cast(pd.Series, df[name])


def rows(df: pd.DataFrame) -> Iterator[Any]:
    """itertuples() with attribute access (namedtuple rows are untyped)."""
    return iter(df.itertuples())


def dtidx(obj: "pd.Series | pd.DataFrame") -> pd.DatetimeIndex:
    """The (DatetimeIndex) index of a Series/DataFrame."""
    return cast(pd.DatetimeIndex, obj.index)


def idx_min(obj: "pd.Series | pd.DataFrame") -> pd.Timestamp:
    """First index timestamp."""
    return cast(pd.Timestamp, obj.index.min())


def idx_max(obj: "pd.Series | pd.DataFrame") -> pd.Timestamp:
    """Last index timestamp."""
    return cast(pd.Timestamp, obj.index.max())


def dt_fields(idx: pd.DatetimeIndex) -> Any:
    """DatetimeIndex with .hour/.minute/.dayofweek visible (dynamic attrs in pandas)."""
    return cast(Any, idx)


def sel(df: pd.DataFrame, mask: Any) -> pd.DataFrame:
    """Boolean-mask row selection that keeps the DataFrame type."""
    return cast(pd.DataFrame, df[mask])


def at_minute_multiple(s: pd.Series, k: int) -> pd.Series:
    """Rows of s whose index minute is a multiple of k (e.g. 15 -> 15m bars)."""
    return cast(pd.Series, s[dt_fields(dtidx(s)).minute % k == 0])


def naive(d: str) -> pd.Timestamp:
    """tz-naive Timestamp from a date string."""
    return cast(pd.Timestamp, pd.Timestamp(d))


def as_ts(v: Any) -> pd.Timestamp:
    """Assert a scalar is a Timestamp (e.g. a value pulled out of a DataFrame row)."""
    return cast(pd.Timestamp, v)


# Non-US-listed / not-a-US-stock symbols (Asia names, pre-IPO tokens, misc)
ASIA = {"SAMSUNG", "SKHYNIX", "HYUNDAI", "CSOPSAMSUNG2L", "CSOPSKHYNIX2L", "SONY",
        "HK0700", "HK1810", "TENCENT", "MEITUAN", "KUAISHOU", "POPMART", "GIGADEV"}
PREIPO = {"MINIMAX", "ZHIPU", "OPENAI", "ANTHROPIC", "SPCX", "SPCXUSD1", "CBRS", "BNC",
          "BOT", "SHAZ", "KSTR", "FWDI", "STRC", "BSP", "AXTI", "QNTX", "PENG", "BBX",
          "USAR", "MUU", "MVLL", "INTW", "SNXX", "SKHY"}
LEV_ETF = {"TQQQ", "SQQQ", "SOXL", "SOXS", "TZA", "TMF", "TBT", "UVXY", "KORU", "BITO"}
US_EXCLUDE = ASIA | PREIPO
KR = {"SAMSUNG", "SKHYNIX", "HYUNDAI"}
HK = {"HK0700", "HK1810", "TENCENT", "POPMART", "MEITUAN", "KUAISHOU", "GIGADEV"}

# --- verified calendars (UTC dates) ------------------------------------------------
# NYSE full-day closures inside the sample (verified: NYSE Group 2026 calendar)
US_HOLIDAYS: List[pd.Timestamp] = [naive("2026-05-25"), naive("2026-06-19"),
                                   naive("2026-07-03")]
# KRX closures inside the sample (verified: KRX notice 2026-05-20; calendarlabs)
KR_HOLIDAYS: List[pd.Timestamp] = [naive("2026-06-03"), naive("2026-07-17"),
                                   naive("2026-08-17")]
# HKEX closures 2026 (verified: calendarlabs) - none fall inside the HK perp sample
HK_HOLIDAYS: List[pd.Timestamp] = [naive("2026-06-19"), naive("2026-07-01"),
                                   naive("2026-10-01")]
# 12:30 UTC macro prints (verified: OMB "Schedule of Release Dates for PFEI 2026")
MACRO_1230: Dict[str, List[str]] = {
    "NFP": ["2026-04-03", "2026-05-08", "2026-06-05", "2026-07-02", "2026-08-07"],
    "CPI": ["2026-04-10", "2026-05-12", "2026-06-10", "2026-07-14", "2026-08-12"],
    "PPI": ["2026-04-14", "2026-05-13", "2026-06-11", "2026-07-15", "2026-08-13"],
    "PCE/GDP": ["2026-04-30", "2026-05-28", "2026-06-25", "2026-07-30"],
    "RETAIL": ["2026-04-16", "2026-05-14", "2026-06-17", "2026-07-16"],
}
# FOMC statement 18:00 UTC (verified: federalreserve.gov)
FOMC_1800 = ["2026-04-29", "2026-06-17", "2026-07-29"]
# monthly opex (3rd Friday; Jun-19 holiday -> Thu Jun-18), Russell recon, S&P rebalance
OPEX = {"2026-04-17": "opex", "2026-05-15": "opex", "2026-06-18": "opex+SPX-rebal(Jun19 hol)",
        "2026-07-17": "opex", "2026-06-26": "Russell-recon"}


def load_px(sym: str) -> pd.DataFrame:
    """Full 5m frame indexed by bar-open UTC time (o,h,l,c,vol,quote_vol,trades)."""
    df = pd.read_parquet(BN / f"{sym}USDT.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["ts"], unit="ms", utc=True))
    df = df[~df.index.duplicated()].sort_index()
    return cast(pd.DataFrame, df.drop(columns=["ts"]))


def load_close(sym: str) -> pd.Series:
    """Close series indexed by bar *end* time (price known at index time)."""
    df = load_px(sym)
    idx = cast(pd.DatetimeIndex, df.index) + mins(5)
    return pd.Series(df["c"].to_numpy(dtype=float), index=idx)


def load_fut(sym: str = "ES") -> pd.DataFrame:
    """IB futures 5m frame indexed by bar *end* time; spliced front contract."""
    df = pd.read_parquet(IB / f"{sym}_5m.parquet")
    # where both contracts overlap keep the later (U6) one - returns are what we use
    df = df.sort_values(["ts", "contract"])
    df.index = pd.DatetimeIndex(pd.to_datetime(df["ts"], unit="ms", utc=True) + mins(5), name="t")
    df = df.drop(columns=["ts"])
    return cast(pd.DataFrame, df[~df.index.duplicated(keep="last")].sort_index())


def fut_close(sym: str = "ES") -> pd.Series:
    df = load_fut(sym)
    return pd.Series(df["c"].to_numpy(dtype=float), index=df.index)


def universe(min_days: int = 20, exclude: Optional[Iterable[str]] = None) -> List[str]:
    """US-listed symbols with at least min_days of history."""
    ex = set(US_EXCLUDE if exclude is None else exclude)
    out: List[str] = []
    for p in sorted(BN.glob("*USDT.parquet")):
        s = p.stem[:-4]
        if s in ex:
            continue
        n = len(pd.read_parquet(p, columns=["ts"]))
        if n >= min_days * 288:
            out.append(s)
    return out


def px_at(s: pd.Series, t: pd.Timestamp, max_gap: Optional[pd.Timedelta] = None) -> float:
    """Price known at time t (last close at or before t); NaN if stale > max_gap."""
    if max_gap is None:
        max_gap = mins(30)
    # Index.searchsorted stub only accepts array-likes; scalar Timestamp works at runtime.
    i = int(cast(Any, s.index).searchsorted(t, side="right")) - 1
    if i < 0:
        return float("nan")
    ts = cast(pd.Timestamp, s.index[i])
    if t - ts > max_gap:
        return float("nan")
    return float(s.iloc[i])


def ret(s: pd.Series, t0: pd.Timestamp, t1: pd.Timestamp,
        max_gap: Optional[pd.Timedelta] = None) -> float:
    """Log return from price at t0 to price at t1."""
    a, b = px_at(s, t0, max_gap), px_at(s, t1, max_gap)
    if not (a > 0 and b > 0):
        return float("nan")
    return float(np.log(b / a))


def sum_between(df: pd.DataFrame, col: str, t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    """Sum of a kline column over bars whose open time is in [t0, t1)."""
    idx = cast(pd.DatetimeIndex, df.index)
    m = np.asarray((idx >= t0) & (idx < t1))
    return float(df[col].to_numpy(dtype=float)[m].sum())


def rv_between(s: pd.Series, t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    """Realised vol (sqrt sum of squared 5m log returns) between t0 and t1, in bps."""
    idx = cast(pd.DatetimeIndex, s.index)
    seg = s.to_numpy(dtype=float)[np.asarray((idx > t0) & (idx <= t1))]
    if len(seg) < 3:
        return float("nan")
    return float(np.sqrt((np.diff(np.log(seg)) ** 2).sum()) * 1e4)


@dataclass
class OLS:
    n: int
    slope: float
    t: float
    r2: float
    intercept: float = 0.0


def ols(x: pd.Series, y: pd.Series, intercept: bool = True,
        cluster: Optional[pd.Series] = None) -> OLS:
    """Simple OLS y~x with (optionally) cluster-robust t (cluster = date labels)."""
    d = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"),
                      "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if cluster is not None:
        d["g"] = cluster.reindex(d.index)
    n = len(d)
    xv = d["x"].to_numpy(dtype=np.float64)
    yv = d["y"].to_numpy(dtype=np.float64)
    if n < 4 or float(np.nanstd(xv)) == 0.0:
        return OLS(n, float("nan"), float("nan"), float("nan"))
    X = np.column_stack([np.ones(n), xv]) if intercept else xv.reshape(-1, 1)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    res = yv - X @ beta
    xtx_inv = np.linalg.inv(X.T @ X)
    if cluster is None or d["g"].nunique() < 3:
        s2 = (res ** 2).sum() / max(n - X.shape[1], 1)
        cov = s2 * xtx_inv
    else:
        meat = np.zeros((X.shape[1], X.shape[1]))
        for _, idx in d.groupby("g").indices.items():
            u = X[idx].T @ res[idx]
            meat += np.outer(u, u)
        cov = xtx_inv @ meat @ xtx_inv
    k = X.shape[1] - 1
    slope = float(beta[k])
    se = float(np.sqrt(cov[k, k]))
    tss = float(((yv - yv.mean()) ** 2).sum())
    r2 = float(1 - (res ** 2).sum() / tss) if tss > 0 else float("nan")
    return OLS(n, slope, slope / se if se > 0 else float("nan"), r2,
               float(beta[0]) if intercept else 0.0)


def beta_5m(perp: pd.Series, fut: pd.Series, mask_fn: Callable[[pd.DatetimeIndex], np.ndarray],
            min_n: int = 200) -> Tuple[float, int]:
    """Pooled 5m-return beta of perp on fut over bars selected by mask_fn(index)."""
    lp = cast(pd.Series, np.log(perp))
    lf = cast(pd.Series, np.log(fut))
    d = pd.DataFrame({"p": lp.diff(), "f": lf.diff()}).dropna()
    d = cast(pd.DataFrame, d[mask_fn(dtidx(d))])
    if len(d) < min_n:
        return float("nan"), len(d)
    p, f = col(d, "p").to_numpy(dtype=float), col(d, "f").to_numpy(dtype=float)
    return float((p * f).sum() / (f ** 2).sum()), len(d)


def minute_of_day(idx: pd.DatetimeIndex) -> np.ndarray:
    """hour*60 + minute for each element of a DatetimeIndex."""
    f = dt_fields(idx)
    return np.asarray(f.hour * 60 + f.minute)


def rth_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    m = minute_of_day(idx)
    dow = np.asarray(dt_fields(idx).dayofweek)
    return np.asarray((dow < 5) & (m > 13 * 60 + 30) & (m <= 20 * 60))


def fmt_bps(v: float) -> str:
    return "nan" if not np.isfinite(v) else f"{v * 1e4:+.0f}"


def utc(d: str, hm: str = "00:00") -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(f"{d} {hm}", tz="UTC"))


def mins(n: float) -> pd.Timedelta:
    return cast(pd.Timedelta, pd.Timedelta(minutes=n))


def hours(n: float) -> pd.Timedelta:
    return cast(pd.Timedelta, pd.Timedelta(hours=n))


def days(n: float) -> pd.Timedelta:
    return cast(pd.Timedelta, pd.Timedelta(days=n))


def trading_days(px_index: pd.DatetimeIndex, holidays: Sequence[pd.Timestamp]) -> List[pd.Timestamp]:
    ds = sorted({cast(pd.Timestamp, pd.Timestamp(t.date())) for t in px_index})
    hol = {cast(pd.Timestamp, pd.Timestamp(h.date())) for h in holidays}
    return [d for d in ds if d.dayofweek < 5 and d not in hol]
