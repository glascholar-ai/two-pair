"""Pure signal computation for the two-pair strategy.

This module is the single source of truth for the signal: both the backtest
and the live engine feed bars through `SignalEngine`. No I/O here.

Signal definition (baseline v3):
    lr    = ln(kr_price) - ln(us_price) - ln(usdkrw)
    mu    = rolling mean of lr over `win_mu` bars (24h at 5m bars)
    resid = lr - mu
    sd    = rolling std (ddof=1) of resid over the last `win_sd` bars of the
            SAME session segment
    z     = resid / sd
"""
from __future__ import annotations

import collections
import dataclasses
import datetime as dt
import math
import zoneinfo
from typing import Deque, Dict, Optional

import numpy as np

# Session segments. KR cash session is fixed at 00:00-06:30 UTC (KST has no
# DST); the US session is 09:30-16:00 America/New_York and therefore shifts
# by an hour in UTC between EDT and EST — resolved via zoneinfo.
SEG_KR_OPEN = "KR_open"
SEG_KR_US_GAP = "KR->US"
SEG_US_OPEN = "US_open"
SEG_US_KR_GAP = "US->KR"
SEG_WEEKEND = "wknd"

_NY = zoneinfo.ZoneInfo("America/New_York")
_US_OPEN_MIN = 9 * 60 + 30    # 09:30 New York
_US_CLOSE_MIN = 16 * 60       # 16:00 New York


def segment_of(ts: dt.datetime) -> str:
    """Maps a UTC timestamp to its session segment (DST-aware for the US).

    Args:
        ts: Timezone-aware UTC timestamp of a bar close.

    Returns:
        One of the SEG_* constants.
    """
    if ts.weekday() >= 5:
        return SEG_WEEKEND
    minutes = ts.hour * 60 + ts.minute
    if minutes < 390:
        return SEG_KR_OPEN
    ny = ts.astimezone(_NY)
    ny_minutes = ny.hour * 60 + ny.minute
    if ny_minutes < _US_OPEN_MIN:
        return SEG_KR_US_GAP
    if ny_minutes < _US_CLOSE_MIN:
        return SEG_US_OPEN
    return SEG_US_KR_GAP


class RollingStat:
    """Fixed-size rolling window with pandas-compatible mean/std.

    Values are kept in a ring buffer; mean/std are computed with numpy on
    each query, which is exact (no drifting running sums) and fast for the
    window sizes used here (<= a few hundred).
    """

    def __init__(self, size: int, min_periods: int) -> None:
        if size <= 0 or min_periods <= 0 or min_periods > size:
            raise ValueError("invalid window sizes")
        self._buf: Deque[float] = collections.deque(maxlen=size)
        self._min_periods = min_periods

    def push(self, value: float) -> None:
        """Appends a value, evicting the oldest when full."""
        self._buf.append(float(value))

    def __len__(self) -> int:
        return len(self._buf)

    def mean(self) -> Optional[float]:
        """Window mean, or None until min_periods values are present."""
        if len(self._buf) < self._min_periods:
            return None
        return float(np.mean(self._buf))

    def std(self) -> Optional[float]:
        """Sample std (ddof=1), or None until min_periods values are present."""
        if len(self._buf) < self._min_periods or len(self._buf) < 2:
            return None
        return float(np.std(self._buf, ddof=1))


@dataclasses.dataclass(frozen=True)
class Bar:
    """One aligned 5-minute observation of the pair."""

    ts: dt.datetime  # bar close time, UTC
    kr: float        # KR-leg perp price
    us: float        # US-leg perp price
    fx: float        # USDKRW spot


@dataclasses.dataclass(frozen=True)
class SignalState:
    """Signal values derived from one bar. z is None during warmup."""

    ts: dt.datetime
    lr: float
    seg: str
    mu: Optional[float]
    sd: Optional[float]
    z: Optional[float]


class SignalEngine:
    """Incremental signal computation over a stream of bars.

    Feed bars in strictly increasing time order via `update`; each call
    returns the SignalState for that bar. Warmup (None z) lasts until the
    anchor has `min_mu` bars and the bar's segment std has `min_sd` residuals.
    """

    def __init__(self, win_mu: int, min_mu: int, win_sd: int,
                 min_sd: int, segmented: bool = True) -> None:
        self._mu = RollingStat(win_mu, min_mu)
        self._sd: Dict[str, RollingStat] = {}
        self._win_sd = win_sd
        self._min_sd = min_sd
        self._segmented = segmented
        self._last_ts: Optional[dt.datetime] = None

    def update(self, bar: Bar) -> SignalState:
        """Consumes one bar and returns the resulting signal state.

        Args:
            bar: The next bar; its timestamp must exceed the previous one.

        Returns:
            SignalState for this bar.

        Raises:
            ValueError: If timestamps are not strictly increasing or prices
                are not positive.
        """
        if self._last_ts is not None and bar.ts <= self._last_ts:
            raise ValueError(f"non-increasing bar ts: {bar.ts} <= {self._last_ts}")
        if min(bar.kr, bar.us, bar.fx) <= 0:
            raise ValueError(f"non-positive price in bar at {bar.ts}")
        self._last_ts = bar.ts

        lr = math.log(bar.kr) - math.log(bar.us) - math.log(bar.fx)
        seg = segment_of(bar.ts)
        self._mu.push(lr)
        mu = self._mu.mean()

        z: Optional[float] = None
        sd: Optional[float] = None
        if mu is not None:
            resid = lr - mu
            bucket = seg if self._segmented else "all"
            stat = self._sd.get(bucket)
            if stat is None:
                stat = RollingStat(self._win_sd, self._min_sd)
                self._sd[bucket] = stat
            stat.push(resid)
            sd = stat.std()
            if sd is not None and sd > 0:
                z = resid / sd
        return SignalState(ts=bar.ts, lr=lr, seg=seg, mu=mu, sd=sd, z=z)
