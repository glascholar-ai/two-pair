"""Shared fixtures: default config and synthetic bar streams."""
from __future__ import annotations

import datetime as dt
import pathlib
import sys
from typing import Iterator, List

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from twopair.config import Config
from twopair.signal import Bar


@pytest.fixture()
def cfg() -> Config:
    """Baseline config with tiny windows for fast synthetic tests."""
    return Config(win_mu=8, min_mu=4, win_sd=6, min_sd=3)


def make_bars(values: List[float], start: dt.datetime,
              step_minutes: int = 5) -> Iterator[Bar]:
    """Builds bars whose kr price encodes the desired lr (us=fx=1)."""
    import math
    for i, val in enumerate(values):
        ts = start + dt.timedelta(minutes=step_minutes * i)
        yield Bar(ts=ts, kr=math.exp(val), us=1.0, fx=1.0)


@pytest.fixture()
def t0() -> dt.datetime:
    """A Monday 01:00 UTC — inside KR_open, no weekend complications."""
    return dt.datetime(2026, 7, 6, 1, 0, tzinfo=dt.timezone.utc)
