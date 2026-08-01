"""Unit tests for twopair.signal."""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from twopair.signal import (Bar, RollingStat, SEG_KR_OPEN, SEG_KR_US_GAP,
                            SEG_US_KR_GAP, SEG_US_OPEN, SEG_WEEKEND,
                            SignalEngine, segment_of)

UTC = dt.timezone.utc


class TestSegmentOf:
    def test_kr_open(self) -> None:
        assert segment_of(dt.datetime(2026, 7, 6, 0, 0, tzinfo=UTC)) == SEG_KR_OPEN
        assert segment_of(dt.datetime(2026, 7, 6, 6, 25, tzinfo=UTC)) == SEG_KR_OPEN

    def test_boundaries(self) -> None:
        assert segment_of(dt.datetime(2026, 7, 6, 6, 30, tzinfo=UTC)) == SEG_KR_US_GAP
        assert segment_of(dt.datetime(2026, 7, 6, 13, 25, tzinfo=UTC)) == SEG_KR_US_GAP
        assert segment_of(dt.datetime(2026, 7, 6, 13, 30, tzinfo=UTC)) == SEG_US_OPEN
        assert segment_of(dt.datetime(2026, 7, 6, 19, 55, tzinfo=UTC)) == SEG_US_OPEN
        assert segment_of(dt.datetime(2026, 7, 6, 20, 0, tzinfo=UTC)) == SEG_US_KR_GAP
        assert segment_of(dt.datetime(2026, 7, 6, 23, 55, tzinfo=UTC)) == SEG_US_KR_GAP

    def test_weekend(self) -> None:
        assert segment_of(dt.datetime(2026, 7, 11, 12, 0, tzinfo=UTC)) == SEG_WEEKEND
        assert segment_of(dt.datetime(2026, 7, 12, 3, 0, tzinfo=UTC)) == SEG_WEEKEND
        # Monday 00:00 is KR_open again.
        assert segment_of(dt.datetime(2026, 7, 13, 0, 0, tzinfo=UTC)) == SEG_KR_OPEN


class TestRollingStat:
    def test_matches_pandas(self) -> None:
        rng = np.random.default_rng(7)
        xs = rng.normal(size=500)
        stat = RollingStat(size=50, min_periods=20)
        ref = pd.Series(xs)
        ref_mean = ref.rolling(50, min_periods=20).mean()
        ref_std = ref.rolling(50, min_periods=20).std()
        for i, x in enumerate(xs):
            stat.push(x)
            mean, std = stat.mean(), stat.std()
            if i + 1 < 20:
                assert mean is None and std is None
            else:
                assert mean == pytest.approx(ref_mean.iloc[i], abs=1e-12)
                assert std == pytest.approx(ref_std.iloc[i], abs=1e-12)

    def test_rejects_bad_sizes(self) -> None:
        with pytest.raises(ValueError):
            RollingStat(0, 1)
        with pytest.raises(ValueError):
            RollingStat(5, 6)


class TestSignalEngine:
    def test_warmup_then_z(self) -> None:
        eng = SignalEngine(win_mu=4, min_mu=2, win_sd=4, min_sd=2)
        start = dt.datetime(2026, 7, 6, 1, 0, tzinfo=UTC)
        zs = []
        for i in range(10):
            bar = Bar(ts=start + dt.timedelta(minutes=5 * i),
                      kr=math.exp(0.01 * (i % 3)), us=1.0, fx=1.0)
            zs.append(eng.update(bar).z)
        assert zs[0] is None          # mu warmup
        assert any(z is not None for z in zs[3:])

    def test_rejects_non_increasing_ts(self) -> None:
        eng = SignalEngine(4, 2, 4, 2)
        ts = dt.datetime(2026, 7, 6, 1, 0, tzinfo=UTC)
        eng.update(Bar(ts=ts, kr=1.0, us=1.0, fx=1.0))
        with pytest.raises(ValueError):
            eng.update(Bar(ts=ts, kr=1.0, us=1.0, fx=1.0))

    def test_rejects_bad_price(self) -> None:
        eng = SignalEngine(4, 2, 4, 2)
        ts = dt.datetime(2026, 7, 6, 1, 0, tzinfo=UTC)
        with pytest.raises(ValueError):
            eng.update(Bar(ts=ts, kr=0.0, us=1.0, fx=1.0))

    def test_lr_definition(self) -> None:
        eng = SignalEngine(4, 2, 4, 2)
        ts = dt.datetime(2026, 7, 6, 1, 0, tzinfo=UTC)
        sig = eng.update(Bar(ts=ts, kr=1200.0, us=150.0, fx=1400.0))
        expected = math.log(1200.0) - math.log(150.0) - math.log(1400.0)
        assert sig.lr == pytest.approx(expected)
        assert sig.seg == SEG_KR_OPEN
