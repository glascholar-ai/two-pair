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

    def test_winter_dst_shift(self) -> None:
        # January (EST, UTC-5): the US session is 14:30-21:00 UTC.
        jan = dt.datetime(2026, 1, 15, 14, 0, tzinfo=UTC)   # 09:00 NY
        assert segment_of(jan) == SEG_KR_US_GAP              # pre-open
        assert segment_of(jan.replace(hour=14, minute=30)) == SEG_US_OPEN
        assert segment_of(jan.replace(hour=20, minute=55)) == SEG_US_OPEN
        assert segment_of(jan.replace(hour=21, minute=0)) == SEG_US_KR_GAP
        # Summer boundaries (EDT) are covered by test_boundaries above.

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
        ref = pd.Series(list(xs))
        ref_mean = list(ref.rolling(50, min_periods=20).mean())
        ref_std = list(ref.rolling(50, min_periods=20).std())
        for i, x in enumerate(xs):
            stat.push(x)
            mean, std = stat.mean(), stat.std()
            if i + 1 < 20:
                assert mean is None and std is None
            else:
                assert mean == pytest.approx(ref_mean[i], abs=1e-12)
                assert std == pytest.approx(ref_std[i], abs=1e-12)

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


class TestPlainSdMode:
    def test_plain_matches_pandas_single_window(self) -> None:
        import pandas as pd
        rng = np.random.default_rng(11)
        vals = rng.normal(0, 0.01, 800).cumsum()
        eng = SignalEngine(win_mu=96, min_mu=48, win_sd=96, min_sd=32,
                           segmented=False)
        start = dt.datetime(2026, 7, 6, 0, 0, tzinfo=UTC)
        got = []
        for i, v in enumerate(vals):
            bar = Bar(ts=start + dt.timedelta(minutes=5 * i),
                      kr=math.exp(v), us=1.0, fx=1.0)
            got.append(eng.update(bar).z)
        ser = pd.Series(vals)
        mu = ser.rolling(96, min_periods=48).mean()
        resid = ser - mu
        sd = resid.rolling(96, min_periods=32).std()
        ref = resid / sd
        for i in range(200, 800):
            assert got[i] == pytest.approx(ref.iloc[i], abs=1e-9)

    def test_segmented_and_plain_differ_across_segments(self) -> None:
        rng = np.random.default_rng(5)
        start = dt.datetime(2026, 7, 6, 0, 0, tzinfo=UTC)
        e_seg = SignalEngine(48, 24, 48, 16, segmented=True)
        e_pln = SignalEngine(48, 24, 48, 16, segmented=False)
        z_seg, z_pln = [], []
        for i in range(600):
            v = math.exp(rng.normal(0, 0.01))
            bar = Bar(ts=start + dt.timedelta(minutes=5 * i), kr=v, us=1.0,
                      fx=1.0)
            z_seg.append(e_seg.update(bar).z)
            z_pln.append(e_pln.update(bar).z)
        diffs = [abs(a - b) for a, b in zip(z_seg, z_pln)
                 if a is not None and b is not None]
        assert diffs and max(diffs) > 1e-6   # genuinely different mechanisms
