"""Tests for the plan-A per-cycle position sync."""
from __future__ import annotations

import datetime as dt
from typing import List

import pytest

from twopair.executor import LiveExecutor, PairView
from twopair.live import SyncAction, classify_sync
from twopair.strategy import Strategy
from twopair.config import Config

UTC = dt.timezone.utc
KR_PX, US_PX = 1100.0, 145.0
DUST, TOL = 10.0, 5.0


def view(kr_qty: float, us_qty: float, pnl: float = 0.0) -> PairView:
    return PairView(kr_qty=kr_qty, us_qty=us_qty, pnl_pct=pnl)


def classify(v: PairView, local_side: int) -> tuple[str, int, str]:
    return classify_sync(v, KR_PX, US_PX, local_side, DUST, TOL)


class TestClassifySync:
    def test_flat_flat(self) -> None:
        action, side, _ = classify(view(0.0, 0.0), 0)
        assert action == SyncAction.NONE and side == 0

    def test_dust_counts_as_flat(self) -> None:
        action, _, _ = classify(view(0.001, -0.01), 0)  # ~$1.1 / ~$1.45
        assert action == SyncAction.NONE

    def test_exchange_flat_local_open_drops(self) -> None:
        action, _, _ = classify(view(0.0, 0.0), 1)
        assert action == SyncAction.DROP

    def test_healthy_pair_adopts_when_local_flat(self) -> None:
        action, side, _ = classify(view(0.9, -6.9), 0)  # ~990 vs ~1000 USD
        assert action == SyncAction.ADOPT and side == 1
        action2, side2, _ = classify(view(-0.9, 6.9), 0)
        assert action2 == SyncAction.ADOPT and side2 == -1

    def test_healthy_pair_tracks_matching_local(self) -> None:
        action, side, _ = classify(view(0.9, -6.9), 1)
        assert action == SyncAction.TRACK and side == 1

    def test_side_conflict_repairs(self) -> None:
        action, _, detail = classify(view(0.9, -6.9), -1)
        assert action == SyncAction.REPAIR and "conflict" in detail

    def test_single_leg_repairs(self) -> None:
        action, _, _ = classify(view(0.9, 0.0), 0)
        assert action == SyncAction.REPAIR

    def test_same_direction_legs_repair(self) -> None:
        action, _, _ = classify(view(0.9, 6.9), 0)
        assert action == SyncAction.REPAIR

    def test_notional_mismatch_beyond_tolerance_repairs(self) -> None:
        # KR ~990 USD vs US ~500 USD -> ~49% mismatch.
        action, _, detail = classify(view(0.9, -3.45), 1)
        assert action == SyncAction.REPAIR and "mismatch" in detail

    def test_mismatch_within_tolerance_tracks(self) -> None:
        # 1000 vs ~980 USD -> ~2%.
        action, _, _ = classify(view(0.909, -6.76), 1)
        assert action == SyncAction.TRACK


class TestStrategyAdoption:
    def test_adopt_and_sync_mtm(self) -> None:
        strat = Strategy(Config())
        ts = dt.datetime(2026, 7, 6, 1, 0, tzinfo=UTC)
        strat.adopt_position(1, ts, mtm_pct=-0.8, seg="KR_open")
        pos = strat.position
        assert pos is not None and pos.side == 1
        assert pos.mtm_pct == pytest.approx(-0.8)
        strat.sync_mtm(-1.4)
        assert pos.mtm_pct == pytest.approx(-1.4)

    def test_adopt_rejects_double(self) -> None:
        strat = Strategy(Config())
        ts = dt.datetime(2026, 7, 6, 1, 0, tzinfo=UTC)
        strat.adopt_position(1, ts, 0.0, "wknd")
        with pytest.raises(ValueError):
            strat.adopt_position(-1, ts, 0.0, "wknd")

    def test_adopt_rejects_bad_side(self) -> None:
        strat = Strategy(Config())
        with pytest.raises(ValueError):
            strat.adopt_position(0, dt.datetime.now(UTC), 0.0, "wknd")

    def test_drop_records_no_trade(self) -> None:
        strat = Strategy(Config())
        strat.adopt_position(1, dt.datetime.now(UTC), 0.0, "wknd")
        strat.drop_position()
        assert strat.position is None and strat.trades == []

    def test_sync_mtm_requires_position(self) -> None:
        strat = Strategy(Config())
        with pytest.raises(ValueError):
            strat.sync_mtm(1.0)

    def test_rearm_latch(self) -> None:
        strat = Strategy(Config())
        assert strat.need_rearm is False
        strat.set_rearm(True)
        assert strat.need_rearm is True

    def test_adopted_position_stop_fires(self) -> None:
        from twopair.signal import SignalState
        strat = Strategy(Config(mtm_stop_pct=2.5))
        ts = dt.datetime(2026, 7, 6, 1, 0, tzinfo=UTC)
        strat.adopt_position(1, ts, mtm_pct=-2.0, seg="KR_open")
        strat.sync_mtm(-2.6)  # exchange truth breaches the stop
        sig = SignalState(ts=ts + dt.timedelta(minutes=5), lr=0.0,
                          seg="KR_open", mu=0.0, sd=1.0, z=1.0)
        decision = strat.on_bar(sig, 0.0, 0.0)
        assert decision.trade is not None
        assert decision.trade.reason.value == "stop"


class _ViewClient:
    """Stub for LiveExecutor.position_view / order-hygiene tests."""

    def __init__(self, rows: List[dict], funding: float) -> None:
        self._rows = rows
        self._funding = funding
        self.income_calls: List[str] = []
        self.cancel_all_calls: List[str] = []
        self.countdown_calls: List[tuple] = []
        self.fail_cancel = False

    def position_risk_all(self) -> List[dict]:
        return self._rows

    def funding_income(self, symbol: str, start_ms: int) -> float:
        self.income_calls.append(symbol)
        return self._funding

    def cancel_all_open(self, symbol: str) -> dict:
        if self.fail_cancel:
            raise ConnectionError("boom")
        self.cancel_all_calls.append(symbol)
        return {}

    def countdown_cancel_all(self, symbol: str, countdown_ms: int) -> dict:
        self.countdown_calls.append((symbol, countdown_ms))
        return {}


class TestPositionView:
    def test_live_combines_unrealized_and_funding(self) -> None:
        rows = [
            {"symbol": "KR", "positionAmt": "0.9", "unRealizedProfit": "-12.0"},
            {"symbol": "US", "positionAmt": "-6.9", "unRealizedProfit": "4.0"},
            {"symbol": "OTHER", "positionAmt": "5", "unRealizedProfit": "99"},
        ]
        client = _ViewClient(rows, funding=3.0)
        ex = LiveExecutor(client, 1000.0, "KR", "US")  # type: ignore[arg-type]
        v = ex.position_view(dt.datetime(2026, 8, 1, tzinfo=UTC))
        assert v is not None
        assert v.kr_qty == pytest.approx(0.9)
        assert v.us_qty == pytest.approx(-6.9)
        # (-12 + 4 + 3*2 legs) / 1000 * 100
        assert v.pnl_pct == pytest.approx((-12.0 + 4.0 + 6.0) / 1000 * 100)
        assert client.income_calls == ["KR", "US"]

    def test_live_flat_skips_income(self) -> None:
        client = _ViewClient([], funding=99.0)
        ex = LiveExecutor(client, 1000.0, "KR", "US")  # type: ignore[arg-type]
        v = ex.position_view(dt.datetime(2026, 8, 1, tzinfo=UTC))
        assert v is not None and v.pnl_pct == pytest.approx(0.0)
        assert client.income_calls == []


class TestOrderHygiene:
    def test_cancel_all_covers_both_legs(self) -> None:
        client = _ViewClient([], 0.0)
        ex = LiveExecutor(client, 1000.0, "KR", "US")  # type: ignore[arg-type]
        ex.cancel_all_open_orders()
        assert client.cancel_all_calls == ["KR", "US"]

    def test_cancel_all_failure_reports_not_raises(self) -> None:
        events: List[str] = []
        client = _ViewClient([], 0.0)
        client.fail_cancel = True
        ex = LiveExecutor(client, 1000.0, "KR", "US",  # type: ignore[arg-type]
                          on_event=events.append)
        ex.cancel_all_open_orders()   # must not raise
        assert len(events) == 2

    def test_deadman_arms_both_legs_in_ms(self) -> None:
        client = _ViewClient([], 0.0)
        ex = LiveExecutor(client, 1000.0, "KR", "US")  # type: ignore[arg-type]
        ex.arm_deadman(900)
        assert client.countdown_calls == [("KR", 900000), ("US", 900000)]
