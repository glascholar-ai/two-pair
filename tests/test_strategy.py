"""Unit tests for the Strategy state machine."""
from __future__ import annotations

import datetime as dt
from typing import Optional

import pytest

from twopair.config import Config
from twopair.signal import SignalState
from twopair.strategy import Action, CloseReason, Strategy

UTC = dt.timezone.utc
T0 = dt.datetime(2026, 7, 6, 1, 0, tzinfo=UTC)


def sig(minutes: int, z: Optional[float], lr: float = 0.0) -> SignalState:
    """Builds a SignalState `minutes` after T0."""
    return SignalState(ts=T0 + dt.timedelta(minutes=minutes), lr=lr,
                       seg="KR_open", mu=0.0, sd=1.0, z=z)


def cfg(**kw) -> Config:
    return Config(**kw)


class TestEntryExit:
    def test_no_entry_during_warmup(self) -> None:
        strat = Strategy(cfg())
        assert strat.on_bar(sig(0, None), 0, 0).action == Action.NONE
        assert strat.position is None

    def test_entry_sides(self) -> None:
        strat = Strategy(cfg())
        d = strat.on_bar(sig(0, 2.5), 0, 0)
        assert d.action == Action.OPEN and d.side == -1  # rich ratio -> short
        strat2 = Strategy(cfg())
        d2 = strat2.on_bar(sig(0, -2.5), 0, 0)
        assert d2.action == Action.OPEN and d2.side == 1

    def test_no_entry_below_threshold(self) -> None:
        strat = Strategy(cfg())
        assert strat.on_bar(sig(0, 1.99), 0, 0).action == Action.NONE

    def test_convergence_exit_and_pnl(self) -> None:
        strat = Strategy(cfg())
        strat.on_bar(sig(0, -2.5, lr=-0.02), 0, 0)   # long ratio at lr=-0.02
        d = strat.on_bar(sig(5, 0.4, lr=0.0), dlr=0.02, funding_long_ratio_pct=0.0)
        assert d.action == Action.CLOSE
        assert d.trade is not None
        assert d.trade.reason == CloseReason.CONVERGED
        assert d.trade.pnl_pct == pytest.approx(2.0)  # +1 side * +0.02 lr * 100

    def test_funding_flows_into_pnl(self) -> None:
        strat = Strategy(cfg())
        strat.on_bar(sig(0, 2.5, lr=0.02), 0, 0)     # short ratio
        d = strat.on_bar(sig(5, 0.0, lr=0.02), dlr=0.0,
                         funding_long_ratio_pct=0.30)
        assert d.trade is not None
        # side=-1: funding for long-ratio holder is +0.30 -> we pay 0.30.
        assert d.trade.pnl_pct == pytest.approx(-0.30)

    def test_timeout_exit(self) -> None:
        strat = Strategy(cfg())
        strat.on_bar(sig(0, 2.5), 0, 0)
        # |z| stays wide; cross the 24h boundary.
        d = strat.on_bar(sig(24 * 60, 2.6), 0, 0)
        assert d.action == Action.CLOSE
        assert d.trade is not None
        assert d.trade.reason == CloseReason.TIMEOUT

    def test_no_reentry_same_bar_after_close(self) -> None:
        strat = Strategy(cfg())
        strat.on_bar(sig(0, 2.5), 0, 0)
        d = strat.on_bar(sig(5, 0.4), 0, 0)
        assert d.action == Action.CLOSE
        assert strat.position is None
        # Timeout exit with |z|>2 re-enters on the NEXT bar, not the same one.
        d2 = strat.on_bar(sig(10, 2.4), 0, 0)
        assert d2.action == Action.OPEN


class TestStopAndRearm:
    def test_stop_triggers_and_rearms(self) -> None:
        strat = Strategy(cfg(mtm_stop_pct=2.5))
        strat.on_bar(sig(0, 2.5, lr=0.02), 0, 0)          # short ratio
        # ratio rips against us: lr +0.03 => mtm -3.0% <= -2.5 -> stop.
        d = strat.on_bar(sig(5, 3.5, lr=0.05), dlr=0.03, funding_long_ratio_pct=0.0)
        assert d.action == Action.CLOSE
        assert d.trade is not None
        assert d.trade.reason == CloseReason.STOP
        # Still wide: re-entry must be suppressed until |z| < z_in.
        assert strat.on_bar(sig(10, 3.4, lr=0.05), 0, 0).action == Action.NONE
        assert strat.on_bar(sig(15, 1.5, lr=0.03), 0, 0).action == Action.NONE
        # Widens again after reset -> entry allowed.
        assert strat.on_bar(sig(20, 2.3, lr=0.04), 0, 0).action == Action.OPEN

    def test_stop_disabled(self) -> None:
        strat = Strategy(cfg(mtm_stop_pct=0.0))
        strat.on_bar(sig(0, 2.5, lr=0.02), 0, 0)
        d = strat.on_bar(sig(5, 3.5, lr=0.10), dlr=0.08, funding_long_ratio_pct=0.0)
        assert d.action == Action.NONE   # -8% but no stop configured

    def test_cancel_entry(self) -> None:
        strat = Strategy(cfg())
        d = strat.on_bar(sig(0, 2.5), 0, 0)
        assert d.action == Action.OPEN
        strat.cancel_entry()
        assert strat.position is None
        assert strat.trades == []


class TestAlert:
    def test_alert_fires_once_per_position(self) -> None:
        strat = Strategy(cfg())
        strat.on_bar(sig(0, 2.5), 0, 0)
        d1 = strat.on_bar(sig(5, 4.6), 0, 0)
        d2 = strat.on_bar(sig(10, 4.8), 0, 0)
        assert d1.z_alert is True
        assert d2.z_alert is False
