"""Unit tests for RiskGuard, Journal, Notifier, and Config."""
from __future__ import annotations

import datetime as dt
import json
import pathlib

import pytest

from twopair.config import Config, load_config
from twopair.journal import Journal
from twopair.notify import Notifier
from twopair.risk import RiskGuard
from twopair.signal import SignalState
from twopair.strategy import CloseReason, Trade

UTC = dt.timezone.utc
MONDAY = dt.datetime(2026, 7, 6, 2, 0, tzinfo=UTC)


class TestConfig:
    def test_defaults_valid(self) -> None:
        cfg = Config()
        assert cfg.z_in == 2.0 and cfg.mode == "paper"

    def test_bad_mode(self) -> None:
        with pytest.raises(ValueError):
            Config(mode="yolo")

    def test_bad_thresholds(self) -> None:
        with pytest.raises(ValueError):
            Config(z_in=1.0, z_out=1.5)

    def test_load_file_and_env(self, tmp_path: pathlib.Path,
                               monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps({"leg_notional_usdt": 5000}))
        monkeypatch.setenv("TWOPAIR_TELEGRAM_TOKEN", "tok")
        monkeypatch.setenv("TWOPAIR_TELEGRAM_CHAT_ID", "42")
        cfg = load_config(str(path))
        assert cfg.leg_notional_usdt == 5000
        assert cfg.telegram_token == "tok" and cfg.telegram_chat_id == "42"

    def test_unknown_key_rejected(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps({"nope": 1}))
        with pytest.raises(ValueError):
            load_config(str(path))


class TestRiskGuard:
    def test_fresh_data_passes(self) -> None:
        guard = RiskGuard(Config())
        out = guard.entry_blocked(MONDAY, MONDAY - dt.timedelta(minutes=5),
                                  MONDAY - dt.timedelta(minutes=10))
        assert out == []

    def test_stale_bar_blocks(self) -> None:
        guard = RiskGuard(Config())
        out = guard.entry_blocked(MONDAY, MONDAY - dt.timedelta(minutes=30),
                                  MONDAY)
        assert any(v.code == "stale_bars" for v in out)

    def test_stale_fx_blocks_on_weekday_only(self) -> None:
        guard = RiskGuard(Config())
        stale_fx = MONDAY - dt.timedelta(hours=5)
        out = guard.entry_blocked(MONDAY, MONDAY - dt.timedelta(minutes=5),
                                  stale_fx)
        assert any(v.code == "stale_fx" for v in out)
        saturday = dt.datetime(2026, 7, 11, 12, 0, tzinfo=UTC)
        out2 = guard.entry_blocked(saturday,
                                   saturday - dt.timedelta(minutes=5),
                                   saturday - dt.timedelta(hours=20))
        assert not any(v.code == "stale_fx" for v in out2)

    def test_daily_loss_halts_and_resets(self) -> None:
        guard = RiskGuard(Config(daily_loss_halt_pct=3.0))
        guard.record_trade_pnl(MONDAY, -3.5)
        out = guard.entry_blocked(MONDAY, MONDAY - dt.timedelta(minutes=5),
                                  MONDAY)
        assert any(v.code == "daily_loss" for v in out)
        next_day = MONDAY + dt.timedelta(days=1)
        out2 = guard.entry_blocked(next_day,
                                   next_day - dt.timedelta(minutes=5),
                                   next_day)
        assert not any(v.code == "daily_loss" for v in out2)

    def test_fx_gap_alert(self) -> None:
        guard = RiskGuard(Config())
        assert guard.fx_gap_alert(1400.0, 1407.0) is not None   # 0.5%
        assert guard.fx_gap_alert(1400.0, 1401.0) is None       # 0.07%


class TestJournal:
    def test_round_trip(self, tmp_path: pathlib.Path) -> None:
        journal = Journal(str(tmp_path / "j.sqlite"))
        sig = SignalState(ts=MONDAY, lr=-7.5, seg="KR_open", mu=-7.49,
                          sd=0.002, z=-2.3)
        journal.record_bar(sig, kr=1100.0, us=145.0, fx=1440.0)
        trade = Trade(entry_ts=MONDAY, exit_ts=MONDAY + dt.timedelta(hours=3),
                      side=1, entry_z=-2.3, max_abs_z=2.9, entry_seg="KR_open",
                      held_hours=3.0, pnl_pct=1.25,
                      reason=CloseReason.CONVERGED)
        journal.record_trade(trade, "paper")
        journal.record_fill(MONDAY, "KRUSDT", "BUY", 0.9, 1100.0, "1", "open")
        journal.record_event(MONDAY, "INFO", "hello")

        assert journal.last_bar_ts() == MONDAY.isoformat()
        trades = journal.query("SELECT side, pnl_pct, reason, mode FROM trades")
        assert trades == [(1, 1.25, "conv", "paper")]
        assert journal.query("SELECT COUNT(*) FROM fills")[0][0] == 1
        journal.close()

    def test_recovery_queries(self, tmp_path: pathlib.Path) -> None:
        journal = Journal(str(tmp_path / "j.sqlite"))
        assert journal.last_open_fill_ts() is None
        assert journal.last_trade("paper") is None
        assert journal.realized_pnl_on_day("2026-07-06", "paper") == 0.0

        journal.record_fill(MONDAY, "KRUSDT", "BUY", 0.9, 1100.0, "1", "open")
        later = MONDAY + dt.timedelta(hours=2)
        journal.record_fill(later, "KRUSDT", "SELL", 0.9, 1105.0, "2", "close")
        assert journal.last_open_fill_ts() == MONDAY.isoformat()

        for pnl, reason, hours in ((-1.2, CloseReason.STOP, 3),
                                   (0.5, CloseReason.CONVERGED, 5)):
            journal.record_trade(
                Trade(entry_ts=MONDAY, exit_ts=MONDAY + dt.timedelta(hours=hours),
                      side=1, entry_z=2.1, max_abs_z=2.5, entry_seg="KR_open",
                      held_hours=float(hours), pnl_pct=pnl, reason=reason),
                "paper")
        last = journal.last_trade("paper")
        assert last is not None and last[1] == "conv"
        assert journal.realized_pnl_on_day(
            "2026-07-06", "paper") == pytest.approx(-0.7)
        assert journal.last_trade("live") is None
        journal.close()

    def test_bar_upsert(self, tmp_path: pathlib.Path) -> None:
        journal = Journal(str(tmp_path / "j.sqlite"))
        sig = SignalState(ts=MONDAY, lr=1.0, seg="KR_open", mu=None, sd=None,
                          z=None)
        journal.record_bar(sig, 1.0, 1.0, 1.0)
        journal.record_bar(sig, 1.0, 1.0, 1.0)
        assert journal.query("SELECT COUNT(*) FROM bars")[0][0] == 1
        journal.close()


class TestNotifier:
    def test_disabled_notifier_no_ops(self) -> None:
        notifier = Notifier("", "")
        assert notifier.enabled is False
        assert notifier.send("msg") is False
