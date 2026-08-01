"""RiskGuard: pre-entry gates and standing alerts.

The guard never places or closes orders itself; it answers two questions:
  * `entry_blocked(...)` — is opening a NEW position currently forbidden?
  * `check_alerts(...)` — any standing conditions the operator must hear about?

Existing positions are managed by the Strategy (stop/timeout) regardless of
guard state; halting entries must not orphan an open position.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
from typing import List, Optional

from twopair.config import Config
from twopair.signal import SEG_WEEKEND, segment_of


@dataclasses.dataclass(frozen=True)
class Violation:
    """One reason why entries are blocked or an alert fired."""

    code: str
    message: str


class RiskGuard:
    """Stateless checks plus a daily realized-loss accumulator."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._day: Optional[dt.date] = None
        self._day_pnl_pct = 0.0

    def record_trade_pnl(self, exit_ts: dt.datetime, pnl_pct: float) -> None:
        """Accumulates realized PnL into the current UTC day's total."""
        if self._day != exit_ts.date():
            self._day = exit_ts.date()
            self._day_pnl_pct = 0.0
        self._day_pnl_pct += pnl_pct

    def daily_pnl_pct(self, now: dt.datetime) -> float:
        """Realized PnL for the current UTC day."""
        if self._day != now.date():
            return 0.0
        return self._day_pnl_pct

    def entry_blocked(self, now: dt.datetime, bar_ts: dt.datetime,
                      fx_ts: dt.datetime) -> List[Violation]:
        """Returns violations that forbid opening a new position now.

        Args:
            now: Current wall-clock time (UTC).
            bar_ts: Open time of the latest completed bar.
            fx_ts: Timestamp of the latest FX observation.
        """
        cfg = self._cfg
        out: List[Violation] = []
        bar_age = (now - bar_ts).total_seconds()
        max_age = cfg.bar_seconds * (cfg.max_data_gap_bars + 1)
        if bar_age > max_age:
            out.append(Violation(
                "stale_bars", f"latest bar is {bar_age / 60:.1f}m old"))
        fx_age_min = (now - fx_ts).total_seconds() / 60.0
        if (segment_of(now) != SEG_WEEKEND
                and fx_age_min > cfg.fx_stale_max_minutes):
            out.append(Violation(
                "stale_fx", f"FX is {fx_age_min:.0f}m old on a weekday"))
        if self.daily_pnl_pct(now) <= -cfg.daily_loss_halt_pct:
            out.append(Violation(
                "daily_loss",
                f"daily realized PnL {self.daily_pnl_pct(now):+.2f}% breached "
                f"-{cfg.daily_loss_halt_pct}%"))
        return out

    def fx_gap_alert(self, prev_fx: float, new_fx: float,
                     threshold_pct: float = 0.3) -> Optional[Violation]:
        """Flags an FX reopen gap larger than threshold_pct."""
        jump = abs(new_fx / prev_fx - 1.0) * 100.0
        if jump >= threshold_pct:
            return Violation("fx_gap", f"USDKRW jumped {jump:.2f}%")
        return None
