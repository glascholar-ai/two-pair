"""Position state machine for the two-pair strategy.

Consumes SignalStates plus funding cashflows and decides entries/exits.
Pure logic — no I/O, no clocks, no exchange knowledge. The same class drives
the backtest and the live engine, which is what guarantees parity.

Conventions:
    side=+1 means long the ratio (long KR leg / short US leg);
    side=-1 means short the ratio (short KR leg / long US leg).
    All PnL figures are percentages of single-leg notional.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import enum
from typing import List, Optional

from twopair.config import Config
from twopair.signal import SignalState


class Action(enum.Enum):
    """What the caller should do after a bar update."""

    NONE = "none"
    OPEN = "open"
    CLOSE = "close"


class CloseReason(enum.Enum):
    """Why a position was closed."""

    CONVERGED = "conv"
    TIMEOUT = "timeout"
    STOP = "stop"
    TAKE_PROFIT = "tp"


@dataclasses.dataclass
class Position:
    """An open ratio position marked to market bar by bar."""

    entry_ts: dt.datetime
    entry_lr: float
    entry_z: float
    entry_seg: str
    side: int          # +1 long ratio, -1 short ratio
    leg_notional_usdt: float = 0.0   # actual per-leg size (MTM denominator)
    mtm_pct: float = 0.0
    max_abs_z: float = 0.0
    max_mtm_pct: float = 0.0

    def held_hours(self, now: dt.datetime) -> float:
        """Hours since entry."""
        return (now - self.entry_ts).total_seconds() / 3600.0


@dataclasses.dataclass(frozen=True)
class Trade:
    """A closed round trip."""

    entry_ts: dt.datetime
    exit_ts: dt.datetime
    side: int
    entry_z: float
    max_abs_z: float
    entry_seg: str
    held_hours: float
    pnl_pct: float
    max_mtm_pct: float
    reason: CloseReason


@dataclasses.dataclass(frozen=True)
class Decision:
    """Result of one bar update."""

    action: Action
    side: int = 0                       # set when action == OPEN
    trade: Optional[Trade] = None       # set when action == CLOSE
    z_alert: bool = False               # |z| crossed the alert threshold


class Strategy:
    """Entry/exit rules of baseline v3 with MTM stop and re-arm.

    Call `on_bar` once per aligned bar, passing the signal state, the change
    in lr since the previous bar (0.0 for the first bar of a position), and
    the net funding accrued on the pair over (prev_bar_ts, bar_ts]
    expressed as +pct for a side=+1 position.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._pos: Optional[Position] = None
        self._need_rearm = False
        self._alerted = False
        self.trades: List[Trade] = []

    @property
    def position(self) -> Optional[Position]:
        """The open position, if any."""
        return self._pos

    def cancel_entry(self) -> None:
        """Rolls back an entry whose execution was blocked or failed.

        Call only immediately after an OPEN decision, before any further
        on_bar calls; no Trade is recorded.
        """
        self._pos = None

    def adopt_position(self, side: int, entry_ts: dt.datetime,
                       mtm_pct: float, seg: str,
                       leg_notional_usdt: float) -> None:
        """Installs a position discovered on the exchange (recovery/sync).

        Signal-space entry metadata (entry_lr, entry_z) is unknowable for an
        adopted position and is zeroed; it is bookkeeping only and drives no
        decision. leg_notional_usdt must be the position's ACTUAL per-leg
        size — it is the denominator for MTM percentages, so the stop fires
        at the same real-money fraction regardless of adopted size.

        Raises:
            ValueError: If a position is already held, side is invalid, or
                the notional is not positive.
        """
        if self._pos is not None:
            raise ValueError("cannot adopt: position already held")
        if side not in (1, -1):
            raise ValueError(f"invalid side {side}")
        if leg_notional_usdt <= 0:
            raise ValueError(f"invalid leg notional {leg_notional_usdt}")
        self._pos = Position(entry_ts=entry_ts, entry_lr=0.0, entry_z=0.0,
                             entry_seg=seg, side=side, mtm_pct=mtm_pct,
                             leg_notional_usdt=leg_notional_usdt)
        self._alerted = False

    def sync_mtm(self, mtm_pct: float) -> None:
        """Overwrites the position MTM with exchange truth (plan A sync)."""
        if self._pos is None:
            raise ValueError("no position to sync")
        self._pos.mtm_pct = mtm_pct

    def drop_position(self) -> None:
        """Discards the local position because the exchange shows none.

        The close happened outside this process (manual intervention or a
        repair); its PnL is unknown here, so no Trade is recorded — the
        caller should journal an event instead.
        """
        self._pos = None

    @property
    def need_rearm(self) -> bool:
        """Whether entries are suppressed until |z| returns below z_in."""
        return self._need_rearm

    def set_rearm(self, value: bool) -> None:
        """Sets the re-arm latch (startup recovery heuristic)."""
        self._need_rearm = value

    def on_bar(self, sig: SignalState, dlr: float,
               funding_long_ratio_pct: float) -> Decision:
        """Advances the state machine by one bar.

        Args:
            sig: Signal state for this bar.
            dlr: lr change since the previous bar (used for MTM).
            funding_long_ratio_pct: Funding accrued over the bar for a
                side=+1 position, in % of single-leg notional (the caller
                computes -kr_funding + us_funding and scales by 100).

        Returns:
            The Decision for this bar. When it is OPEN the caller must place
            the two legs; when CLOSE it must flatten them.
        """
        cfg = self._cfg
        if self._pos is not None:
            self._pos.mtm_pct += self._pos.side * (dlr * 100.0
                                                   + funding_long_ratio_pct)
            self._pos.max_mtm_pct = max(self._pos.max_mtm_pct,
                                        self._pos.mtm_pct)
        if sig.z is None:
            return Decision(Action.NONE)
        abs_z = abs(sig.z)

        if self._need_rearm and abs_z < cfg.z_in:
            self._need_rearm = False

        if self._pos is None:
            if not self._need_rearm and abs_z > cfg.z_in:
                self._pos = Position(entry_ts=sig.ts, entry_lr=sig.lr,
                                     entry_z=sig.z, entry_seg=sig.seg,
                                     side=-1 if sig.z > 0 else 1,
                                     leg_notional_usdt=cfg.leg_notional_usdt,
                                     max_abs_z=abs_z)
                self._alerted = False
                return Decision(Action.OPEN, side=self._pos.side)
            return Decision(Action.NONE)

        pos = self._pos
        pos.max_abs_z = max(pos.max_abs_z, abs_z)
        alert = False
        if not self._alerted and abs_z >= cfg.z_alert:
            self._alerted = True
            alert = True

        stop_hit = (cfg.mtm_stop_pct > 0
                    and pos.mtm_pct <= -cfg.mtm_stop_pct)
        tp_hit = (cfg.mtm_take_profit_pct > 0
                  and pos.mtm_pct >= cfg.mtm_take_profit_pct)
        timed_out = pos.held_hours(sig.ts) >= cfg.max_hold_hours
        converged = abs_z < cfg.z_out
        if not (converged or timed_out or stop_hit or tp_hit):
            return Decision(Action.NONE, z_alert=alert)

        if converged:
            reason = CloseReason.CONVERGED
        elif tp_hit:
            reason = CloseReason.TAKE_PROFIT
        elif stop_hit:
            reason = CloseReason.STOP
            self._need_rearm = True
        else:
            reason = CloseReason.TIMEOUT
        trade = Trade(entry_ts=pos.entry_ts, exit_ts=sig.ts, side=pos.side,
                      entry_z=pos.entry_z, max_abs_z=pos.max_abs_z,
                      entry_seg=pos.entry_seg,
                      held_hours=pos.held_hours(sig.ts),
                      pnl_pct=pos.mtm_pct, max_mtm_pct=pos.max_mtm_pct,
                      reason=reason)
        self.trades.append(trade)
        self._pos = None
        return Decision(Action.CLOSE, trade=trade, z_alert=alert)
