"""SQLite journal: every bar's signal state, trades, fills, and events.

The journal is append-only and doubles as the research dataset for judging
deferred rules (session tags, max-z, stop hits) once enough live samples
accumulate.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from typing import Any, Optional, Sequence

from twopair.signal import SignalState
from twopair.strategy import Trade

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    ts TEXT PRIMARY KEY, kr REAL, us REAL, fx REAL,
    lr REAL, mu REAL, sd REAL, z REAL, seg TEXT
);
CREATE TABLE IF NOT EXISTS trades (
    entry_ts TEXT, exit_ts TEXT, side INTEGER, entry_z REAL,
    max_abs_z REAL, entry_seg TEXT, held_hours REAL, pnl_pct REAL,
    reason TEXT, mode TEXT
);
CREATE TABLE IF NOT EXISTS fills (
    ts TEXT, symbol TEXT, side TEXT, qty REAL, price REAL,
    order_id TEXT, purpose TEXT
);
CREATE TABLE IF NOT EXISTS events (
    ts TEXT, level TEXT, message TEXT
);
"""


class Journal:
    """Thin, typed wrapper over a SQLite database."""

    def __init__(self, path: str) -> None:
        self._conn = sqlite3.connect(path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Closes the underlying connection."""
        self._conn.close()

    def record_bar(self, sig: SignalState, kr: float, us: float,
                   fx: float) -> None:
        """Stores one bar's raw inputs and derived signal values."""
        self._conn.execute(
            "INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?,?)",
            (sig.ts.isoformat(), kr, us, fx, sig.lr, sig.mu, sig.sd, sig.z,
             sig.seg))
        self._conn.commit()

    def record_trade(self, trade: Trade, mode: str) -> None:
        """Stores a closed round trip."""
        self._conn.execute(
            "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?)",
            (trade.entry_ts.isoformat(), trade.exit_ts.isoformat(),
             trade.side, trade.entry_z, trade.max_abs_z, trade.entry_seg,
             trade.held_hours, trade.pnl_pct, trade.reason.value, mode))
        self._conn.commit()

    def record_fill(self, ts: dt.datetime, symbol: str, side: str, qty: float,
                    price: float, order_id: str, purpose: str) -> None:
        """Stores one leg fill (purpose: open/close/repair)."""
        self._conn.execute(
            "INSERT INTO fills VALUES (?,?,?,?,?,?,?)",
            (ts.isoformat(), symbol, side, qty, price, order_id, purpose))
        self._conn.commit()

    def record_event(self, ts: dt.datetime, level: str, message: str) -> None:
        """Stores a log event (level: INFO/WARN/ERROR)."""
        self._conn.execute("INSERT INTO events VALUES (?,?,?)",
                           (ts.isoformat(), level, message))
        self._conn.commit()

    def query(self, sql: str,
              params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        """Runs a read-only query and returns all rows."""
        return list(self._conn.execute(sql, params))

    def last_bar_ts(self) -> Optional[str]:
        """ISO timestamp of the most recent recorded bar, if any."""
        rows = self.query("SELECT MAX(ts) FROM bars")
        return rows[0][0] if rows and rows[0][0] else None
