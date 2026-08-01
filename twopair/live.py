"""Live/paper trading loop.

Synchronous 5-minute polling — deliberately simple. Each cycle:
  1. sleep to the next bar boundary (+grace), fetch the just-closed bar
     for both legs and the latest FX;
  2. push the bar through SignalEngine + Strategy (identical code to the
     backtest);
  3. act on the Decision via the Executor, consult RiskGuard for entries;
  4. journal everything, notify on trades/alerts/errors.

Warmup replays recent history through the engine before the first live bar,
so the strategy starts with full windows.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Optional, cast

import pandas as pd

from twopair import data as datamod
from twopair.config import Config
from twopair.executor import Executor
from twopair.journal import Journal
from twopair.notify import Notifier
from twopair.risk import RiskGuard
from twopair.signal import Bar, SignalEngine, SignalState
from twopair.strategy import Action, Decision, Strategy

logger = logging.getLogger(__name__)


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class LiveApp:
    """Owns the polling loop and wires all components together."""

    def __init__(self, cfg: Config, executor: Executor, journal: Journal,
                 notifier: Notifier) -> None:
        self._cfg = cfg
        self._exec = executor
        self._journal = journal
        self._notify = notifier
        self._engine = SignalEngine(cfg.win_mu, cfg.min_mu, cfg.win_sd,
                                    cfg.min_sd)
        self._strategy = Strategy(cfg)
        self._guard = RiskGuard(cfg)
        self._funding_kr = pd.Series(dtype=float)
        self._funding_us = pd.Series(dtype=float)
        self._prev_lr: Optional[float] = None
        self._prev_ts: Optional[dt.datetime] = None
        self._last_fx: Optional[float] = None
        self._fx_ts: Optional[dt.datetime] = None

    # ------------------------------------------------------------------ setup
    def warmup(self, days: int = 7) -> None:
        """Replays recent history so windows are full before going live."""
        cfg = self._cfg
        start_ms = int((_utcnow() - dt.timedelta(days=days)).timestamp() * 1000)
        kr = datamod.fetch_klines(cfg.kr_symbol, "5m", start_ms,
                                  cfg.binance_base)
        us = datamod.fetch_klines(cfg.us_symbol, "5m", start_ms,
                                  cfg.binance_base)
        fx = datamod.fetch_fx_yahoo()
        pair = datamod.build_pair_dataset(kr, us, fx)
        # Drop the final row: its bar may still be forming.
        pair = pair.iloc[:-1]
        self._refresh_funding(start_ms)
        kr_col = pair["kr"].to_numpy(dtype=float)
        us_col = pair["us"].to_numpy(dtype=float)
        fx_col = pair["fx"].to_numpy(dtype=float)
        for i, raw_ts in enumerate(pair.index):
            ts = cast(dt.datetime, raw_ts)
            sig = self._engine.update(Bar(ts=ts, kr=kr_col[i], us=us_col[i],
                                          fx=fx_col[i]))
            self._prev_ts, self._prev_lr = ts, sig.lr
        self._last_fx = float(fx_col[-1])
        self._fx_ts = cast(dt.datetime, pair.index[-1])
        logger.info("warmup done: %d bars to %s", len(pair), self._prev_ts)

    def _refresh_funding(self, start_ms: int) -> None:
        cfg = self._cfg
        self._funding_kr = datamod.fetch_funding(cfg.kr_symbol, start_ms,
                                                 cfg.binance_base)
        self._funding_us = datamod.fetch_funding(cfg.us_symbol, start_ms,
                                                 cfg.binance_base)

    # ------------------------------------------------------------------- loop
    def run_forever(self) -> None:
        """Blocks, processing one bar per cycle. Ctrl-C to stop."""
        self._notify.send(f"twopair {self._cfg.mode} loop starting")
        while True:
            self._sleep_to_next_bar()
            try:
                self.step()
            except Exception as err:  # noqa: BLE001 — loop must survive
                logger.exception("cycle failed")
                self._journal.record_event(_utcnow(), "ERROR", str(err))
                self._notify.send(f"cycle error: {err}")

    def _sleep_to_next_bar(self) -> None:
        cfg = self._cfg
        now = _utcnow().timestamp()
        next_close = (int(now // cfg.bar_seconds) + 1) * cfg.bar_seconds
        time.sleep(max(0.0, next_close + cfg.poll_grace_seconds - now))

    def step(self) -> None:
        """Fetches the latest closed bar and advances the strategy."""
        bar = self._fetch_latest_bar()
        if bar is None:
            return
        funding_start = int((bar.ts - dt.timedelta(hours=2)).timestamp() * 1000)
        self._refresh_funding(min(
            funding_start,
            int((self._prev_ts or bar.ts).timestamp() * 1000)))
        sig = self._engine.update(bar)

        dlr = 0.0
        funding_pct = 0.0
        prev_ts, prev_lr = self._prev_ts, self._prev_lr
        if (self._strategy.position is not None and prev_lr is not None
                and prev_ts is not None):
            dlr = sig.lr - prev_lr
            kr_f = datamod.funding_between(self._funding_kr, prev_ts, bar.ts)
            us_f = datamod.funding_between(self._funding_us, prev_ts, bar.ts)
            funding_pct = (-kr_f + us_f) * 100.0
        decision = self._strategy.on_bar(sig, dlr, funding_pct)
        self._journal.record_bar(sig, bar.kr, bar.us, bar.fx)
        self._prev_ts, self._prev_lr = bar.ts, sig.lr

        if decision.z_alert:
            msg = f"|z| alert: z={sig.z:.2f} at {sig.ts}"
            self._journal.record_event(sig.ts, "WARN", msg)
            self._notify.send(msg)
        if decision.action == Action.OPEN:
            self._handle_open(decision.side, bar, sig)
        elif decision.action == Action.CLOSE:
            self._handle_close(bar, sig, decision)

    # ---------------------------------------------------------------- actions
    def _handle_open(self, side: int, bar: Bar, sig: SignalState) -> None:
        blocked = self._guard.entry_blocked(_utcnow(), bar.ts,
                                            self._fx_ts or bar.ts)
        if blocked:
            # Roll back the entry: the strategy opened a position internally,
            # but risk forbids it. Discard silently and re-arm cleanly.
            self._strategy.cancel_entry()
            msgs = "; ".join(v.message for v in blocked)
            self._journal.record_event(bar.ts, "WARN", f"entry blocked: {msgs}")
            self._notify.send(f"entry blocked: {msgs}")
            return
        result = self._exec.open_ratio(side, bar.kr, bar.us)
        for fill in result.fills:
            self._journal.record_fill(bar.ts, fill.symbol, fill.side,
                                      fill.qty, fill.price, fill.order_id,
                                      "open")
        if not result.ok:
            self._strategy.cancel_entry()
            self._journal.record_event(bar.ts, "ERROR",
                                       f"open failed: {result.error}")
            self._notify.send(f"OPEN FAILED (repaired): {result.error}")
            return
        self._notify.send(
            f"OPEN side={side:+d} z={sig.z:.2f} seg={sig.seg} "
            f"kr={bar.kr:.2f} us={bar.us:.2f}")

    def _handle_close(self, bar: Bar, sig: SignalState,
                      decision: Decision) -> None:
        result = self._exec.close_all(bar.kr, bar.us)
        for fill in result.fills:
            self._journal.record_fill(bar.ts, fill.symbol, fill.side,
                                      fill.qty, fill.price, fill.order_id,
                                      "close")
        trade = decision.trade
        if trade is None:  # unreachable on a CLOSE decision; keep types honest
            return
        self._journal.record_trade(trade, self._cfg.mode)
        self._guard.record_trade_pnl(trade.exit_ts, trade.pnl_pct)
        status = "" if result.ok else f" (EXEC ERROR: {result.error})"
        self._notify.send(
            f"CLOSE {trade.reason.value} pnl={trade.pnl_pct:+.2f}% "
            f"held={trade.held_hours:.1f}h{status}")
        if not result.ok:
            self._journal.record_event(bar.ts, "ERROR",
                                       f"close failed: {result.error}")

    # ------------------------------------------------------------------ data
    def _fetch_latest_bar(self) -> Optional[Bar]:
        """Fetches the most recent CLOSED 5m bar for both legs plus FX."""
        cfg = self._cfg
        now = _utcnow()
        last_close = (int(now.timestamp() // cfg.bar_seconds)
                      * cfg.bar_seconds)
        open_ts = dt.datetime.fromtimestamp(last_close - cfg.bar_seconds,
                                            dt.timezone.utc)
        start_ms = int(open_ts.timestamp() * 1000)
        kr = datamod.fetch_klines(cfg.kr_symbol, "5m", start_ms,
                                  cfg.binance_base)
        us = datamod.fetch_klines(cfg.us_symbol, "5m", start_ms,
                                  cfg.binance_base)
        if open_ts not in kr.index or open_ts not in us.index:
            self._journal.record_event(now, "WARN",
                                       f"bar {open_ts} missing on a leg")
            return None
        try:
            fx = datamod.fetch_fx_yahoo()
            prev_fx = self._last_fx
            self._last_fx = float(fx.to_numpy(dtype=float)[-1])
            self._fx_ts = cast(pd.Timestamp, fx.index[-1]).to_pydatetime()
            if prev_fx is not None:
                gap = RiskGuard(cfg).fx_gap_alert(prev_fx, self._last_fx)
                if gap is not None:
                    self._notify.send(gap.message)
        except ConnectionError as err:
            self._journal.record_event(now, "WARN", f"fx fetch failed: {err}")
        if self._last_fx is None:
            return None
        if self._prev_ts is not None and open_ts <= self._prev_ts:
            return None  # already processed
        return Bar(ts=open_ts, kr=float(kr.loc[open_ts]),
                   us=float(us.loc[open_ts]), fx=self._last_fx)
