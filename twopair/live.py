"""Live trading loop (prod or Binance testnet).

Synchronous 5-minute polling — deliberately simple. Each cycle:
  1. sleep to the next bar boundary (+grace), fetch the just-closed bar
     for both legs and the latest FX;
  2. sync position state with the exchange (source of truth: positionRisk
     + funding income; see classify_sync);
  3. push the bar through SignalEngine + Strategy (identical signal path to
     the backtest);
  4. act on the Decision via the executor, consult RiskGuard for entries;
  5. journal everything, notify on trades/alerts/errors.

There is no local paper mode: use the Binance testnet (system rehearsal;
our symbols may be absent there) or prod with a tiny leg notional
(strategy rehearsal) instead. Position MTM comes solely from the exchange
sync; the loop never accrues PnL locally.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Optional, cast

import pandas as pd

from twopair import data as datamod
from twopair.config import Config
from twopair.executor import LiveExecutor, PairView
from twopair.journal import Journal
from twopair.notify import Notifier
from twopair.risk import RiskGuard
from twopair.signal import Bar, SignalEngine, SignalState, segment_of
from twopair.strategy import Action, Decision, Strategy

logger = logging.getLogger(__name__)


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class SyncAction:
    """What the per-cycle position sync decided (string constants)."""

    NONE = "none"        # exchange flat, local flat — nothing to do
    TRACK = "track"      # healthy pair matching local side — refresh MTM
    ADOPT = "adopt"      # healthy pair, no local position — install it
    DROP = "drop"        # exchange flat but local has one — discard local
    REPAIR = "repair"    # orphan leg / same-sign / size or side mismatch


def classify_sync(view: PairView, kr_price: float, us_price: float,
                  local_side: int, dust_usdt: float,
                  tolerance_pct: float) -> tuple[str, int, str]:
    """Classifies exchange state against local state (pure function).

    Args:
        view: Exchange truth for the two legs.
        kr_price: Current KR-leg price (to value quantities).
        us_price: Current US-leg price.
        local_side: Strategy's position side, 0 when flat.
        dust_usdt: Leg notionals below this count as flat.
        tolerance_pct: Max abs notional mismatch between legs, in %.

    Returns:
        (action, side, detail) where action is a SyncAction constant and
        side is the exchange pair's ratio side (0 unless a healthy pair).
    """
    kr_notional = view.kr_qty * kr_price
    us_notional = view.us_qty * us_price
    kr_on = abs(kr_notional) >= dust_usdt
    us_on = abs(us_notional) >= dust_usdt
    if not kr_on and not us_on:
        if local_side == 0:
            return SyncAction.NONE, 0, ""
        return SyncAction.DROP, 0, "exchange flat but local position exists"
    if kr_on and us_on and kr_notional * us_notional < 0:
        bigger = max(abs(kr_notional), abs(us_notional))
        mismatch = abs(abs(kr_notional) - abs(us_notional)) / bigger * 100.0
        if mismatch <= tolerance_pct:
            side = 1 if view.kr_qty > 0 else -1
            if local_side == 0:
                return SyncAction.ADOPT, side, f"mismatch {mismatch:.1f}%"
            if local_side == side:
                return SyncAction.TRACK, side, ""
            return (SyncAction.REPAIR, side,
                    f"side conflict: local {local_side:+d} vs exchange "
                    f"{side:+d}")
        return (SyncAction.REPAIR, 0,
                f"leg notional mismatch {mismatch:.1f}% > "
                f"{tolerance_pct:.1f}%")
    return SyncAction.REPAIR, 0, "orphan or same-direction legs"


class LiveApp:
    """Owns the polling loop and wires all components together."""

    def __init__(self, cfg: Config, executor: LiveExecutor, journal: Journal,
                 notifier: Notifier) -> None:
        self._cfg = cfg
        self._exec = executor
        self._journal = journal
        self._notify = notifier
        self._engine = SignalEngine(cfg.win_mu, cfg.min_mu, cfg.win_sd,
                                    cfg.min_sd, cfg.segmented_sd)
        self._strategy = Strategy(cfg)
        self._guard = RiskGuard(cfg)
        self._prev_ts: Optional[dt.datetime] = None
        self._last_fx: Optional[float] = None
        self._fx_ts: Optional[dt.datetime] = None
        self._last_sig: Optional[SignalState] = None
        self._last_digest_day: Optional[dt.date] = None
        self._blocked_entries = 0

    # ------------------------------------------------------------------ setup
    def warmup(self, days: int = 0) -> None:
        """Replays recent history so windows are full before going live.

        days=0 derives the depth from the configured signal windows.
        """
        cfg = self._cfg
        if days <= 0:
            days = cfg.warmup_days()
        start_ms = int((_utcnow() - dt.timedelta(days=days)).timestamp() * 1000)
        kr = datamod.fetch_klines(cfg.kr_symbol, "5m", start_ms,
                                  cfg.binance_base)
        us = datamod.fetch_klines(cfg.us_symbol, "5m", start_ms,
                                  cfg.binance_base)
        if cfg.fx_source == "usdkrw":
            fx = datamod.fetch_fx_yahoo()
        else:
            fx = pd.Series(1.0, index=kr.index)
        pair = datamod.build_pair_dataset(kr, us, fx)
        # Drop the final row: its bar may still be forming.
        pair = pair.iloc[:-1]
        kr_col = pair["kr"].to_numpy(dtype=float)
        us_col = pair["us"].to_numpy(dtype=float)
        fx_col = pair["fx"].to_numpy(dtype=float)
        for i, raw_ts in enumerate(pair.index):
            ts = cast(dt.datetime, raw_ts)
            self._engine.update(Bar(ts=ts, kr=kr_col[i], us=us_col[i],
                                    fx=fx_col[i]))
            self._prev_ts = ts
        self._last_fx = float(fx_col[-1])
        self._fx_ts = cast(dt.datetime, pair.index[-1])
        logger.info("warmup done: %d bars to %s", len(pair), self._prev_ts)
        self._exec.cancel_all_open_orders()
        logger.info("startup: cancelled all open orders")
        self._recover_state()

    def _recover_state(self) -> None:
        """Seeds the daily-loss counter and re-arm latch after a restart.

        Positions are recovered by the first per-cycle sync; the daily-loss
        counter comes from the exchange income API (includes manual trades);
        only the re-arm latch falls back to the journal, best effort.
        """
        cfg = self._cfg
        now = _utcnow()
        try:
            today_pnl = self._exec.realized_pnl_today_pct(now)
        except Exception as err:  # noqa: BLE001 — recovery is best effort
            logger.warning("daily-PnL recovery failed: %s", err)
            today_pnl = 0.0
        if today_pnl != 0.0:
            self._guard.record_trade_pnl(now, today_pnl)
            logger.info("recovered daily realized PnL %+.2f%%", today_pnl)
        last = self._journal.last_trade(cfg.mode_label())
        if last is not None and last[1] == "stop":
            exit_ts = dt.datetime.fromisoformat(last[0])
            age_h = (now - exit_ts).total_seconds() / 3600.0
            if age_h <= cfg.rearm_recovery_hours:
                self._strategy.set_rearm(True)
                logger.info("re-arm latch restored (stop %.1fh ago)", age_h)

    # ------------------------------------------------------------------- loop
    def run_forever(self) -> None:
        """Blocks, processing one bar per cycle. Ctrl-C to stop."""
        self._notify.send(f"twopair {self._cfg.mode_label()} loop starting")
        while True:
            self._sleep_to_next_bar()
            try:
                self.step()
            except Exception as err:  # noqa: BLE001 — loop must survive
                logger.exception("cycle failed")
                self._notify.send(f"cycle error: {err}")

    def _sleep_to_next_bar(self) -> None:
        cfg = self._cfg
        now = _utcnow().timestamp()
        next_close = (int(now // cfg.bar_seconds) + 1) * cfg.bar_seconds
        time.sleep(max(0.0, next_close + cfg.poll_grace_seconds - now))

    def step(self) -> None:
        """Syncs with the exchange, then advances the strategy by one bar."""
        bar = self._fetch_latest_bar()
        if bar is None:
            return
        entry_safe = self._sync_position(bar)
        if self._cfg.deadman_seconds > 0:
            self._exec.arm_deadman(self._cfg.deadman_seconds)
        sig = self._engine.update(bar)
        # MTM comes exclusively from the exchange sync (real-money truth,
        # funding included via income). Local deltas are always zero; if a
        # sync fetch fails the MTM is simply one bar stale.
        decision = self._strategy.on_bar(sig, 0.0, 0.0)
        self._journal.record_bar(sig, bar.kr, bar.us, bar.fx)
        self._prev_ts = bar.ts
        self._last_sig = sig
        pos = self._strategy.position
        logger.info(
            "bar %s z=%s seg=%s kr=%.2f us=%.2f fx=%.1f pos=%s decision=%s",
            f"{bar.ts:%m-%d %H:%M}",
            f"{sig.z:+.2f}" if sig.z is not None else "warmup",
            sig.seg, bar.kr, bar.us, bar.fx,
            (f"{pos.side:+d}@{pos.mtm_pct:+.2f}%" if pos is not None
             else "flat"),
            decision.action.value)

        if decision.z_alert:
            msg = f"|z| alert: z={sig.z:.2f} at {sig.ts}"
            logger.warning(msg)
            self._notify.send(msg)
        if decision.action == Action.OPEN:
            self._handle_open(decision.side, bar, sig, entry_safe)
        elif decision.action == Action.CLOSE:
            self._handle_close(bar, sig, decision)
        self._maybe_send_digest()

    # ---------------------------------------------------------------- digest
    def _maybe_send_digest(self) -> None:
        """Sends one health/state digest per UTC day (also after restarts).

        The digest doubles as a dead-man signal for the operator: its
        absence at the expected time means the loop is not running.
        """
        cfg = self._cfg
        if cfg.digest_utc_hour < 0:
            return
        now = _utcnow()
        if now.hour < cfg.digest_utc_hour or self._last_digest_day == now.date():
            return
        self._last_digest_day = now.date()
        self._notify.send(self._digest_text(now))
        self._blocked_entries = 0

    def _digest_text(self, now: dt.datetime) -> str:
        """Builds the digest message from in-memory and journal state."""
        sig = self._last_sig
        if sig is not None and sig.z is not None:
            sig_line = f"z={sig.z:+.2f} ({sig.seg}) @ {sig.ts:%m-%d %H:%M}"
        elif sig is not None:
            sig_line = f"z=warmup ({sig.seg}) @ {sig.ts:%m-%d %H:%M}"
        else:
            sig_line = "z=n/a (no bars yet)"
        pos = self._strategy.position
        if pos is None:
            pos_line = "position: FLAT"
        else:
            pos_line = (f"position: side={pos.side:+d} "
                        f"mtm={pos.mtm_pct:+.2f}% "
                        f"held={pos.held_hours(now):.1f}h")
        bars_24h = "n/a"
        try:
            rows = self._journal.query(
                "SELECT COUNT(*) FROM bars WHERE ts >= ?",
                ((now - dt.timedelta(hours=24)).isoformat(),))
            bars_24h = str(rows[0][0])
        except Exception as err:  # noqa: BLE001 — digest is best effort
            logger.warning("digest journal query failed: %s", err)
        # Trailing 24h window from the exchange: the guard counter resets
        # at UTC midnight, so at digest time (shortly after) it always read
        # ~0 — the original always-zero digest bug.
        realized_24h = "n/a"
        try:
            usd = self._exec.realized_pnl_usd(now - dt.timedelta(hours=24))
            realized_24h = f"{usd:+.2f} USDT"
        except Exception as err:  # noqa: BLE001 — digest is best effort
            logger.warning("digest income query failed: %s", err)
        return (f"digest {now:%Y-%m-%d} [{self._cfg.mode_label()}]\n"
                f"{sig_line}\n{pos_line}\n"
                f"realized 24h: {realized_24h}\n"
                f"bars 24h: {bars_24h} | blocked entries: "
                f"{self._blocked_entries}")

    # ------------------------------------------------------------------ sync
    def _sync_position(self, bar: Bar) -> bool:
        """Reconciles local position state with the exchange (plan A).

        Returns True only when the sync succeeded AND left a clean state —
        the precondition for opening NEW positions this cycle. False when
        the view fetch failed (exchange state unknown: an unseen position
        may exist, so entries are unsafe; MTM stays one bar stale) or when
        a REPAIR was needed (something was wrong; do not re-enter on the
        same bar).
        """
        pos = self._strategy.position
        entry_ts = pos.entry_ts if pos is not None else None
        try:
            view = self._exec.position_view(entry_ts)
        except Exception as err:  # noqa: BLE001 — sync must not kill the loop
            logger.warning("position sync failed: %s", err)
            return False
        local_side = pos.side if pos is not None else 0
        action, side, detail = classify_sync(
            view, bar.kr, bar.us, local_side,
            self._cfg.dust_usdt, self._cfg.sync_tolerance_pct)
        if action == SyncAction.NONE:
            return True
        if action == SyncAction.TRACK:
            assert pos is not None
            self._strategy.sync_mtm(
                view.pnl_usd / pos.leg_notional_usdt * 100.0)
            return True
        if action == SyncAction.ADOPT:
            self._adopt(view, bar, side, detail)
            return True
        if action == SyncAction.DROP:
            self._strategy.drop_position()
            msg = f"position closed externally ({detail}); local state dropped"
            logger.warning(msg)
            self._notify.send(msg)
            return True
        # REPAIR: flatten everything, then run flat.
        result = self._exec.close_all(bar.kr, bar.us)
        for fill in result.fills:
            logger.info("fill[repair] %s %s qty=%g px=%.4f id=%s",
                        fill.symbol, fill.side, fill.qty, fill.price,
                        fill.order_id)
            self._journal.record_fill(bar.ts, fill.symbol, fill.side,
                                      fill.qty, fill.price, fill.order_id,
                                      "repair")
        if self._strategy.position is not None:
            self._strategy.drop_position()
        msg = f"sync repair: {detail}; flattened (ok={result.ok})"
        logger.error(msg)
        self._notify.send(msg)
        return False  # something was wrong — no new entries this cycle

    def _adopt(self, view: PairView, bar: Bar, side: int,
               detail: str) -> None:
        """Installs an exchange position, resizing DOWN if over config.

        Config-notional changes are a normal operation: an oversized pair
        is trimmed to the configured size with reduce-only orders (no full
        flatten-and-reopen churn); an undersized pair is adopted as-is and
        runs its course — the next fresh entry uses the new size. The MTM
        denominator is always the position's ACTUAL leg notional, so the
        stop fires at the same real-money fraction at any size.
        """
        cfg = self._cfg
        measured = (abs(view.kr_qty) * bar.kr
                    + abs(view.us_qty) * bar.us) / 2.0
        leg_notional = measured
        if measured > cfg.leg_notional_usdt * (
                1.0 + cfg.sync_tolerance_pct / 100.0):
            fills = self._exec.trim_to_notional(cfg.leg_notional_usdt,
                                                bar.kr, bar.us)
            for fill in fills:
                logger.info("fill[trim] %s %s qty=%g px=%.4f id=%s",
                            fill.symbol, fill.side, fill.qty, fill.price,
                            fill.order_id)
                self._journal.record_fill(bar.ts, fill.symbol, fill.side,
                                          fill.qty, fill.price,
                                          fill.order_id, "trim")
            leg_notional = cfg.leg_notional_usdt
            self._notify.send(
                f"trimmed adopted position {measured:.0f} -> "
                f"{leg_notional:.0f} USDT/leg ({len(fills)} reduce-only "
                "fills)")
        entry_ts: Optional[dt.datetime] = None
        try:
            entry_ts = self._exec.estimate_entry_ts()
        except Exception as err:  # noqa: BLE001 — hint is best effort
            logger.warning("entry-ts reconstruction failed: %s", err)
        if entry_ts is None:
            entry_ts = _utcnow()
        mtm_pct = view.pnl_usd / leg_notional * 100.0
        self._strategy.adopt_position(side, entry_ts, mtm_pct,
                                      segment_of(bar.ts), leg_notional)
        msg = (f"adopted exchange position side={side:+d} "
               f"leg~{leg_notional:.0f}USDT pnl={mtm_pct:+.2f}% "
               f"entry~{entry_ts:%m-%d %H:%M} ({detail})")
        logger.warning(msg)
        self._notify.send(msg)

    # ---------------------------------------------------------------- actions
    def _handle_open(self, side: int, bar: Bar, sig: SignalState,
                     entry_safe: bool) -> None:
        if not entry_safe:
            self._strategy.cancel_entry()
            msg = "entry suppressed: position sync unavailable or repaired"
            logger.warning(msg)
            self._notify.send(msg)
            return
        blocked = self._guard.entry_blocked(_utcnow(), bar.ts,
                                            self._fx_ts or bar.ts)
        if blocked:
            # Roll back the entry: the strategy opened a position internally,
            # but risk forbids it. Discard silently and re-arm cleanly.
            self._strategy.cancel_entry()
            self._blocked_entries += 1
            msgs = "; ".join(v.message for v in blocked)
            logger.warning("entry blocked: %s", msgs)
            self._notify.send(f"entry blocked: {msgs}")
            return
        result = self._exec.open_ratio(side, bar.kr, bar.us)
        for fill in result.fills:
            logger.info("fill[open] %s %s qty=%g px=%.4f id=%s",
                        fill.symbol, fill.side, fill.qty, fill.price,
                        fill.order_id)
            self._journal.record_fill(bar.ts, fill.symbol, fill.side,
                                      fill.qty, fill.price, fill.order_id,
                                      "open")
        if not result.ok:
            self._strategy.cancel_entry()
            logger.error("open failed: %s", result.error)
            self._notify.send(f"OPEN FAILED (repaired): {result.error}")
            return
        self._notify.send(
            f"OPEN side={side:+d} z={sig.z:.2f} seg={sig.seg} "
            f"kr={bar.kr:.2f} us={bar.us:.2f}")

    def _handle_close(self, bar: Bar, sig: SignalState,
                      decision: Decision) -> None:
        result = self._exec.close_all(bar.kr, bar.us)
        for fill in result.fills:
            logger.info("fill[close] %s %s qty=%g px=%.4f id=%s",
                        fill.symbol, fill.side, fill.qty, fill.price,
                        fill.order_id)
            self._journal.record_fill(bar.ts, fill.symbol, fill.side,
                                      fill.qty, fill.price, fill.order_id,
                                      "close")
        trade = decision.trade
        if trade is None:  # unreachable on a CLOSE decision; keep types honest
            return
        if not result.ok:
            # Execution not confirmed: record nothing. The local position is
            # already dropped; if legs actually remain on the exchange the
            # next sync ADOPTs them and management continues.
            logger.error("close failed: %s", result.error)
            self._notify.send(
                f"CLOSE FAILED ({trade.reason.value}); repaired to flat; "
                f"trade NOT recorded: {result.error}")
            return
        self._journal.record_trade(trade, self._cfg.mode_label())
        self._guard.record_trade_pnl(trade.exit_ts, trade.pnl_pct)
        self._notify.send(
            f"CLOSE {trade.reason.value} pnl={trade.pnl_pct:+.2f}% "
            f"held={trade.held_hours:.1f}h")

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
            logger.warning("bar %s missing on a leg", open_ts)
            return None
        if cfg.fx_source == "none":
            self._last_fx = 1.0
            self._fx_ts = now
        else:
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
                logger.warning("fx fetch failed: %s", err)
        if self._last_fx is None:
            return None
        if self._prev_ts is not None and open_ts <= self._prev_ts:
            return None  # already processed
        return Bar(ts=open_ts, kr=float(kr.loc[open_ts]),
                   us=float(us.loc[open_ts]), fx=self._last_fx)
