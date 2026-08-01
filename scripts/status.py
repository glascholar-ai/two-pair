#!/usr/bin/env python3
"""On-demand status: signal state from the journal, positions from Binance.

Usage:
    python3 scripts/status.py [--config cfg.json] [--testnet]

Works without API keys (journal-only view); with BINANCE_API_KEY/SECRET it
also shows exchange positions and today's realized PnL.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import os
import pathlib
import sys
from typing import Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from twopair.config import Config, load_config
from twopair.executor import BinanceClient, LiveExecutor
from twopair.journal import Journal

TESTNET_BASE = "https://testnet.binancefuture.com"


def journal_section(cfg: Config) -> None:
    """Prints the latest signal state and recent trades from the journal."""
    if not pathlib.Path(cfg.db_path).exists():
        print(f"journal: {cfg.db_path} not found (loop never ran here?)")
        return
    journal = Journal(cfg.db_path)
    rows = journal.query(
        "SELECT ts, z, seg, lr, fx FROM bars ORDER BY ts DESC LIMIT 1")
    if rows:
        ts, z, seg, lr, fx = rows[0]
        z_txt = f"{z:+.2f}" if z is not None else "warmup"
        age_min = (dt.datetime.now(dt.timezone.utc)
                   - dt.datetime.fromisoformat(str(ts))).total_seconds() / 60
        print(f"last bar : {ts}  ({age_min:.0f}m ago)")
        print(f"signal   : z={z_txt}  seg={seg}  lr={lr:.5f}  fx={fx:.1f}")
    else:
        print("journal: no bars recorded yet")
    count = journal.query(
        "SELECT COUNT(*) FROM bars WHERE ts >= ?",
        ((dt.datetime.now(dt.timezone.utc)
          - dt.timedelta(hours=24)).isoformat(),))[0][0]
    print(f"bars 24h : {count}")
    trades = journal.query(
        "SELECT exit_ts, side, pnl_pct, reason, held_hours FROM trades "
        "WHERE mode = ? ORDER BY exit_ts DESC LIMIT 5", (cfg.mode_label(),))
    if trades:
        print("recent trades:")
        for exit_ts, side, pnl, reason, held in trades:
            print(f"  {exit_ts}  side={side:+d}  {pnl:+.2f}%  "
                  f"{reason}  {held:.1f}h")
    journal.close()


def exchange_section(cfg: Config) -> None:
    """Prints exchange positions and today's realized PnL, if keys exist."""
    key = os.environ.get("BINANCE_API_KEY", "")
    secret = os.environ.get("BINANCE_API_SECRET", "")
    if not key or not secret:
        print("exchange : (no API keys in env — journal view only)")
        return
    client = BinanceClient(key, secret, cfg.binance_base)
    executor = LiveExecutor(client, cfg.leg_notional_usdt, cfg.kr_symbol,
                            cfg.us_symbol)
    entry_ts: Optional[dt.datetime] = None
    try:
        entry_ts = executor.estimate_entry_ts()
    except Exception:  # noqa: BLE001 — cosmetic
        pass
    view = executor.position_view(entry_ts)
    if abs(view.kr_qty) < 1e-12 and abs(view.us_qty) < 1e-12:
        print("exchange : FLAT")
    else:
        held = ""
        if entry_ts is not None:
            hours = (dt.datetime.now(dt.timezone.utc)
                     - entry_ts).total_seconds() / 3600
            held = f"  held={hours:.1f}h"
        print(f"exchange : kr={view.kr_qty:+g}  us={view.us_qty:+g}  "
              f"pnl={view.pnl_pct:+.2f}%{held}")
    pnl = executor.realized_pnl_today_pct(dt.datetime.now(dt.timezone.utc))
    print(f"today    : realized {pnl:+.2f}% of leg notional")


def main() -> None:
    """Prints a one-shot status snapshot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--testnet", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.testnet:
        cfg = dataclasses.replace(cfg, binance_base=TESTNET_BASE)
    print(f"twopair status [{cfg.mode_label()}]  "
          f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S} UTC")
    journal_section(cfg)
    exchange_section(cfg)


if __name__ == "__main__":
    main()
