#!/usr/bin/env python3
"""Entry point for the paper/live trading loop.

Usage:
    python3 run_live.py                     # paper mode, defaults
    python3 run_live.py --config cfg.json   # config overrides
    python3 run_live.py --live              # real orders (needs BINANCE_API_KEY/SECRET)
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from typing import Optional

from twopair.config import Config, load_config
from twopair.executor import BinanceClient, Executor, LiveExecutor, PaperExecutor
from twopair.journal import Journal
from twopair.live import LiveApp
from twopair.notify import Notifier


def build_executor(cfg: Config, notifier: Notifier) -> Executor:
    """Constructs the executor matching cfg.mode."""
    if cfg.mode == "paper":
        return PaperExecutor(cfg.leg_notional_usdt, cfg.kr_symbol,
                             cfg.us_symbol)
    key = os.environ.get("BINANCE_API_KEY", "")
    secret = os.environ.get("BINANCE_API_SECRET", "")
    if not key or not secret:
        raise SystemExit("live mode needs BINANCE_API_KEY / BINANCE_API_SECRET")
    client = BinanceClient(key, secret, cfg.binance_base)
    return LiveExecutor(client, cfg.leg_notional_usdt, cfg.kr_symbol,
                        cfg.us_symbol, on_event=lambda m: notifier.send(m))


def main(argv: Optional[list[str]] = None) -> None:
    """Parses arguments, wires components, and runs the loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="JSON config path")
    parser.add_argument("--live", action="store_true",
                        help="place real orders (default: paper)")
    parser.add_argument("--warmup-days", type=int, default=7)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.live:
        cfg = dataclasses.replace(cfg, mode="live")
    notifier = Notifier(cfg.telegram_token, cfg.telegram_chat_id)
    journal = Journal(cfg.db_path)
    app = LiveApp(cfg, build_executor(cfg, notifier), journal, notifier)
    app.warmup(days=args.warmup_days)
    app.run_forever()


if __name__ == "__main__":
    main()
