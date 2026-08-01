#!/usr/bin/env python3
"""Entry point for the trading loop (prod or Binance testnet).

Usage:
    python3 run_live.py                     # prod, needs BINANCE_API_KEY/SECRET
    python3 run_live.py --testnet           # Binance futures testnet
    python3 run_live.py --config cfg.json   # config overrides

There is no built-in paper mode: rehearse the machinery on the testnet
(note our stock-perp symbols may not be listed there — override symbols in
the config), or rehearse the strategy on prod with a tiny leg_notional_usdt.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from typing import Optional

from twopair.config import Config, load_config
from twopair.executor import BinanceClient, ChasePolicy, LiveExecutor
from twopair.journal import Journal
from twopair.live import LiveApp
from twopair.notify import Notifier

TESTNET_BASE = "https://testnet.binancefuture.com"


def build_executor(cfg: Config, notifier: Notifier) -> LiveExecutor:
    """Constructs the executor from environment credentials."""
    key = os.environ.get("BINANCE_API_KEY", "")
    secret = os.environ.get("BINANCE_API_SECRET", "")
    if not key or not secret:
        raise SystemExit("set BINANCE_API_KEY / BINANCE_API_SECRET "
                         "(testnet keys for --testnet)")
    client = BinanceClient(key, secret, cfg.binance_base)
    policy = ChasePolicy(style=cfg.order_style,
                         chase_interval_seconds=cfg.chase_interval_seconds,
                         max_chases=cfg.max_chases,
                         fill_poll_seconds=cfg.fill_poll_seconds)
    return LiveExecutor(client, cfg.leg_notional_usdt, cfg.kr_symbol,
                        cfg.us_symbol, policy=policy,
                        on_event=lambda m: notifier.send(m))


def main(argv: Optional[list[str]] = None) -> None:
    """Parses arguments, wires components, and runs the loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="JSON config path")
    parser.add_argument("--testnet", action="store_true",
                        help="run against the Binance futures testnet")
    parser.add_argument("--warmup-days", type=int, default=7)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.testnet:
        cfg = dataclasses.replace(cfg, binance_base=TESTNET_BASE)
    notifier = Notifier(cfg.telegram_token, cfg.telegram_chat_id)
    journal = Journal(cfg.db_path)
    app = LiveApp(cfg, build_executor(cfg, notifier), journal, notifier)
    app.warmup(days=args.warmup_days)
    app.run_forever()


if __name__ == "__main__":
    main()
