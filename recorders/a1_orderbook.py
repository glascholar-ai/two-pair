#!/usr/bin/env python3
"""A1 recorder: Binance stock-perp books/trades in dead zone + weekends.

Purpose: validate the fill-rate assumption of far (k>=4 sigma) passive
quotes before piloting the off-hours reversal strategy — aggTrades tell
whether prints cross a hypothetical resting level; depth20 snapshots bound
the queue ahead; bookTicker feeds the EWMA/sigma baseline.

No API key required (public market data). Records ONLY during the A1
windows (00:00-08:00 UTC weekdays, all weekend); sleeps otherwise.

Usage:
    python3 recorders/a1_orderbook.py [--config recorders/a1_config.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib
import sys
import urllib.request
from typing import List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from recorders.common import (a1_window_active, a1_window_end, chunk_streams,
                              make_binance_writers, next_a1_window_start,
                              record_connection, utcnow)

logger = logging.getLogger("recorder.a1")

STOCK_TYPES = {"EQUITY", "KR_EQUITY", "HK_EQUITY"}


def load_config(path: str) -> dict:
    """Loads the recorder config (universe mode, exclusions, output root)."""
    cfg = json.loads(pathlib.Path(path).read_text())
    cfg.setdefault("universe", "auto")
    cfg.setdefault("exclude", [])
    cfg.setdefault("output_root", "data/recordings/a1")
    cfg.setdefault("depth_stream", "depth20@500ms")
    return cfg


def resolve_universe(cfg: dict) -> List[str]:
    """Returns the symbols to record.

    "auto" = every TRADING Binance stock perp minus the exclusion list
    (the exclusion list must contain all symbols traded by our strategies:
    a recorder never conflicts, but the eventual A1 pilot must not).
    """
    if isinstance(cfg["universe"], list):
        symbols = list(cfg["universe"])
    else:
        with urllib.request.urlopen(
                "https://fapi.binance.com/fapi/v1/exchangeInfo",
                timeout=30) as resp:
            info = json.load(resp)
        symbols = [s["symbol"] for s in info["symbols"]
                   if s.get("underlyingType") in STOCK_TYPES
                   and s.get("status") == "TRADING"]
    excluded = set(cfg["exclude"])
    return sorted(s for s in symbols if s not in excluded)


def build_streams(symbols: List[str], depth_stream: str) -> List[str]:
    """Stream names for one symbol universe."""
    out: List[str] = []
    for sym in symbols:
        low = sym.lower()
        out += [f"{low}@bookTicker", f"{low}@aggTrade",
                f"{low}@{depth_stream}"]
    return out


async def run_window(cfg: dict, symbols: List[str]) -> None:
    """Records until the current A1 window closes."""
    stop = a1_window_end(utcnow())
    root = pathlib.Path(cfg["output_root"])
    writers = make_binance_writers(root, include_depth=True)
    streams = build_streams(symbols, cfg["depth_stream"])
    logger.info("window until %s: %d symbols, %d streams", stop,
                len(symbols), len(streams))
    tasks = [asyncio.create_task(record_connection(chunk, writers, stop))
             for chunk in chunk_streams(streams)]
    try:
        await asyncio.gather(*tasks)
    finally:
        for writer in writers.values():
            writer.close()


async def main_async(cfg: dict) -> None:
    """Forever: sleep to the next window, record through it."""
    symbols = resolve_universe(cfg)
    logger.info("universe: %d symbols (excluded %d)", len(symbols),
                len(cfg["exclude"]))
    while True:
        now = utcnow()
        if not a1_window_active(now):
            start = next_a1_window_start(now)
            logger.info("outside window; sleeping until %s", start)
            await asyncio.sleep((start - now).total_seconds() + 1)
            continue
        await run_window(cfg, symbols)


def main() -> None:
    """CLI entry."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="recorders/a1_config.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(main_async(load_config(args.config)))


if __name__ == "__main__":
    main()
