#!/usr/bin/env python3
"""A2 recorder: home-line (IBKR) ticks + Binance perp quotes/trades.

Purpose: build the dataset for the ADR/home-line lag trade
(extends/docs/scan/adr_homeline_basis.md). Primary target is ASML/ASML.AS
(lag2 14.8 bps/5m, t 7.4); the HK lines (700/1810 vs HK0700/TENCENT/HK1810)
had only 17 days of history and need another month of accumulation before
the re-audit; NVO is a candidate pending CPH data permission.

Two independent capture tasks:
  - Binance perp bookTicker + aggTrade for the configured symbols, 24/7
    (public WS, no API key).
  - IBKR home-line market data via ib_insync (``--with-ib``): requires a
    running IB gateway; ticks only arrive while the home market quotes, so
    no window logic is needed. Without the flag the recorder runs
    Binance-only — deployable before the gateway exists.

Usage:
    python3 recorders/a2_homeline.py [--config recorders/a2_config.json]
                                     [--with-ib]
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import pathlib
import sys
from typing import Any, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from recorders.common import (IB_TICKER_SCHEMA, ParquetBufferWriter,
                              make_binance_writers, record_connection, utcnow)

logger = logging.getLogger("recorder.a2")


def load_config(path: str) -> dict:
    """Loads the recorder config (Binance symbols, IB contracts, output)."""
    cfg = json.loads(pathlib.Path(path).read_text())
    cfg.setdefault("output_root", "data/recordings/a2")
    cfg.setdefault("binance_symbols", [])
    cfg.setdefault("ib", {})
    cfg["ib"].setdefault("host", "127.0.0.1")
    cfg["ib"].setdefault("port", 4002)
    cfg["ib"].setdefault("client_id", 17)
    cfg["ib"].setdefault("contracts", [])
    return cfg


def build_binance_streams(symbols: List[str]) -> List[str]:
    """bookTicker + aggTrade stream names (no depth: A2 trades taker/BBO)."""
    out: List[str] = []
    for sym in symbols:
        low = sym.lower()
        out += [f"{low}@bookTicker", f"{low}@aggTrade"]
    return out


def ib_contract_label(spec: dict) -> str:
    """Stable symbol label for stored rows, e.g. ASML.AEB or EUR.IDEALPRO."""
    return f"{spec['symbol']}.{spec['exchange']}"


def _f(value: Any) -> float:
    """IB tick field -> float (None/-1 sentinels become NaN-safe floats)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


async def run_ib(cfg: dict, writer: ParquetBufferWriter) -> None:
    """Streams IBKR ticks for the configured contracts (reconnects forever)."""
    # Optional server-side dependency; importlib keeps local pyright/test
    # environments free of it.
    ib_insync = importlib.import_module("ib_insync")
    ib_cfg = cfg["ib"]
    while True:
        ib = ib_insync.IB()
        try:
            await ib.connectAsync(ib_cfg["host"], ib_cfg["port"],
                                  clientId=ib_cfg["client_id"])
            contracts = []
            labels: dict = {}
            for spec in ib_cfg["contracts"]:
                contract = ib_insync.Contract(
                    secType=spec.get("sec_type", "STK"),
                    symbol=spec["symbol"], exchange=spec["exchange"],
                    currency=spec["currency"])
                contracts.append(contract)
            qualified = await ib.qualifyContractsAsync(*contracts)
            for spec, contract in zip(ib_cfg["contracts"], qualified):
                labels[contract.conId] = ib_contract_label(spec)
                ib.reqMktData(contract, "", False, False)
            logger.info("IB connected: %d contracts", len(qualified))

            def on_tickers(tickers: Any) -> None:
                t_ms = int(utcnow().timestamp() * 1000)
                for tk in tickers:
                    label = labels.get(tk.contract.conId)
                    if label is None:
                        continue
                    ts_ib = int(tk.time.timestamp() * 1000) if tk.time else 0
                    writer.append({
                        "t": t_ms, "s": label,
                        "bid": _f(tk.bid), "bid_sz": _f(tk.bidSize),
                        "ask": _f(tk.ask), "ask_sz": _f(tk.askSize),
                        "last": _f(tk.last), "last_sz": _f(tk.lastSize),
                        "volume": _f(tk.volume), "ts_ib": ts_ib})

            ib.pendingTickersEvent += on_tickers
            while ib.isConnected():
                await asyncio.sleep(5)
            raise ConnectionError("IB gateway disconnected")
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001 — reconnect forever
            logger.warning("IB error (%s); reconnecting in 30s", err)
            writer.flush()
            await asyncio.sleep(30)
        finally:
            if ib.isConnected():
                ib.disconnect()


async def main_async(cfg: dict, with_ib: bool) -> None:
    """Runs Binance capture (always) and IB capture (behind --with-ib)."""
    root = pathlib.Path(cfg["output_root"])
    tasks = []
    streams = build_binance_streams(cfg["binance_symbols"])
    if streams:
        writers = make_binance_writers(root, include_depth=False)
        tasks.append(asyncio.create_task(
            record_connection(streams, writers, stop_at=None)))
    if with_ib:
        ib_writer = ParquetBufferWriter(root, "ib_ticker", IB_TICKER_SCHEMA)
        tasks.append(asyncio.create_task(run_ib(cfg, ib_writer)))
    if not tasks:
        raise SystemExit("nothing to record: empty binance_symbols, no --with-ib")
    await asyncio.gather(*tasks)


def main() -> None:
    """CLI entry."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="recorders/a2_config.json")
    parser.add_argument("--with-ib", action="store_true",
                        help="also record IBKR home-line ticks (needs gateway)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(main_async(load_config(args.config), args.with_ib))


if __name__ == "__main__":
    main()
