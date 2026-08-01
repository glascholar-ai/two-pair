#!/usr/bin/env python3
"""Live integration test: order placement, cancellation, and a round trip.

Runs against the REAL account configured in the environment. Phases:
  1. per leg: place a post-only limit 2% below the bid (unfillable in
     practice), verify it rests, cancel it, verify cancellation;
  2. cancel_all_open with two resting orders;
  3. full executor round trip: open the pair (BBO chase, tiny notional),
     verify both legs on the exchange, close, verify flat.

Any position or order left behind is cleaned up in a finally block.
STOP the twopair service before running this — its per-cycle sync would
treat test positions as orphans and repair them.

Usage:
    BINANCE_API_KEY=... BINANCE_API_SECRET=... \
        python3 scripts/integration_test.py --config deploy/cfg.json \
        [--skip-trade]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from twopair.config import Config, load_config
from twopair.executor import (BinanceClient, ChasePolicy, LiveExecutor,
                              format_step)

PASS, FAIL = "PASS", "FAIL"
_results: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """Records and prints one assertion."""
    _results.append(ok)
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))


def wait_status(client: BinanceClient, symbol: str, client_id: str,
                target: str, timeout: float = 6.0) -> str:
    """Polls an order until it reaches `target` status or times out.

    papi has read-after-write lag: immediate queries can 400 or return a
    stale status for ~1s after any mutation.
    """
    deadline = time.time() + timeout
    status = "?"
    while time.time() < deadline:
        try:
            status = str(client.query_order(symbol, client_id).get("status"))
            if status == target:
                return status
        except Exception:  # noqa: BLE001 — propagation lag
            status = "unqueryable"
        time.sleep(0.5)
    return status


def far_limit_roundtrip(client: BinanceClient, symbol: str) -> None:
    """Places an unfillable post-only order, verifies it rests, cancels."""
    bid, _ask = client.book_ticker(symbol)
    step, tick = client.symbol_filters(symbol)
    price = format_step(bid * 0.98, tick)
    qty = format_step(max(60.0, 25.0) / bid, step)
    client_id = f"itest-{symbol[:4].lower()}-{int(time.time())}"
    client.limit_order_post_only(symbol, "BUY", qty, price, client_id)
    status = wait_status(client, symbol, client_id, "NEW")
    check(f"{symbol} far limit rests", status == "NEW",
          f"qty={qty} price={price} status={status}")
    client.cancel_order(symbol, client_id)
    status = wait_status(client, symbol, client_id, "CANCELED")
    check(f"{symbol} cancel confirmed", status == "CANCELED",
          f"status={status}")


def cancel_all_test(client: BinanceClient, symbol: str) -> None:
    """Rests two far orders, then cancels everything at once."""
    bid, _ask = client.book_ticker(symbol)
    step, tick = client.symbol_filters(symbol)
    ids = []
    for i, off in enumerate((0.975, 0.97)):
        price = format_step(bid * off, tick)
        qty = format_step(60.0 / bid, step)
        client_id = f"itest-all-{i}-{int(time.time())}"
        client.limit_order_post_only(symbol, "BUY", qty, price, client_id)
        ids.append(client_id)
    client.cancel_all_open(symbol)
    statuses = [wait_status(client, symbol, cid, "CANCELED") for cid in ids]
    check(f"{symbol} cancel_all clears both",
          all(s == "CANCELED" for s in statuses), f"statuses={statuses}")


def flat(view_kr: float, view_us: float) -> bool:
    return abs(view_kr) < 1e-9 and abs(view_us) < 1e-9


def round_trip_test(executor: LiveExecutor, client: BinanceClient,
                    cfg: Config) -> None:
    """Opens the pair for real (tiny notional), verifies legs, closes."""
    kr_bid, kr_ask = client.book_ticker(cfg.kr_symbol)
    us_bid, us_ask = client.book_ticker(cfg.us_symbol)
    result = executor.open_ratio(1, (kr_bid + kr_ask) / 2,
                                 (us_bid + us_ask) / 2)
    check("open_ratio ok", result.ok and len(result.fills) == 2,
          f"fills={[(f.symbol, f.side, f.qty, f.price) for f in result.fills]}"
          f" err={result.error}")
    view = executor.position_view(None)
    check("exchange shows long KR / short US",
          view.kr_qty > 0 and view.us_qty < 0,
          f"kr={view.kr_qty} us={view.us_qty} pnl={view.pnl_pct:+.3f}%")
    result = executor.close_all(kr_bid, us_bid)
    check("close_all ok", result.ok, f"err={result.error}")
    time.sleep(3)
    view = executor.position_view(None)
    check("flat after close", flat(view.kr_qty, view.us_qty),
          f"kr={view.kr_qty} us={view.us_qty}")


def cleanup(executor: LiveExecutor, client: BinanceClient,
            cfg: Config) -> None:
    """Best-effort: cancel every order, flatten every position."""
    executor.cancel_all_open_orders()
    kr_bid, _ = client.book_ticker(cfg.kr_symbol)
    us_bid, _ = client.book_ticker(cfg.us_symbol)
    executor.close_all(kr_bid, us_bid)
    view = executor.position_view(None)
    if not flat(view.kr_qty, view.us_qty):
        print(f"  !! cleanup left kr={view.kr_qty} us={view.us_qty} — "
              "flatten manually")


def main() -> None:
    """Runs all phases; exits 1 if any check failed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-trade", action="store_true",
                        help="skip the real open/close round trip")
    args = parser.parse_args()
    cfg = load_config(args.config)
    key = os.environ.get("BINANCE_API_KEY", "")
    secret = os.environ.get("BINANCE_API_SECRET", "")
    if not key or not secret:
        raise SystemExit("BINANCE_API_KEY / BINANCE_API_SECRET required")
    client = BinanceClient(key, secret, cfg.binance_base,
                           pm=cfg.portfolio_margin)
    executor = LiveExecutor(client, cfg.leg_notional_usdt, cfg.kr_symbol,
                            cfg.us_symbol,
                            policy=ChasePolicy(chase_interval_seconds=3.0),
                            on_event=lambda m: print(f"  (exec) {m}"))
    print(f"integration test [{cfg.mode_label()}"
          f"{' pm' if cfg.portfolio_margin else ''}] "
          f"{cfg.kr_symbol}/{cfg.us_symbol} leg={cfg.leg_notional_usdt}")
    try:
        print("phase 1: far-limit place/cancel per leg")
        far_limit_roundtrip(client, cfg.kr_symbol)
        far_limit_roundtrip(client, cfg.us_symbol)
        print("phase 2: cancel_all_open")
        cancel_all_test(client, cfg.kr_symbol)
        if not args.skip_trade:
            print("phase 3: real round trip (BBO chase open -> close)")
            round_trip_test(executor, client, cfg)
    finally:
        print("cleanup")
        cleanup(executor, client, cfg)
    passed, total = sum(_results), len(_results)
    print(f"result: {passed}/{total} checks passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
