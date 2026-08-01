"""Order execution: paper simulator and Binance USDT-M live executor.

Both implement the same interface: open/close an equal-notional two-leg
ratio position. The live executor places both market orders concurrently,
verifies fills, and if exactly one leg fails it immediately flattens the
filled leg (never run a naked single leg).
"""
from __future__ import annotations

import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import hmac
import json
import logging
import math
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class LegFill:
    """One executed leg."""

    symbol: str
    side: str          # BUY | SELL
    qty: float
    price: float
    order_id: str


@dataclasses.dataclass(frozen=True)
class PairExecution:
    """Result of a two-leg action. ok=False means we are (or were) broken."""

    ok: bool
    fills: List[LegFill]
    error: str = ""


class Executor:
    """Interface: open or close the ratio position with equal leg notionals."""

    def open_ratio(self, side: int, kr_price: float,
                   us_price: float) -> PairExecution:
        """Opens side=+1 (long KR / short US) or side=-1 (reverse)."""
        raise NotImplementedError

    def close_all(self, kr_price: float, us_price: float) -> PairExecution:
        """Flattens both legs of the currently held position."""
        raise NotImplementedError


class PaperExecutor(Executor):
    """Fills instantly at the provided reference prices."""

    def __init__(self, leg_notional_usdt: float, kr_symbol: str,
                 us_symbol: str) -> None:
        self._notional = leg_notional_usdt
        self._kr = kr_symbol
        self._us = us_symbol
        self._open: Dict[str, LegFill] = {}

    def open_ratio(self, side: int, kr_price: float,
                   us_price: float) -> PairExecution:
        if self._open:
            return PairExecution(False, [], "position already open")
        kr_side = "BUY" if side > 0 else "SELL"
        us_side = "SELL" if side > 0 else "BUY"
        fills = [
            LegFill(self._kr, kr_side, self._notional / kr_price, kr_price,
                    f"paper-{uuid.uuid4().hex[:8]}"),
            LegFill(self._us, us_side, self._notional / us_price, us_price,
                    f"paper-{uuid.uuid4().hex[:8]}"),
        ]
        self._open = {f.symbol: f for f in fills}
        return PairExecution(True, fills)

    def close_all(self, kr_price: float, us_price: float) -> PairExecution:
        if not self._open:
            return PairExecution(True, [])
        prices = {self._kr: kr_price, self._us: us_price}
        fills = [
            LegFill(f.symbol, "SELL" if f.side == "BUY" else "BUY", f.qty,
                    prices[f.symbol], f"paper-{uuid.uuid4().hex[:8]}")
            for f in self._open.values()
        ]
        self._open = {}
        return PairExecution(True, fills)


class BinanceClient:
    """Minimal signed REST client for Binance USDT-M futures."""

    def __init__(self, api_key: str, api_secret: str,
                 base: str = "https://fapi.binance.com") -> None:
        self._key = api_key
        self._secret = api_secret.encode()
        self._base = base

    def sign(self, params: Dict[str, str]) -> str:
        """Returns the signed query string for the given params."""
        query = urllib.parse.urlencode(params)
        sig = hmac.new(self._secret, query.encode(), hashlib.sha256).hexdigest()
        return f"{query}&signature={sig}"

    def request(self, method: str, path: str,
                params: Optional[Dict[str, str]] = None,
                signed: bool = False) -> dict:
        """Performs one REST call and returns the parsed JSON body."""
        params = dict(params or {})
        if signed:
            params["timestamp"] = str(int(time.time() * 1000))
            params["recvWindow"] = "5000"
            query = self.sign(params)
        else:
            query = urllib.parse.urlencode(params)
        url = f"{self._base}{path}"
        data: Optional[bytes] = None
        if method == "GET":
            url = f"{url}?{query}" if query else url
        else:
            data = query.encode()
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"X-MBX-APIKEY": self._key} if self._key else {})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.load(resp)

    def market_order(self, symbol: str, side: str, qty: float,
                     client_id: str) -> dict:
        """Places a MARKET order and returns the exchange response."""
        return self.request("POST", "/fapi/v1/order", {
            "symbol": symbol, "side": side, "type": "MARKET",
            "quantity": f"{qty}", "newClientOrderId": client_id,
        }, signed=True)

    def position_amt(self, symbol: str) -> float:
        """Returns the signed position amount for a symbol."""
        rows = self.request("GET", "/fapi/v2/positionRisk",
                            {"symbol": symbol}, signed=True)
        return float(rows[0]["positionAmt"]) if rows else 0.0

    def step_size(self, symbol: str) -> float:
        """Returns the LOT_SIZE step for a symbol."""
        info = self.request("GET", "/fapi/v1/exchangeInfo",
                            {"symbol": symbol})
        for sym in info["symbols"]:
            if sym["symbol"] == symbol:
                for flt in sym["filters"]:
                    if flt["filterType"] == "LOT_SIZE":
                        return float(flt["stepSize"])
        raise ValueError(f"no LOT_SIZE for {symbol}")


def round_step(qty: float, step: float) -> float:
    """Rounds a quantity DOWN to the exchange step size."""
    if step <= 0:
        raise ValueError("step must be positive")
    return math.floor(qty / step + 1e-9) * step


class LiveExecutor(Executor):
    """Two-leg market execution on Binance with broken-leg repair."""

    def __init__(self, client: BinanceClient, leg_notional_usdt: float,
                 kr_symbol: str, us_symbol: str,
                 on_event: Optional[Callable[[str], None]] = None) -> None:
        self._client = client
        self._notional = leg_notional_usdt
        self._kr = kr_symbol
        self._us = us_symbol
        self._on_event = on_event or (lambda msg: None)
        self._steps: Dict[str, float] = {}

    def _step(self, symbol: str) -> float:
        if symbol not in self._steps:
            self._steps[symbol] = self._client.step_size(symbol)
        return self._steps[symbol]

    def _place(self, symbol: str, side: str, qty: float,
               purpose: str) -> LegFill:
        client_id = f"tp-{purpose}-{uuid.uuid4().hex[:10]}"
        resp = self._client.market_order(symbol, side, qty, client_id)
        price = float(resp.get("avgPrice") or 0.0) or float(
            resp.get("price") or 0.0)
        return LegFill(symbol, side, float(resp.get("executedQty") or qty),
                       price, str(resp.get("orderId", client_id)))

    def _both(self, orders: List[tuple], purpose: str) -> PairExecution:
        """Places two legs concurrently; repairs if exactly one fails."""
        fills: List[LegFill] = []
        errors: List[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futs = {pool.submit(self._place, s, side, q, purpose): (s, side)
                    for s, side, q in orders}
            for fut in concurrent.futures.as_completed(futs):
                try:
                    fills.append(fut.result())
                except Exception as err:  # noqa: BLE001 — must not hang a leg
                    errors.append(f"{futs[fut][0]}: {err}")
        if not errors:
            return PairExecution(True, fills)
        if len(fills) == 1:  # exactly one leg on — flatten it immediately
            leg = fills[0]
            repair_side = "SELL" if leg.side == "BUY" else "BUY"
            self._on_event(f"single-leg failure ({errors}); repairing "
                           f"{leg.symbol}")
            try:
                repair = self._place(leg.symbol, repair_side, leg.qty,
                                     "repair")
                fills.append(repair)
            except Exception as err:  # noqa: BLE001
                errors.append(f"repair failed: {err}")
        return PairExecution(False, fills, "; ".join(errors))

    def open_ratio(self, side: int, kr_price: float,
                   us_price: float) -> PairExecution:
        kr_qty = round_step(self._notional / kr_price, self._step(self._kr))
        us_qty = round_step(self._notional / us_price, self._step(self._us))
        if kr_qty <= 0 or us_qty <= 0:
            return PairExecution(False, [], "quantity rounds to zero")
        kr_side = "BUY" if side > 0 else "SELL"
        us_side = "SELL" if side > 0 else "BUY"
        return self._both([(self._kr, kr_side, kr_qty),
                           (self._us, us_side, us_qty)], "open")

    def close_all(self, kr_price: float, us_price: float) -> PairExecution:
        orders: List[tuple] = []
        for symbol in (self._kr, self._us):
            amt = self._client.position_amt(symbol)
            if abs(amt) > 0:
                orders.append((symbol, "SELL" if amt > 0 else "BUY", abs(amt)))
        if not orders:
            return PairExecution(True, [])
        return self._both(orders, "close")
