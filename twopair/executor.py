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


@dataclasses.dataclass(frozen=True)
class PairView:
    """Exchange truth for the pair (plan-A per-cycle sync).

    pnl_pct = (sum of unrealized PnL + funding income since entry_ts)
    as a percentage of single-leg notional.
    """

    kr_qty: float   # signed position amount, KR leg
    us_qty: float   # signed position amount, US leg
    pnl_pct: float


class Executor:
    """Interface: open or close the ratio position with equal leg notionals."""

    def open_ratio(self, side: int, kr_price: float,
                   us_price: float) -> PairExecution:
        """Opens side=+1 (long KR / short US) or side=-1 (reverse)."""
        raise NotImplementedError

    def close_all(self, kr_price: float, us_price: float) -> PairExecution:
        """Flattens both legs of the currently held position."""
        raise NotImplementedError

    def position_view(self,
                      entry_ts: Optional[dt.datetime]) -> Optional[PairView]:
        """Reads exchange truth for the pair.

        Returns None when no external source of truth exists (paper mode);
        the caller must then trust its local state.
        """
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

    def position_view(self,
                      entry_ts: Optional[dt.datetime]) -> Optional[PairView]:
        """Paper mode has no external truth; local state is authoritative."""
        return None


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
                signed: bool = False):
        """Performs one REST call and returns the parsed JSON body.

        Returns:
            Parsed JSON: dict for most endpoints, list for a few
            (positionRisk, income).
        """
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

    def limit_order_post_only(self, symbol: str, side: str, qty: float,
                              price: float, client_id: str) -> dict:
        """Places a post-only (GTX) LIMIT order.

        GTX is rejected by the exchange if it would cross the book, which is
        exactly what we want: the order either joins the queue or fails fast.
        """
        return self.request("POST", "/fapi/v1/order", {
            "symbol": symbol, "side": side, "type": "LIMIT",
            "timeInForce": "GTX", "quantity": f"{qty}", "price": f"{price}",
            "newClientOrderId": client_id,
        }, signed=True)

    def cancel_order(self, symbol: str, client_id: str) -> dict:
        """Cancels an order by client id; returns the exchange response."""
        return self.request("DELETE", "/fapi/v1/order", {
            "symbol": symbol, "origClientOrderId": client_id,
        }, signed=True)

    def query_order(self, symbol: str, client_id: str) -> dict:
        """Fetches order state (status / executedQty / avgPrice)."""
        return self.request("GET", "/fapi/v1/order", {
            "symbol": symbol, "origClientOrderId": client_id,
        }, signed=True)

    def book_ticker(self, symbol: str) -> tuple:
        """Returns (best_bid, best_ask) for a symbol."""
        data = self.request("GET", "/fapi/v1/ticker/bookTicker",
                            {"symbol": symbol})
        return float(data["bidPrice"]), float(data["askPrice"])

    def position_amt(self, symbol: str) -> float:
        """Returns the signed position amount for a symbol."""
        rows = self.request("GET", "/fapi/v2/positionRisk",
                            {"symbol": symbol}, signed=True)
        return float(rows[0]["positionAmt"]) if rows else 0.0

    def position_risk_all(self) -> List[dict]:
        """Returns positionRisk rows for all symbols (single call)."""
        rows = self.request("GET", "/fapi/v2/positionRisk", {}, signed=True)
        return list(rows) if isinstance(rows, list) else []

    def funding_income(self, symbol: str, start_ms: int) -> float:
        """Sums FUNDING_FEE income for a symbol since start_ms (USDT)."""
        rows = self.request("GET", "/fapi/v1/income", {
            "symbol": symbol, "incomeType": "FUNDING_FEE",
            "startTime": str(start_ms), "limit": "1000",
        }, signed=True)
        if not isinstance(rows, list):
            return 0.0
        return sum(float(r.get("income", 0.0)) for r in rows)

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


@dataclasses.dataclass(frozen=True)
class ChasePolicy:
    """Passive-execution parameters for the BBO chase loop."""

    style: str = "bbo"                 # "bbo" | "market"
    chase_interval_seconds: float = 4.0
    max_chases: int = 5
    fill_poll_seconds: float = 0.5


class LiveExecutor(Executor):
    """Two-leg execution on Binance with broken-leg repair.

    Legs are worked passively: a post-only limit joins the touch (BUY at
    best bid / SELL at best ask). If the quote is not fully filled within
    the chase interval it is cancelled and re-pegged to the fresh BBO; after
    max_chases the remainder is taken with a market order — a pair leg must
    always complete.
    """

    def __init__(self, client: BinanceClient, leg_notional_usdt: float,
                 kr_symbol: str, us_symbol: str,
                 policy: Optional[ChasePolicy] = None,
                 on_event: Optional[Callable[[str], object]] = None) -> None:
        self._client = client
        self._notional = leg_notional_usdt
        self._kr = kr_symbol
        self._us = us_symbol
        self._policy = policy or ChasePolicy()
        self._on_event = on_event or (lambda msg: None)
        self._steps: Dict[str, float] = {}

    def _step(self, symbol: str) -> float:
        if symbol not in self._steps:
            self._steps[symbol] = self._client.step_size(symbol)
        return self._steps[symbol]

    def _market(self, symbol: str, side: str, qty: float,
                purpose: str) -> LegFill:
        client_id = f"tp-{purpose}-{uuid.uuid4().hex[:10]}"
        resp = self._client.market_order(symbol, side, qty, client_id)
        price = float(resp.get("avgPrice") or 0.0) or float(
            resp.get("price") or 0.0)
        return LegFill(symbol, side, float(resp.get("executedQty") or qty),
                       price, str(resp.get("orderId", client_id)))

    def _await_fill(self, symbol: str, client_id: str,
                    deadline: float) -> dict:
        """Polls an order until FILLED or the deadline passes."""
        while True:
            order = self._client.query_order(symbol, client_id)
            if order.get("status") == "FILLED" or time.time() >= deadline:
                return order
            time.sleep(self._policy.fill_poll_seconds)

    def _reap(self, symbol: str, client_id: str) -> tuple:
        """Cancels a working order; returns (filled_qty, avg_price).

        A cancel that races a fill ("order does not exist" / already FILLED)
        is resolved by querying final state.
        """
        try:
            order = self._client.cancel_order(symbol, client_id)
        except Exception:  # noqa: BLE001 — cancel/fill race
            order = self._client.query_order(symbol, client_id)
        qty = float(order.get("executedQty") or 0.0)
        price = float(order.get("avgPrice") or 0.0)
        return qty, price

    def _place(self, symbol: str, side: str, qty: float,
               purpose: str) -> LegFill:
        """Executes one leg per the policy (passive chase, market fallback)."""
        if self._policy.style == "market":
            return self._market(symbol, side, qty, purpose)
        step = self._step(symbol)
        remaining = qty
        filled_qty = 0.0
        filled_notional = 0.0
        last_id = ""
        for _ in range(self._policy.max_chases):
            bid, ask = self._client.book_ticker(symbol)
            price = bid if side == "BUY" else ask
            client_id = f"tp-{purpose}-{uuid.uuid4().hex[:10]}"
            try:
                self._client.limit_order_post_only(symbol, side, remaining,
                                                   price, client_id)
            except Exception:  # noqa: BLE001 — GTX reject: book moved; re-peg
                continue
            last_id = client_id
            deadline = time.time() + self._policy.chase_interval_seconds
            order = self._await_fill(symbol, client_id, deadline)
            if order.get("status") == "FILLED":
                got = float(order.get("executedQty") or remaining)
                filled_qty += got
                filled_notional += got * float(order.get("avgPrice") or price)
                remaining = 0.0
                break
            got, avg = self._reap(symbol, client_id)
            if got > 0:
                filled_qty += got
                filled_notional += got * (avg or price)
                remaining = round_step(remaining - got, step)
            if remaining <= 0:
                break
        if remaining > 0:
            self._on_event(f"{symbol}: {remaining} unfilled after "
                           f"{self._policy.max_chases} chases; taking market")
            fill = self._market(symbol, side, remaining, purpose)
            filled_qty += fill.qty
            filled_notional += fill.qty * fill.price
            last_id = fill.order_id
        avg_price = filled_notional / filled_qty if filled_qty > 0 else 0.0
        return LegFill(symbol, side, filled_qty, avg_price, last_id)

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

    def position_view(self,
                      entry_ts: Optional[dt.datetime]) -> Optional[PairView]:
        """Reads both legs and real-money PnL from the exchange.

        pnl_pct combines unRealizedProfit of both legs with FUNDING_FEE
        income accrued since entry_ts (settled funding leaves the position
        and lands in the wallet, so unrealized alone would miss it).
        """
        rows = self._client.position_risk_all()
        qty = {self._kr: 0.0, self._us: 0.0}
        unreal = 0.0
        for row in rows:
            symbol = str(row.get("symbol", ""))
            if symbol in qty:
                qty[symbol] = float(row.get("positionAmt") or 0.0)
                unreal += float(row.get("unRealizedProfit") or 0.0)
        funding = 0.0
        if entry_ts is not None and (qty[self._kr] != 0 or qty[self._us] != 0):
            start_ms = int(entry_ts.timestamp() * 1000)
            for symbol in (self._kr, self._us):
                funding += self._client.funding_income(symbol, start_ms)
        pnl_pct = (unreal + funding) / self._notional * 100.0
        return PairView(kr_qty=qty[self._kr], us_qty=qty[self._us],
                        pnl_pct=pnl_pct)
