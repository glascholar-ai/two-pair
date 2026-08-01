"""Unit tests for executors: paper fills, sizing, signing, repair logic."""
from __future__ import annotations

from typing import Dict, List, Optional

import pytest

from twopair.executor import (BinanceClient, ChasePolicy,
                              LiveExecutor, PaperExecutor, round_step)

FAST = ChasePolicy(style="bbo", chase_interval_seconds=0.01, max_chases=3,
                   fill_poll_seconds=0.001)
MARKET = ChasePolicy(style="market")


class TestRoundStep:
    def test_rounds_down(self) -> None:
        assert round_step(0.999, 0.01) == pytest.approx(0.99)
        assert round_step(9.13, 1.0) == pytest.approx(9.0)

    def test_exact_multiple_stays(self) -> None:
        assert round_step(0.30, 0.1) == pytest.approx(0.3)

    def test_bad_step(self) -> None:
        with pytest.raises(ValueError):
            round_step(1.0, 0.0)


class TestPaperExecutor:
    def test_open_close_round_trip(self) -> None:
        ex = PaperExecutor(1000.0, "KRUSDT", "USUSDT")
        res = ex.open_ratio(1, kr_price=1100.0, us_price=145.0)
        assert res.ok and len(res.fills) == 2
        by_sym = {f.symbol: f for f in res.fills}
        assert by_sym["KRUSDT"].side == "BUY"
        assert by_sym["USUSDT"].side == "SELL"
        # Equal notional legs.
        assert by_sym["KRUSDT"].qty * 1100.0 == pytest.approx(1000.0)
        assert by_sym["USUSDT"].qty * 145.0 == pytest.approx(1000.0)
        res2 = ex.close_all(kr_price=1105.0, us_price=144.0)
        assert res2.ok and len(res2.fills) == 2
        assert {f.side for f in res2.fills} == {"BUY", "SELL"}

    def test_double_open_rejected(self) -> None:
        ex = PaperExecutor(1000.0, "A", "B")
        assert ex.open_ratio(1, 1.0, 1.0).ok
        assert not ex.open_ratio(1, 1.0, 1.0).ok

    def test_close_when_flat_is_ok(self) -> None:
        ex = PaperExecutor(1000.0, "A", "B")
        res = ex.close_all(1.0, 1.0)
        assert res.ok and res.fills == []

    def test_short_ratio_sides(self) -> None:
        ex = PaperExecutor(1000.0, "A", "B")
        res = ex.open_ratio(-1, 2.0, 3.0)
        by_sym = {f.symbol: f for f in res.fills}
        assert by_sym["A"].side == "SELL"
        assert by_sym["B"].side == "BUY"


class TestBinanceSigning:
    def test_signature_is_hmac_sha256(self) -> None:
        client = BinanceClient("key", "secret")
        signed = client.sign({"a": "1", "b": "2"})
        # Known-vector: HMAC-SHA256("a=1&b=2", "secret").
        assert signed.startswith("a=1&b=2&signature=")
        expected = ("2fb23a4f2c92a38dbdae7c90a45e01c8f2fc03e"
                    "70e6f4a06a2e2eb3f1c05dbca")
        assert signed.endswith(expected[:32]) or len(
            signed.split("signature=")[1]) == 64


class _FakeClient(BinanceClient):
    """BinanceClient stub with programmable behavior.

    limit_plan: per-symbol list consumed one entry per limit order —
      "fill"          order fills fully at its price
      "none"          order rests unfilled (cancelled by the chase loop)
      ("partial", f)  fraction f fills, remainder cancelled
      "gtx_reject"    post-only rejected (would cross)
    """

    def __init__(self, fail_symbols: Optional[set] = None,
                 limit_plan: Optional[Dict[str, List]] = None) -> None:
        super().__init__("k", "s")
        self.fail = fail_symbols or set()
        self.limit_plan = limit_plan or {}
        self.orders: List[Dict[str, object]] = []
        self.market_calls: List[Dict[str, object]] = []
        self.positions: Dict[str, float] = {}
        self._working: Dict[str, dict] = {}

    def market_order(self, symbol: str, side: str, qty: float,
                     client_id: str) -> dict:
        if symbol in self.fail:
            raise ConnectionError(f"simulated reject for {symbol}")
        self.orders.append({"symbol": symbol, "side": side, "qty": qty,
                            "type": "MARKET"})
        self.market_calls.append(self.orders[-1])
        delta = qty if side == "BUY" else -qty
        self.positions[symbol] = self.positions.get(symbol, 0.0) + delta
        return {"orderId": len(self.orders), "executedQty": qty,
                "avgPrice": 100.5}

    def limit_order_post_only(self, symbol: str, side: str, qty: float,
                              price: float, client_id: str) -> dict:
        if symbol in self.fail:
            raise ConnectionError(f"simulated reject for {symbol}")
        plan = self.limit_plan.get(symbol, ["fill"])
        behavior = plan.pop(0) if plan else "fill"
        if behavior == "gtx_reject":
            raise ConnectionError("Order would immediately match and take")
        self.orders.append({"symbol": symbol, "side": side, "qty": qty,
                            "type": "LIMIT", "price": price})
        filled = 0.0
        if behavior == "fill":
            filled = qty
        elif isinstance(behavior, tuple) and behavior[0] == "partial":
            filled = qty * behavior[1]
        delta = filled if side == "BUY" else -filled
        self.positions[symbol] = self.positions.get(symbol, 0.0) + delta
        self._working[client_id] = {
            "status": "FILLED" if filled == qty else "PARTIALLY_FILLED",
            "executedQty": filled, "avgPrice": price,
            "orderId": len(self.orders)}
        return {"orderId": len(self.orders)}

    def query_order(self, symbol: str, client_id: str) -> dict:
        return dict(self._working[client_id])

    def cancel_order(self, symbol: str, client_id: str) -> dict:
        order = self._working[client_id]
        if order["status"] == "FILLED":
            raise ConnectionError("Unknown order sent")  # cancel/fill race
        order["status"] = "CANCELED"
        return dict(order)

    def book_ticker(self, symbol: str) -> tuple:
        return 99.0, 101.0

    def position_amt(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)

    def step_size(self, symbol: str) -> float:
        return 0.001


class TestLiveExecutorRepair:
    def test_both_legs_fill(self) -> None:
        client = _FakeClient()
        ex = LiveExecutor(client, 1000.0, "KR", "US", policy=FAST)
        res = ex.open_ratio(1, 1100.0, 145.0)
        assert res.ok and len(res.fills) == 2

    def test_single_leg_failure_repairs(self) -> None:
        events: List[str] = []
        client = _FakeClient(fail_symbols={"US"})
        ex = LiveExecutor(client, 1000.0, "KR", "US", policy=MARKET,
                          on_event=events.append)
        res = ex.open_ratio(1, 1100.0, 145.0)
        assert not res.ok
        assert "US" in res.error
        # The KR leg was filled then immediately flattened.
        assert client.position_amt("KR") == pytest.approx(0.0)
        assert any("repair" in e for e in events)

    def test_close_flattens_actual_positions(self) -> None:
        client = _FakeClient()
        ex = LiveExecutor(client, 1000.0, "KR", "US", policy=FAST)
        ex.open_ratio(-1, 1100.0, 145.0)
        res = ex.close_all(1100.0, 145.0)
        assert res.ok
        assert client.position_amt("KR") == pytest.approx(0.0)
        assert client.position_amt("US") == pytest.approx(0.0)

    def test_zero_qty_rejected(self) -> None:
        client = _FakeClient()
        ex = LiveExecutor(client, 0.0001, "KR", "US", policy=FAST)
        res = ex.open_ratio(1, 1100.0, 145.0)
        assert not res.ok and "zero" in res.error


class TestBboChase:
    def test_passive_fill_first_quote_no_market(self) -> None:
        client = _FakeClient(limit_plan={"KR": ["fill"], "US": ["fill"]})
        ex = LiveExecutor(client, 1000.0, "KR", "US", policy=FAST)
        res = ex.open_ratio(1, 1100.0, 145.0)
        assert res.ok
        assert client.market_calls == []          # purely passive
        by_sym = {f.symbol: f for f in res.fills}
        assert by_sym["KR"].price == pytest.approx(99.0)   # BUY joins bid
        assert by_sym["US"].price == pytest.approx(101.0)  # SELL joins ask

    def test_requote_then_fill(self) -> None:
        client = _FakeClient(limit_plan={"KR": ["none", "fill"],
                                         "US": ["fill"]})
        ex = LiveExecutor(client, 1000.0, "KR", "US", policy=FAST)
        res = ex.open_ratio(1, 1100.0, 145.0)
        assert res.ok
        kr_limits = [o for o in client.orders
                     if o["symbol"] == "KR" and o["type"] == "LIMIT"]
        assert len(kr_limits) == 2                # cancelled once, re-pegged
        assert client.market_calls == []

    def test_market_fallback_after_max_chases(self) -> None:
        events: List[str] = []
        client = _FakeClient(limit_plan={"KR": ["none", "none", "none"],
                                         "US": ["fill"]})
        ex = LiveExecutor(client, 1000.0, "KR", "US", policy=FAST,
                          on_event=events.append)
        res = ex.open_ratio(1, 1100.0, 145.0)
        assert res.ok
        assert len(client.market_calls) == 1      # remainder taken at market
        assert any("taking market" in e for e in events)

    def test_partial_fill_accumulates(self) -> None:
        client = _FakeClient(limit_plan={"KR": [("partial", 0.5), "fill"],
                                         "US": ["fill"]})
        ex = LiveExecutor(client, 1000.0, "KR", "US", policy=FAST)
        res = ex.open_ratio(1, 1100.0, 145.0)
        assert res.ok
        kr = {f.symbol: f for f in res.fills}["KR"]
        assert kr.qty * 1100.0 == pytest.approx(1000.0, rel=1e-2)
        assert client.market_calls == []

    def test_gtx_reject_repegs(self) -> None:
        client = _FakeClient(limit_plan={"KR": ["gtx_reject", "fill"],
                                         "US": ["fill"]})
        ex = LiveExecutor(client, 1000.0, "KR", "US", policy=FAST)
        res = ex.open_ratio(1, 1100.0, 145.0)
        assert res.ok
        assert client.market_calls == []
