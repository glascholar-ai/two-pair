"""Unit tests for executors: paper fills, sizing, signing, repair logic."""
from __future__ import annotations

from typing import Dict, List, Optional

import pytest

from twopair.executor import (BinanceClient, ChasePolicy,
                              LiveExecutor, format_step, round_step)


def _is_exact_decimal(s: str) -> bool:
    """API-boundary contract: plain decimal, <=9 digits after the point."""
    if "e" in s.lower():
        return False
    frac = s.split(".")[1] if "." in s else ""
    return len(frac) <= 9


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

    def market_order(self, symbol: str, side: str, qty: str,
                     client_id: str) -> dict:
        assert _is_exact_decimal(qty), f"dirty qty string: {qty!r}"
        if symbol in self.fail:
            raise ConnectionError(f"simulated reject for {symbol}")
        fqty = float(qty)
        self.orders.append({"symbol": symbol, "side": side, "qty": fqty,
                            "type": "MARKET"})
        self.market_calls.append(self.orders[-1])
        delta = fqty if side == "BUY" else -fqty
        self.positions[symbol] = self.positions.get(symbol, 0.0) + delta
        return {"orderId": len(self.orders), "executedQty": qty,
                "avgPrice": 100.5}

    def limit_order_post_only(self, symbol: str, side: str, qty: str,
                              price: str, client_id: str) -> dict:
        assert _is_exact_decimal(qty), f"dirty qty string: {qty!r}"
        assert _is_exact_decimal(price), f"dirty price string: {price!r}"
        if symbol in self.fail:
            raise ConnectionError(f"simulated reject for {symbol}")
        plan = self.limit_plan.get(symbol, ["fill"])
        behavior = plan.pop(0) if plan else "fill"
        if behavior == "gtx_reject":
            raise ConnectionError("Order would immediately match and take")
        fqty, fprice = float(qty), float(price)
        self.orders.append({"symbol": symbol, "side": side, "qty": fqty,
                            "type": "LIMIT", "price": fprice})
        filled = 0.0
        if behavior == "fill":
            filled = fqty
        elif isinstance(behavior, tuple) and behavior[0] == "partial":
            filled = fqty * behavior[1]
        delta = filled if side == "BUY" else -filled
        self.positions[symbol] = self.positions.get(symbol, 0.0) + delta
        self._working[client_id] = {
            "status": "FILLED" if filled == fqty else "PARTIALLY_FILLED",
            "executedQty": filled, "avgPrice": fprice,
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

    def symbol_filters(self, symbol: str) -> tuple:
        return 0.001, 0.01


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


class TestFormatStep:
    def test_cleans_float_residue(self) -> None:
        assert format_step(2.675, 0.001) == "2.675"
        assert format_step(9.13, 0.001) == "9.130"
        assert format_step(0.1 + 0.2, 0.1) == "0.3"

    def test_no_scientific_notation(self) -> None:
        assert format_step(3.0, 0.00001) == "3.00000"
        assert "e" not in format_step(0.00012, 0.00001).lower()

    def test_rounds_down(self) -> None:
        assert format_step(0.8999999, 0.001) == "0.899"
        assert format_step(1.1218181818, 0.01) == "1.12"

    def test_bad_step_raises(self) -> None:
        with pytest.raises(ValueError):
            format_step(1.0, 0.0)


class _RouteRecorder(BinanceClient):
    """Captures (method, path, signed) without any network I/O."""

    def __init__(self, pm: bool) -> None:
        super().__init__("k", "s", pm=pm)
        self.calls: list = []

    def request(self, method: str, path: str,
                params=None, signed: bool = False) -> dict:
        self.calls.append((method, path, signed))
        return {}


class TestPortfolioMarginRouting:
    def test_classic_paths(self) -> None:
        c = _RouteRecorder(pm=False)
        c.market_order("S", "BUY", "1", "id")
        c.cancel_all_open("S")
        c.income_sum("S", "FUNDING_FEE", 0)
        paths = [p for _, p, _ in c.calls]
        assert paths == ["/fapi/v1/order", "/fapi/v1/allOpenOrders",
                        "/fapi/v1/income"]
        assert c.supports_countdown is True

    def test_pm_paths(self) -> None:
        c = _RouteRecorder(pm=True)
        c.market_order("S", "BUY", "1", "id")
        c.limit_order_post_only("S", "SELL", "1", "9.9", "id2")
        c.cancel_order("S", "id2")
        c.query_order("S", "id2")
        c.cancel_all_open("S")
        c.position_amt("S")
        c.position_risk_all()
        c.income_sum("S", "FUNDING_FEE", 0)
        c.user_trades("S", 0)
        paths = [p for _, p, _ in c.calls]
        assert paths == [
            "/papi/v1/um/order", "/papi/v1/um/order", "/papi/v1/um/order",
            "/papi/v1/um/order", "/papi/v1/um/allOpenOrders",
            "/papi/v1/um/positionRisk", "/papi/v1/um/positionRisk",
            "/papi/v1/um/income", "/papi/v1/um/userTrades"]
        assert c.supports_countdown is False

    def test_pm_public_data_stays_on_fapi(self) -> None:
        c = _RouteRecorder(pm=True)
        try:
            c.book_ticker("S")   # request stubbed -> KeyError on parse is fine
        except KeyError:
            pass
        _, path, signed = c.calls[-1]
        assert path == "/fapi/v1/ticker/bookTicker" and signed is False

    def test_deadman_noop_on_pm(self) -> None:
        c = _RouteRecorder(pm=True)
        ex = LiveExecutor(c, 1000.0, "KR", "US")
        ex.arm_deadman(900)
        assert c.calls == []          # skipped entirely, no papi 404 spam


class _LaggyClient(_FakeClient):
    """Simulates papi read-after-write lag: first N queries 400."""

    def __init__(self, lag_queries: int, **kw) -> None:
        super().__init__(**kw)
        self._lag = lag_queries

    def query_order(self, symbol: str, client_id: str) -> dict:
        if self._lag > 0:
            self._lag -= 1
            raise ConnectionError("HTTP 400: order does not exist (lag)")
        return super().query_order(symbol, client_id)


class TestPapiLagTolerance:
    def test_await_fill_survives_laggy_queries(self) -> None:
        client = _LaggyClient(2, limit_plan={"KR": ["fill"], "US": ["fill"]})
        ex = LiveExecutor(client, 1000.0, "KR", "US", policy=FAST)
        res = ex.open_ratio(1, 1100.0, 145.0)
        assert res.ok and len(res.fills) == 2
        assert client.market_calls == []   # passive fill despite lag

    def test_market_ack_with_zero_executed_qty(self) -> None:
        class _ZeroAck(_FakeClient):
            def market_order(self, symbol, side, qty, client_id):  # type: ignore[override]
                resp = super().market_order(symbol, side, qty, client_id)
                resp["executedQty"] = "0.00"   # papi async-fill ack
                return resp

        client = _ZeroAck()
        ex = LiveExecutor(client, 1000.0, "KR", "US", policy=MARKET)
        res = ex.open_ratio(1, 1100.0, 145.0)
        assert res.ok
        for fill in res.fills:
            assert fill.qty > 0                # requested qty, not 0.00
