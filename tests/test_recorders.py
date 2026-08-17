"""Unit tests for the A1/A2 recorders (no network)."""
from __future__ import annotations

import datetime as dt
import pathlib
from typing import List

import pyarrow.parquet as pq
import pytest

from recorders.a1_orderbook import build_streams, load_config, resolve_universe
from recorders.a2_homeline import (build_binance_streams, ib_contract_label,
                                   load_config as load_a2_config)
from recorders.common import (AGG_TRADE_SCHEMA, BOOK_TICKER_SCHEMA,
                              DEPTH_SCHEMA, ParquetBufferWriter,
                              a1_window_active, a1_window_end, chunk_streams,
                              next_a1_window_start, parse_agg_trade,
                              parse_book_ticker, parse_depth, route_kind)

UTC = dt.timezone.utc


def ts(day: int, hour: int, minute: int = 0) -> dt.datetime:
    """August 2026: the 3rd is a Monday."""
    return dt.datetime(2026, 8, day, hour, minute, tzinfo=UTC)


class FakeClock:
    def __init__(self, now: dt.datetime) -> None:
        self.now = now

    def __call__(self) -> dt.datetime:
        return self.now


class TestWindows:
    def test_weekday_dead_zone(self) -> None:
        assert a1_window_active(ts(3, 0))          # Mon 00:00
        assert a1_window_active(ts(3, 7, 59))
        assert not a1_window_active(ts(3, 8))
        assert not a1_window_active(ts(3, 23, 59))

    def test_weekend_all_day(self) -> None:
        assert a1_window_active(ts(8, 12))         # Sat
        assert a1_window_active(ts(9, 23, 59))     # Sun

    def test_next_start(self) -> None:
        assert next_a1_window_start(ts(3, 2)) == ts(3, 2)      # already in
        assert next_a1_window_start(ts(3, 9)) == ts(4, 0)      # Mon -> Tue
        assert next_a1_window_start(ts(7, 15)) == ts(8, 0)     # Fri -> Sat

    def test_window_end(self) -> None:
        assert a1_window_end(ts(3, 2)) == ts(3, 8)             # Mon dead zone
        assert a1_window_end(ts(7, 5)) == ts(7, 8)             # Fri dead zone
        assert a1_window_end(ts(8, 12)) == ts(10, 8)           # Sat -> Mon 08
        assert a1_window_end(ts(9, 1)) == ts(10, 8)            # Sun -> Mon 08
        with pytest.raises(ValueError):
            a1_window_end(ts(3, 9))

    def test_chunking(self) -> None:
        streams = [f"s{i}" for i in range(310)]
        chunks = chunk_streams(streams, size=150)
        assert [len(c) for c in chunks] == [150, 150, 10]
        assert sum(chunks, []) == streams


class TestParsers:
    def test_book_ticker(self) -> None:
        row = parse_book_ticker(1000, {
            "s": "ASMLUSDT", "u": 7, "b": "812.5", "B": "3.2",
            "a": "812.9", "A": "1.1", "E": 5, "T": 4})
        assert row == {"t": 1000, "s": "ASMLUSDT", "u": 7, "bp": 812.5,
                       "bq": 3.2, "ap": 812.9, "aq": 1.1, "E": 5, "T": 4}

    def test_agg_trade(self) -> None:
        row = parse_agg_trade(1000, {
            "s": "MUUSDT", "a": 42, "p": "111.5", "q": "2", "m": True,
            "T": 999})
        assert row == {"t": 1000, "s": "MUUSDT", "a": 42, "p": 111.5,
                       "q": 2.0, "m": True, "T": 999}

    def test_depth(self) -> None:
        row = parse_depth(1000, {
            "s": "MUUSDT", "u": 9, "E": 2, "T": 1,
            "b": [["100.0", "1.5"], ["99.9", "2.0"]],
            "a": [["100.1", "0.5"]]})
        assert row["bp"] == [100.0, 99.9] and row["bq"] == [1.5, 2.0]
        assert row["ap"] == [100.1] and row["aq"] == [0.5]

    def test_route_kind(self) -> None:
        book = route_kind("bookTicker")
        depth = route_kind("depth20")
        assert book is not None and book[0] == "bookticker"
        assert depth is not None and depth[0] == "depth"
        assert route_kind("markPrice") is None


class TestParquetBufferWriter:
    def _rows(self, n: int) -> List[dict]:
        return [parse_book_ticker(i, {"s": "X", "u": i, "b": "1", "B": "1",
                                      "a": "1", "A": "1"}) for i in range(n)]

    def test_flush_on_row_count(self, tmp_path: pathlib.Path) -> None:
        clock = FakeClock(ts(3, 2))
        writer = ParquetBufferWriter(tmp_path, "bookticker",
                                     BOOK_TICKER_SCHEMA, flush_rows=5,
                                     clock=clock)
        for row in self._rows(5):
            writer.append(row)
        parts = list(tmp_path.glob("2026-08-03/bookticker/part-*.parquet"))
        assert len(parts) == 1
        table = pq.read_table(parts[0])
        assert table.num_rows == 5
        assert table.column("t").to_pylist() == [0, 1, 2, 3, 4]

    def test_flush_on_age(self, tmp_path: pathlib.Path) -> None:
        clock = FakeClock(ts(3, 2))
        writer = ParquetBufferWriter(tmp_path, "bookticker",
                                     BOOK_TICKER_SCHEMA, flush_rows=10_000,
                                     flush_seconds=300, clock=clock)
        writer.append(self._rows(1)[0])
        clock.now += dt.timedelta(seconds=301)
        writer.append(self._rows(1)[0])
        assert len(list(tmp_path.rglob("*.parquet"))) == 1

    def test_close_flushes_remainder(self, tmp_path: pathlib.Path) -> None:
        writer = ParquetBufferWriter(tmp_path, "aggtrade", AGG_TRADE_SCHEMA,
                                     clock=FakeClock(ts(3, 2)))
        writer.append(parse_agg_trade(5, {"s": "X", "a": 1, "p": "1",
                                          "q": "1", "m": False, "T": 2}))
        writer.close()
        parts = list(tmp_path.rglob("*.parquet"))
        assert len(parts) == 1 and pq.read_table(parts[0]).num_rows == 1

    def test_no_empty_files(self, tmp_path: pathlib.Path) -> None:
        writer = ParquetBufferWriter(tmp_path, "depth", DEPTH_SCHEMA,
                                     clock=FakeClock(ts(3, 2)))
        writer.flush()
        writer.close()
        assert list(tmp_path.rglob("*.parquet")) == []

    def test_same_second_parts_get_distinct_names(
            self, tmp_path: pathlib.Path) -> None:
        clock = FakeClock(ts(3, 2))
        writer = ParquetBufferWriter(tmp_path, "bookticker",
                                     BOOK_TICKER_SCHEMA, flush_rows=2,
                                     clock=clock)
        for row in self._rows(4):
            writer.append(row)
        assert len(list(tmp_path.rglob("*.parquet"))) == 2

    def test_daily_rotation_by_flush_date(self, tmp_path: pathlib.Path) -> None:
        clock = FakeClock(ts(8, 23, 59))
        writer = ParquetBufferWriter(tmp_path, "depth", DEPTH_SCHEMA,
                                     clock=clock)
        writer.append(parse_depth(1, {"s": "X", "b": [], "a": []}))
        writer.flush()
        clock.now = ts(9, 0, 1)
        writer.append(parse_depth(2, {"s": "X", "b": [], "a": []}))
        writer.flush()
        assert (tmp_path / "2026-08-08" / "depth").exists()
        assert (tmp_path / "2026-08-09" / "depth").exists()

    def test_list_columns_round_trip(self, tmp_path: pathlib.Path) -> None:
        writer = ParquetBufferWriter(tmp_path, "depth", DEPTH_SCHEMA,
                                     clock=FakeClock(ts(3, 2)))
        writer.append(parse_depth(1, {
            "s": "X", "b": [["1.0", "2.0"]], "a": [["1.1", "3.0"]]}))
        writer.close()
        table = pq.read_table(next(tmp_path.rglob("*.parquet")))
        assert table.column("bp").to_pylist() == [[1.0]]
        assert table.column("aq").to_pylist() == [[3.0]]


class TestA1Config:
    def test_load_and_universe_from_list(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "cfg.json"
        path.write_text('{"universe": ["AUSDT", "BUSDT", "SKHYUSDT"],'
                        ' "exclude": ["SKHYUSDT"]}')
        cfg = load_config(str(path))
        assert resolve_universe(cfg) == ["AUSDT", "BUSDT"]
        assert cfg["depth_stream"] == "depth20@500ms"

    def test_build_streams(self) -> None:
        streams = build_streams(["AUSDT"], "depth20@500ms")
        assert streams == ["ausdt@bookTicker", "ausdt@aggTrade",
                           "ausdt@depth20@500ms"]

    def test_repo_config_excludes_all_traded_symbols(self) -> None:
        cfg = load_config("recorders/a1_config.json")
        import glob
        import json as _json
        for f in glob.glob("deploy/cfg-*.json"):
            pair = _json.loads(pathlib.Path(f).read_text())
            for key in ("kr_symbol", "us_symbol"):
                if key in pair:
                    assert pair[key] in cfg["exclude"], (f, key)
        # A2 pilot perp legs must not be quoted by A1 either.
        a2 = load_a2_config("recorders/a2_config.json")
        for sym in a2["binance_symbols"]:
            assert sym in cfg["exclude"]


class TestA2Config:
    def test_defaults(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "cfg.json"
        path.write_text('{"binance_symbols": ["ASMLUSDT"]}')
        cfg = load_a2_config(str(path))
        assert cfg["ib"]["host"] == "127.0.0.1"
        assert cfg["ib"]["contracts"] == []
        assert build_binance_streams(cfg["binance_symbols"]) == [
            "asmlusdt@bookTicker", "asmlusdt@aggTrade"]

    def test_repo_config_contracts(self) -> None:
        cfg = load_a2_config("recorders/a2_config.json")
        labels = [ib_contract_label(c) for c in cfg["ib"]["contracts"]]
        assert "ASML.AEB" in labels and "700.SEHK" in labels
        assert len(labels) == len(set(labels))
