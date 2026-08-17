"""Shared recorder infrastructure: parquet writers, windows, WS capture.

Recorders are read-only (no API keys) and independent from the trading
system: they may die and restart freely; every record carries its local
receive timestamp so downstream analysis never depends on recorder uptime
bookkeeping.

Storage format: typed columnar parquet with zstd compression (raw JSONL
explodes on book data — repeated keys + stringified numbers). Rows buffer
in memory and flush as part files under <root>/<YYYY-MM-DD>/<stream>/;
a crash loses at most one buffer (flush_seconds worth of data).
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import pathlib
from typing import Callable, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

BINANCE_WS = "wss://fstream.binance.com/stream"
MAX_STREAMS_PER_CONN = 150

Row = Dict[str, object]
Parser = Callable[[int, dict], Row]

# t = local receive time (ms); E/T = exchange event/transaction time (ms).
BOOK_TICKER_SCHEMA = pa.schema([
    ("t", pa.int64()), ("s", pa.string()), ("u", pa.int64()),
    ("bp", pa.float64()), ("bq", pa.float64()),
    ("ap", pa.float64()), ("aq", pa.float64()),
    ("E", pa.int64()), ("T", pa.int64()),
])
AGG_TRADE_SCHEMA = pa.schema([
    ("t", pa.int64()), ("s", pa.string()), ("a", pa.int64()),
    ("p", pa.float64()), ("q", pa.float64()), ("m", pa.bool_()),
    ("T", pa.int64()),
])
DEPTH_SCHEMA = pa.schema([
    ("t", pa.int64()), ("s", pa.string()), ("u", pa.int64()),
    ("E", pa.int64()), ("T", pa.int64()),
    ("bp", pa.list_(pa.float64())), ("bq", pa.list_(pa.float64())),
    ("ap", pa.list_(pa.float64())), ("aq", pa.list_(pa.float64())),
])
# IBKR home-line ticker snapshots (NaN where the field is absent).
IB_TICKER_SCHEMA = pa.schema([
    ("t", pa.int64()), ("s", pa.string()),
    ("bid", pa.float64()), ("bid_sz", pa.float64()),
    ("ask", pa.float64()), ("ask_sz", pa.float64()),
    ("last", pa.float64()), ("last_sz", pa.float64()),
    ("volume", pa.float64()), ("ts_ib", pa.int64()),
])


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class ParquetBufferWriter:
    """Buffers typed rows; flushes zstd parquet part files.

    Part files land at <root>/<YYYY-MM-DD>/<stream>/part-HHMMSS-NNNN.parquet
    (date = flush time, UTC). Flush triggers on row count or age of the
    oldest buffered row; restarts are safe (new part file, never appends).
    """

    def __init__(self, root: pathlib.Path, stream: str, schema: pa.Schema,
                 flush_rows: int = 20_000, flush_seconds: float = 300.0,
                 clock: Callable[[], dt.datetime] = utcnow) -> None:
        self._root = root
        self._stream = stream
        self._schema = schema
        self._flush_rows = flush_rows
        self._flush_seconds = flush_seconds
        self._clock = clock
        self._buf: List[Row] = []
        self._first_ts: Optional[dt.datetime] = None
        self._seq = 0

    def append(self, row: Row) -> None:
        """Adds one row, flushing when the buffer is full or old enough."""
        now = self._clock()
        if not self._buf:
            self._first_ts = now
        self._buf.append(row)
        age = (now - self._first_ts).total_seconds() \
            if self._first_ts is not None else 0.0
        if len(self._buf) >= self._flush_rows or age >= self._flush_seconds:
            self.flush()

    def flush(self) -> None:
        """Writes the buffer as one part file (no-op when empty)."""
        if not self._buf:
            return
        now = self._clock()
        part_dir = self._root / now.date().isoformat() / self._stream
        part_dir.mkdir(parents=True, exist_ok=True)
        cols = {name: [row.get(name) for row in self._buf]
                for name in self._schema.names}
        table = pa.Table.from_pydict(cols, schema=self._schema)
        while True:
            path = part_dir / f"part-{now:%H%M%S}-{self._seq:04d}.parquet"
            if not path.exists():
                break
            self._seq += 1
        pq.write_table(table, path, compression="zstd")
        logger.debug("flushed %d rows -> %s", len(self._buf), path)
        self._seq += 1
        self._buf = []
        self._first_ts = None

    def close(self) -> None:
        """Flushes any remaining rows."""
        self.flush()


def parse_book_ticker(t_ms: int, d: dict) -> Row:
    """Binance futures bookTicker payload -> typed row."""
    return {"t": t_ms, "s": d["s"], "u": int(d["u"]),
            "bp": float(d["b"]), "bq": float(d["B"]),
            "ap": float(d["a"]), "aq": float(d["A"]),
            "E": int(d.get("E", 0)), "T": int(d.get("T", 0))}


def parse_agg_trade(t_ms: int, d: dict) -> Row:
    """Binance futures aggTrade payload -> typed row."""
    return {"t": t_ms, "s": d["s"], "a": int(d["a"]),
            "p": float(d["p"]), "q": float(d["q"]), "m": bool(d["m"]),
            "T": int(d["T"])}


def parse_depth(t_ms: int, d: dict) -> Row:
    """Binance futures partial book depth payload -> typed row."""
    bids = d.get("b", [])
    asks = d.get("a", [])
    return {"t": t_ms, "s": d["s"], "u": int(d.get("u", 0)),
            "E": int(d.get("E", 0)), "T": int(d.get("T", 0)),
            "bp": [float(px) for px, _ in bids],
            "bq": [float(q) for _, q in bids],
            "ap": [float(px) for px, _ in asks],
            "aq": [float(q) for _, q in asks]}


def route_kind(kind: str) -> Optional[Tuple[str, Parser]]:
    """Maps a stream-type suffix (after '@') to (writer name, parser)."""
    if kind == "bookTicker":
        return "bookticker", parse_book_ticker
    if kind == "aggTrade":
        return "aggtrade", parse_agg_trade
    if kind.startswith("depth"):
        return "depth", parse_depth
    return None


def make_binance_writers(root: pathlib.Path, include_depth: bool,
                         clock: Callable[[], dt.datetime] = utcnow,
                         ) -> Dict[str, ParquetBufferWriter]:
    """Standard writer set for Binance capture, keyed by route_kind names."""
    writers = {
        "bookticker": ParquetBufferWriter(root, "bookticker",
                                          BOOK_TICKER_SCHEMA, clock=clock),
        "aggtrade": ParquetBufferWriter(root, "aggtrade", AGG_TRADE_SCHEMA,
                                        clock=clock),
    }
    if include_depth:
        writers["depth"] = ParquetBufferWriter(root, "depth", DEPTH_SCHEMA,
                                               clock=clock)
    return writers


def a1_window_active(ts: dt.datetime) -> bool:
    """A1 recording window: weekend (all day) or weekday dead zone 00-08 UTC."""
    if ts.weekday() >= 5:
        return True
    return ts.hour < 8


def next_a1_window_start(ts: dt.datetime) -> dt.datetime:
    """Earliest instant at/after ts when the A1 window is active."""
    if a1_window_active(ts):
        return ts
    # Weekday 08:00-24:00: next window is tomorrow 00:00 UTC (Sat included).
    return (ts + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0,
                                               microsecond=0)


def a1_window_end(ts: dt.datetime) -> dt.datetime:
    """End of the active window containing ts.

    Weekday dead zone ends 08:00 same day; the weekend window runs through
    Monday 08:00 (Mon 00-08 is contiguous with it). Requires ts active.
    """
    if not a1_window_active(ts):
        raise ValueError(f"{ts} is outside the A1 window")
    if ts.weekday() >= 5:
        days = 7 - ts.weekday()
        ts = ts + dt.timedelta(days=days)
    return ts.replace(hour=8, minute=0, second=0, microsecond=0)


def chunk_streams(streams: List[str],
                  size: int = MAX_STREAMS_PER_CONN) -> List[List[str]]:
    """Splits stream names into connection-sized chunks."""
    return [streams[i:i + size] for i in range(0, len(streams), size)]


async def record_connection(streams: List[str],
                            writers: Dict[str, ParquetBufferWriter],
                            stop_at: Optional[dt.datetime],
                            clock: Callable[[], dt.datetime] = utcnow) -> None:
    """Records one combined-stream connection until stop_at (reconnects).

    Messages route to a writer via route_kind on the stream-type suffix
    (the part after '@'); unknown types are dropped.
    """
    import websockets  # deferred: keeps offline tests dependency-free

    url = f"{BINANCE_WS}?streams={'/'.join(streams)}"
    try:
        while stop_at is None or clock() < stop_at:
            try:
                async with websockets.connect(url, ping_interval=60,
                                              max_size=2 ** 22) as ws:
                    logger.info("connected: %d streams", len(streams))
                    while stop_at is None or clock() < stop_at:
                        raw = await asyncio.wait_for(ws.recv(), timeout=90)
                        msg = json.loads(raw)
                        stream = msg.get("stream", "")
                        kind = (stream.split("@", 1)[1]
                                if "@" in stream else "")
                        kind = kind.split("@", 1)[0]  # depth20@500ms -> depth20
                        route = route_kind(kind)
                        if route is None:
                            continue
                        name, parser = route
                        writer = writers.get(name)
                        if writer is not None:
                            t_ms = int(clock().timestamp() * 1000)
                            writer.append(parser(t_ms, msg.get("data") or {}))
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001 — reconnect forever
                logger.warning("ws error (%s); reconnecting in 5s", err)
                await asyncio.sleep(5)
    finally:
        for writer in writers.values():
            writer.flush()
