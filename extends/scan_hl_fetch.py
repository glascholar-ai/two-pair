#!/usr/bin/env python3
"""Fetch Hyperliquid HIP-3 "xyz" stock-perp 5m candles + funding history for names
that also trade on Binance TradFi perps; cache to data/hl/.

Usage:
    python scan_hl_fetch.py            # top-N by HL 24h volume that exist on Binance
    python scan_hl_fetch.py NVDA MU    # explicit HL names

Cache layout (all parquet):
    data/hl/xyz_<NAME>_5m.parquet       cols ts,o,h,l,c,vol,trades   (ts = open ms UTC)
    data/hl/xyz_<NAME>_funding.parquet  cols ts,funding_rate,premium  (hourly)
    data/hl/universe.json               HL xyz meta+ctx snapshot + chosen name map
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd

ROOT = Path(__file__).parent
CACHE = ROOT / "data" / "hl"
BN_DIR = ROOT / "data" / "bn5m"
API = "https://api.hyperliquid.xyz/info"
CHUNK_MS = 15 * 86_400_000       # 4320 5m bars per request (< 5000 API cap)
EARLIEST_MS = 1_772_323_200_000  # 2026-03-01 UTC
TOP_N = 22
# HL xyz name -> Binance TradFi symbol root where they differ.
NAME_MAP: Dict[str, str] = {"SKHX": "SKHYNIX", "SMSN": "SAMSUNG"}


def post(payload: Dict[str, Any], tries: int = 5) -> Any:
    """POST to HL info endpoint with simple retry/backoff."""
    body = json.dumps(payload).encode()
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                API, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except Exception as ex:  # noqa: BLE001 - network retry
            print(f"  retry {attempt}: {ex!r}", flush=True)
            time.sleep(2.0 * (attempt + 1))
    return None


def load_universe() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (universe, assetCtxs) for dex xyz."""
    out = post({"type": "metaAndAssetCtxs", "dex": "xyz"})
    if not isinstance(out, list) or len(out) != 2:
        raise RuntimeError("bad metaAndAssetCtxs response")
    meta, ctxs = out
    return list(meta["universe"]), list(ctxs)


def bn_symbol(hl_name: str) -> Optional[str]:
    """Map an HL xyz name to a locally cached Binance symbol root, if any."""
    root = NAME_MAP.get(hl_name, hl_name)
    return root if (BN_DIR / f"{root}USDT.parquet").exists() else None


def choose_names(universe: List[Dict[str, Any]], ctxs: List[Dict[str, Any]],
                 top_n: int) -> List[Tuple[str, str, float]]:
    """Top-N HL names by 24h notional volume that also exist on Binance."""
    rows: List[Tuple[str, str, float]] = []
    for u, c in zip(universe, ctxs):
        name = str(u["name"]).split(":")[-1]
        bn = bn_symbol(name)
        if bn is None:
            continue
        rows.append((name, bn, float(c.get("dayNtlVlm", 0.0))))
    rows.sort(key=lambda r: -r[2])
    return rows[:top_n]


def fetch_candles(coin: str, start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
    out = post({"type": "candleSnapshot",
                "req": {"coin": coin, "interval": "5m",
                        "startTime": start_ms, "endTime": end_ms}})
    return out if isinstance(out, list) else []


def fetch_candle_history(name: str, since_ms: int) -> pd.DataFrame:
    """Walk backwards in 15d chunks until data runs out (2 empty chunks)."""
    coin = f"xyz:{name}"
    end = int(time.time() * 1000)
    frames: List[pd.DataFrame] = []
    empty_streak = 0
    while end > since_ms and empty_streak < 2:
        start = end - CHUNK_MS
        rows = fetch_candles(coin, start, end)
        if rows:
            empty_streak = 0
            frames.append(pd.DataFrame(rows))
        else:
            empty_streak += 1
        end = start - 1
        time.sleep(0.3)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames)
    df = pd.DataFrame({
        "ts": raw["t"].astype("int64"),
        "o": raw["o"].astype(float), "h": raw["h"].astype(float),
        "l": raw["l"].astype(float), "c": raw["c"].astype(float),
        "vol": raw["v"].astype(float), "trades": raw["n"].astype("int64")})
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def fetch_funding_history(name: str, since_ms: int) -> pd.DataFrame:
    """Page forward through fundingHistory (<=500 rows/call)."""
    coin = f"xyz:{name}"
    start = since_ms
    rows: List[Dict[str, Any]] = []
    while True:
        out = post({"type": "fundingHistory", "coin": coin, "startTime": start})
        if not isinstance(out, list) or not out:
            break
        rows.extend(out)
        last = int(out[-1]["time"])
        if len(out) < 500 or last <= start:
            break
        start = last + 1
        time.sleep(0.3)
    if not rows:
        return pd.DataFrame({"ts": pd.Series(dtype="int64"),
                             "funding_rate": pd.Series(dtype=float),
                             "premium": pd.Series(dtype=float)})
    df = pd.DataFrame({
        "ts": [int(r["time"]) for r in rows],
        "funding_rate": [float(r["fundingRate"]) for r in rows],
        "premium": [float(r.get("premium", "nan")) for r in rows]})
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def refresh_parquet(path: Path, fetch: Any, name: str) -> int:
    """Fetch (incrementally if cached) and write parquet; return row count."""
    since = EARLIEST_MS
    old: Optional[pd.DataFrame] = None
    if path.exists():
        old = pd.read_parquet(path)
        if len(old):
            since = int(old["ts"].max()) - 86_400_000  # 1d overlap for late fixes
    new = fetch(name, since)
    if old is not None and len(old):
        new = pd.concat([old, new]) if len(new) else old
        new = new.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
    if len(new):
        new.to_parquet(path, index=False)
    return int(len(new))


def refresh_binance(symbol_root: str) -> int:
    """Append recent Binance 5m klines (public fapi) to the local parquet."""
    path = BN_DIR / f"{symbol_root}USDT.parquet"
    if not path.exists():
        return 0
    old = pd.read_parquet(path)
    start = int(old["ts"].max()) + 1
    rows: List[List[Any]] = []
    while True:
        url = (f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol_root}USDT"
               f"&interval=5m&startTime={start}&limit=1500")
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                data = json.load(resp)
        except Exception as ex:  # noqa: BLE001
            print(f"  bn {symbol_root}: {ex!r}")
            break
        if not data:
            break
        rows.extend(data)
        if len(data) < 1500:
            break
        start = int(data[-1][0]) + 1
        time.sleep(0.3)
    if not rows:
        return len(old)
    new = pd.DataFrame(rows).iloc[:, [0, 1, 2, 3, 4, 5, 7, 8]]
    new.columns = ["ts", "o", "h", "l", "c", "vol", "quote_vol", "trades"]
    new = new.astype({"ts": "int64", "o": float, "h": float, "l": float, "c": float,
                      "vol": float, "quote_vol": float, "trades": "int64"})
    out = pd.concat([old, new]).drop_duplicates("ts").sort_values("ts")
    out.to_parquet(path, index=False)
    return len(out)


def fetch_binance_ref(symbol_root: str, since_ms: int) -> None:
    """Cache Binance 5m mark-price and index-price klines to data/hl/bnref_<root>.parquet."""
    out_p = CACHE / f"bnref_{symbol_root}.parquet"
    if out_p.exists():
        return
    series: Dict[str, pd.DataFrame] = {}
    for kind, col, key in (("markPriceKlines", "mark", "symbol"),
                           ("indexPriceKlines", "index", "pair")):
        rows: List[List[Any]] = []
        start = since_ms
        while True:
            url = (f"https://fapi.binance.com/fapi/v1/{kind}?{key}={symbol_root}USDT"
                   f"&interval=5m&startTime={start}&limit=1500")
            try:
                with urllib.request.urlopen(url, timeout=20) as resp:
                    data = json.load(resp)
            except Exception as ex:  # noqa: BLE001
                print(f"  bn {kind} {symbol_root}: {ex!r}")
                break
            if not data:
                break
            rows.extend(data)
            if len(data) < 1500:
                break
            start = int(data[-1][0]) + 1
            time.sleep(0.3)
        if rows:
            df = pd.DataFrame(rows).iloc[:, [0, 4]]
            df.columns = ["ts", col]
            series[col] = df.astype({"ts": "int64", col: float})
    if len(series) == 2:
        merged = series["mark"].merge(series["index"], on="ts", how="inner")
        merged.to_parquet(out_p, index=False)


def fetch_usdc_usdt(since_ms: int) -> None:
    """Cache Binance spot USDCUSDT 5m closes (stablecoin basis between the two venues'
    quote currencies: Binance TradFi perps are USDT-quoted with index / USDTUSD, HL xyz
    is USDC-collateralised) to data/hl/usdcusdt_5m.parquet."""
    out_p = CACHE / "usdcusdt_5m.parquet"
    start = since_ms
    old: Optional[pd.DataFrame] = None
    if out_p.exists():
        old = pd.read_parquet(out_p)
        start = int(old["ts"].max()) + 1
    rows: List[List[Any]] = []
    while True:
        url = (f"https://api.binance.com/api/v3/klines?symbol=USDCUSDT&interval=5m"
               f"&startTime={start}&limit=1000")
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                data = json.load(resp)
        except Exception as ex:  # noqa: BLE001
            print(f"  usdcusdt: {ex!r}")
            break
        if not data:
            break
        rows.extend(data)
        if len(data) < 1000:
            break
        start = int(data[-1][0]) + 1
        time.sleep(0.2)
    if not rows:
        return
    new = pd.DataFrame(rows).iloc[:, [0, 4]]
    new.columns = ["ts", "usdcusdt"]
    new = new.astype({"ts": "int64", "usdcusdt": float})
    if old is not None:
        new = pd.concat([old, new]).drop_duplicates("ts").sort_values("ts")
    new.to_parquet(out_p, index=False)


def refresh_binance_funding(symbol_roots: List[str]) -> None:
    """Append recent Binance funding rows for the given roots to _funding.parquet."""
    path = BN_DIR / "_funding.parquet"
    old = pd.read_parquet(path)
    frames: List[pd.DataFrame] = [old]
    for root in symbol_roots:
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={root}USDT&limit=100"
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                data = json.load(resp)
        except Exception as ex:  # noqa: BLE001
            print(f"  bn funding {root}: {ex!r}")
            continue
        if data:
            frames.append(cast(pd.DataFrame, pd.DataFrame(data)[list(old.columns)]))
        time.sleep(0.2)
    out = pd.concat(frames).drop_duplicates(["symbol", "fundingTime"])
    out = out.sort_values(["symbol", "fundingTime"]).reset_index(drop=True)
    out.to_parquet(path, index=False)


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    universe, ctxs = load_universe()
    if len(sys.argv) > 1:
        chosen = [(n, bn_symbol(n) or n, 0.0) for n in sys.argv[1:]]
    else:
        chosen = choose_names(universe, ctxs, TOP_N)
    (CACHE / "universe.json").write_text(json.dumps(
        {"fetched_at": int(time.time() * 1000), "universe": universe, "ctxs": ctxs,
         "chosen": [{"hl": n, "bn": b, "dayNtlVlm": v} for n, b, v in chosen]}, indent=1))
    for name, bn, vol in chosen:
        t0 = time.time()
        n_c = refresh_parquet(CACHE / f"xyz_{name}_5m.parquet", fetch_candle_history, name)
        n_f = refresh_parquet(CACHE / f"xyz_{name}_funding.parquet",
                              fetch_funding_history, name)
        n_b = refresh_binance(bn)
        hl_bars = pd.read_parquet(CACHE / f"xyz_{name}_5m.parquet")
        fetch_binance_ref(bn, int(hl_bars["ts"].min()))
        print(f"{name:8s} -> {bn:8s} vol24h={vol/1e6:8.1f}M  candles={n_c:6d}  "
              f"funding={n_f:5d}  bn_bars={n_b:6d}  ({time.time()-t0:.0f}s)", flush=True)
    refresh_binance_funding([b for _, b, _ in chosen])
    fetch_usdc_usdt(EARLIEST_MS)


if __name__ == "__main__":
    main()
