#!/usr/bin/env python3
"""Fetch commodity/FX perp data: Binance TradFi commodity perps (5m klines +
funding + depth snapshot) and Hyperliquid xyz commodity/FX/index perps (5m
candles + funding history). Also writes an inventory JSON.

Cache layout:
  data/bn5m_cmdty/<SYM>.parquet, _funding.parquet, depth_<ts>.json, inventory.json
  data/hl/xyz_<NAME>_5m.parquet, xyz_<NAME>_funding.parquet
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pandas as pd

ROOT = Path(__file__).parent / "data"
BN_DIR = ROOT / "bn5m_cmdty"
HL_DIR = ROOT / "hl"
FAPI = "https://fapi.binance.com/fapi/v1"
HL_API = "https://api.hyperliquid.xyz/info"
HL_NAMES = ["GOLD", "SILVER", "CL", "BRENTOIL", "COPPER", "NATGAS", "PLATINUM",
            "PALLADIUM", "EUR", "GBP", "JPY", "SP500"]
HL_CHUNK_MS = 15 * 86_400_000
HL_EARLIEST_MS = 1_756_684_800_000  # 2025-09-01
KCOLS = ["ts", "o", "h", "l", "c", "vol", "close_ts", "quote_vol", "trades",
         "tb_base", "tb_quote", "ig"]


def bn_get(url: str, tries: int = 6) -> Optional[Any]:
    for a in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                used = int(r.headers.get("X-MBX-USED-WEIGHT-1M", 0))
                d = json.load(r)
            if used > 2000:
                time.sleep(30)
            return d
        except Exception as ex:  # noqa: BLE001
            print(f"retry {a}: {ex!r}", flush=True)
            time.sleep(5 * (a + 1))
    return None


def hl_post(payload: Dict[str, Any], tries: int = 5) -> Any:
    for a in range(tries):
        try:
            req = urllib.request.Request(HL_API, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as ex:  # noqa: BLE001
            print(f"hl retry {a}: {ex!r}", flush=True)
            time.sleep(3 * (a + 1))
    return None


def bn_inventory() -> List[Dict[str, Any]]:
    info = bn_get(f"{FAPI}/exchangeInfo")
    t24 = {t["symbol"]: t for t in bn_get(f"{FAPI}/ticker/24hr") or []}
    out: List[Dict[str, Any]] = []
    for s in (info or {}).get("symbols", []):
        if s.get("contractType") != "TRADIFI_PERPETUAL":
            continue
        if s.get("underlyingType") in ("EQUITY", "PREMARKET", "KR_EQUITY", "HK_EQUITY",
                                       "JP_EQUITY"):
            continue
        out.append({"symbol": s["symbol"], "underlyingType": s.get("underlyingType"),
                    "status": s.get("status"), "onboardDate": int(s["onboardDate"]),
                    "quoteVolume24h": float(t24.get(s["symbol"], {}).get("quoteVolume", 0)),
                    "lastPrice": float(t24.get(s["symbol"], {}).get("lastPrice", 0))})
    return out


def bn_klines(sym: str, start: int) -> pd.DataFrame:
    rows: List[List[Any]] = []
    while True:
        d = bn_get(f"{FAPI}/klines?symbol={sym}&interval=5m&startTime={start}&limit=1500")
        if not d:
            break
        rows.extend(d)
        if len(d) < 1500:
            break
        start = int(d[-1][0]) + 1
        time.sleep(0.25)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.columns = pd.Index(KCOLS)
    keep = ["ts", "o", "h", "l", "c", "vol", "quote_vol", "trades", "tb_quote"]
    return cast(pd.DataFrame, df[keep]).astype(
        {"ts": "int64", "o": float, "h": float, "l": float, "c": float, "vol": float,
         "quote_vol": float, "trades": "int64", "tb_quote": float})


def bn_funding(sym: str, start: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    while True:
        d = bn_get(f"{FAPI}/fundingRate?symbol={sym}&startTime={start}&limit=1000")
        if not d:
            break
        rows.extend(d)
        if len(d) < 1000:
            break
        start = int(d[-1]["fundingTime"]) + 1
        time.sleep(0.25)
    return pd.DataFrame(rows)


def bn_depth(symbols: List[str]) -> None:
    snap: Dict[str, Any] = {"ts": int(time.time() * 1000), "books": {}}
    for s in symbols:
        d = bn_get(f"{FAPI}/depth?symbol={s}&limit=100")
        if d:
            snap["books"][s] = d
        time.sleep(0.2)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    (BN_DIR / f"depth_{stamp}.json").write_text(json.dumps(snap))
    print(f"depth snapshot saved {stamp}", flush=True)


def fetch_binance() -> None:
    BN_DIR.mkdir(parents=True, exist_ok=True)
    inv = bn_inventory()
    (BN_DIR / "inventory.json").write_text(json.dumps(inv, indent=1))
    funds: List[pd.DataFrame] = []
    for i, it in enumerate(inv):
        sym, onboard = it["symbol"], it["onboardDate"]
        f = BN_DIR / f"{sym}.parquet"
        if not f.exists():
            df = bn_klines(sym, onboard)
            if not df.empty:
                df.to_parquet(f, index=False)
            print(f"[{i+1}/{len(inv)}] {sym}: {len(df)} bars", flush=True)
        fr = bn_funding(sym, onboard)
        if not fr.empty:
            funds.append(fr)
        print(f"  {sym}: {len(fr)} funding rows", flush=True)
    if funds:
        pd.concat(funds).to_parquet(BN_DIR / "_funding.parquet", index=False)
    bn_depth([it["symbol"] for it in inv])


def hl_candles(coin: str) -> pd.DataFrame:
    end = int(time.time() * 1000)
    frames: List[pd.DataFrame] = []
    empty = 0
    while end > HL_EARLIEST_MS and empty < 2:
        start = end - HL_CHUNK_MS
        rows = hl_post({"type": "candleSnapshot",
                        "req": {"coin": coin, "interval": "5m", "startTime": start,
                                "endTime": end}})
        if isinstance(rows, list) and rows:
            frames.append(pd.DataFrame(rows))
            empty = 0
        else:
            empty += 1
        end = start - 1
        time.sleep(0.25)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    out = pd.DataFrame({"ts": df["t"].astype("int64"), "o": df["o"].astype(float),
                        "h": df["h"].astype(float), "l": df["l"].astype(float),
                        "c": df["c"].astype(float), "vol": df["v"].astype(float),
                        "trades": df["n"].astype("int64")})
    return out.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def hl_funding(coin: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    start = HL_EARLIEST_MS
    while True:
        d = hl_post({"type": "fundingHistory", "coin": coin, "startTime": start})
        if not isinstance(d, list) or not d:
            break
        rows.extend(d)
        nxt = int(d[-1]["time"]) + 1
        if len(d) < 500 or nxt <= start:
            break
        start = nxt
        time.sleep(0.25)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates("time")


def fetch_hl() -> None:
    HL_DIR.mkdir(parents=True, exist_ok=True)
    meta = hl_post({"type": "metaAndAssetCtxs", "dex": "xyz"})
    if meta:
        (HL_DIR / "xyz_meta.json").write_text(json.dumps(meta))
    for n in HL_NAMES:
        coin = f"xyz:{n}"
        p = HL_DIR / f"xyz_{n}_5m.parquet"
        if not p.exists():
            df = hl_candles(coin)
            if not df.empty:
                df.to_parquet(p, index=False)
            print(f"HL {n}: {len(df)} bars", flush=True)
        pf = HL_DIR / f"xyz_{n}_funding.parquet"
        if not pf.exists():
            fr = hl_funding(coin)
            if not fr.empty:
                fr.to_parquet(pf, index=False)
            print(f"HL {n}: {len(fr)} funding rows", flush=True)


if __name__ == "__main__":
    fetch_binance()
    fetch_hl()
    print("ALL_DONE", flush=True)
