#!/usr/bin/env python3
"""Map dynamic-carry Type-A underlyings to IBKR contracts, verify against the
Binance index price, and fetch stock bars from TWS (read-only, 127.0.0.1:7496).

Mapping sources (no guessing): Binance /fapi/v1/constituents dxfeed/pyth
symbols give the real ticker (US: 'XXX:USLF24', HK: pyth 'NNNNHK'); KR via
kaiko 'KK_RFR_ANNNNNNKRW_USD'; JP names (HL-only) via a small static map.
Verification: IBKR last daily close vs Binance index price (FX-adjusted),
|log diff| < 6% accepts (catches wrong-contract mappings, tolerates 1d drift).

Output: data/dyn/ib/<TICKER>_1m.parquet  (32d, all hours, TRADES)
        data/dyn/ib/<TICKER>_10s.parquet (32d, RTH, MIDPOINT; top names only)
        data/dyn/ib/fx_<PAIR>_1m.parquet (USDHKD/USDKRW/USDJPY, MIDPOINT)
        data/dyn/ib_map.json             (mapping + verification report)
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ib_insync import IB, Contract, Forex, Stock, util

ROOT = Path(__file__).parent
DYN = ROOT / "data" / "dyn"
OUT = DYN / "ib"
BN_FAPI = "https://fapi.binance.com"
JP_MAP = {"SOFTBANK": ("9984", "TSEJ", "JPY"), "KIOXIA": ("285A", "TSEJ", "JPY")}
KR_MAP = {"SAMSUNG": "005930", "HYUNDAI": "005380", "NAVER": "035420",
          "LGELECTRONICS": "066570", "HANMI": "042700", "SAMSUNGEM": "009150",
          "KODEX200": "069500"}
N_DAYS = 32
TOP_10S = 12          # names also fetched at 10s MIDPOINT
PACE_S = 2.0


def get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.load(r)


def resolve_contract(ticker: str, kind: str,
                     bn_symbol: str) -> Optional[Tuple[Contract, str]]:
    """(contract, real_ticker) from constituents / static maps."""
    if ticker in JP_MAP:
        sym, exch, ccy = JP_MAP[ticker]
        return Stock(sym, exch, ccy), sym
    if kind == "KR_EQUITY":
        code = KR_MAP.get(ticker)
        return (Stock(code, "KRX", "KRW"), code) if code else None
    cons = None
    if bn_symbol:
        try:
            cons = get(f"{BN_FAPI}/fapi/v1/constituents?symbol={bn_symbol}")
        except Exception:  # noqa: BLE001
            cons = None
    symbols = [str(c.get("symbol", "")) for c in (cons or {}).get(
        "constituents", [])]
    if kind == "HK_EQUITY":
        for s in symbols:
            mm = re.match(r"(\d{3,5})HK//", s)
            if mm:
                return Stock(str(int(mm.group(1))), "SEHK", "HKD"), mm.group(1)
        for s in symbols:
            mm = re.match(r"KK_RFR_(\d+)HKD_USD", s)
            if mm:
                return Stock(str(int(mm.group(1))), "SEHK", "HKD"), mm.group(1)
        return None
    for s in symbols:                       # US: dxfeed 'XXX:USLF24//...'
        mm = re.match(r"([A-Z.]+):USLF24", s)
        if mm:
            return Stock(mm.group(1).replace(".", " "), "SMART", "USD"), mm.group(1)
    for s in symbols:
        mm = re.match(r"KK_RFR_([A-Z.]+)USD//", s)
        if mm:
            return Stock(mm.group(1), "SMART", "USD"), mm.group(1)
    return Stock(ticker, "SMART", "USD"), ticker      # last resort: same name


def bn_index_prices() -> Dict[str, float]:
    return {m["symbol"]: float(m["indexPrice"])
            for m in get(f"{BN_FAPI}/fapi/v1/premiumIndex")}


def fetch_chunks(ib: IB, contract: Contract, bar: str, what: str,
                 use_rth: bool, days: int, chunk_days: int) -> pd.DataFrame:
    """Walk backwards in chunks; return frame ts(ms),o,h,l,c,vol."""
    frames: List[pd.DataFrame] = []
    end = datetime.now(timezone.utc)
    stop = end - timedelta(days=days)
    while end > stop:
        try:
            bars = ib.reqHistoricalData(
                contract, endDateTime=end, durationStr=f"{chunk_days} D",
                barSizeSetting=bar, whatToShow=what, useRTH=use_rth,
                formatDate=2, timeout=90)
        except Exception as ex:  # noqa: BLE001
            print(f"    chunk fail {contract.symbol} {end:%m-%d}: {ex!r}",
                  flush=True)
            bars = []
        if bars:
            df = util.df(bars)
            if df is not None:
                frames.append(df)
            first = min(pd.Timestamp(b.date) for b in bars)
            if first.tzinfo is None:
                first = first.tz_localize(timezone.utc)
            new_end = first.to_pydatetime()
            if isinstance(new_end, datetime) and new_end < end:
                end = new_end
            else:
                end = end - timedelta(days=chunk_days)
        else:
            end = end - timedelta(days=chunk_days)
        time.sleep(PACE_S)
    if not frames:
        return pd.DataFrame()
    allf = pd.concat(frames)
    ts = pd.to_datetime(allf["date"], utc=True).astype("int64") // 10**6
    out = pd.DataFrame({"ts": ts, "o": allf["open"].astype(float),
                        "h": allf["high"].astype(float),
                        "l": allf["low"].astype(float),
                        "c": allf["close"].astype(float),
                        "vol": allf["volume"].astype(float)})
    return out.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cand = json.loads((DYN / "candidates.json").read_text())
    rows = [r for r in cand["type_a"]]
    # unique underlyings, ranked by deployable edge for the 10s tier
    seen: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        t = str(r["ticker"])
        edge = min(float(r["slot_kusd"]), 1500.0) * abs(float(r["apr"]))
        if t not in seen or edge > seen[t]["edge"]:
            seen[t] = {"kind": str(r.get("kind") or ""), "edge": edge,
                       "apr": r["apr"], "slot_kusd": r["slot_kusd"],
                       "bn_symbol": str(r.get("bn_symbol") or "")}
    order = sorted(seen, key=lambda t: -seen[t]["edge"])
    idx = bn_index_prices()

    ib = IB()
    ib.connect("127.0.0.1", 7496, clientId=211, timeout=20, readonly=True)
    fx_rate: Dict[str, float] = {"USD": 1.0}
    for pair, ccy in (("USDHKD", "HKD"), ("USDKRW", "KRW"), ("USDJPY", "JPY")):
        fxp = OUT / f"fx_{pair}_1m.parquet"
        if fxp.exists():
            df = pd.read_parquet(fxp)
        else:
            df = fetch_chunks(ib, Forex(pair), "1 min", "MIDPOINT", False,
                              N_DAYS, 10)
            if len(df):
                df.to_parquet(fxp, index=False)
        if len(df):
            fx_rate[ccy] = float(df["c"].iloc[-1])
        print(f"fx {pair}: {len(df)} bars last={fx_rate.get(ccy)}", flush=True)

    report: List[Dict[str, Any]] = []
    for rank, t in enumerate(order):
        kind = seen[t]["kind"]
        if kind == "CN_EQUITY":
            report.append({"ticker": t, "status": "skip_cn_equity"})
            continue
        res = resolve_contract(t, kind, str(seen[t]["bn_symbol"]))
        if res is None:
            report.append({"ticker": t, "status": "no_mapping"})
            print(f"{t}: no mapping", flush=True)
            continue
        contract, real = res
        try:
            q = ib.qualifyContracts(contract)
        except Exception as ex:  # noqa: BLE001
            q = []
            print(f"{t}: qualify error {ex!r}", flush=True)
        if not q:
            report.append({"ticker": t, "status": "no_contract", "real": real})
            print(f"{t}: no IBKR contract for {real}", flush=True)
            continue
        daily = fetch_chunks(ib, contract, "1 day", "TRADES", True, 6, 6)
        ccy = str(contract.currency)
        ok, close_usd = False, float("nan")
        bn_idx = idx.get(str(seen[t]["bn_symbol"]))
        if len(daily):
            close_usd = float(daily["c"].iloc[-1]) / fx_rate.get(ccy, 1.0)
            if bn_idx:
                ok = abs(np.log(close_usd / bn_idx)) < 0.06
            elif t in JP_MAP:          # no BN symbol; static map is trusted
                ok = True
        status = "ok" if ok else "price_mismatch"
        report.append({"ticker": t, "status": status, "real": real,
                       "exchange": str(contract.primaryExchange or
                                       contract.exchange),
                       "ccy": ccy, "close_usd": round(close_usd, 3),
                       "bn_index": bn_idx})
        print(f"{t:12s} -> {real:8s} {status} close_usd={close_usd:.2f} "
              f"idx={bn_idx}", flush=True)
        if not ok:
            continue
        m1 = fetch_chunks(ib, contract, "1 min", "TRADES", False, N_DAYS, 5)
        if len(m1):
            m1.to_parquet(OUT / f"{t}_1m.parquet", index=False)
        print(f"{t:12s} 1m bars: {len(m1)}", flush=True)
        if rank < TOP_10S:
            s10 = fetch_chunks(ib, contract, "10 secs", "MIDPOINT", True,
                               N_DAYS, 1)
            if len(s10):
                s10.to_parquet(OUT / f"{t}_10s.parquet", index=False)
            print(f"{t:12s} 10s bars: {len(s10)}", flush=True)
    (DYN / "ib_map.json").write_text(json.dumps(report, indent=1))
    ib.disconnect()
    n_ok = sum(1 for r in report if r["status"] == "ok")
    print(f"done: {n_ok}/{len(report)} mapped+verified", flush=True)


if __name__ == "__main__":
    main()
