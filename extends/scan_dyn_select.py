#!/usr/bin/env python3
"""Dynamic funding-carry universe selector: join 30d funding APR (perpfund.db,
both venues) with current OI and Binance underlyingType; emit candidate lists.

Type A (perp vs underlying stock carry): |APR| >= MIN_APR on a venue whose
15% OI slot >= MIN_SLOT.  Type B (BN-vs-HL funding differential, perp-perp):
|APR_bn - APR_hl| >= MIN_SPREAD with slot = 15% x min(OI_bn, OI_hl).

Writes data/dyn/candidates.json + prints markdown tables.
"""
from __future__ import annotations

import json
import sqlite3
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, cast

import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "dyn"
DB = Path.home() / "app" / "stocka" / "data" / "perpfund.db"
BN_FAPI = "https://fapi.binance.com"
HL_API = "https://api.hyperliquid.xyz/info"
HL_DEXES = ("xyz", "para")
CANONICAL = {"STXX": "STX", "SKHX": "SKHYNIX", "SMSN": "SAMSUNG"}
EXCLUDE = {"SKHYNIX", "ANTHROPIC", "OPENAI"}   # user-excluded + pre-IPO
MIN_APR = 8.0          # % p.a., 30d window, either venue
MIN_SLOT = 80_000.0    # USD, 15% of venue OI
MIN_SPREAD = 15.0      # % p.a. BN-HL differential for type B
OI_CAP_SHARE = 0.15


def get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def post(payload: Dict[str, Any]) -> Any:
    req = urllib.request.Request(HL_API, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def bn_universe() -> pd.DataFrame:
    """BN TradFi perps: ticker, symbol, kind (underlyingType)."""
    info = get(f"{BN_FAPI}/fapi/v1/exchangeInfo")
    rows: List[Dict[str, object]] = []
    for s in info["symbols"]:
        if (s.get("contractType") == "TRADIFI_PERPETUAL"
                and s.get("status") == "TRADING"):
            base = str(s["baseAsset"])
            rows.append({"ticker": CANONICAL.get(base, base),
                         "symbol": str(s["symbol"]),
                         "kind": str(s.get("underlyingType"))})
    return pd.DataFrame(rows)


def funding_apr_30d() -> pd.DataFrame:
    """ticker x venue -> APR% over last 30d (>=20d of data required)."""
    conn = sqlite3.connect(DB)
    now = int(time.time() * 1000)
    f = pd.read_sql("SELECT venue,ticker,ts,rate FROM funding WHERE ts>?",
                    conn, params=(now - 31 * 86_400_000,))
    conn.close()
    g = f.groupby(["venue", "ticker"]).agg(
        cum=("rate", "sum"),
        days=("ts", lambda s: (s.max() - s.min()) / 86_400_000)).reset_index()
    g = cast(pd.DataFrame, g[g["days"] >= 20.0]).copy()
    g["apr"] = g["cum"] / g["days"] * 365 * 100
    piv = g.pivot_table(index="ticker", columns="venue", values="apr")
    piv = piv.rename(columns={"binance": "apr_bn", "hyperliquid": "apr_hl"})
    return piv.reset_index()


def oi_snapshot(bn: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Current OI USD per ticker x venue for the given tickers."""
    marks = {m["symbol"]: float(m["markPrice"])
             for m in get(f"{BN_FAPI}/fapi/v1/premiumIndex")}
    sym_of = dict(zip(bn["ticker"], bn["symbol"]))
    rows: List[Dict[str, object]] = []
    for t in tickers:
        sym = sym_of.get(t)
        if sym and sym in marks:
            try:
                oi = get(f"{BN_FAPI}/fapi/v1/openInterest?symbol={sym}")
                rows.append({"ticker": t, "venue": "BN",
                             "oi": float(oi["openInterest"]) * marks[sym]})
            except Exception as ex:  # noqa: BLE001
                print(f"  oi {t}: {ex!r}")
            time.sleep(0.12)
    for dex in HL_DEXES:
        meta, ctxs = post({"type": "metaAndAssetCtxs", "dex": dex})
        for u, c in zip(meta["universe"], ctxs):
            base = str(u["name"]).split(":", 1)[1]
            t = CANONICAL.get(base, base)
            if t in tickers and not u.get("isDelisted"):
                try:
                    rows.append({"ticker": t, "venue": "HL", "dex": dex,
                                 "hl_coin": str(u["name"]),
                                 "oi": float(c["openInterest"]) * float(c["markPx"])})
                except (KeyError, TypeError, ValueError):
                    continue
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bn = bn_universe()
    apr = funding_apr_30d()
    apr = cast(pd.DataFrame, apr[~apr["ticker"].isin(list(EXCLUDE))])
    hot = cast(pd.DataFrame, apr[
        (apr["apr_bn"].abs() >= MIN_APR) | (apr["apr_hl"].abs() >= MIN_APR)
        | ((apr["apr_bn"] - apr["apr_hl"]).abs() >= MIN_SPREAD)])
    tickers = sorted(hot["ticker"])
    oi = oi_snapshot(bn, tickers)
    oi_piv = oi.pivot_table(index="ticker", columns="venue", values="oi",
                            aggfunc="max")
    m = hot.merge(oi_piv.reset_index(), on="ticker", how="left")
    m = m.merge(bn[["ticker", "symbol", "kind"]], on="ticker", how="left")
    hl_meta = cast(pd.DataFrame, oi[oi["venue"] == "HL"]).sort_values(
        "oi", ascending=False).drop_duplicates("ticker")
    if len(hl_meta):
        m = m.merge(hl_meta[["ticker", "dex", "hl_coin"]], on="ticker", how="left")
    for v in ("BN", "HL"):
        if v not in m.columns:
            m[v] = float("nan")
    m["slot_bn"] = m["BN"] * OI_CAP_SHARE
    m["slot_hl"] = m["HL"] * OI_CAP_SHARE

    # Type A: pick the venue with the stronger |APR|; slot must clear MIN_SLOT.
    a_rows: List[Dict[str, object]] = []
    for _, r in m.iterrows():
        for venue, apr_c, slot_c in (("BN", "apr_bn", "slot_bn"),
                                     ("HL", "apr_hl", "slot_hl")):
            a, s = r[apr_c], r[slot_c]
            if bool(pd.notna(a)) and bool(pd.notna(s)) \
                    and abs(float(a)) >= MIN_APR and float(s) >= MIN_SLOT:
                a_rows.append({
                    "ticker": r["ticker"], "venue": venue, "kind": r["kind"],
                    "apr": round(float(a), 1), "slot_kusd": round(float(s) / 1e3),
                    "side": "short_perp" if float(a) > 0 else "long_perp",
                    "bn_symbol": r["symbol"],
                    "hl_coin": r.get("hl_coin"), "dex": r.get("dex")})
    a_df = pd.DataFrame(a_rows).sort_values("apr", key=abs, ascending=False)

    b = cast(pd.DataFrame, m.dropna(subset=["apr_bn", "apr_hl"])).copy()
    b["spread"] = b["apr_bn"] - b["apr_hl"]
    b["slot"] = b[["slot_bn", "slot_hl"]].min(axis=1)
    b = cast(pd.DataFrame, b[(b["spread"].abs() >= MIN_SPREAD)
                             & (b["slot"] >= MIN_SLOT)])
    b_df = pd.DataFrame({
        "ticker": b["ticker"], "kind": b["kind"],
        "apr_bn": b["apr_bn"].round(1), "apr_hl": b["apr_hl"].round(1),
        "spread": b["spread"].round(1),
        "slot_kusd": (b["slot"] / 1e3).round(0),
        "short_leg": ["BN" if s > 0 else "HL" for s in b["spread"]],
        "bn_symbol": b["symbol"], "hl_coin": b["hl_coin"], "dex": b["dex"],
    }).sort_values("spread", key=abs, ascending=False)

    payload = {"generated": int(time.time() * 1000),
               "type_a": a_df.to_dict("records"),
               "type_b": b_df.to_dict("records")}
    (OUT / "candidates.json").write_text(json.dumps(payload, indent=1))
    pd.set_option("display.width", 220)
    print(f"## Type A (perp vs stock carry)  n={len(a_df)}")
    print(a_df.to_string(index=False))
    print(f"\n## Type B (BN-HL funding differential)  n={len(b_df)}")
    print(b_df.to_string(index=False))


if __name__ == "__main__":
    main()
