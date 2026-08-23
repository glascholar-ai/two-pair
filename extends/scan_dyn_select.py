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
MIN_APR_7D = 20.0      # % p.a., 7d window — catches fresh movers the 30d
                       # average hasn't warmed up to (NCLD/VST lesson, 08-23)
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


def _apr_from_events(f: pd.DataFrame, by: List[str], min_days: float
                     ) -> pd.DataFrame:
    g = f.groupby(by).agg(
        cum=("rate", "sum"),
        days=("ts", lambda s: (s.max() - s.min()) / 86_400_000)).reset_index()
    g = cast(pd.DataFrame, g[g["days"] >= min_days]).copy()
    g["apr"] = g["cum"] / g["days"] * 365 * 100
    return g


def funding_apr(window_d: int, min_days: float, suffix: str) -> pd.DataFrame:
    """ticker -> BN/HL APR% over the trailing window, from the self-contained
    caches (data/dyn/funding_{bn,hl}.parquet — see scan_dyn_fetch_funding.py).
    HL APR per ticker = the dex coin with max |cum| in the window (funding can
    differ materially between xyz and para for the same name, e.g. AVGO);
    the chosen coin is carried in column hl_coin{suffix}.
    """
    now = int(time.time() * 1000)
    t0 = now - (window_d + 1) * 86_400_000
    bn = pd.read_parquet(ROOT / "data" / "dyn" / "funding_bn.parquet")
    hl = pd.read_parquet(ROOT / "data" / "dyn" / "funding_hl.parquet")
    bn = cast(pd.DataFrame, bn[bn["ts"] > t0])
    hl = cast(pd.DataFrame, hl[hl["ts"] > t0])
    gb = _apr_from_events(bn, ["ticker"], min_days).rename(
        columns={"apr": f"apr_bn{suffix}"})
    gh = _apr_from_events(hl, ["ticker", "dex", "coin"], min_days)
    gh = gh.sort_values("apr", key=lambda s: s.abs(), ascending=False)
    gh = cast(pd.DataFrame, gh.drop_duplicates("ticker")).rename(
        columns={"apr": f"apr_hl{suffix}", "coin": f"hl_coin{suffix}",
                 "dex": f"hl_dex{suffix}"})
    out = gb[["ticker", f"apr_bn{suffix}"]].merge(
        gh[["ticker", f"apr_hl{suffix}", f"hl_coin{suffix}",
            f"hl_dex{suffix}"]], on="ticker", how="outer")
    return out


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
    apr = funding_apr(30, 20.0, "")
    apr7 = funding_apr(7, 4.0, "_7d")
    apr = apr.merge(apr7, on="ticker", how="outer")
    apr = cast(pd.DataFrame, apr[~apr["ticker"].isin(list(EXCLUDE))])
    for c in ("apr_bn", "apr_hl", "apr_bn_7d", "apr_hl_7d",
              "hl_coin", "hl_dex", "hl_coin_7d", "hl_dex_7d"):
        if c not in apr.columns:
            apr[c] = float("nan")
    # 30d screen (persistent) OR 7d screen (fresh movers) OR B-type spread
    hot = cast(pd.DataFrame, apr[
        (apr["apr_bn"].abs() >= MIN_APR) | (apr["apr_hl"].abs() >= MIN_APR)
        | (apr["apr_bn_7d"].abs() >= MIN_APR_7D)
        | (apr["apr_hl_7d"].abs() >= MIN_APR_7D)
        | ((apr["apr_bn"] - apr["apr_hl"]).abs() >= MIN_SPREAD)])
    # names passing only the 7d screen: use the 7d APR as their working apr
    for v in ("bn", "hl"):
        base, fresh = f"apr_{v}", f"apr_{v}_7d"
        take7 = apr[base].isna() | (apr[base].abs() < MIN_APR)
        hot = hot.copy()
        hot.loc[take7, base] = hot.loc[take7, fresh]
    # HL coin/dex: whichever the funding screen picked (7d choice preferred —
    # it reflects the current regime); OI slot must be that exact coin's.
    hot = hot.copy()
    hot["coin_pick"] = hot["hl_coin_7d"].fillna(hot["hl_coin"])
    hot["dex_pick"] = hot["hl_dex_7d"].fillna(hot["hl_dex"])
    tickers = sorted(hot["ticker"])
    oi = oi_snapshot(bn, tickers)
    bn_oi = cast(pd.DataFrame, oi[oi["venue"] == "BN"]).set_index("ticker")["oi"]
    coin_oi: Dict[str, float] = {}
    for _, r in cast(pd.DataFrame, oi[oi["venue"] == "HL"]).iterrows():
        c = r.get("hl_coin")
        if isinstance(c, str):
            coin_oi[c] = max(coin_oi.get(c, 0.0), float(r["oi"]))
    m = hot.merge(bn[["ticker", "symbol", "kind"]], on="ticker", how="left")
    m = cast(pd.DataFrame, m)
    m["slot_bn"] = cast(pd.Series, m["ticker"]).map(
        bn_oi.to_dict()) * OI_CAP_SHARE
    m["slot_hl"] = cast(pd.Series, m["coin_pick"]).map(
        lambda c: coin_oi.get(c) if isinstance(c, str) else None
    ).astype(float) * OI_CAP_SHARE

    # Type A: pick the venue with the stronger |APR|; slot must clear MIN_SLOT.
    # COMMODITY perps have no stock hedge (separate line, commodity_fx_perps.md)
    # and e.g. BZUSDT (Brent) would mis-map to the BZ ADR — hard-exclude.
    m = cast(pd.DataFrame, m[m["kind"] != "COMMODITY"])
    a_rows: List[Dict[str, object]] = []
    for _, r in m.iterrows():
        for venue, apr_c, slot_c in (("BN", "apr_bn", "slot_bn"),
                                     ("HL", "apr_hl", "slot_hl")):
            a, s = r[apr_c], r[slot_c]
            if bool(pd.notna(a)) and bool(pd.notna(s)) \
                    and abs(float(a)) >= MIN_APR and float(s) >= MIN_SLOT:
                a7 = r[f"apr_{venue.lower()}_7d"]
                a_rows.append({
                    "ticker": r["ticker"], "venue": venue, "kind": r["kind"],
                    "apr": round(float(a), 1),
                    "apr_7d": round(float(a7), 1)
                    if bool(pd.notna(a7)) else None,
                    "slot_kusd": round(float(s) / 1e3),
                    "side": "short_perp" if float(a) > 0 else "long_perp",
                    "bn_symbol": r["symbol"],
                    "hl_coin": r.get("coin_pick"), "dex": r.get("dex_pick")})
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
        "bn_symbol": b["symbol"], "hl_coin": b["coin_pick"],
        "dex": b["dex_pick"],
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
