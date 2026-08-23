#!/usr/bin/env python3
"""Dynamic funding-carry portfolio backtest, past month, $10M ($5M stock cash +
$5M perp margin at 1x), OI-capped.

Type A (carry): short rich perp / long stock when funding-side premium clears a
cost-based threshold; exit when premium dies. Long-perp/short-stock on the
discount side (borrow cost charged). Decisions on stock 1m bars during the
underlying's regular session; perp leg priced from BN 1s aggTrade bars or HL
5m/15m candles (asof join).
Type B (BN-HL funding differential): short the high-funding venue, long the
other, enter/exit on trailing-7d funding spread; both legs perp.

Inputs: data/dyn/candidates.json, data/dyn/bn1s/, data/dyn/hl/, data/dyn/ib/,
data/dyn/ib_map.json, data/oi/, perpfund.db.
Outputs: docs/scan/dyn_positions.csv + markdown summary on stdout.
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DYN = ROOT / "data" / "dyn"
OI_DIR = ROOT / "data" / "oi"
OUT_DIR = ROOT / "docs" / "scan"
DB = Path.home() / "app" / "stocka" / "data" / "perpfund.db"

WINDOW_DAYS = 30
CAPITAL_STOCK = 5_000_000.0      # USD cash for stock legs (type A)
CAPITAL_PERP = 5_000_000.0       # USD perp margin, 1x notional
OI_CAP = 0.15
CAP_PER_NAME = 1500000.0
MIN_TICKET = 50_000.0
CUSHION = 1.5                    # entry premium >= CUSHION x round-trip cost
MIN_PREM_BPS = 20.0
EXIT_BPS = 0.0
MIN_TRAIL_APR = 8.0              # trailing-7d funding APR gate (type A)
FAT_APR = 25.0                   # 30d APR above which funding-first entry opens
FAT_ENTRY_APR = 20.0             # trailing-7d APR to enter without premium edge
FAT_EXIT_APR = 8.0               # trailing-7d APR below which fat position exits
FAT_STOP_BPS = -100.0            # basis blowout stop for fat entries (smoothed)
FAT_KINDS = {"EQUITY", "KR_EQUITY", "JP"}   # stock legs cheap enough for
# funding-first entries; HK excluded (26bp friction + flaky funding regimes)
# In production the fat-mode universe is a HUMAN-CURATED whitelist (funding
# persistence record, borrow/access sanity) — FAT_KINDS+FAT_APR is its backtest
# proxy. FAT_PREM_FLOOR: entry premium must be >= this multiple of the
# round-trip cost, so the basis leg alone pays for the trip.
FAT_PREM_FLOOR = 1.0
B_ENTRY_APR = 15.0               # type B trailing spread entry
B_EXIT_APR = 5.0
BORROW_APR = 1.0                 # % p.a. charged on short-stock legs
FILL_LAG_BARS = 1
SMOOTH_BARS = 12                 # 1h median on the 5m decision grid
COOLDOWN_MS = 2 * 3_600_000      # re-entry cooldown per name after exit
# Round-trip friction (both directions, one-leg notional bps).
# Perp legs executed as MAKER (fat + most prem entries are not time-critical):
# BN TradFi maker 0 (current promo); HL xyz growth maker 0.003%/side; para
# (standard HIP-3) maker 0.03%/side. The 2x effective-half-spread charge stays
# — it models adverse selection / fill quality, not fees.
PERP_RT = {"BN": 0.0, "xyz": 0.6, "para": 6.0}
# Stock legs at actual IBKR tiered rates + statutory taxes:
#   US: $0.0035/share/side (converted per name via its price) + SEC 0.28bp sell
#   HK: stamp 0.10%/side x2 + levies ~0.17bp + commission 0.03%/side
#   KR: sell tax 0.20% (0.05% STT + 0.15% rural, 2026) + commission 0.04%/side
#   JP: commission 0.05%/side, no stamp
STOCK_RT_FIXED = {"HK_EQUITY": 26.2, "KR_EQUITY": 28.0, "JP": 10.0}
US_COMM_PER_SHARE = 0.0035
US_SELL_REG_BPS = 0.28
SESSIONS = {                     # UTC minutes-of-day windows, Aug 2026 (DST)
    "EQUITY": [(810, 1200)],
    "HK_EQUITY": [(90, 240), (300, 480)],
    "KR_EQUITY": [(0, 380)],
    "JP": [(0, 150), (210, 390)],
}
HOLIDAYS = {"KR_EQUITY": {"2026-08-17"}, "JP": {"2026-08-11"},
            "EQUITY": set(), "HK_EQUITY": set()}
JP_NAMES = {"SOFTBANK", "KIOXIA"}


def col(df: pd.DataFrame, name: str) -> pd.Series:
    return cast(pd.Series, df[name])


class FundingBook:
    """Per (venue,ticker) funding events + trailing daily APR lookup.

    Prefers the self-contained API caches (data/dyn/funding_{bn,hl}.parquet,
    scan_dyn_fetch_funding.py) over perpfund.db: the db may be mid-migration
    in another session, and it keys one dex per HL ticker while funding can
    differ materially between xyz and para (AVGO). HL rows are filtered to the
    exact coin each candidate trades (candidates.json hl_coin).
    """

    def __init__(self) -> None:
        bn_p = DYN / "funding_bn.parquet"
        hl_p = DYN / "funding_hl.parquet"
        if bn_p.exists() and hl_p.exists():
            bn = pd.read_parquet(bn_p)
            bn["venue"] = "binance"
            hl = pd.read_parquet(hl_p)
            cand_p = DYN / "candidates.json"
            coin_of: Dict[str, str] = {}
            if cand_p.exists():
                cand = json.loads(cand_p.read_text())
                for r in cand["type_a"] + cand["type_b"]:
                    if r.get("hl_coin"):
                        coin_of[str(r["ticker"])] = str(r["hl_coin"])
            keep = col(hl, "ticker").map(
                lambda t: coin_of.get(str(t), "")) == col(hl, "coin")
            fallback = ~col(hl, "ticker").isin(list(coin_of))
            hl = cast(pd.DataFrame, hl[keep | fallback]).copy()
            hl = cast(pd.DataFrame,
                      hl.sort_values("ts")).drop_duplicates(["ticker", "ts"])
            hl["venue"] = "hyperliquid"
            f = pd.concat([bn[["venue", "ticker", "ts", "rate"]],
                           hl[["venue", "ticker", "ts", "rate"]]])
        else:
            conn = sqlite3.connect(DB)
            f = pd.read_sql("SELECT venue,ticker,ts,rate FROM funding", conn)
            conn.close()
        self.ev: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
        self.daily: Dict[Tuple[str, str], pd.Series] = {}
        for key_vt, g in f.groupby(["venue", "ticker"]):
            v, t = cast(Tuple[str, str], key_vt)
            g = cast(pd.DataFrame, g).sort_values("ts")
            ts = g["ts"].to_numpy(dtype="int64")
            rt = g["rate"].to_numpy(dtype=float)
            key = (str(v), str(t))
            self.ev[key] = (ts, rt)
            day = pd.to_datetime(g["ts"], unit="ms", utc=True).dt.floor("D")
            self.daily[key] = cast(
                pd.Series, g.assign(day=day).groupby("day")["rate"].sum())

    def accrue(self, venue: str, ticker: str, t0: int, t1: int) -> float:
        """Sum of funding rates in (t0, t1]."""
        key = (venue, ticker)
        if key not in self.ev:
            return 0.0
        ts, rt = self.ev[key]
        i0 = int(np.searchsorted(ts, t0, side="right"))
        i1 = int(np.searchsorted(ts, t1, side="right"))
        return float(rt[i0:i1].sum())

    def trail7_apr(self, venue: str, ticker: str, t: int) -> float:
        key = (venue, ticker)
        d = self.daily.get(key)
        if d is None or len(d) == 0:
            return 0.0
        day = pd.to_datetime(t, unit="ms", utc=True).floor("D")
        w = cast(pd.Series, d.loc[:day - pd.Timedelta(days=1)]).tail(7)
        if len(w) < 4:
            return 0.0
        return float(w.mean() * 365 * 100)


def in_session(ts_ms: np.ndarray, kind: str) -> np.ndarray:
    dt = pd.Series(pd.to_datetime(ts_ms, unit="ms", utc=True))
    minute = (dt.dt.hour * 60 + dt.dt.minute).to_numpy()
    ok = np.zeros(len(ts_ms), dtype=bool)
    for lo, hi in SESSIONS[kind]:
        ok |= (minute >= lo) & (minute < hi)
    ok &= dt.dt.dayofweek.to_numpy() < 5
    hol = HOLIDAYS[kind]
    if hol:
        ok &= ~dt.dt.strftime("%Y-%m-%d").isin(sorted(hol)).to_numpy()
    return ok


def load_perp_px(venue: str, bn_symbol: str, hl_coin: str,
                 dex: str) -> Optional[pd.DataFrame]:
    """Perp price series: ts(ms), px. BN 1s bars or HL 5m+15m closes."""
    if venue == "BN":
        p = DYN / "bn1s" / f"{bn_symbol}.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        return pd.DataFrame({"ts": col(df, "ts").astype("int64") * 1000,
                             "px": col(df, "px").astype(float),
                             "n": 999})
    stem = str(hl_coin).replace(":", "_")
    frames: List[pd.DataFrame] = []
    for iv in ("15m", "5m"):
        p = DYN / "hl" / f"{stem}_{iv}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            end_of_bar = {"15m": 900_000, "5m": 300_000}[iv]
            frames.append(pd.DataFrame(
                {"ts": col(d, "ts").astype("int64") + end_of_bar,
                 "px": col(d, "c").astype(float),
                 "n": col(d, "trades").astype("int64"),
                 "pri": 0 if iv == "5m" else 1}))
    if not frames:
        return None
    allf = pd.concat(frames).sort_values(["ts", "pri"])
    allf = allf.drop_duplicates("ts", keep="first")
    return cast(pd.DataFrame, allf[["ts", "px", "n"]].reset_index(drop=True))


def load_stock(ticker: str, ccy: str) -> Optional[pd.DataFrame]:
    p = DYN / "ib" / f"{ticker}_1m.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    out = pd.DataFrame({"ts": col(df, "ts").astype("int64") + 60_000,
                        "px": col(df, "c").astype(float)})
    if ccy != "USD":
        fxp = DYN / "ib" / f"fx_USD{ccy}_1m.parquet"
        if not fxp.exists():
            return None
        fx = pd.read_parquet(fxp)
        fxs = pd.DataFrame({"ts": col(fx, "ts").astype("int64") + 60_000,
                            "fx": col(fx, "c").astype(float)})
        out = pd.merge_asof(out.sort_values("ts"), fxs.sort_values("ts"),
                            on="ts", tolerance=3_600_000, direction="backward")
        out["px"] = col(out, "px") / col(out, "fx")
        out = cast(pd.DataFrame, out.dropna(subset=["px"])[["ts", "px"]])
    return out.reset_index(drop=True)


def load_oi_series(bn_symbol: str) -> Optional[pd.DataFrame]:
    p = OI_DIR / f"{bn_symbol}_1h.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return pd.DataFrame({"ts": col(df, "ts").astype("int64"),
                         "oi_usd": col(df, "oi_usd").astype(float)})


def build_a_frame(r: Dict[str, Any], ib_ok: Dict[str, Dict[str, Any]],
                  t0: int, t1: int) -> Optional[pd.DataFrame]:
    """Decision frame for one type-A row: ts, prem_bps (smoothed), prem_raw,
    slot_usd."""
    ticker = str(r["ticker"])
    info = ib_ok.get(ticker)
    if info is None:
        return None
    kind = "JP" if ticker in JP_NAMES else str(r.get("kind") or "EQUITY")
    if kind not in SESSIONS:
        return None
    stock = load_stock(ticker, str(info["ccy"]))
    perp = load_perp_px(str(r["venue"]), str(r.get("bn_symbol") or ""),
                        str(r.get("hl_coin") or ""), str(r.get("dex") or ""))
    if stock is None or perp is None or len(stock) < 500 or len(perp) < 500:
        return None
    # freshness: perp quote must be from the same 5m window (HL 15m-era bars
    # simply yield fewer decision points) and, for HL, from a bar with real
    # trading — a stale thin-name candle close is not a tradeable price.
    tol = 120_000 if r["venue"] == "BN" else 300_000
    m = pd.merge_asof(stock.sort_values("ts"),
                      perp.sort_values("ts").rename(columns={"px": "perp"}),
                      on="ts", tolerance=tol, direction="backward")
    m = cast(pd.DataFrame, m.dropna(subset=["perp"]))
    m = cast(pd.DataFrame, m[col(m, "n") >= 3])
    m = cast(pd.DataFrame, m[(col(m, "ts") >= t0) & (col(m, "ts") <= t1)])
    if len(m) < 500:
        return None
    # 5m decision grid: last obs per bucket kills 1m-level microstructure churn
    m["ts"] = (col(m, "ts") // 300_000) * 300_000 + 300_000
    m = m.drop_duplicates("ts", keep="last")
    m["prem_raw"] = np.log(col(m, "perp") / col(m, "px")) * 1e4
    # session-filter FIRST, then smooth within contiguous blocks only: a 1h
    # median that spans the pre-market or the overnight gap manufactures
    # phantom open dislocations. Full window required -> no signals in the
    # first hour of each session block.
    ok = in_session(col(m, "ts").to_numpy(dtype="int64"), kind)
    m = cast(pd.DataFrame, m[ok]).copy()
    if len(m) < 200:
        return None
    block = (col(m, "ts").diff() > 900_000).cumsum()
    m["prem_bps"] = col(m, "prem_raw").groupby(block).transform(
        lambda s: s.rolling(SMOOTH_BARS, min_periods=SMOOTH_BARS).median())
    # effective half-spread of the perp leg, estimated from 5m close-to-close
    # moves (thin xyz names trade inside a wide book; last-price bounce is not
    # capturable). Charged as 2x per round trip on top of fee cost.
    dperp = np.abs(np.diff(np.log(col(m, "perp").to_numpy(dtype=float)))) * 1e4
    half_spread = max(float(np.median(dperp[dperp > 0])) if len(dperp) else 1.0,
                      1.0)
    m.attrs["half_spread"] = half_spread
    oi = None if r["venue"] == "HL" else load_oi_series(str(r["bn_symbol"]))
    if oi is not None:
        m = pd.merge_asof(m.sort_values("ts"), oi.sort_values("ts"),
                          on="ts", tolerance=7_200_000, direction="backward")
        m["slot_usd"] = col(m, "oi_usd") * OI_CAP
    else:
        m["slot_usd"] = float(r["slot_kusd"]) * 1e3    # HL: snapshot constant
    m["kind"] = kind
    return cast(pd.DataFrame, m.reset_index(drop=True))


def cost_rt_bps(kind: str, venue: str, dex: str,
                stock_px_usd: float) -> float:
    perp = PERP_RT["BN"] if venue == "BN" else PERP_RT.get(dex, 6.0)
    if kind in STOCK_RT_FIXED:
        stock = STOCK_RT_FIXED[kind]
    else:   # US: per-share commission both sides + sell-side regulatory
        stock = 2 * US_COMM_PER_SHARE / max(stock_px_usd, 1.0) * 1e4 \
            + US_SELL_REG_BPS
    return stock + perp


def run_type_a(cand: List[Dict[str, Any]], ib_ok: Dict[str, Dict[str, Any]],
               fb: FundingBook, t0: int, t1: int
               ) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Greedy portfolio sim over merged decision bars. Returns (positions,
    utilization frame)."""
    frames: Dict[str, Dict[str, Any]] = {}
    for r in cand:
        key = f"{r['ticker']}@{r['venue']}"
        df = build_a_frame(r, ib_ok, t0, t1)
        if df is None:
            continue
        dirn = 1 if float(r["apr"]) > 0 else -1     # +1 = short perp/long stock
        venue = "binance" if r["venue"] == "BN" else "hyperliquid"
        cost = cost_rt_bps(str(df["kind"].iloc[0]), str(r["venue"]),
                           str(r.get("dex") or ""),
                           float(col(df, "px").median()))
        cost += 2.0 * float(df.attrs.get("half_spread", 1.0))
        frames[key] = {
            "r": r, "df": df, "dir": dirn, "venue_db": venue, "cost": cost,
            "entry_thr": max(CUSHION * cost, MIN_PREM_BPS),
            # funding-first mode: only for fat positive-funding names (short
            # perp / long stock; the discount side would need stock borrow)
            "fat": dirn == 1 and abs(float(r["apr"])) >= FAT_APR
                and str(df["kind"].iloc[0]) in FAT_KINDS}
    # merged event grid
    events: List[Tuple[int, str, int]] = []       # (ts, key, row_idx)
    for key, f in frames.items():
        ts = col(f["df"], "ts").to_numpy(dtype="int64")
        events.extend((int(t), key, i) for i, t in enumerate(ts))
    events.sort()
    open_pos: Dict[str, Dict[str, Any]] = {}
    cooldown: Dict[str, int] = {}
    stock_used = 0.0
    perp_used = 0.0
    closed: List[Dict[str, Any]] = []
    util_rows: List[Dict[str, Any]] = []
    last_util_day = ""
    for ts, key, i in events:
        f = frames[key]
        df = f["df"]
        prem_s = float(df["prem_bps"].iloc[i])
        if not np.isfinite(prem_s):
            continue
        dirn = f["dir"]
        sig = dirn * prem_s
        if key in open_pos:
            pos = open_pos[key]
            if pos.get("mode") == "fat":
                apr_now = fb.trail7_apr(f["venue_db"], str(f["r"]["ticker"]),
                                        ts) * dirn
                do_exit = apr_now <= FAT_EXIT_APR or sig <= FAT_STOP_BPS
            else:
                do_exit = sig <= EXIT_BPS
            if do_exit and i + FILL_LAG_BARS < len(df):
                j = i + FILL_LAG_BARS
                xts = int(df["ts"].iloc[j])
                xprem = float(df["prem_bps"].iloc[j])
                fnd = fb.accrue(f["venue_db"], str(f["r"]["ticker"]),
                                pos["t_in"], xts) * dirn * 1e4
                hold_d = (xts - pos["t_in"]) / 86_400_000
                borrow = (BORROW_APR * 100 / 365 * hold_d
                          if dirn == -1 else 0.0)
                gross = dirn * (pos["prem_in"] - xprem)
                net = gross + fnd - f["cost"] - borrow
                closed.append({
                    "type": "A", "key": key, "mode": pos.get("mode", "prem"),
                    "t_in": pos["t_in"], "t_out": xts,
                    "days": round(hold_d, 2), "notional": pos["notional"],
                    "prem_in": round(pos["prem_in"], 1),
                    "prem_out": round(xprem, 1), "dir": dirn,
                    "basis_bps": round(gross, 1), "fund_bps": round(fnd, 1),
                    "cost_bps": f["cost"], "net_bps": round(net, 1),
                    "pnl_usd": round(net / 1e4 * pos["notional"], 0)})
                stock_used -= pos["notional"]
                perp_used -= pos["notional"]
                cooldown[key] = xts + COOLDOWN_MS
                del open_pos[key]
        else:
            if i + FILL_LAG_BARS >= len(df) or ts < cooldown.get(key, 0):
                continue
            apr_tr = fb.trail7_apr(f["venue_db"], str(f["r"]["ticker"]),
                                   ts) * dirn
            mode = ""
            if sig >= f["entry_thr"] and apr_tr >= MIN_TRAIL_APR:
                mode = "prem"
            elif f["fat"] and apr_tr >= FAT_ENTRY_APR \
                    and sig >= FAT_PREM_FLOOR * f["cost"]:
                mode = "fat"     # funding-first: basis covers the round trip
            if mode:
                j = i + FILL_LAG_BARS
                slot = float(df["slot_usd"].iloc[j]) if np.isfinite(
                    float(df["slot_usd"].iloc[j])) else 0.0
                room = min(CAPITAL_STOCK - stock_used,
                           CAPITAL_PERP - perp_used)
                notional = min(slot, CAP_PER_NAME, room)
                if notional < MIN_TICKET:
                    continue
                open_pos[key] = {
                    "t_in": int(df["ts"].iloc[j]),
                    "prem_in": float(df["prem_bps"].iloc[j]),
                    "notional": notional, "mode": mode}
                stock_used += notional
                perp_used += notional
        day = datetime.fromtimestamp(ts / 1000, timezone.utc).strftime(
            "%Y-%m-%d")
        if day != last_util_day:
            util_rows.append({"day": day, "stock_used": stock_used,
                              "n_open": len(open_pos)})
            last_util_day = day
    # force-close remaining at last bar
    for key, pos in list(open_pos.items()):
        f = frames[key]
        df = f["df"]
        xts = int(df["ts"].iloc[-1])
        xprem = float(df["prem_bps"].iloc[-1])
        dirn = f["dir"]
        fnd = fb.accrue(f["venue_db"], str(f["r"]["ticker"]),
                        pos["t_in"], xts) * dirn * 1e4
        hold_d = (xts - pos["t_in"]) / 86_400_000
        borrow = BORROW_APR * 100 / 365 * hold_d if dirn == -1 else 0.0
        gross = dirn * (pos["prem_in"] - xprem)
        net = gross + fnd - f["cost"] - borrow
        closed.append({"type": "A", "key": key, "mode": pos.get("mode", "prem"),
                       "t_in": pos["t_in"],
                       "t_out": xts, "days": round(hold_d, 2),
                       "notional": pos["notional"],
                       "prem_in": round(pos["prem_in"], 1),
                       "prem_out": round(xprem, 1), "dir": dirn,
                       "basis_bps": round(gross, 1), "fund_bps": round(fnd, 1),
                       "cost_bps": f["cost"], "net_bps": round(net, 1),
                       "pnl_usd": round(net / 1e4 * pos["notional"], 0),
                       "open_at_end": True})
    return closed, pd.DataFrame(util_rows)


def run_type_b(cand: List[Dict[str, Any]], fb: FundingBook,
               t0: int, t1: int) -> List[Dict[str, Any]]:
    """Daily-grid sim of BN-vs-HL funding-spread positions (no stock leg)."""
    closed: List[Dict[str, Any]] = []
    for r in cand:
        ticker = str(r["ticker"])
        bn_px = load_perp_px("BN", str(r["bn_symbol"]), "", "")
        hl_px = load_perp_px("HL", "", str(r["hl_coin"]), str(r["dex"]))
        if bn_px is None or hl_px is None:
            continue
        m = pd.merge_asof(
            hl_px.rename(columns={"px": "hl"}).sort_values("ts"),
            bn_px.rename(columns={"px": "bn"}).sort_values("ts"),
            on="ts", tolerance=600_000, direction="backward")
        m = cast(pd.DataFrame, m.dropna())
        m = cast(pd.DataFrame, m[(col(m, "ts") >= t0) & (col(m, "ts") <= t1)])
        if len(m) < 200:
            continue
        m["sprd_raw"] = np.log(col(m, "bn") / col(m, "hl")) * 1e4
        m["sprd"] = col(m, "sprd_raw").rolling(6, min_periods=2).median()
        m = cast(pd.DataFrame, m.dropna(subset=["sprd"]))
        cost = PERP_RT["BN"] + PERP_RT.get(str(r["dex"]), 9.0)
        for leg in ("bn", "hl"):
            dl = np.abs(np.diff(np.log(col(m, leg).to_numpy(dtype=float)))) * 1e4
            cost += 2.0 * max(float(np.median(dl[dl > 0])) if len(dl) else 1.0,
                              1.0)
        slot = float(r["slot_kusd"]) * 1e3
        notional = min(slot, CAP_PER_NAME)
        pos: Optional[Dict[str, Any]] = None
        ts_arr = col(m, "ts").to_numpy(dtype="int64")
        sp_arr = col(m, "sprd").to_numpy(dtype=float)
        day_marks = np.searchsorted(
            ts_arr, np.arange(t0, t1, 21_600_000))          # 6h grid
        for k in day_marks:
            if k >= len(ts_arr):
                break
            ts = int(ts_arr[k])
            gap = (fb.trail7_apr("binance", ticker, ts)
                   - fb.trail7_apr("hyperliquid", ticker, ts))
            if pos is None:
                if abs(gap) >= B_ENTRY_APR and notional >= MIN_TICKET:
                    pos = {"t_in": ts, "dir": 1 if gap < 0 else -1,
                           "sprd_in": float(sp_arr[k])}
                    # dir=+1: short HL / long BN (HL funding richer)
            else:
                if abs(gap) <= B_EXIT_APR or np.sign(gap) == pos["dir"]:
                    fnd = (fb.accrue("hyperliquid", ticker, pos["t_in"], ts)
                           - fb.accrue("binance", ticker, pos["t_in"], ts)
                           ) * pos["dir"] * 1e4
                    basis = pos["dir"] * (float(sp_arr[k]) - pos["sprd_in"])
                    net = fnd + basis - cost
                    closed.append({
                        "type": "B", "key": ticker, "t_in": pos["t_in"],
                        "t_out": ts,
                        "days": round((ts - pos["t_in"]) / 86_400_000, 2),
                        "notional": notional, "dir": pos["dir"],
                        "basis_bps": round(basis, 1), "fund_bps": round(fnd, 1),
                        "cost_bps": cost, "net_bps": round(net, 1),
                        "pnl_usd": round(net / 1e4 * notional, 0)})
                    pos = None
        if pos is not None:
            ts = int(ts_arr[-1])
            fnd = (fb.accrue("hyperliquid", ticker, pos["t_in"], ts)
                   - fb.accrue("binance", ticker, pos["t_in"], ts)
                   ) * pos["dir"] * 1e4
            basis = pos["dir"] * (float(sp_arr[-1]) - pos["sprd_in"])
            net = fnd + basis - cost
            closed.append({"type": "B", "key": ticker, "t_in": pos["t_in"],
                           "t_out": ts,
                           "days": round((ts - pos["t_in"]) / 86_400_000, 2),
                           "notional": notional, "dir": pos["dir"],
                           "basis_bps": round(basis, 1),
                           "fund_bps": round(fnd, 1), "cost_bps": cost,
                           "net_bps": round(net, 1),
                           "pnl_usd": round(net / 1e4 * notional, 0),
                           "open_at_end": True})
    return closed


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(str(r[c]) for c in cols) + " |\n"
                          for _, r in df.iterrows())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t1 = (int(time.time() * 1000) // 86_400_000) * 86_400_000   # today 00:00
    t0 = t1 - WINDOW_DAYS * 86_400_000
    cand = json.loads((DYN / "candidates.json").read_text())
    ib_map = json.loads((DYN / "ib_map.json").read_text())
    ib_ok = {r["ticker"]: r for r in ib_map if r.get("status") == "ok"}
    fb = FundingBook()
    a_closed, util = run_type_a(cand["type_a"], ib_ok, fb, t0, t1)
    b_closed = run_type_b(cand["type_b"], fb, t0, t1)
    allp = pd.DataFrame(a_closed + b_closed)
    if allp.empty:
        print("no positions")
        return
    allp["t_in"] = pd.to_datetime(col(allp, "t_in"), unit="ms", utc=True)
    allp["t_out"] = pd.to_datetime(col(allp, "t_out"), unit="ms", utc=True)
    allp.to_csv(OUT_DIR / "dyn_positions.csv", index=False)
    for typ, g in allp.groupby("type"):
        tot = float(col(g, "pnl_usd").sum())
        print(f"\n## Type {typ}: {len(g)} positions, total PnL "
              f"${tot:,.0f} over {WINDOW_DAYS}d")
        agg = g.groupby("key").agg(
            n=("net_bps", "size"), notional_avg=("notional", "mean"),
            days=("days", "sum"), basis=("basis_bps", "sum"),
            fund=("fund_bps", "sum"), net_bps=("net_bps", "sum"),
            pnl_usd=("pnl_usd", "sum")).round(1).reset_index()
        print(md(agg.sort_values("pnl_usd", ascending=False)))
    tot = float(col(allp, "pnl_usd").sum())
    ann = tot / (WINDOW_DAYS / 365) / 10_000_000 * 100
    print(f"\n## TOTAL: ${tot:,.0f} on $10M over {WINDOW_DAYS}d "
          f"=> {ann:.1f}% annualized")
    if len(util):
        print("\n## utilization (daily snapshots)")
        u = cast(pd.DataFrame, util.tail(30)).copy()
        u["stock_used_m"] = (col(u, "stock_used") / 1e6).round(2)
        print(md(cast(pd.DataFrame, u[["day", "stock_used_m", "n_open"]])))


if __name__ == "__main__":
    main()
