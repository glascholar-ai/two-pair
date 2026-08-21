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
CAP_PER_NAME = 1_500_000.0
MIN_TICKET = 50_000.0
CUSHION = 1.5                    # entry premium >= CUSHION x round-trip cost
MIN_PREM_BPS = 20.0
EXIT_BPS = 0.0
MIN_TRAIL_APR = 8.0              # trailing-7d funding APR gate (type A)
B_ENTRY_APR = 15.0               # type B trailing spread entry
B_EXIT_APR = 5.0
BORROW_APR = 1.0                 # % p.a. charged on short-stock legs
FILL_LAG_BARS = 1
SMOOTH_BARS = 5
# Round-trip friction (both directions, one-leg notional bps): stock leg by
# market + perp leg by venue (BN taker 4x2; HL xyz growth 0.9x2, para 4.5x2).
STOCK_RT = {"EQUITY": 6.0, "HK_EQUITY": 26.0, "KR_EQUITY": 24.0, "JP": 6.0}
PERP_RT = {"BN": 8.0, "xyz": 1.8, "para": 9.0}
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
    """Per (venue,ticker) funding events + trailing daily APR lookup."""

    def __init__(self) -> None:
        conn = sqlite3.connect(DB)
        f = pd.read_sql("SELECT venue,ticker,ts,rate FROM funding", conn)
        conn.close()
        self.ev: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
        self.daily: Dict[Tuple[str, str], pd.Series] = {}
        for key_vt, g in f.groupby(["venue", "ticker"]):
            v, t = cast(Tuple[str, str], key_vt)
            g = g.sort_values("ts")
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
                             "px": col(df, "px").astype(float)})
    stem = str(hl_coin).replace(":", "_")
    frames: List[pd.DataFrame] = []
    for iv in ("15m", "5m"):
        p = DYN / "hl" / f"{stem}_{iv}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            end_of_bar = {"15m": 900_000, "5m": 300_000}[iv]
            frames.append(pd.DataFrame(
                {"ts": col(d, "ts").astype("int64") + end_of_bar,
                 "px": col(d, "c").astype(float), "pri": 0 if iv == "5m" else 1}))
    if not frames:
        return None
    allf = pd.concat(frames).sort_values(["ts", "pri"])
    allf = allf.drop_duplicates("ts", keep="first")
    return cast(pd.DataFrame, allf[["ts", "px"]].reset_index(drop=True))


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
    tol = 120_000 if r["venue"] == "BN" else 1_200_000
    m = pd.merge_asof(stock.sort_values("ts"),
                      perp.sort_values("ts").rename(columns={"px": "perp"}),
                      on="ts", tolerance=tol, direction="backward")
    m = cast(pd.DataFrame, m.dropna(subset=["perp"]))
    m = cast(pd.DataFrame, m[(col(m, "ts") >= t0) & (col(m, "ts") <= t1)])
    if len(m) < 500:
        return None
    m["prem_raw"] = np.log(col(m, "perp") / col(m, "px")) * 1e4
    m["prem_bps"] = col(m, "prem_raw").rolling(
        SMOOTH_BARS, min_periods=3).median()
    ok = in_session(col(m, "ts").to_numpy(dtype="int64"), kind)
    m = cast(pd.DataFrame, m[ok])
    if len(m) < 300:
        return None
    oi = None if r["venue"] == "HL" else load_oi_series(str(r["bn_symbol"]))
    if oi is not None:
        m = pd.merge_asof(m.sort_values("ts"), oi.sort_values("ts"),
                          on="ts", tolerance=7_200_000, direction="backward")
        m["slot_usd"] = col(m, "oi_usd") * OI_CAP
    else:
        m["slot_usd"] = float(r["slot_kusd"]) * 1e3    # HL: snapshot constant
    m["kind"] = kind
    return cast(pd.DataFrame, m.reset_index(drop=True))


def cost_rt_bps(kind: str, venue: str, dex: str) -> float:
    perp = PERP_RT["BN"] if venue == "BN" else PERP_RT.get(dex, 9.0)
    return STOCK_RT[kind] + perp


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
        frames[key] = {
            "r": r, "df": df, "dir": dirn, "venue_db": venue,
            "cost": cost_rt_bps(str(df["kind"].iloc[0]), str(r["venue"]),
                                str(r.get("dex") or "")),
            "entry_thr": max(CUSHION * cost_rt_bps(
                str(df["kind"].iloc[0]), str(r["venue"]),
                str(r.get("dex") or "")), MIN_PREM_BPS)}
    # merged event grid
    events: List[Tuple[int, str, int]] = []       # (ts, key, row_idx)
    for key, f in frames.items():
        ts = col(f["df"], "ts").to_numpy(dtype="int64")
        events.extend((int(t), key, i) for i, t in enumerate(ts))
    events.sort()
    open_pos: Dict[str, Dict[str, Any]] = {}
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
            if sig <= EXIT_BPS and i + FILL_LAG_BARS < len(df):
                j = i + FILL_LAG_BARS
                xts = int(df["ts"].iloc[j])
                xprem = float(df["prem_raw"].iloc[j])
                fnd = fb.accrue(f["venue_db"], str(f["r"]["ticker"]),
                                pos["t_in"], xts) * dirn * 1e4
                hold_d = (xts - pos["t_in"]) / 86_400_000
                borrow = (BORROW_APR * 100 / 365 * hold_d
                          if dirn == -1 else 0.0)
                gross = dirn * (pos["prem_in"] - xprem)
                net = gross + fnd - f["cost"] - borrow
                closed.append({
                    "type": "A", "key": key, "t_in": pos["t_in"], "t_out": xts,
                    "days": round(hold_d, 2), "notional": pos["notional"],
                    "prem_in": round(pos["prem_in"], 1),
                    "prem_out": round(xprem, 1), "dir": dirn,
                    "basis_bps": round(gross, 1), "fund_bps": round(fnd, 1),
                    "cost_bps": f["cost"], "net_bps": round(net, 1),
                    "pnl_usd": round(net / 1e4 * pos["notional"], 0)})
                stock_used -= pos["notional"]
                perp_used -= pos["notional"]
                del open_pos[key]
        else:
            if sig >= f["entry_thr"] and i + FILL_LAG_BARS < len(df):
                apr_tr = fb.trail7_apr(f["venue_db"], str(f["r"]["ticker"]), ts)
                if dirn * apr_tr < MIN_TRAIL_APR:
                    continue
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
                    "prem_in": float(df["prem_raw"].iloc[j]),
                    "notional": notional}
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
        xprem = float(df["prem_raw"].iloc[-1])
        dirn = f["dir"]
        fnd = fb.accrue(f["venue_db"], str(f["r"]["ticker"]),
                        pos["t_in"], xts) * dirn * 1e4
        hold_d = (xts - pos["t_in"]) / 86_400_000
        borrow = BORROW_APR * 100 / 365 * hold_d if dirn == -1 else 0.0
        gross = dirn * (pos["prem_in"] - xprem)
        net = gross + fnd - f["cost"] - borrow
        closed.append({"type": "A", "key": key, "t_in": pos["t_in"],
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
        m["sprd"] = np.log(col(m, "bn") / col(m, "hl")) * 1e4
        cost = PERP_RT["BN"] + PERP_RT.get(str(r["dex"]), 9.0)
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
