#!/usr/bin/env python3
"""Dynamic cash-and-carry timing study: Binance stock-perp premium vs funding.

Motivating live trade: skfunding SK Hynix carry (2026-07-29 -> 08-21, +$187k on
$5.3M: funding $164k + entry/exit basis $27.5k). Question: does a dynamic rule
(enter carry when perp premium / funding is high, exit when basis dies or
funding flips) beat static hold, net of per-cycle stock-leg friction
(KR sell tax ~20 bps dominates)?

Data: Binance premiumIndexKlines 5m (perp vs index; during KRX open the index
is the vendor-priced home line, so premium = true basis at decision times),
cached to data/prem/<SYMBOL>.parquet; funding history from perpfund.db
(~/app/stocka/data/perpfund.db, synced by the perpfund tracker).

Outputs markdown tables on stdout + docs/scan/dyncarry_cycles.csv.
"""
from __future__ import annotations

import json
import sqlite3
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
PREM_DIR = ROOT / "data" / "prem"
OUT_DIR = ROOT / "docs" / "scan"
DB = Path.home() / "app" / "stocka" / "data" / "perpfund.db"
KR_PAIRS: List[Tuple[str, str]] = [("SKHYNIX", "SKHYNIXUSDT"),
                                   ("SAMSUNG", "SAMSUNGUSDT")]
LIST_MS = 1_780_358_400_000        # 2026-06-02 UTC, KR names listed
KRX_END_MIN = 380                  # 06:20 UTC
# KRX-closed weekdays inside sample: 6/3 local-election holiday, 7/17 holiday
# (events_calendar.md B4), 8/17 Liberation Day observed.
KR_HOLIDAYS = {"2026-06-03", "2026-07-17", "2026-08-17"}
ENTRY_BPS = (10.0, 20.0, 30.0, 50.0)
EXIT_BPS = (0.0, -10.0)
# Per-cycle round-trip cost of one-leg notional, bps: stock leg (KR commission
# ~2x2 + sell-side tax ~20) + perp leg (maker 0 / taker 4x2).
COST_LO, COST_HI = 25.0, 43.0


def col(df: pd.DataFrame, name: str) -> pd.Series:
    return cast(pd.Series, df[name])


def fetch_premium(symbol: str, start_ms: int) -> pd.DataFrame:
    """5m premiumIndexKlines from fapi, cached+incremental."""
    PREM_DIR.mkdir(parents=True, exist_ok=True)
    path = PREM_DIR / f"{symbol}.parquet"
    old: Optional[pd.DataFrame] = None
    start = start_ms
    if path.exists():
        old = pd.read_parquet(path)
        start = int(old["ts"].max()) + 1
    rows: List[List[object]] = []
    while True:
        url = (f"https://fapi.binance.com/fapi/v1/premiumIndexKlines?symbol="
               f"{symbol}&interval=5m&startTime={start}&limit=1500")
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                data = json.load(resp)
        except Exception as ex:  # noqa: BLE001
            print(f"  prem {symbol}: {ex!r}")
            break
        if not data:
            break
        rows.extend(data)
        if len(data) < 1500:
            break
        start = int(data[-1][0]) + 1
        time.sleep(0.25)
    new = pd.DataFrame()
    if rows:
        raw = pd.DataFrame(rows).iloc[:, [0, 4]]
        raw.columns = ["ts", "prem"]
        new = raw.astype({"ts": "int64", "prem": float})
    if old is not None:
        new = pd.concat([old, new]) if len(new) else old
    new = new.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    new.to_parquet(path, index=False)
    return new


def funding_series(ticker: str, venue: str = "binance") -> pd.DataFrame:
    conn = sqlite3.connect(DB)
    df = pd.read_sql(
        "SELECT ts, rate FROM funding WHERE venue=? AND ticker=? ORDER BY ts",
        conn, params=(venue, ticker))
    conn.close()
    return df.astype({"ts": "int64", "rate": float})


def build_frame(symbol: str) -> pd.DataFrame:
    """5m premium frame with KRX-open decision flag and 1h rolling premium."""
    df = fetch_premium(symbol, LIST_MS)
    df["dt"] = pd.to_datetime(col(df, "ts"), unit="ms", utc=True)
    dt = col(df, "dt")
    minute = dt.dt.hour * 60 + dt.dt.minute
    open_flag = (dt.dt.dayofweek < 5) & (minute < KRX_END_MIN)
    open_flag &= ~dt.dt.strftime("%Y-%m-%d").isin(KR_HOLIDAYS)
    df["krx_open"] = open_flag
    df["prem_bps"] = col(df, "prem") * 1e4
    df["prem_1h"] = col(df, "prem_bps").rolling(12, min_periods=6).mean()
    return df


def prem_funding_relation(df: pd.DataFrame, fund: pd.DataFrame,
                          ticker: str) -> pd.DataFrame:
    """Daily KRX-open premium vs daily funding sum: level corr + lead/lag."""
    d = cast(pd.DataFrame, df[col(df, "krx_open")]).copy()
    d["day"] = col(d, "dt").dt.floor("D")
    daily_prem = cast(pd.Series, d.groupby("day")["prem_bps"].mean())
    f = fund.copy()
    f["day"] = pd.to_datetime(col(f, "ts"), unit="ms", utc=True).dt.floor("D")
    daily_fund = cast(pd.Series, f.groupby("day")["rate"].sum()) * 1e4
    m = pd.DataFrame({"prem": daily_prem, "fund": daily_fund}).dropna()
    rows: List[Dict[str, object]] = []
    for lag in (-2, -1, 0, 1, 2):
        c = float(col(m, "prem").corr(cast(pd.Series, col(m, "fund").shift(-lag))))
        rows.append({"ticker": ticker, "fund_lag_days": lag,
                     "corr": round(c, 3), "n_days": len(m)})
    return pd.DataFrame(rows)


def persistence_table(ticker: str, fund: pd.DataFrame) -> pd.DataFrame:
    """Given trailing-7d funding APR bucket, realized next-7d funding APR."""
    f = fund.copy()
    f["day"] = pd.to_datetime(col(f, "ts"), unit="ms", utc=True).dt.floor("D")
    daily = cast(pd.Series, f.groupby("day")["rate"].sum())
    idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D", tz="UTC")
    daily = daily.reindex(idx, fill_value=0.0)
    trail = daily.rolling(7).sum() / 7 * 365 * 100
    rev = cast(pd.Series, cast(pd.Series, daily[::-1]).rolling(7).sum()[::-1])
    fwd = rev.shift(-1) / 7 * 365 * 100
    m = pd.DataFrame({"trail": trail, "fwd": fwd}).dropna()
    bins = [-1e9, 0, 20, 40, 1e9]
    labels = ["<0%", "0-20%", "20-40%", ">40%"]
    m["bucket"] = pd.cut(col(m, "trail"), bins=bins, labels=labels)
    g = m.groupby("bucket", observed=True)["fwd"].agg(["count", "mean", "median"])
    out = cast(pd.DataFrame, g.round(1)).reset_index()
    out.insert(0, "ticker", ticker)
    return out


def cross_sectional_persistence() -> pd.DataFrame:
    """All BN tickers: Spearman rank corr of trailing-7d vs next-7d funding."""
    conn = sqlite3.connect(DB)
    f = pd.read_sql("SELECT ticker, ts, rate FROM funding WHERE venue='binance'",
                    conn)
    conn.close()
    f["day"] = pd.to_datetime(col(f, "ts"), unit="ms", utc=True).dt.floor("D")
    daily = cast(pd.DataFrame,
                 f.groupby(["ticker", "day"])["rate"].sum().unstack("ticker"))
    daily = daily.fillna(0.0)
    trail = cast(pd.DataFrame, daily.rolling(7).sum())
    fwd = cast(pd.DataFrame, trail.shift(-7))
    rows: List[Dict[str, object]] = []
    for day in trail.index[6:-7:7]:
        a = cast(pd.Series, trail.loc[day])
        b = cast(pd.Series, fwd.loc[day])
        mask = a.notna() & b.notna() & ((a != 0) | (b != 0))
        if int(mask.sum()) < 10:
            continue
        ar = cast(pd.Series, a[mask]).rank()
        br = cast(pd.Series, b[mask]).rank()
        rho = float(ar.corr(br))
        rows.append({"week_of": str(day.date()), "n_names": int(mask.sum()),
                     "spearman": round(rho, 3)})
    return pd.DataFrame(rows)


def run_cycles(df: pd.DataFrame, fund: pd.DataFrame, ticker: str,
               entry: float, exit_thr: float, fund_exit: bool) -> pd.DataFrame:
    """State machine on KRX-open bars: short-perp/long-stock carry cycles."""
    d = cast(pd.DataFrame, df[col(df, "krx_open")]).reset_index(drop=True)
    ts = col(d, "ts").to_numpy(dtype="int64")
    p1h = col(d, "prem_1h").to_numpy(dtype=float)
    prem = col(d, "prem_bps").to_numpy(dtype=float)
    f_ts = col(fund, "ts").to_numpy(dtype="int64")
    f_rate = col(fund, "rate").to_numpy(dtype=float)
    f_day = pd.to_datetime(col(fund, "ts"), unit="ms", utc=True).dt.floor("D")
    daily_fund = cast(pd.Series,
                      fund.assign(day=f_day).groupby("day")["rate"].sum())
    cycles: List[Dict[str, object]] = []
    in_pos = False
    e_i = 0
    for i in range(len(d)):
        if not np.isfinite(p1h[i]):
            continue
        if not in_pos:
            if p1h[i] >= entry:
                in_pos, e_i = True, i
        else:
            sig_exit = p1h[i] <= exit_thr
            if fund_exit:
                day = pd.to_datetime(int(ts[i]), unit="ms", utc=True).floor("D")
                last3 = float(cast(pd.Series, daily_fund.loc[:day]).tail(3).sum())
                sig_exit = sig_exit or (last3 <= 0)
            if sig_exit:
                cycles.append(_close_cycle(ticker, entry, exit_thr, fund_exit,
                                           ts, prem, e_i, i, f_ts, f_rate))
                in_pos = False
    if in_pos:
        c = _close_cycle(ticker, entry, exit_thr, fund_exit, ts, prem,
                         e_i, len(d) - 1, f_ts, f_rate)
        c["open"] = True
        cycles.append(c)
    return pd.DataFrame(cycles)


def _close_cycle(ticker: str, entry: float, exit_thr: float, fund_exit: bool,
                 ts: np.ndarray, prem: np.ndarray, e_i: int, x_i: int,
                 f_ts: np.ndarray, f_rate: np.ndarray) -> Dict[str, object]:
    fnd = float(f_rate[(f_ts > ts[e_i]) & (f_ts <= ts[x_i])].sum()) * 1e4
    basis = float(prem[e_i] - prem[x_i])
    return {"ticker": ticker, "entry_thr": entry, "exit_thr": exit_thr,
            "fund_exit": fund_exit,
            "t_in": pd.Timestamp(int(ts[e_i]), unit="ms", tz="UTC"),
            "t_out": pd.Timestamp(int(ts[x_i]), unit="ms", tz="UTC"),
            "days": round((ts[x_i] - ts[e_i]) / 86_400_000, 1),
            "prem_in": round(float(prem[e_i]), 1),
            "prem_out": round(float(prem[x_i]), 1),
            "basis_bps": round(basis, 1), "funding_bps": round(fnd, 1),
            "gross_bps": round(basis + fnd, 1),
            "net_lo": round(basis + fnd - COST_LO, 1),
            "net_hi": round(basis + fnd - COST_HI, 1), "open": False}


def static_benchmark(df: pd.DataFrame, fund: pd.DataFrame,
                     ticker: str) -> Dict[str, object]:
    """Hold carry from first to last KRX-open bar, one round trip."""
    d = cast(pd.DataFrame, df[col(df, "krx_open")]).reset_index(drop=True)
    ts = col(d, "ts").to_numpy(dtype="int64")
    prem = col(d, "prem_bps").to_numpy(dtype=float)
    fnd = float(col(fund, "rate")[(col(fund, "ts") > int(ts[0]))
                                  & (col(fund, "ts") <= int(ts[-1]))].sum()) * 1e4
    basis = float(prem[0] - prem[-1])
    days = (int(ts[-1]) - int(ts[0])) / 86_400_000
    return {"ticker": ticker, "days": round(days, 1),
            "basis_bps": round(basis, 1), "funding_bps": round(fnd, 0),
            "net_lo": round(basis + fnd - COST_LO, 0),
            "net_hi": round(basis + fnd - COST_HI, 0)}


def summarize(cyc: pd.DataFrame, span_days: float) -> pd.DataFrame:
    def stats(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "cycles": len(g),
            "days_in": round(float(col(g, "days").sum()), 1),
            "in_share": round(float(col(g, "days").sum()) / span_days, 2),
            "basis_sum": round(float(col(g, "basis_bps").sum()), 0),
            "fund_sum": round(float(col(g, "funding_bps").sum()), 0),
            "net_lo_sum": round(float(col(g, "net_lo").sum()), 0),
            "net_hi_sum": round(float(col(g, "net_hi").sum()), 0),
            "worst_net_lo": round(float(col(g, "net_lo").min()), 0),
        })
    return cyc.groupby(["ticker", "entry_thr", "exit_thr", "fund_exit"]).apply(
        stats).reset_index()


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(str(r[c]) for c in cols) + " |\n"
                          for _, r in df.iterrows())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_cycles: List[pd.DataFrame] = []
    for ticker, symbol in KR_PAIRS:
        df = build_frame(symbol)
        fund = funding_series(ticker)
        dtc = col(df, "dt")
        span = (dtc.max() - dtc.min()).total_seconds() / 86_400
        print(f"\n# {ticker}: {dtc.min()} .. {dtc.max()}  bars={len(df)}")
        opens = cast(pd.DataFrame, df[col(df, "krx_open")])
        print(f"KRX-open premium bps: mean {col(opens, 'prem_bps').mean():.1f}  "
              f"std {col(opens, 'prem_bps').std():.1f}  "
              f"p10/p50/p90 {col(opens, 'prem_bps').quantile(0.1):.0f}/"
              f"{col(opens, 'prem_bps').quantile(0.5):.0f}/"
              f"{col(opens, 'prem_bps').quantile(0.9):.0f}")
        print("\n## daily premium vs funding lead/lag\n"
              + md(prem_funding_relation(df, fund, ticker)))
        print("## funding persistence (trailing 7d APR -> next 7d APR)\n"
              + md(persistence_table(ticker, fund)))
        print("## static benchmark (full-sample carry)\n"
              + md(pd.DataFrame([static_benchmark(df, fund, ticker)])))
        for entry in ENTRY_BPS:
            for exit_thr in EXIT_BPS:
                for fexit in (False, True):
                    cyc = run_cycles(df, fund, ticker, entry, exit_thr, fexit)
                    if len(cyc):
                        all_cycles.append(cyc)
        cyc_all = pd.concat([c for c in all_cycles
                             if c["ticker"].iloc[0] == ticker])
        print("## dynamic grid\n" + md(summarize(cyc_all, span)))
    cycles = pd.concat(all_cycles).reset_index(drop=True)
    cycles.to_csv(OUT_DIR / "dyncarry_cycles.csv", index=False)
    best = cast(pd.DataFrame, cycles[
        (cycles["entry_thr"] == 30.0) & (cycles["exit_thr"] == 0.0)
        & (~col(cycles, "fund_exit").astype(bool))]).copy()
    print("\n## cycles detail (entry 30 / exit 0 / no fund-exit)\n" + md(
        cast(pd.DataFrame, best[["ticker", "t_in", "t_out", "days", "prem_in",
                                 "prem_out", "basis_bps", "funding_bps",
                                 "net_lo", "net_hi", "open"]])))
    best["month"] = col(best, "t_in").dt.strftime("%Y-%m")
    monthly = best.groupby(["ticker", "month"]).agg(
        cycles=("days", "size"), days_in=("days", "sum"),
        basis=("basis_bps", "sum"), funding=("funding_bps", "sum"),
        net_lo=("net_lo", "sum"), net_hi=("net_hi", "sum")).round(0).reset_index()
    print("## monthly split (entry 30 / exit 0)\n" + md(monthly))
    xs = cross_sectional_persistence()
    print("\n## cross-sectional persistence, all BN names "
          "(rank corr trailing-7d vs next-7d funding)\n" + md(xs))


if __name__ == "__main__":
    main()
