#!/usr/bin/env python3
"""Item 2: 12:30 UTC macro prints (NFP/CPI/PPI/PCE-GDP/retail) and 18:00 UTC FOMC.

At 12:30 UTC the perp and ES trade but cash equities do not (premarket only).
For each event x symbol: perp and ES return over 12:30->12:45, ES-implied move
(beta_RTH x ES), residual, and what happens to the residual by 13:30 (cash open) and
in the first RTH hour.  Reaction beta (pooled over events) vs normal RTH beta and vs
the same 12:30-12:45 window on non-event days.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, cast

import numpy as np
import pandas as pd

import scan_events_common as C


def _events() -> List[Dict[str, str]]:
    ev = [{"kind": k, "date": d, "hm": "12:30"} for k, ds in C.MACRO_1230.items() for d in ds]
    ev += [{"kind": "FOMC", "date": d, "hm": "18:00"} for d in C.FOMC_1800]
    return ev


def _control_days(idx: pd.DatetimeIndex) -> List[str]:
    ev = {e["date"] for e in _events()}
    hol = {str(h.date()) for h in C.US_HOLIDAYS}
    return [str(d.date()) for d in C.trading_days(idx, C.US_HOLIDAYS)
            if str(d.date()) not in ev and str(d.date()) not in hol]


def event_rows(sym: str, es: pd.Series, beta_rth: float) -> List[Dict[str, float | str]]:
    s = C.load_close(sym)
    rows: List[Dict[str, float | str]] = []
    ctrl = _control_days(C.dtidx(s))
    items = [(e["kind"], e["date"], e["hm"]) for e in _events()]
    items += [("CTRL1230", d, "12:30") for d in ctrl]
    items += [("CTRL1800", d, "18:00") for d in ctrl]
    for kind, d, hm in items:
        t0 = C.utc(d, hm)
        if t0 < C.idx_min(s) or t0 > C.idx_max(s) or t0.dayofweek >= 5:
            continue
        g = C.mins(10)
        r: Dict[str, float | str] = {"sym": sym, "kind": kind, "date": d}
        r["p15"] = C.ret(s, t0, t0 + C.mins(15), g)
        r["es15"] = C.ret(es, t0, t0 + C.mins(15), g)
        r["p5"] = C.ret(s, t0, t0 + C.mins(5), g)
        r["es5"] = C.ret(es, t0, t0 + C.mins(5), g)
        r["p_pre"] = C.ret(s, t0 - C.mins(30), t0, g)
        r["es_pre"] = C.ret(es, t0 - C.mins(30), t0, g)
        if hm == "12:30":
            r["p_to_open"] = C.ret(s, t0 + C.mins(15), C.utc(d, "13:30"), g)
            r["es_to_open"] = C.ret(es, t0 + C.mins(15), C.utc(d, "13:30"), g)
            r["p_open1h"] = C.ret(s, C.utc(d, "13:30"), C.utc(d, "14:30"), g)
            r["es_open1h"] = C.ret(es, C.utc(d, "13:30"), C.utc(d, "14:30"), g)
        else:
            r["p_to_open"] = C.ret(s, t0 + C.mins(15), t0 + C.mins(60), g)
            r["es_to_open"] = C.ret(es, t0 + C.mins(15), t0 + C.mins(60), g)
            r["p_open1h"] = C.ret(s, t0 + C.mins(60), C.utc(d, "20:00"), g)
            r["es_open1h"] = C.ret(es, t0 + C.mins(60), C.utc(d, "20:00"), g)
        r["beta_rth"] = beta_rth
        rows.append(r)
    return rows


def summarize(df: pd.DataFrame, label: str) -> None:
    """Reaction beta, residual, and residual follow-through for one event group."""
    d = df.dropna(subset=["p15", "es15"]).copy()
    if len(d) < 10:
        print(f"  {label}: n={len(d)} (too few)")
        return
    def c(name: str) -> pd.Series:
        return C.col(d, name)

    b15 = C.ols(c("es15"), c("p15"), intercept=False, cluster=c("date"))
    b5 = C.ols(c("es5"), c("p5"), intercept=False, cluster=c("date"))
    # ratio of realised reaction to RTH-beta-implied reaction (pooled)
    d["implied"] = c("beta_rth") * c("es15")
    d["resid"] = c("p15") - c("implied")
    ratio = C.ols(c("implied"), c("p15"), intercept=False, cluster=c("date"))
    fo = C.ols(c("resid"), c("p_to_open"), cluster=c("date"))
    fo2 = C.ols(c("resid"), c("p_open1h"), cluster=c("date"))
    # is the perp's total move to 13:30 more or less than the ES-implied?
    d["p_tot"] = c("p15") + c("p_to_open")
    d["es_tot"] = c("es15") + c("es_to_open")
    btot = C.ols(c("es_tot"), c("p_tot"), intercept=False, cluster=c("date"))
    print(f"  {label:9s} n={len(d):5d} ev={d['date'].nunique():3d} | 5m beta {b5.slope:+.2f} "
          f"| 15m beta {b15.slope:+.2f} (t {b15.t:.1f}) vs RTH beta med {d['beta_rth'].median():.2f}"
          f" | react/implied {ratio.slope:+.2f} | resid->13:30 slope {fo.slope:+.2f} (t {fo.t:+.1f})"
          f" resid->open1h {fo2.slope:+.2f} (t {fo2.t:+.1f}) | to-13:30 beta {btot.slope:+.2f}"
          f" | |ES15| med {d['es15'].abs().median() * 1e4:.0f} bps")


def main() -> None:
    es = C.fut_close("ES")
    syms = C.universe(min_days=20)
    rows: List[Dict[str, float | str]] = []
    b15s: Dict[str, float] = {}
    for sym in syms:
        s = C.load_close(sym)
        b, _ = C.beta_5m(s, es, C.rth_mask, min_n=500)
        # 15m-return RTH beta (control for Epps/staleness in the 5m beta)
        s15 = C.at_minute_multiple(s, 15)
        es15 = C.at_minute_multiple(es, 15)
        b15s[sym], _ = C.beta_5m(s15, es15, C.rth_mask, min_n=150)
        rows.extend(event_rows(sym, es, b))
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c not in ("sym", "kind", "date"):
            df[c] = pd.to_numeric(df[c])
    df["beta15_rth"] = C.col(df, "sym").map(b15s)
    print(f"symbols {df['sym'].nunique()}, rows {len(df)}")
    print(f"  median RTH beta: 5m {df.groupby('sym')['beta_rth'].first().median():.2f}  "
          f"15m {df.groupby('sym')['beta15_rth'].first().median():.2f}")
    print("\n== 2a. 12:30 UTC prints: perp reaction vs ES (pooled across symbols, cluster by date) ==")
    print("  columns: 5m/15m realised beta of perp on ES over the reaction window; RTH beta = "
          "normal 5m beta; react/implied = slope of perp15 on (beta_rth*ES15) (1 = perp moves "
          "exactly its RTH beta); resid->13:30 = slope of 12:45->13:30 perp move on the residual "
          "(negative = the over/under-reaction is corrected before cash open)")
    for k in ["NFP", "CPI", "PPI", "PCE/GDP", "RETAIL", "CTRL1230"]:
        summarize(C.sel(df, df["kind"] == k), k)
    big = C.sel(df, C.col(df, "kind").isin(["NFP", "CPI", "PPI", "PCE/GDP", "RETAIL"]))
    summarize(big, "ALL1230")
    # only events where ES actually moved
    m = C.col(big, "es15").abs() > 0.0015
    summarize(C.sel(big, m), "ALL|ES>15")
    # economic size of the correction: fade the residual from 12:45 to 13:30
    d = C.sel(big, m).dropna(subset=["p15", "es15", "p_to_open"]).copy()
    d["resid"] = C.col(d, "p15") - C.col(d, "beta15_rth") * C.col(d, "es15")
    d["fade"] = -np.sign(C.col(d, "resid")) * C.col(d, "p_to_open")
    big_r = C.sel(d, C.col(d, "resid").abs() > 0.003)
    print(f"  fade-the-residual (|ES15|>15bps): mean|resid| {d['resid'].abs().mean() * 1e4:.0f} bps, "
          f"fade pnl 12:45->13:30 mean {d['fade'].mean() * 1e4:+.1f} bps hit {(d['fade'] > 0).mean():.2f} "
          f"(n {len(d)}); |resid|>30bps: n {len(big_r)} fade {big_r['fade'].mean() * 1e4:+.1f} bps "
          f"hit {(big_r['fade'] > 0).mean():.2f}")
    ctl = C.sel(df, df["kind"] == "CTRL1230").dropna(subset=["p15", "es15", "p_to_open"]).copy()
    ctl["resid"] = C.col(ctl, "p15") - C.col(ctl, "beta15_rth") * C.col(ctl, "es15")
    ctl["fade"] = -np.sign(C.col(ctl, "resid")) * C.col(ctl, "p_to_open")
    cr = C.sel(ctl, C.col(ctl, "resid").abs() > 0.003)
    print(f"  same fade rule on CONTROL days (no print): |resid|>30bps n {len(cr)} fade "
          f"{cr['fade'].mean() * 1e4:+.1f} bps hit {(cr['fade'] > 0).mean():.2f}; all n {len(ctl)} "
          f"fade {ctl['fade'].mean() * 1e4:+.1f} bps -> bid/ask bounce baseline")
    print("  per-event fade (|resid|>30bps): " + "; ".join(
        f"{d0}: n{len(g)} {g['fade'].mean() * 1e4:+.0f}bps"
        for d0, g in big_r.groupby("date")))
    print("\n  per-event: ES 15m move, cross-sectional median perp move, react/implied slope")
    for key, g in big.groupby(["kind", "date"]):
        k, d0 = cast(Tuple[str, str], key)
        r = C.ols(C.col(g, "beta_rth") * C.col(g, "es15"), C.col(g, "p15"), intercept=False)
        fo = C.ols(C.col(g, "p15") - C.col(g, "beta_rth") * C.col(g, "es15"), C.col(g, "p_to_open"))
        print(f"    {d0} {k:8s} ES15 {C.fmt_bps(float(C.col(g, 'es15').iloc[0]))} bps ES->13:30 "
              f"{C.fmt_bps(float(C.col(g, 'es_to_open').iloc[0]))} | perp15 med {g['p15'].median() * 1e4:+.0f}"
              f" | react/implied {r.slope:+.2f} (n {r.n}) | resid->13:30 {fo.slope:+.2f} (t {fo.t:+.1f})")
    print("\n== 2b. FOMC 18:00 UTC (both cash and perp open) vs same-time control ==")
    summarize(C.sel(df, df["kind"] == "FOMC"), "FOMC")
    summarize(C.sel(df, df["kind"] == "CTRL1800"), "CTRL1800")
    for d0, g in C.sel(df, df["kind"] == "FOMC").groupby("date"):
        r = C.ols(C.col(g, "beta_rth") * C.col(g, "es15"), C.col(g, "p15"), intercept=False)
        print(f"    {d0} FOMC ES15 {C.fmt_bps(float(C.col(g, 'es15').iloc[0]))} bps ES 18:15-19:00 "
              f"{C.fmt_bps(float(C.col(g, 'es_to_open').iloc[0]))} | perp15 med {g['p15'].median() * 1e4:+.0f}"
              f" | react/implied {r.slope:+.2f}")
    print("\n== 2c. index perps only (SPY/QQQ/IWM/TQQQ/SOXL) at 12:30 events ==")
    idx = C.sel(big, C.col(big, "sym").isin(["SPY", "QQQ", "IWM", "TQQQ", "SOXL", "SQQQ"]))
    for sym, g in idx.groupby("sym"):
        b = C.ols(C.col(g, "es15"), C.col(g, "p15"), intercept=False)
        fo = C.ols(C.col(g, "p15") - C.col(g, "beta_rth") * C.col(g, "es15"), C.col(g, "p_to_open"))
        print(f"    {sym:5s} n={len(g):3d} 15m beta {b.slope:+.2f} (RTH {C.col(g, 'beta_rth').iloc[0]:.2f}) "
              f"resid->13:30 {fo.slope:+.2f} (t {fo.t:+.1f})")


if __name__ == "__main__":
    main()
