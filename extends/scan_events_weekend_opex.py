#!/usr/bin/env python3
"""Items 4+5: opex / rebalance days, and weekend structure vs the ES Sunday open.

Weekend segments (UTC): FriAH Fri20:00->Sat00:00 | Sat 00:00->24:00 | Sun 00:00->22:00 |
SunCME Sun22:00->Mon00:00 (ES reopens 22:00) | MonPre Mon00:00->13:30.
ES weekend gap = ES(Sun 22:05) - ES(Fri 20:55, close of the last cash-hours bar).
"""
from __future__ import annotations

from typing import Dict, List, cast

import numpy as np
import pandas as pd

import scan_events_common as C

WK = [("FriAH", -1, "20:00", 0, "00:00"), ("Sat", 0, "00:00", 1, "00:00"),
      ("Sun", 1, "00:00", 1, "22:00"), ("SunCME", 1, "22:00", 2, "00:00"),
      ("MonPre", 2, "00:00", 2, "13:30")]


def weekend_rows(sym: str, es: pd.Series) -> List[Dict[str, float | str]]:
    df = C.load_px(sym)
    s = C.load_close(sym)
    rows: List[Dict[str, float | str]] = []
    sats = sorted({t.normalize() for t in C.dtidx(s) if t.dayofweek == 5})
    hol = {h.date() for h in C.US_HOLIDAYS}
    g = C.hours(2)
    for sat in sats:
        fri = sat - C.days(1)
        mon = sat + C.days(2)
        if fri.date() in hol or mon.date() in hol:
            continue  # long weekends handled in the holiday scan
        if C.utc(str(fri.date()), "20:00") < C.idx_min(s) or C.utc(str(mon.date()), "14:30") > C.idx_max(s):
            continue
        r: Dict[str, float | str] = {"sym": sym, "wk": str(sat.date())}
        for name, d0, h0, d1, h1 in WK:
            t0 = C.utc(str((sat + C.days(d0)).date()), h0)
            t1 = C.utc(str((sat + C.days(d1)).date()), h1)
            r["r_" + name] = C.ret(s, t0, t1, g)
            r["v_" + name] = C.sum_between(df, "quote_vol", t0, t1)
            r["rv_" + name] = C.rv_between(s, t0, t1)
        r["v_FriRTH"] = C.sum_between(df, "quote_vol", C.utc(str(fri.date()), "13:30"),
                                      C.utc(str(fri.date()), "20:00"))
        r["r_es_gap"] = C.ret(es, C.utc(str(fri.date()), "20:55"), C.utc(str(sat.date()), "22:05")
                              + C.days(1), C.days(3))
        r["r_es_sun"] = C.ret(es, C.utc(str(mon.date()), "22:05") - C.days(1),
                              C.utc(str(mon.date()), "00:00"), g)
        r["r_es_monpre"] = C.ret(es, C.utc(str(mon.date()), "00:00"), C.utc(str(mon.date()), "13:30"), g)
        r["r_MonOpen"] = C.ret(s, C.utc(str(mon.date()), "13:30"), C.utc(str(mon.date()), "14:30"), g)
        r["r_MonRTH"] = C.ret(s, C.utc(str(mon.date()), "13:30"), C.utc(str(mon.date()), "20:00"), g)
        r["r_es_MonRTH"] = C.ret(es, C.utc(str(mon.date()), "13:30"), C.utc(str(mon.date()), "20:00"), g)
        rows.append(r)
    return rows


def weekend_section(syms: List[str], es: pd.Series, betas: Dict[str, float],
                    liquid: Dict[str, float]) -> None:
    rows: List[Dict[str, float | str]] = []
    for sym in syms:
        rows.extend(weekend_rows(sym, es))
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c not in ("sym", "wk"):
            df[c] = pd.to_numeric(df[c])
    df["beta"] = C.col(df, "sym").map(betas)
    print(f"weekends: {C.col(df, 'wk').nunique()}, symbols {C.col(df, 'sym').nunique()}, rows {len(df)}")
    print("\n== 5a. weekend volume share (quote vol, sum over syms) and per-hour rate vs Fri RTH ==")
    hrs = {"FriAH": 4, "Sat": 24, "Sun": 22, "SunCME": 2, "MonPre": 13.5}
    tot = {k: df["v_" + k].sum() for k in hrs}
    fri_rth = df["v_FriRTH"].sum()
    for k, h in hrs.items():
        print(f"  {k:7s} share of Fri20:00-Mon13:30 vol {tot[k] / sum(tot.values()):.3f}  "
              f"per-hour rate / FriRTH per-hour {(tot[k] / h) / (fri_rth / 6.5):.3f}  "
              f"median RV(bps) {df['rv_' + k].median():.0f}")
    print("\n== 5b. Sun 22:00-24:00 perp return ~ ES weekend gap (Fri close -> Sun 22:05) ==")
    d = df.dropna(subset=["r_es_gap", "r_SunCME", "r_Sat", "r_Sun"])

    def c(name: str) -> pd.Series:
        return C.col(d, name)

    d["r_wkend_pre"] = c("r_FriAH") + c("r_Sat") + c("r_Sun")      # perp move before CME opens
    d["r_wkend_all"] = c("r_wkend_pre") + c("r_SunCME")
    d["es_impl"] = c("beta") * c("r_es_gap")
    for y in ["r_SunCME", "r_wkend_pre", "r_wkend_all", "r_MonPre", "r_MonRTH"]:
        o = C.ols(c("es_impl"), c(y), cluster=c("wk"))
        print(f"  {y:12s} ~ beta*ES_gap: n={o.n} slope {o.slope:+.2f} t={o.t:+.1f} R2={o.r2:.3f}")
    # residual: perp move before CME open minus what ES gap implies -> does SunCME correct it?
    d["resid_pre"] = c("r_wkend_pre") - c("es_impl")
    for y in ["r_SunCME", "r_MonPre", "r_MonRTH"]:
        o = C.ols(c("resid_pre"), c(y), cluster=c("wk"))
        print(f"  {y:12s} ~ (perp Fri20-Sun22 - beta*ES_gap): slope {o.slope:+.2f} t={o.t:+.1f}")
    d["fade"] = -np.sign(c("resid_pre")) * (c("r_SunCME") + c("r_MonPre"))
    d["fade_open"] = -np.sign(c("resid_pre")) * (c("r_SunCME") + c("r_MonPre") + c("r_MonOpen"))
    d["adv"] = c("sym").map(liquid)
    for lab, dd in [("all syms", d), ("ADV>$10M", C.sel(d, c("adv") > 10e6)),
                    ("ADV>$30M", C.sel(d, c("adv") > 30e6))]:
        for thr in [0.0, 0.005, 0.01]:
            g = C.sel(dd, C.col(dd, "resid_pre").abs() > thr)
            by_wk = cast(pd.Series, g.groupby("wk")["fade"].mean())
            print(f"  [{lab:8s}] fade residual at Sun22:00 -> Mon13:30, |resid|>{thr * 1e4:.0f}bps: n={len(g)} "
                  f"mean {g['fade'].mean() * 1e4:+.0f} bps hit {(g['fade'] > 0).mean():.2f} "
                  f"(to 14:30: {g['fade_open'].mean() * 1e4:+.0f}) weekends+ {(by_wk > 0).sum()}/{len(by_wk)}")
    print("\n== 5c. does weekend perp trading predict the ES Sunday open? (index perps) ==")
    for sym in ["SPY", "QQQ", "IWM", "TQQQ", "SOXL"]:
        g = C.sel(d, c("sym") == sym)
        if len(g) < 5:
            continue
        o = C.ols(C.col(g, "r_wkend_pre"), C.col(g, "r_es_gap"))
        o2 = C.ols(C.col(g, "r_es_gap"), C.col(g, "r_SunCME"))
        o3 = C.ols(C.col(g, "r_es_gap"), C.col(g, "r_wkend_pre"))
        print(f"  {sym:5s} n={o.n:2d} ES_gap ~ perp(Fri20->Sun22): slope {o.slope:+.2f} t={o.t:+.1f} "
              f"R2={o.r2:.2f} | perp(Fri20->Sun22) ~ ES_gap slope {o3.slope:+.2f} (RTH beta "
              f"{betas.get(sym, float('nan')):.2f}) | SunCME ~ ES_gap slope {o2.slope:+.2f} t={o2.t:+.1f}")
        print("     per weekend (bps): " + " ".join(
            f"[{w[5:]}: perp {p * 1e4:+.0f}/ES {e * 1e4:+.0f}/Sun22 {c * 1e4:+.0f}]"
            for w, p, e, c in zip(g["wk"], g["r_wkend_pre"], g["r_es_gap"], g["r_SunCME"])))
    # pooled all symbols: cross-sectional demeaned (idiosyncratic) weekend move -> Sun CME / Monday
    print("\n== 5d. pooled: does the pre-CME weekend perp move continue or revert (all syms) ==")
    for y in ["r_SunCME", "r_MonPre", "r_MonOpen", "r_MonRTH"]:
        o = C.ols(c("r_wkend_pre"), c(y), cluster=c("wk"))
        print(f"  {y:9s} ~ perp(Fri20->Sun22): slope {o.slope:+.3f} t={o.t:+.1f}")
    dm = d.copy()
    for k in ["r_wkend_pre", "r_SunCME", "r_MonPre", "r_MonOpen", "r_MonRTH"]:
        dm[k] = C.col(dm, k) - C.col(dm, k).groupby(C.col(dm, "wk")).transform("mean")
    for y in ["r_SunCME", "r_MonPre", "r_MonOpen", "r_MonRTH"]:
        o = C.ols(C.col(dm, "r_wkend_pre"), C.col(dm, y), cluster=C.col(dm, "wk"))
        print(f"  demeaned {y:9s} ~ perp(Fri20->Sun22): slope {o.slope:+.3f} t={o.t:+.1f}")


def opex_section(syms: List[str], es: pd.Series) -> None:
    print("\n\n== 4. opex / rebalance days: last-30m into 20:00, AH, DEAD; vs normal Fridays ==")
    rows: List[Dict[str, float | str]] = []
    for sym in syms:
        df = C.load_px(sym)
        s = C.load_close(sym)
        days = C.trading_days(C.dtidx(s), C.US_HOLIDAYS)
        for dd in days:
            d = str(dd.date())
            kind = C.OPEX.get(d)
            if kind is None and dd.dayofweek != 4:
                continue
            nxt = [x for x in days if x > dd]
            if not nxt:
                continue
            n2 = str(nxt[0].date())
            g = C.mins(30)
            r: Dict[str, float | str] = {"sym": sym, "day": d, "kind": kind or "fri"}
            r["last30"] = C.ret(s, C.utc(d, "19:30"), C.utc(d, "20:00"), g)
            r["last5"] = C.ret(s, C.utc(d, "19:55"), C.utc(d, "20:00"), g)
            r["ah"] = C.ret(s, C.utc(d, "20:00"), C.utc(n2, "00:00") if nxt[0] - dd == C.days(1)
                            else C.utc(d, "23:59") + C.mins(1), g)
            r["ah1h"] = C.ret(s, C.utc(d, "20:00"), C.utc(d, "21:00"), g)
            r["to_open"] = C.ret(s, C.utc(d, "20:00"), C.utc(n2, "13:30"), C.hours(3))
            r["next_open"] = C.ret(s, C.utc(n2, "13:30"), C.utc(n2, "14:30"), g)
            r["v_last30"] = C.sum_between(df, "quote_vol", C.utc(d, "19:30"), C.utc(d, "20:00"))
            r["v_rth"] = C.sum_between(df, "quote_vol", C.utc(d, "13:30"), C.utc(d, "20:00"))
            r["v_ah"] = C.sum_between(df, "quote_vol", C.utc(d, "20:00"), C.utc(d, "23:59"))
            r["es_last30"] = C.ret(es, C.utc(d, "19:30"), C.utc(d, "20:00"), g)
            r["es_ah"] = C.ret(es, C.utc(d, "20:00"), C.utc(d, "21:00"), g)
            rows.append(r)
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c not in ("sym", "day", "kind"):
            df[c] = pd.to_numeric(df[c])
    ev_days = C.col(C.sel(df, df["kind"] != "fri"), "day").unique()
    print(f"  rows {len(df)}, days: " + ", ".join(sorted(ev_days)))
    for kind, g in df.groupby("kind"):
        o1 = C.ols(C.col(g, "last30"), C.col(g, "ah1h"), cluster=C.col(g, "day"))
        o2 = C.ols(C.col(g, "last30"), C.col(g, "to_open"), cluster=C.col(g, "day"))
        o3 = C.ols(C.col(g, "last5"), C.col(g, "ah1h"), cluster=C.col(g, "day"))
        print(f"  {kind:26s} n={len(g):4d} days={C.col(g, 'day').nunique():2d} | last30 vol share of RTH "
              f"{(g['v_last30'].sum() / g['v_rth'].sum()):.3f} | AH vol/RTH vol {(g['v_ah'].sum() / g['v_rth'].sum()):.3f}"
              f" | |last30| med {g['last30'].abs().median() * 1e4:.0f} bps |ah1h| med "
              f"{g['ah1h'].abs().median() * 1e4:.0f} | ah1h~last30 slope {o1.slope:+.2f} (t {o1.t:+.1f})"
              f" | to_open~last30 {o2.slope:+.2f} (t {o2.t:+.1f}) | ah1h~last5 {o3.slope:+.2f} (t {o3.t:+.1f})")
    print("  per event day (cross-section):")
    for d0, g in C.sel(df, df["kind"] != "fri").groupby("day"):
        o1 = C.ols(C.col(g, "last30"), C.col(g, "ah1h"))
        o2 = C.ols(C.col(g, "last30"), C.col(g, "to_open"))
        print(f"    {d0} {C.col(g, 'kind').iloc[0]:24s} n={len(g):3d} ES last30 "
              f"{C.fmt_bps(float(C.col(g, 'es_last30').iloc[0]))} "
              f"ES 20-21 {C.fmt_bps(float(C.col(g, 'es_ah').iloc[0]))} | perp last30 med "
              f"{C.col(g, 'last30').median() * 1e4:+.0f} "
              f"ah1h med {C.col(g, 'ah1h').median() * 1e4:+.0f} to_open med {C.col(g, 'to_open').median() * 1e4:+.0f} | "
              f"ah1h~last30 {o1.slope:+.2f} (t {o1.t:+.1f}) to_open~last30 {o2.slope:+.2f} (t {o2.t:+.1f})")


def main() -> None:
    es = C.fut_close("ES")
    syms = C.universe(min_days=20)
    betas: Dict[str, float] = {}
    liquid: Dict[str, float] = {}
    for sym in syms:
        px = C.load_px(sym)
        liquid[sym] = float(px["quote_vol"].sum() / len(px) * 288)
        s = C.load_close(sym)
        s15 = C.at_minute_multiple(s, 15)
        es15 = C.at_minute_multiple(es, 15)
        betas[sym], _ = C.beta_5m(s15, es15, C.rth_mask, min_n=150)
    weekend_section(syms, es, betas, liquid)
    opex_section(syms, es)


if __name__ == "__main__":
    main()
