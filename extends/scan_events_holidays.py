#!/usr/bin/env python3
"""Item 1: market-holiday behaviour of Binance stock perps.

US: NYSE closed 2026-05-25 (Mon), 06-19 (Fri), 07-03 (Fri); CME ES trades until 17:00 UTC
    on those days then reopens 22:00 UTC.  Perp trades 24/7.
KR: KRX closed 2026-06-03 (local election), 07-17 (Constitution Day) for SAMSUNG/SKHYNIX/
    HYUNDAI perps (KRX session 00:00-06:30 UTC).  HK perps: no HKEX holiday in sample.

Outputs (stdout): volume/vol/drift on the holiday, perp-vs-ES beta on the holiday,
reopen regression (does the holiday-period perp move predict or revert at reopen).
"""
from __future__ import annotations

from typing import Dict, List, cast

import numpy as np
import pandas as pd

import scan_events_common as C


def _norm_days(idx: pd.DatetimeIndex, center: pd.Timestamp, k: int = 15) -> List[pd.Timestamp]:
    """Up to k normal weekdays before and after `center` (excluding US holidays)."""
    days = C.trading_days(idx, C.US_HOLIDAYS)
    c = center.tz_localize(None).normalize() if center.tzinfo else center
    before = [d for d in days if d < c][-k:]
    after = [d for d in days if d > c][:k]
    return before + after


def _next_trading_day(d: pd.Timestamp) -> pd.Timestamp:
    hol = {h.normalize() for h in C.US_HOLIDAYS}
    n = d + C.days(1)
    while n.dayofweek >= 5 or n.normalize() in hol:
        n += C.days(1)
    return n


def _prev_trading_day(d: pd.Timestamp) -> pd.Timestamp:
    hol = {h.normalize() for h in C.US_HOLIDAYS}
    n = d - C.days(1)
    while n.dayofweek >= 5 or n.normalize() in hol:
        n -= C.days(1)
    return n


def us_holiday_rows(sym: str, es: pd.Series) -> List[Dict[str, float | str]]:
    """Per (symbol, holiday) metrics + a set of matched normal-day rows for baselines."""
    df = C.load_px(sym)
    s = C.load_close(sym)
    rows: List[Dict[str, float | str]] = []
    for h in C.US_HOLIDAYS:
        if C.idx_min(s) > C.utc(str(h.date())) - C.days(20):
            continue
        d = str(h.date())
        prev = str(_prev_trading_day(h).date())
        nxt = str(_next_trading_day(h).date())
        row: Dict[str, float | str] = {"sym": sym, "hol": d, "kind": "hol"}
        row["vol_rth"] = C.sum_between(df, "quote_vol", C.utc(d, "13:30"), C.utc(d, "20:00"))
        row["rv_rth"] = C.rv_between(s, C.utc(d, "13:30"), C.utc(d, "20:00"))
        row["r_rth"] = C.ret(s, C.utc(d, "13:30"), C.utc(d, "20:00"))
        row["r_es_1330_1700"] = C.ret(es, C.utc(d, "13:30"), C.utc(d, "17:00"))
        row["r_p_1330_1700"] = C.ret(s, C.utc(d, "13:30"), C.utc(d, "17:00"))
        row["r_day"] = C.ret(s, C.utc(d, "00:00"), C.utc(nxt, "00:00"))
        row["r_closed"] = C.ret(s, C.utc(prev, "20:00"), C.utc(nxt, "13:30"))
        row["r_es_closed"] = C.ret(es, C.utc(prev, "20:55"), C.utc(nxt, "13:30"),
                                   C.hours(6))
        row["r_open"] = C.ret(s, C.utc(nxt, "13:30"), C.utc(nxt, "14:30"))
        row["r_rth_next"] = C.ret(s, C.utc(nxt, "13:30"), C.utc(nxt, "20:00"))
        rows.append(row)
        # matched normal days: same metrics with the "closed period" = plain overnight
        for nd in _norm_days(C.dtidx(s), h):
            n2 = str(nd.date())
            p2 = str(_prev_trading_day(nd).date())
            r2: Dict[str, float | str] = {"sym": sym, "hol": d, "kind": "norm", "day": n2}
            r2["vol_rth"] = C.sum_between(df, "quote_vol", C.utc(n2, "13:30"), C.utc(n2, "20:00"))
            r2["rv_rth"] = C.rv_between(s, C.utc(n2, "13:30"), C.utc(n2, "20:00"))
            r2["r_closed"] = C.ret(s, C.utc(p2, "20:00"), C.utc(n2, "13:30"))
            r2["r_es_closed"] = C.ret(es, C.utc(p2, "20:55"), C.utc(n2, "13:30"),
                                      C.hours(6))
            r2["r_open"] = C.ret(s, C.utc(n2, "13:30"), C.utc(n2, "14:30"))
            r2["r_rth_next"] = C.ret(s, C.utc(n2, "13:30"), C.utc(n2, "20:00"))
            rows.append(r2)
    return rows


def holiday_beta(syms: List[str], es: pd.Series) -> pd.DataFrame:
    """5m beta of perp on ES: normal RTH vs holiday 13:30-17:00 (ES open, cash closed)."""
    hol_days = {h.date() for h in C.US_HOLIDAYS}
    out = []
    for sym in syms:
        s = C.load_close(sym)

        def hol_mask(idx: pd.DatetimeIndex) -> np.ndarray:
            m = C.minute_of_day(idx)
            return np.asarray([(t.date() in hol_days) for t in idx]) & np.asarray(
                (m > 13 * 60 + 30) & (m <= 17 * 60))

        def norm_mask(idx: pd.DatetimeIndex) -> np.ndarray:
            return C.rth_mask(idx) & ~np.asarray([(t.date() in hol_days) for t in idx])

        bh, nh = C.beta_5m(s, es, hol_mask, min_n=60)
        bn, nn = C.beta_5m(s, es, norm_mask, min_n=500)
        out.append({"sym": sym, "beta_hol": bh, "n_hol": nh, "beta_rth": bn, "n_rth": nn})
    return pd.DataFrame(out)


def us_section(syms: List[str], es: pd.Series) -> None:
    rows: List[Dict[str, float | str]] = []
    for sym in syms:
        rows.extend(us_holiday_rows(sym, es))
    df = pd.DataFrame(rows)
    hol = cast(pd.DataFrame, df[df["kind"] == "hol"].copy())
    norm = cast(pd.DataFrame, df[df["kind"] == "norm"].copy())
    print(f"US holidays: {C.col(hol, 'sym').nunique()} symbols, {len(hol)} symbol-holidays, "
          f"{len(norm)} matched normal symbol-days")

    # 1a. volume / RV in the would-be RTH window relative to the symbol's normal median
    med = cast(pd.DataFrame, norm.groupby(["sym", "hol"])[["vol_rth", "rv_rth"]].median())
    med = med.rename(columns={"vol_rth": "vol_med", "rv_rth": "rv_med"})
    hol = hol.join(med, on=["sym", "hol"])
    hol["vol_ratio"] = hol["vol_rth"] / hol["vol_med"]
    hol["rv_ratio"] = hol["rv_rth"] / hol["rv_med"]
    print("\n== 1a. holiday 13:30-20:00 UTC (would-be RTH): perp volume & realised vol vs "
          "normal-day median (per symbol), median across symbols ==")
    g = hol.groupby("hol").agg(n=("sym", "size"), vol_ratio=("vol_ratio", "median"),
                              rv_ratio=("rv_ratio", "median"),
                              r_rth_med_bps=("r_rth", lambda v: v.median() * 1e4),
                              r_es_1330_1700_bps=("r_es_1330_1700", lambda v: v.median() * 1e4),
                              r_p_1330_1700_bps=("r_p_1330_1700", lambda v: v.median() * 1e4))
    print(g.round(3).to_string())
    print("  (SPY perp on each holiday: r_p 13:30-17:00 vs ES)")
    for _, r in hol[hol["sym"].isin(["SPY", "QQQ", "IWM"])].iterrows():
        print(f"    {r['hol']} {r['sym']:4s} perp {C.fmt_bps(float(r['r_p_1330_1700']))} bps "
              f"vs ES {C.fmt_bps(float(r['r_es_1330_1700']))} bps; perp full-day "
              f"{C.fmt_bps(float(r['r_day']))} bps; vol_ratio {r['vol_ratio']:.2f}")

    # 1b. beta to ES on holidays
    print("\n== 1b. 5m beta of perp on ES: normal RTH vs holiday 13:30-17:00 (median across syms) ==")
    b = holiday_beta(syms, es)
    print(f"  median beta_rth {b['beta_rth'].median():.2f}  median beta_holiday "
          f"{b['beta_hol'].median():.2f}   (n syms with holiday bars>=60: "
          f"{b['beta_hol'].notna().sum()})")
    for sym in ["SPY", "QQQ", "NVDA", "TSLA", "SOXL"]:
        r = cast(pd.DataFrame, b[b["sym"] == sym])
        if len(r):
            print(f"    {sym:5s} beta_rth {float(C.col(r, 'beta_rth').iloc[0]):.2f} "
                  f"beta_hol {float(C.col(r, 'beta_hol').iloc[0]):.2f}")

    # 1c. reopen regression
    print("\n== 1c. reopen: next-day OPEN(13:30-14:30) / RTH ~ perp move over the closed "
          "period (prev close 20:00 -> reopen 13:30); slope, cluster-t (by holiday) ==")
    for name, d in [("holiday", hol), ("normal (matched days)", norm)]:
        d = d.dropna(subset=["r_closed", "r_open", "r_rth_next"])
        cl = C.col(d, "hol") if name == "holiday" else C.col(d, "day")
        for y in ["r_open", "r_rth_next"]:
            o = C.ols(C.col(d, "r_closed"), C.col(d, y), cluster=cl)
            print(f"  {name:22s} {y:10s} ~ r_closed: n={o.n:4d} slope {o.slope:+.3f} "
                  f"t={o.t:+.1f} R2={o.r2:.3f}")
    # residual vs ES: perp move not explained by ES over the same closed period
    d = hol.dropna(subset=["r_closed", "r_es_closed", "r_open"]).copy()
    d = d.join(C.col(b.set_index("sym"), "beta_rth"), on="sym")
    d["resid"] = C.col(d, "r_closed") - C.col(d, "beta_rth") * C.col(d, "r_es_closed")
    for y in ["r_open", "r_rth_next"]:
        o = C.ols(C.col(d, "resid"), C.col(d, y), cluster=C.col(d, "hol"))
        print(f"  holiday {y:10s} ~ (r_closed - beta*ES_closed): n={o.n} slope {o.slope:+.3f} "
              f"t={o.t:+.1f}")
    # cross-sectional (within-holiday demeaned): idiosyncratic holiday move -> idiosyncratic open
    dm = hol.copy()
    for c in ["r_closed", "r_open", "r_rth_next"]:
        v = cast(pd.Series, pd.to_numeric(C.col(dm, c)))
        dm[c] = v - v.groupby(C.col(dm, "hol")).transform("mean")
    for y in ["r_open", "r_rth_next"]:
        o = C.ols(C.col(dm, "r_closed"), C.col(dm, y))
        print(f"  holiday, within-holiday demeaned: {y:10s} ~ r_closed: n={o.n} slope {o.slope:+.3f} "
              f"t={o.t:+.1f}")
    print("  per-holiday cross-sectional slope of r_open on r_closed:")
    for h, g2 in hol.groupby("hol"):
        o = C.ols(C.col(g2, "r_closed"), C.col(g2, "r_open"))
        o2 = C.ols(C.col(g2, "r_closed"), C.col(g2, "r_rth_next"))
        print(f"    {h}: n={o.n:3d} open slope {o.slope:+.3f} (t {o.t:+.1f}); "
              f"rth slope {o2.slope:+.3f} (t {o2.t:+.1f}); mean r_closed "
              f"{g2['r_closed'].mean() * 1e4:+.0f} bps, mean r_open {g2['r_open'].mean() * 1e4:+.0f}")


def kr_section() -> None:
    print("\n\n== KRX holidays (SAMSUNG/SKHYNIX/HYUNDAI perps; KRX session 00:00-06:30 UTC) ==")
    ewy = C.load_close("EWY")
    for sym in sorted(C.KR):
        df = C.load_px(sym)
        s = C.load_close(sym)
        days = C.trading_days(C.dtidx(s), C.KR_HOLIDAYS)
        for h in C.KR_HOLIDAYS:
            d = str(h.date())
            if C.utc(d) < C.idx_min(s) or C.utc(d) > C.idx_max(s):
                continue
            nxt = [x for x in days if x > h][0]
            prv = [x for x in days if x < h]
            n2 = str(nxt.date())
            vol_h = C.sum_between(df, "quote_vol", C.utc(d, "00:00"), C.utc(d, "06:30"))
            rv_h = C.rv_between(s, C.utc(d, "00:00"), C.utc(d, "06:30"))
            norm = [x for x in days if abs((x - h).days) <= 21 and x != h]
            vols = [C.sum_between(df, "quote_vol", C.utc(str(x.date()), "00:00"),
                                  C.utc(str(x.date()), "06:30")) for x in norm]
            rvs = [C.rv_between(s, C.utc(str(x.date()), "00:00"),
                                C.utc(str(x.date()), "06:30")) for x in norm]
            r_sess = C.ret(s, C.utc(d, "00:00"), C.utc(d, "06:30"))
            r_day = C.ret(s, C.utc(d, "00:00"), C.utc(n2, "00:00"))
            r_open_next = C.ret(s, C.utc(n2, "00:00"), C.utc(n2, "01:00"))
            r_sess_next = C.ret(s, C.utc(n2, "00:00"), C.utc(n2, "06:30"))
            r_ewy = C.ret(ewy, C.utc(d, "13:30"), C.utc(d, "20:00"))
            print(f"  {sym:8s} {d} (n_norm={len(norm)}, first_day={'yes' if not prv else 'no'}): "
                  f"vol_ratio {vol_h / np.median(vols):.2f}  rv_ratio {rv_h / np.nanmedian(rvs):.2f}  "
                  f"perp 00:00-06:30 {C.fmt_bps(r_sess)} bps, full day {C.fmt_bps(r_day)} bps, "
                  f"EWY RTH same day {C.fmt_bps(r_ewy)} bps | next KRX open 1h {C.fmt_bps(r_open_next)} "
                  f"session {C.fmt_bps(r_sess_next)} bps")
    print("  HK perps (HK0700/HK1810/TENCENT/...): listed 2026-07-22; no HKEX holiday until "
          "2026-10-01 -> no sample.")


def main() -> None:
    es = C.fut_close("ES")
    syms = C.universe(min_days=20)
    us_section(syms, es)
    kr_section()


if __name__ == "__main__":
    main()
